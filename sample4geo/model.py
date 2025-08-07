import open_clip
import torch
import timm
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from lavis.models import load_model_and_preprocess
from lavis.models.blip2_models.blip2 import disabled_train
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
# from mmpretrain import get_model

class MVCV4GeoTextLLM(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self, img_size=224,freeze_reference=True):
        super(MVCV4GeoTextLLM, self).__init__()

        self.grd_size = img_size
        self.sat_size = img_size
   
        # self.cross_query_model, _, self.cross_query_preprocess = open_clip.create_model_and_transforms("ViT-L-14",pretrained="openai")
        # self.cross_query_tokenizer = open_clip.get_tokenizer("ViT-L-14")

        self.cross_query_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip2_cir_align_prompt", model_type="pretrain_vitL", is_eval=False, device="cuda")

        for name, param in self.cross_query_model.visual_encoder.named_parameters():
            param.requires_grad = False
        self.cross_query_model.visual_encoder = self.cross_query_model.visual_encoder.eval()
        self.cross_query_model.visual_encoder.train = disabled_train
        print("freeze query vision encoder")

        self.cross_reference_model, _, self.cross_reference_preprocess = open_clip.create_model_and_transforms("ViT-L-14")
        self.cross_reference_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        ckpt = torch.load(f"path/to/RemoteCLIP-ViT-L-14.pt", map_location="cpu") 
        msg = self.cross_reference_model.load_state_dict(ckpt)

        if freeze_reference:
            for name, param in self.cross_reference_model.named_parameters():
                param.requires_grad = False
            self.cross_reference_model = self.cross_reference_model.eval()
            self.cross_reference_model.train = disabled_train
            print("freeze cross_reference_model vision encoder")


        # print("参数精度：")
        # for name, param in self.named_parameters():
        #     print(f"Parameter: {name}, dtype: {param.dtype}")

        self.cross_query_model = self.cross_query_model.float()
        # print("参数精度（调整后）：")
        # for name, param in self.named_parameters():
        #     print(f"Parameter: {name}, dtype: {param.dtype}")

        self.query_proj = nn.Linear(self.cross_query_model.visual_encoder.num_features, self.cross_reference_model.visual.output_dim)
        # for name, param in self.query_proj.named_parameters():
        #     param.requires_grad = False
        # self.query_proj = self.query_proj.eval()
        # self.query_proj.train = disabled_train
        # print("freeze query_proj")


        self.hidden_dim = self.cross_reference_model.visual.output_dim

        self.text_dim = 4096

        self.dropout_rate = 0.5


        self.text_fc = nn.Sequential(nn.Linear(self.text_dim, self.hidden_dim),
                                nn.ReLU(),
                                nn.Dropout(self.dropout_rate),
                                nn.Linear(self.hidden_dim, self.hidden_dim))

        self.dropout= nn.Dropout(self.dropout_rate)

        self.combiner_fc = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                         nn.ReLU())

        self.scaler_fc = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Dropout(self.dropout_rate),
                                       nn.Linear(self.hidden_dim, 1),
                                       nn.Sigmoid())
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        trainable_params = [name for name, param in self.named_parameters() if param.requires_grad]

        print("可训练参数：")
        for param in trainable_params:
            print(param)

    def _split_list(self, lst, device):
        if not lst:  # 如果 lst 是空列表
            return []

        # 获取当前设备的索引
        world_size = torch.cuda.device_count()  # 获取 GPU 的数量
        device_id = device.index  # 直接获取 GPU 的索引

        # 计算切分大小
        total_size = len(lst)
        split_size = total_size // world_size  # 每个 GPU 应该处理的样本数量
        remainder = total_size % world_size  # 计算余数

        # 确定当前设备的起始和结束索引
        start_idx = device_id * split_size + min(device_id, remainder)
        end_idx = start_idx + split_size + (1 if device_id < remainder else 0)

        # 切分并返回当前设备的部分
        return lst[start_idx:end_idx]

    def fusion_img_text(self,img,text):
        textual_query = F.normalize(self.text_fc(text), dim=-1)
        visual_query = F.normalize(img, dim=-1)
        combined_feature = self.combiner_fc(torch.cat([textual_query, visual_query], dim=-1))
        dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        query = dynamic_scaler * textual_query + (1 - dynamic_scaler) * visual_query
        return F.normalize(query, dim=-1)

    def forward(self, img1, img2=None,caption=None,mode="grd"):
        if img2 is not None:

            img_features = self.query_proj(self.cross_query_model.encode_images(img1))
            fusion_features = self.fusion_img_text(img_features,caption)

            image_features2 = self.cross_reference_model(img2)[0] #torch.Size([8, 768])
            
            return F.normalize(fusion_features,dim=-1), F.normalize(image_features2,dim=-1)            
              
        else:
            if mode =="grd":
                image_features = self.query_proj(self.cross_query_model.encode_images(img1))
            elif mode=="text":
                img_features = self.query_proj(self.cross_query_model.encode_images(img1))
                image_features = self.fusion_img_text(img_features,caption)
            else:
                image_features = self.cross_reference_model(img1)[0] #torch.Size([8, 768])
             
            return F.normalize(image_features,dim=-1)

     

class MVCV4GeoComplex2(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self, img_size=224,freeze_reference=False):
        super(MVCV4GeoComplex2, self).__init__()

        self.grd_size = img_size
        self.sat_size = img_size
   
        # self.cross_query_model, _, self.cross_query_preprocess = open_clip.create_model_and_transforms("ViT-L-14",pretrained="openai")
        # self.cross_query_tokenizer = open_clip.get_tokenizer("ViT-L-14")

        self.cross_query_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip2_cir_align_prompt", model_type="pretrain_vitL", is_eval=False, device="cuda")


        self.cross_reference_model, _, self.cross_reference_preprocess = open_clip.create_model_and_transforms("ViT-L-14")
        self.cross_reference_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        ckpt = torch.load(f"pata/to/RemoteCLIP-ViT-L-14.pt", map_location="cpu") 
        msg = self.cross_reference_model.load_state_dict(ckpt)
        print(msg)

        # print("ckpt")
        # print(ckpt.keys())
        # print(ckpt['head.weight'].shape)
        # print("model")
        # print(self.cross_reference_model)
        # print(ckpt['head.weig

        if freeze_reference:
            for name, param in self.cross_reference_model.named_parameters():
                param.requires_grad = False
            self.cross_reference_model = self.cross_reference_model.eval()
            self.cross_reference_model.train = disabled_train
            print("freeze cross_reference_model vision encoder")

        trainable_params = [name for name, param in self.named_parameters() if param.requires_grad]

        print("可训练参数：")
        for param in trainable_params:
            print(param)

        # print("参数精度：")
        # for name, param in self.named_parameters():
        #     print(f"Parameter: {name}, dtype: {param.dtype}")

        self.cross_query_model = self.cross_query_model.float()
        # print("参数精度（调整后）：")
        # for name, param in self.named_parameters():
        #     print(f"Parameter: {name}, dtype: {param.dtype}")

        self.query_proj = nn.Linear(self.cross_query_model.visual_encoder.num_features, self.cross_reference_model.visual.output_dim)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def forward(self, img1, img2=None,img3=None,img4=None,mode="grd"):
        if img4 is not None:
            image_features1 = self.query_proj(self.cross_query_model.encode_images(img1)) #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features2 = self.query_proj(self.cross_query_model.encode_images(img2)) #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features3 = self.cross_reference_model(img3)[0] #torch.Size([8, 768])
            image_features4 = self.cross_reference_model(img4)[0] #torch.Size([8, 768])
            
            return F.normalize(image_features1,dim=-1), F.normalize(image_features2,dim=-1),F.normalize(image_features3,dim=-1),F.normalize(image_features4,dim=-1)   
        elif img3 is not None:
            image_features1 = self.query_proj(self.cross_query_model.encode_images(img1)) #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features2 = self.query_proj(self.cross_query_model.encode_images(img2)) #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features3 = self.cross_reference_model(img3)[0] #torch.Size([8, 768])
            
            return F.normalize(image_features1,dim=-1), F.normalize(image_features2,dim=-1),F.normalize(image_features3,dim=-1)
        elif img2 is not None:
        
            image_features1 = self.query_proj(self.cross_query_model.encode_images(img1)) #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features2 = self.cross_reference_model(img2)[0] #torch.Size([8, 768])
            
            return F.normalize(image_features1,dim=-1), F.normalize(image_features2,dim=-1)            
              
        else:
            if mode =="grd":
                image_features = self.query_proj(self.cross_query_model.encode_images(img1))
            else:
                image_features = self.cross_reference_model(img1)[0] #torch.Size([8, 768])
             
            return F.normalize(image_features,dim=-1)



class DINO4Geo(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self, img_size=224,vit_size="large"):
        super(DINO4Geo, self).__init__()

        self.grd_size = img_size
        self.sat_size = img_size
   
        self.dino_processor = AutoImageProcessor.from_pretrained(f"facebook/dinov2-{vit_size}")
        self.dino_model = AutoModel.from_pretrained(f"facebook/dinov2-{vit_size}")
        print(self.dino_processor)
        print(self.dino_model)


    def forward(self, img1, img2=None, img3=None, mode="grd"):
        if img3 is not None:
            image_features1 = self.dino_model(img1) #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features2 = self.dino_model(img2) #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features3 = self.dino_model(img3) #torch.Size([8, 768])
            print(image_features1)
            
            return F.normalize(image_features1,dim=-1), F.normalize(image_features2,dim=-1),F.normalize(image_features3,dim=-1)   
        elif img2 is not None:
        
            image_features1 = self.dino_model(img1) #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features2 = self.dino_model(img2) #torch.Size([8, 768])
            
            return F.normalize(image_features1,dim=-1), F.normalize(image_features2,dim=-1)            
              
        else:
            image_features = self.dino_model(img1)
            # print(image_features.last_hidden_state)
            # print(image_features.last_hidden_state.shape)
            image_features = image_features.last_hidden_state.mean(dim=1)
            # print(image_features.shape)
            return F.normalize(image_features,dim=-1)
        

class MOCO4Geo(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self, img_size=224,vit_size="large"):
        super(MOCO4Geo, self).__init__()

        self.grd_size = img_size
        self.sat_size = img_size
   
        # self.moco_model = get_model('mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k', pretrained=True).to("cuda:0")

        print(self.moco_model)


    def forward(self, img1, img2=None, img3=None, mode="grd"):
        if img3 is not None:
            image_features1 = self.moco_model(img1)[0] #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features2 = self.moco_model(img2)[0] #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features3 = self.moco_model(img3)[0] #torch.Size([8, 768])
            print(image_features1)
            
            return F.normalize(image_features1,dim=-1), F.normalize(image_features2,dim=-1),F.normalize(image_features3,dim=-1)   
        elif img2 is not None:
        
            image_features1 = self.moco_model(img1)[0] #torch.Size([8, 1024]) torch.Size([8, 768])
            image_features2 = self.moco_model(img2)[0] #torch.Size([8, 768])
            
            return F.normalize(image_features1,dim=-1), F.normalize(image_features2,dim=-1)            
              
        else:
            image_features = self.moco_model(img1)[0]
            # print(image_features.last_hidden_state)
            # print(image_features.last_hidden_state.shape)
            # image_features = image_features.last_hidden_state.mean(dim=1)
            # print(image_features.shape)
            return F.normalize(image_features,dim=-1)