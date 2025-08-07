import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # 这个镜像网站可能也可以换掉

from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download

for model_name in ['ViT-L-14']:
    checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir='checkpoints',local_dir="pretrained_models",local_dir_use_symlinks="false")
    print(f'{model_name} is downloaded to {checkpoint_path}.')
