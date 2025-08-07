import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sample4geo.model import MVCV4GeoComplex2
from sample4geo.transforms import get_transforms_val
from torch.cuda.amp import autocast
import torch.nn.functional as F
from PIL import Image

def load_features(feature_path):
    """Load saved features from a file."""
    # Assuming the features are saved as a .pt file (PyTorch tensor)
    return torch.load(feature_path)

def calculate_similarity(query_feature, reference_features):
    """Calculate similarity between a single query feature and reference features."""
    # Compute similarity
    similarity = query_feature @ reference_features.T
    return similarity

def find_image_for_coordinates(json_file, lon, lat):
    with open(json_file, 'r') as f:
        bounds_dict = json.load(f)
    
    for image, bounds in bounds_dict.items():
        if bounds['west'] <= lon <= bounds['east'] and bounds['south'] <= lat <= bounds['north']:
            return image
    
    return None

def visualize_similarity_on_image(similarity, original_image_path,query_image_path):
    """Visualize similarity values as a heatmap over the original image."""
    # Reshape similarity vector to (404, 872)
    similarity = similarity.cpu().numpy().reshape((404, 872))
    
    scale_factor = 3
    similarity = np.exp(similarity*scale_factor)

    meta_csv_path = "/home/zhuxy/codes/MultiViewGeo/tif/zhengzhou_multi_view.csv"
    meta_image_path = "/home/zhuxy/codes/MultiViewGeo/tif/zhengzhou_multi_view_filtered"
    bounds_json_file = r'/home/zhuxy/codes/MultiViewGeo/BaiduPanoramaSpider/resources/output_bounds.json'  # 替换为实际的JSON文件路径
    # tif_image = find_image_for_coordinates(bounds_json_file, lon, lat)

    parts = query_image_path.split('/')
    folder_prefix = parts[-2]  # 获取 27418 作为文件夹前缀
    file_suffix = parts[-1]    # 获取 1.jpg 作为文件名

    # 获取文件的数字后缀（去掉扩展名）
    file_number = os.path.splitext(file_suffix)[0]  # 提取 '1'
    query_key = f"{folder_prefix}_{file_number}"

    # 在 target_folder 中查找以 folder_prefix 开头的文件夹
    target_folder="/home/zhuxy/codes/MultiViewGeo/tif/zhengzhou_multi_view_filtered"
    matching_folder = None
    for folder_name in os.listdir(target_folder):
        if folder_name.startswith(folder_prefix) and os.path.isdir(os.path.join(target_folder, folder_name)):
            matching_folder = os.path.join(target_folder, folder_name)
            break

    target_name = None
    for file_name in os.listdir(matching_folder):
        if file_name.endswith(f"{file_number}.jpg"):
            target_name = file_name
    print(target_name)
    # exit()
            
    print(np.max(similarity))
    print(np.min(similarity))

    # Normalize similarity for visualization (0-255)
    norm_similarity = cv2.normalize(similarity, None, 0, 255, cv2.NORM_MINMAX)
    norm_similarity = norm_similarity.astype(np.uint8)


    print(np.max(norm_similarity))
    print(np.min(norm_similarity))

    # Create a heatmap
    heatmap = cv2.applyColorMap(norm_similarity, cv2.COLORMAP_JET)

    # Load the original image (without resizing)
    Image.MAX_IMAGE_PIXELS = None  # 设置为 None 表示移除限制
    original_image = Image.open(original_image_path)

    # Convert image to RGB (Pillow uses RGB by default, but this ensures it's consistent)
    original_image = original_image.convert("RGB")

    # Convert the image to a numpy array (Pillow's output is HxWxC, which matches OpenCV's BGR->RGB conversion)
    original_image = np.array(original_image)

    # Resize the heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Overlay the heatmap on the original image
    overlayed_image = cv2.addWeighted(original_image, 0.6, heatmap_resized, 0.4, 0)
    output_path = f"vis/output_heatmap_{query_key}_scale{scale_factor}.jpg"
    # Save the overlayed image directly using OpenCV
    cv2.imwrite(output_path, overlayed_image)  # 转换为BGR格式以便 OpenCV 正确保存

def save_similarity(similarity, query_key):
    """Save similarity to a file."""
    filename = f"cache/similarity_{query_key}.pt"
    torch.save(similarity, filename)

def load_similarity(query_key):
    """Load similarity from a file if it exists."""
    filename = f"cache/similarity_{query_key}.pt"
    if os.path.exists(filename):
        return torch.load(filename)
    return None

if __name__ == "__main__":
    # Paths to saved features
    reference_feature_path = "./vis/remote_features.pt"
    query_image_path = "./data/MVCV/streetview/27418/1.jpg"
    original_image_path = "./stitched_image_scaled.jpg"
    checkpoint_start = "path/to/weights_end.pth"
    device = "cuda"

    parts = query_image_path.split('/')
    folder_prefix = parts[-2]  # 获取 27418 作为文件夹前缀
    file_suffix = parts[-1]  # 获取 27418 作为文件夹前缀
    file_number = os.path.splitext(file_suffix)[0]  # 提取 '1'
    # Load existing similarity if available
    query_key = f"{folder_prefix}_{file_number}"
    similarity = load_similarity(query_key)

    if similarity is None:
        # Load features
        print("Load features")
        reference_features = load_features(reference_feature_path)  # Load reference features

        model = MVCV4GeoComplex2(img_size=224)
        # Load pretrained checkpoint
        print("Start from:", checkpoint_start)
        model_state_dict = torch.load(checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

        # Eval
        data_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        print(data_config)
        mean = data_config["mean"]
        std = data_config["std"]
        sat_transforms_val, ground_transforms_val = get_transforms_val(
            [224, 224],
            [224, 224],
            mean=mean,
            std=std,
        )
        img = cv2.imread(query_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Image transforms
        img = ground_transforms_val(image=img)["image"]
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            with autocast():
                img = img.to(device)
                img = img.unsqueeze(0)
                query_feature = model(img, mode="grd")
                query_feature = F.normalize(query_feature, dim=-1)

        # Calculate similarity
        similarity = calculate_similarity(query_feature, reference_features.to(device))
        # Save the calculated similarity for future use
        save_similarity(similarity, query_key)
    else:
        print("Loaded existing similarity for:", query_key)

    # Visualize the similarity as a heatmap over the original image
    visualize_similarity_on_image(similarity, original_image_path,query_image_path)
