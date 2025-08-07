import json
import os
import cv2
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sample4geo.model import MVCV4GeoComplex2
from sample4geo.transforms import get_transforms_val
from torch.cuda.amp import autocast
import torch.nn.functional as F
from PIL import Image, ImageDraw
from osgeo import gdal

def load_features(feature_path):
    """Load saved features from a file."""
    # Assuming the features are saved as a .pt file (PyTorch tensor)
    return torch.load(feature_path)

def calculate_similarity(query_feature, reference_features):
    """Calculate similarity between a single query feature and reference features."""
    # Compute similarity
    similarity = query_feature @ reference_features.T
    return similarity

def crop_image(tif_dataset, lon, lat, size=1024):
    # Get geotransform parameters
    geo_transform = tif_dataset.GetGeoTransform()
    if geo_transform is None:
        raise RuntimeError("Unable to get geotransform parameters")

    # Convert longitude and latitude to pixel coordinates
    px = int((lon - geo_transform[0]) / geo_transform[1])
    py = int((lat - geo_transform[3]) / geo_transform[5])

    # Calculate cropping region boundaries
    half_size = size // 2
    x_min = px - half_size
    x_max = px + half_size
    y_min = py - half_size
    y_max = py + half_size

    # Check if the cropping region exceeds image boundaries
    if x_min < 0 or y_min < 0 or x_max > tif_dataset.RasterXSize or y_max > tif_dataset.RasterYSize:
        raise ValueError(f"Cropping region exceeds image boundaries at coordinates (lon: {lon}, lat: {lat})")

    # Read RGB band data
    bands = [tif_dataset.GetRasterBand(i) for i in range(1, 4)]
    data = [band.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min) for band in bands]

    # Combine RGB band data
    rgb_data = np.stack(data, axis=-1)

    # Convert NumPy array to PIL image
    img = Image.fromarray(rgb_data.astype('uint8'))

    return img

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
    
    scale_factor = 10
    similarity = np.exp(similarity*scale_factor)

    meta_csv_path = "/home/***/codes/MultiViewGeo/tif/zhengzhou_multi_view.csv"
    bounds_json_file = r'/home/***/codes/MultiViewGeo/BaiduPanoramaSpider/resources/output_bounds.json'  # 替换为实际的JSON文件路径

    parts = query_image_path.split('/')
    folder_prefix = parts[-2]  # 获取 27418 作为文件夹前缀
    file_suffix = parts[-1]    # 获取 1.jpg 作为文件名

    # 获取文件的数字后缀（去掉扩展名）
    file_number = os.path.splitext(file_suffix)[0]  # 提取 '1'
    query_key = f"{folder_prefix}_{file_number}"

    # 在 target_folder 中查找以 folder_prefix 开头的文件夹
    target_folder="/home/***/codes/MultiViewGeo/tif/zhengzhou_multi_view_filtered"
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

    meta_df = pd.read_csv(meta_csv_path)

    # 构造要查找的文件名
    search_image_name = f"{folder_prefix}_"

    matching_rows = meta_df[meta_df['Image'].str.startswith(search_image_name)].iloc[0]
    target_parts = target_name.split('_')
    target_prefix = '_'.join(target_parts[:-1])
    target_lon = matching_rows[f"{target_prefix}_lon"]
    target_lat = matching_rows[f"{target_prefix}_lat"]
    # exit()
    tif_image = find_image_for_coordinates(bounds_json_file, target_lon, target_lat)
    tif_folder = r'/home/***/codes/MultiViewGeo/zhengzhou'  # 存放遥感图像的文件夹
    tif_path = os.path.join(tif_folder, tif_image)
    tif_dataset = gdal.Open(tif_path)
    geo_transform = tif_dataset.GetGeoTransform()
    local_origin = crop_image(tif_dataset,target_lon,target_lat,3000)
    if geo_transform is None:
        raise RuntimeError("Unable to get geotransform parameters")

    # Convert longitude and latitude to pixel coordinates
    px = int((target_lon - geo_transform[0]) / geo_transform[1])
    py = int((target_lat - geo_transform[3]) / geo_transform[5])

    print(tif_image)
    print(px)
    print(py)
    # exit()
    tif_image_parts = tif_image.split("_")[0].split("-")
    tif_image_parts_row = tif_image_parts[1]
    tif_image_parts_col = tif_image_parts[2]
    target_px = int(tif_image_parts_col)*2034.9 + px/10
    target_py = int(13-int(tif_image_parts_row))*2022.5 + py/10

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
    heatmap_weight = 0.5
    overlayed_image = cv2.addWeighted(original_image, 1-heatmap_weight, heatmap_resized, heatmap_weight, 0)

    output_path = f"vis/heatmap/{query_key}/{scale_factor}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    outside_half_size = 1000 // 2
    # 计算矩形框的左上角和右下角坐标，并转换为整数
    outside_top_left = (int(target_px - outside_half_size*1.9), int(target_py - outside_half_size))
    outside_bottom_right = (int(target_px + outside_half_size*1.9), int(target_py + outside_half_size))
    # Draw a white rectangle on the overlayed image

    x1, y1 = outside_top_left
    x2, y2 = outside_bottom_right
    crop_sat_image = overlayed_image[y1:y2, x1:x2]
    crop_sat_image_origin = original_image[y1:y2, x1:x2]


    inside_half_size = 80 // 2
    # 计算矩形框的左上角和右下角坐标，并转换为整数
    top_left = (int(target_px - inside_half_size), int(target_py - inside_half_size))
    bottom_right = (int(target_px + inside_half_size), int(target_py + inside_half_size))
    # Draw a white rectangle on the overlayed image
    cv2.rectangle(overlayed_image, top_left, bottom_right, (255, 255, 255), 5)
    cv2.imwrite(os.path.join(output_path,f"local.jpg"), crop_sat_image)  # 转换为BGR格式以便 OpenCV 正确保存
    cv2.imwrite(os.path.join(output_path,f"local_origin.jpg"), crop_sat_image_origin)  # 转换为BGR格式以便 OpenCV 正确保存


    cv2.rectangle(overlayed_image, outside_top_left, outside_bottom_right, (255, 255, 255), 100)


    output_scale_factor = 20  # 缩小为1/10
    height, width = overlayed_image.shape[:2]
    overlayed_image_part = overlayed_image[int(0.2*height): , int(0.2*width):]
    new_width = width // output_scale_factor
    new_height = height // output_scale_factor
    small_image = cv2.resize(overlayed_image_part, (new_width, new_height), interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(output_path,f"global.jpg"), small_image)  # 转换为BGR格式以便 OpenCV 正确保存
    original_query_image = Image.open(query_image_path)
    original_query_image.save(os.path.join(output_path,f"query.jpg"))  # 转换为BGR格式以便 OpenCV 正确保存
    original_ref_image = Image.open(f"/home/***/codes/MultiViewGeo/Sample4Geo/data/MVCV/sat/{folder_prefix}.jpg")
    original_ref_image.save(os.path.join(output_path,f"ref.jpg"))  # 转换为BGR格式以便 OpenCV 正确保存
    local_origin.save(os.path.join(output_path,f"local_bigger.jpg"), format='JPEG')




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
    reference_feature_path = "/home/***/codes/MultiViewGeo/Sample4Geo/vis/remote_features.pt"
    # 1699 2103 3902 4756 9626
    # query_image_path = "/home/***/codes/MultiViewGeo/Sample4Geo/data/MVCV/streetview/02103/1.jpg"
    query_image_path = "/home/***/codes/MultiViewGeo/Sample4Geo/data/MVCV/streetview/38854/1.jpg"
    original_image_path = "/home/***/codes/MultiViewGeo/Sample4Geo/stitched_image_scaled.jpg"
    checkpoint_start = "/home/***/codes/MultiViewGeo/Sample4Geo/weights_end.pth"
    device = "cuda"

    parts = query_image_path.split('/')
    folder_prefix = parts[-2]  # 获取 27418 作为文件夹前缀
    file_suffix = parts[-1]  # 获取 27418 作为文件夹前缀
    file_number = os.path.splitext(file_suffix)[0]  # 提取 '1'
    # Load existing similarity if available
    # query_key = f"{folder_prefix}_{file_number}"
    query_key = "02103_1"
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
