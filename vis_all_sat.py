from PIL import Image
import os
from tqdm import tqdm

def get_sorted_image_paths(folder_path):
    rows = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))],
                  key=lambda x: int(x), reverse=True)
    sorted_paths = []
    
    for row in rows:
        row_path = os.path.join(folder_path, row)
        rows_inside = sorted([f for f in os.listdir(row_path) if os.path.isdir(os.path.join(row_path, f))],
                             key=lambda x: int(x))
        for row_inside in rows_inside:
            row_inside_path = os.path.join(row_path, row_inside)
            cols = sorted([f for f in os.listdir(row_inside_path) if f.endswith('.jpg')],
                          key=lambda x: int(os.path.splitext(x)[0]))
            row_images = [os.path.join(row_inside_path, col) for col in cols]
            sorted_paths.append(row_images)
    
    return sorted_paths

def stitch_images(image_paths, scale=0.5):
    rows_images = []
    
    for row in tqdm(image_paths, desc="Stitching Rows"):
        images = [Image.open(img_path) for img_path in row]
        images = [img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS) for img in images]
        
        row_width = sum(img.width for img in images)
        row_height = images[0].height
        row_image = Image.new('RGB', (row_width, row_height))
        
        x_offset = 0
        for img in images:
            row_image.paste(img, (x_offset, 0))
            x_offset += img.width
        rows_images.append(row_image)
    
    total_width = rows_images[0].width
    total_height = sum(img.height for img in rows_images)
    full_image = Image.new('RGB', (total_width, total_height))
    
    y_offset = 0
    for row_image in tqdm(rows_images, desc="Stitching Final Image"):
        full_image.paste(row_image, (0, y_offset))
        y_offset += row_image.height
    
    return full_image

# 设置图片文件夹路径和缩放比例
folder_path = "/home/***/codes/MultiViewGeo/zhengzhou_700_feather"  # 图片文件夹路径
scale = 0.1  # 缩放比例，例如0.5表示缩小为一半
image_paths = get_sorted_image_paths(folder_path)

# 拼接图片
final_image = stitch_images(image_paths, scale=scale)

# 保存最终拼接图像
final_image.save(f"stitched_image_feather_new_scale_{scale}.jpg")
