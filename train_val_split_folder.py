import os
import shutil
import pandas as pd
from tqdm import tqdm  # 导入 tqdm

def load_selected_folders(file_path):
    """加载选定的文件夹名称"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def process_folders(base_folder, selected_folders, output_folder):
    """处理原始文件夹，复制并重命名图片"""
    streetview_folder = os.path.join(output_folder, 'streetview')
    sat_folder = os.path.join(output_folder, 'sat')
    
    os.makedirs(streetview_folder, exist_ok=True)
    os.makedirs(sat_folder, exist_ok=True)
    
    train_data = []
    val_data = []

    selected_set = set(selected_folders)  # 转换为集合以提高查找效率

    # 使用 tqdm 显示进度条
    all_folders = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    
    for folder in tqdm(all_folders, desc="Processing folders"):
        folder_path = os.path.join(base_folder, folder)
        
        # 提取序号
        folder_number = folder.split('_')[0]  # 提取序号部分
        streetview_subfolder = os.path.join(streetview_folder, folder_number)
        os.makedirs(streetview_subfolder, exist_ok=True)

        # 查找以 point1, point2 和 sat 结尾的文件
        point1_path = None
        point2_path = None
        sat_path = None

        for file in os.listdir(folder_path):
            if file.endswith('point1.jpg'):
                point1_path = os.path.join(folder_path, file)
            elif file.endswith('point2.jpg'):
                point2_path = os.path.join(folder_path, file)
            elif file.endswith('sat.jpg'):
                sat_path = os.path.join(folder_path, file)

        # 复制并重命名图片
        if point1_path and os.path.exists(point1_path):
            new_point1_path = os.path.join(streetview_subfolder, "1.jpg")
            shutil.copy(point1_path, new_point1_path)
        
        if point2_path and os.path.exists(point2_path):
            new_point2_path = os.path.join(streetview_subfolder, "2.jpg")
            shutil.copy(point2_path, new_point2_path)

        if sat_path and os.path.exists(sat_path):
            new_sat_path = os.path.join(sat_folder, f"{folder_number}.jpg")
            shutil.copy(sat_path, new_sat_path)

        # 将数据分类到 train_data 或 val_data
        if folder in selected_set:
            val_data.append((os.path.join('streetview', folder_number, '1.jpg'),
                             os.path.join('streetview', folder_number, '2.jpg'),
                             os.path.join('sat', f"{folder_number}.jpg")))
        else:
            train_data.append((os.path.join('streetview', folder_number, '1.jpg'),
                               os.path.join('streetview', folder_number, '2.jpg'),
                               os.path.join('sat', f"{folder_number}.jpg")))

    # 创建 DataFrame 并保存到 CSV
    train_df = pd.DataFrame(train_data, columns=['streetview1', 'streetview2', 'sat'])
    val_df = pd.DataFrame(val_data, columns=['streetview1', 'streetview2', 'sat'])

    train_df.to_csv(os.path.join(output_folder, 'train.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(output_folder, 'val.csv'), index=False, header=False)


def main():
    base_folder = r'path/to/data/zhengzhou_multi_view_filtered'  # 修改为你的原始文件夹路径
    output_folder = r'path/to/code/Sample4Geo/data/MVCV_Random'  # 修改为处理后文件夹的路径
    selected_folders_file = 'selected_folders.txt'

    selected_folders = load_selected_folders(selected_folders_file)
    process_folders(base_folder, selected_folders, output_folder)

    print("处理完成，图片和CSV文件已保存。")

if __name__ == "__main__":
    main()
