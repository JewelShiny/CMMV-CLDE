import os
import random
import numpy as np

def get_folders(base_folder):
    """获取所有子文件夹的经纬度信息"""
    folders = []
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            try:
                _, lon, lat = folder.split('_')
                folders.append((folder, float(lon), float(lat)))
            except ValueError:
                continue
    return folders

def create_grid(lon_min, lon_max, lat_min, lat_max, grid_size):
    """创建经纬度网格"""
    lon_edges = np.linspace(lon_min, lon_max, grid_size[0] + 1)
    lat_edges = np.linspace(lat_min, lat_max, grid_size[1] + 1)
    return lon_edges, lat_edges

def count_folders_in_grid(folders, lon_edges, lat_edges):
    """统计每个小矩形中的文件夹数量"""
    grid_counts = np.zeros((len(lon_edges) - 1, len(lat_edges) - 1), dtype=int)
    folder_map = {}

    for folder, lon, lat in folders:
        lon_index = np.searchsorted(lon_edges, lon) - 1
        lat_index = np.searchsorted(lat_edges, lat) - 1
        
        if 0 <= lon_index < grid_counts.shape[0] and 0 <= lat_index < grid_counts.shape[1]:
            grid_counts[lon_index, lat_index] += 1
            grid_key = (lon_index, lat_index)
            if grid_key not in folder_map:
                folder_map[grid_key] = []
            folder_map[grid_key].append(folder)
    
    return grid_counts, folder_map

def select_folders(folder_map, total_folders, fraction=0.1):
    """随机选择文件夹"""
    target_count = int(total_folders * fraction)
    selected_folders = set()
    
    while len(selected_folders) < target_count:
        grid_key = random.choice(list(folder_map.keys()))
        selected_folders.update(folder_map[grid_key])
        if len(selected_folders) >= target_count:
            break
    
    return list(selected_folders)

def main():
    base_folder = r'path/to/data/zhengzhou_multi_view_filtered'  # 修改为你的文件夹路径
    folders = get_folders(base_folder)

    if not folders:
        print("没有找到文件夹。")
        return

    lon_min = min(lon for _, lon, _ in folders)
    lon_max = max(lon for _, lon, _ in folders)
    lat_min = min(lat for _, _, lat in folders)
    lat_max = max(lat for _, _, lat in folders)

    # grid_size = (100, 100)  # 100x100的小矩形
    grid_size = (20, 20)  # 100x100的小矩形
    lon_edges, lat_edges = create_grid(lon_min, lon_max, lat_min, lat_max, grid_size)

    grid_counts, folder_map = count_folders_in_grid(folders, lon_edges, lat_edges)
    print(grid_counts)
    print(grid_counts[0])
    print(grid_counts[1])
        # 计算非零值的平均值
    non_zero_values = grid_counts[grid_counts != 0]
    average_non_zero = non_zero_values.mean()

    print("非零值的平均值:", average_non_zero.item())  # 使用 item() 方法获取 Python 标量

    total_folders = len(folders)
    selected_folders = select_folders(folder_map, total_folders,0.2)

    # 保存选择的文件夹名称
    with open('selected_folders.txt', 'w') as f:
        for folder in selected_folders:
            f.write(folder + '\n')

    print(f"选定的文件夹数量: {len(selected_folders)}")
    print("已保存选定的文件夹名称到 selected_folders.txt")

def main_random():
    base_folder = r'pata/to/data/zhengzhou_multi_view_filtered'  # 修改为你的文件夹路径
    folders = get_folders(base_folder)

    if not folders:
        print("没有找到文件夹。")
        return

    total_folders = len(folders)
    selection_ratio = 0.2  # 选择20%的文件夹
    selected_count = int(total_folders * selection_ratio)
    # 随机选择文件夹
    selected_folders = random.sample([folder[0] for folder in folders], selected_count)

    # 保存选择的文件夹名称
    with open('selected_folders.txt', 'w') as f:
        for folder in selected_folders:
            f.write(folder + '\n')

    print(f"选定的文件夹数量: {len(selected_folders)}")
    print("已保存选定的文件夹名称到 selected_folders.txt")

if __name__ == "__main__":
    main()
    # main_random()
