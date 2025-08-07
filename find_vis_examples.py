import json

def find_mismatched_entries(file1_path, file2_path, output_path):
    # 读取第一个JSON文件
    with open(file1_path, 'r') as file1:
        json1 = json.load(file1)

    # 读取第二个JSON文件
    with open(file2_path, 'r') as file2:
        json2 = json.load(file2)

    # 用于存储不匹配的结果
    mismatched_entries = {}

    # 遍历第二个JSON文件
    for key, value in json2.items():
        if len(value) == 0:
            continue  # 跳过空数组
        
        # 检查key是否与数组第一个元素相同
        if str(value[0]) == key:
            # 检查第一个JSON文件中相同key的值是否不一样
            if key in json1 and value[0] != json1[key][0]:
                mismatched_entries[key] = {
                    "image_only": json1[key],
                    "with_text": value
                }

    # 将结果保存为JSON文件
    with open(output_path, 'w') as output_file:
        json.dump(mismatched_entries, output_file, indent=4, ensure_ascii=False)

    print(f"匹配到的结果已保存为 {output_path}")

# 示例调用
file1_path = '/home/***/codes/MultiViewGeo/Sample4Geo/nearest_dict_image.json'
file2_path = '/home/***/codes/MultiViewGeo/Sample4Geo/nearest_dict_text.json'
output_path = 'nearest_mismatched_entries_all.json'

find_mismatched_entries(file1_path, file2_path, output_path)
