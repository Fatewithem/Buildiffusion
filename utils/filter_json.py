import json


def get_matching_folders(json_file_path, folder_prefix):
    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 筛选出以指定前缀开头的路径
    matching_folders = [folder for folder in data.keys() if folder.startswith(folder_prefix)]

    return matching_folders


# 示例用法
json_file_path = '/path/to/your/file.json'  # 替换为你的 JSON 文件路径
folder_prefix = '0005-0019'  # 要查找的前缀

matching_folders = get_matching_folders(json_file_path, folder_prefix)
print("Matching folders:", matching_folders)