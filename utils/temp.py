import json

# 读取已有的 JSON 文件
with open("/home/code/Buildiffusion/data/Qingdao/filter.json", "r") as infile:
    json_data = json.load(infile)

# 为每个对象添加 "untitled.mtl": null 和 "untitled.obj": null，如果它们不存在
for key, value in json_data.items():
    if "untitled.mtl" not in value and "untitled.obj" not in value:
        json_data[key]["untitled.mtl"] = None
        json_data[key]["untitled.obj"] = None

# 将修改后的 JSON 数据写入新的文件
with open("/home/code/Buildiffusion/data/Qingdao/filter_up.json", "w") as outfile:
    json.dump(json_data, outfile, indent=4)

print("JSON 文件已成功读取、更新并保存，已过滤已有对象的键。")