import bpy
import os
import mathutils
import math
import csv
import numpy as np
import json


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def enable_metal_gpu_rendering():
    """
    启用 Metal GPU 渲染（适用于 Apple Silicon，如 M1/M2）。
    """
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'  # 设置为 GPU 渲染

    # 获取用户首选项中的 Cycles 渲染设置
    preferences = bpy.context.preferences.addons["cycles"].preferences

    # 设置 Metal 设备
    preferences.compute_device_type = 'METAL'  # Metal 后端
    preferences.get_devices()  # 获取设备信息

    # 启用所有 Metal 设备
    for device in preferences.devices:
        if "Metal" in device.name:
            device.use = True
            print(f"Enabled Metal GPU device: {device.name}")
        else:
            print(f"Skipped device: {device.name}")


def calculate_center_of_mass(objects):
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = -float('inf')

    for obj in objects:
        bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        min_x = min(min_x, *[corner.x for corner in bbox_corners])
        min_y = min(min_y, *[corner.y for corner in bbox_corners])
        min_z = min(min_z, *[corner.z for corner in bbox_corners])
        max_x = max(max_x, *[corner.x for corner in bbox_corners])
        max_y = max(max_y, *[corner.y for corner in bbox_corners])
        max_z = max(max_z, *[corner.z for corner in bbox_corners])

    center = mathutils.Vector((min_x + max_x, min_y + max_y, min_z + max_z)) / 2
    print(f"Center: {center}")
    return center


def import_and_process_obj(file_path, output_base_path, target_scale=20.0, render_settings=None, focal_length=35.0, sensor_width=32.0):
    clear_scene()

    dic_name = 'blender'
    obj_name = os.path.basename(os.path.dirname(file_path))
    output_path = os.path.join(output_base_path, dic_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Import OBJ file
    if os.path.isfile(file_path):
        try:
            bpy.ops.wm.obj_import(filepath=file_path)
            print(f"OBJ file imported successfully: {file_path}")
        except Exception as e:
            print(f"Import failed for {file_path}: {e}")
            return
    else:
        print(f"Specified OBJ file does not exist: {file_path}")
        return

    imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.pass_index = 1  # 将对象ID设置为1

    # Scale objects
    for obj in imported_objects:
        dimensions = obj.dimensions
        max_dimension = max(dimensions)
        print(f"Before: {max_dimension}")
        scale_factor = target_scale / max_dimension
        obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(scale=True)
        dimensions = obj.dimensions
        max_dimension = max(dimensions)
        print(f"After: {max_dimension}")

    # Move objects to origin
    center_of_mass = calculate_center_of_mass(imported_objects)
    for obj in imported_objects:
        obj.location -= center_of_mass

    camera_positions = [
        ("_1", (45, 0, 9), (math.radians(75), math.radians(0), math.radians(90))),
        ("_2", (-45, 0, 9), (math.radians(255), math.radians(180), math.radians(90))),
        ("_3", (0, 45, 9), (math.radians(75), math.radians(0), math.radians(180))),
        ("_4", (0, -45, 9), (math.radians(255), math.radians(180), math.radians(180))),
        ("_0", (0, 0, 55), (0, 0, 0))
    ]

    # Add camera settings for clip_start and clip_end
    clip_start = 0.1  # 设置 Clip Start 为 0.1
    clip_end = 1000.0  # 设置 Clip End 为 1000

    # Apply render settings if provided
    if render_settings is not None:
        bpy.context.scene.render.engine = render_settings.get('engine', 'CYCLES')
        bpy.context.scene.render.resolution_x = render_settings.get('resolution_x', 1920)
        bpy.context.scene.render.resolution_y = render_settings.get('resolution_y', 1080)
        bpy.context.scene.view_settings.view_transform = render_settings.get('view_transform', 'Standard')
        bpy.context.scene.view_settings.look = render_settings.get('look', 'None')
        bpy.context.scene.view_settings.exposure = render_settings.get('exposure', 0)
        bpy.context.scene.view_settings.gamma = render_settings.get('gamma', 1)
        bpy.context.scene.cycles.samples = render_settings.get('samples', 128)  # 降低采样次数以提高渲染速度
        bpy.context.scene.cycles.use_adaptive_sampling = render_settings.get('use_adaptive_sampling', True)
        bpy.context.scene.cycles.max_bounces = render_settings.get('max_bounces', 4)  # 减少反射次数

        # Set background color
        bpy.context.scene.world.use_nodes = True
        bg = bpy.context.scene.world.node_tree.nodes['Background']
        bg.inputs['Color'].default_value = render_settings.get('background_color', (1, 1, 1, 1))
    else:
        # Default render settings
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.view_settings.look = 'None'
        bpy.context.scene.view_settings.exposure = 0
        bpy.context.scene.view_settings.gamma = 1
        bpy.context.scene.world.use_nodes = True
        bg = bpy.context.scene.world.node_tree.nodes['Background']
        bg.inputs['Color'].default_value = (1, 1, 1, 1)
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.max_bounces = 4

    for name, location, rotation in camera_positions:
        clear_existing_cameras()

        bpy.ops.object.camera_add(location=location, rotation=rotation)
        camera = bpy.context.object
        camera.name = name
        bpy.context.scene.camera = camera
        # set_camera_fov(camera, fov=40)

        camera.data.lens = focal_length
        camera.data.sensor_width = sensor_width
        camera.data.clip_start = clip_start
        camera.data.clip_end = clip_end

        output_file_path = os.path.join(output_path, f"{obj_name}{name}.png")
        bpy.context.scene.render.filepath = output_file_path

        # 渲染图片
        bpy.context.scene.render.image_settings.file_format = 'PNG'  # 设置为 PNG 格式
        bpy.ops.render.render(write_still=True)

        # 保存深度图
        depth_file_path = os.path.join(output_path, f"{obj_name}{name}_depth.exr")
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.filepath = depth_file_path
        bpy.context.scene.render.use_compositing = True

        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # 清除默认节点
        for n in tree.nodes:
            tree.nodes.remove(n)

        # 创建输入渲染层节点
        rl = tree.nodes.new('CompositorNodeRLayers')

        # 创建输出文件节点
        of = tree.nodes.new('CompositorNodeOutputFile')
        of.format.file_format = 'OPEN_EXR'
        of.base_path = output_path
        of.file_slots[0].path = f"{obj_name}{name}_depth"

        # 连接节点
        links.new(rl.outputs['Depth'], of.inputs[0])

        bpy.ops.render.render(write_still=True)

        # 读取渲染的深度图数据并存储为 .npy 格式
        depth_image = bpy.data.images.load(depth_file_path)
        depth_pixels = np.array(depth_image.pixels[:], dtype=np.float32)
        depth_pixels = depth_pixels[::4]  # 只取每第4个值 (R 通道)
        depth_pixels = depth_pixels.reshape(
            (bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x))

        npy_depth_path = os.path.join(output_path, f"{obj_name}{name}_depth.npy")
        np.save(npy_depth_path, depth_pixels)

        # 删除 EXR 文件
        if os.path.exists(depth_file_path):
            os.remove(depth_file_path)
            print(f"Deleted EXR file: {depth_file_path}")

        # 渲染完深度图后，切换回 PNG 格式，避免后续渲染产生 EXR 文件
        bpy.context.scene.render.image_settings.file_format = 'PNG'

    for name, location, rotation in camera_positions:
        clear_existing_cameras()

        bpy.ops.object.camera_add(location=location, rotation=rotation)
        camera = bpy.context.object
        camera.name = name
        bpy.context.scene.camera = camera
        camera.data.lens = focal_length
        camera.data.sensor_width = sensor_width

        # 生成 mask 并保存为 PNG 格式
        mask_file_path = os.path.join(output_path, f"{obj_name}{name}_mask.png")
        save_mask(mask_file_path)
        print(f"Mask saved: {mask_file_path}")

    # save_camera_data(pose_data, output_path)
    print(f"Processing and rendering completed for {file_path}")


def save_mask(output_file_path):
    scene = bpy.context.scene
    objs = [obj for obj in bpy.data.objects if obj.type in ("MESH", "CURVE")]

    # 为每个物体分配不同的颜色，用于mask渲染
    for obj_idx, obj in enumerate(objs):
        color = (obj_idx / len(objs), 0, 0, 0)  # 分配不同颜色，使用RGB值

        material_name = "auto.material." + obj.name
        material = bpy.data.materials.new(material_name)

        material["is_auto"] = True
        material.use_nodes = True
        material.node_tree.nodes.clear()

        emission = material.node_tree.nodes.new(type="ShaderNodeEmission")
        emission.inputs['Color'].default_value = color  # 给不同物体设置不同颜色

        output = material.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
        material.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])

        obj.data.materials.clear()  # 删除物体之前的所有材质
        obj.data.materials.append(material)  # 将刚刚创建并初始化颜色的材质加入物体

    # 渲染并保存mask图像
    scene.render.filepath = output_file_path  # 设置mask图像保存路径
    bpy.context.scene.render.image_settings.file_format = 'PNG'  # 设置保存格式为PNG
    bpy.ops.render.render(write_still=True)


# Helper function to get depth data
def get_depth_data():
    # 将深度缓冲区数据提取为 NumPy 数组
    depth_buffer = bpy.data.images['Render Result'].pixels[:]
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    depth = np.array(depth_buffer).reshape(height, width, 4)[:, :, 0]  # 提取深度通道
    return depth


def clear_existing_cameras():
    # 删除场景中的所有相机对象，确保不会有多个相机参与渲染
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)


def process_selected_folders(folder_list, output_base_path, render_settings=None):
    """
    仅从给定的文件夹列表中查找并处理 .obj 文件
    :param folder_list: 包含所有要处理的文件夹的列表
    :param output_base_path: 输出文件的基础路径
    :param render_settings: 渲染设置
    """
    for folder in folder_list:
        # 遍历每个给定的文件夹
        for file in os.listdir(folder):
            # 跳过以 "._" 开头的文件
            if file.startswith("._"):
                continue

            if file.endswith(".obj"):
                # 获取 .obj 文件的完整路径
                file_path = os.path.join(folder, file)
                print(f"Processing file: {file_path}")  # 输出每个文件的路径，检查是否有重复

                # 确定输出路径，以保持文件夹结构
                relative_path = os.path.relpath(folder, folder_list[0])  # 使用第一个文件夹作为基准
                output_path = os.path.join(folder, relative_path)

                # 创建输出文件夹
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                print("Processing OBJ file...")
                import_and_process_obj(file_path, output_path, render_settings=render_settings, focal_length=35.0, sensor_width=32.0)


def get_matching_folders_with_prefix(json_file_path, folder_prefixes, additional_prefix):
    """
    从 JSON 文件中获取匹配的文件夹路径，根据多个前缀进行筛选。
    :param json_file_path: JSON 文件路径
    :param folder_prefixes: 包含多个前缀的列表
    :param additional_prefix: 要添加的路径前缀
    :return: 匹配文件夹的完整路径列表
    """
    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 筛选出以任一指定前缀开头的路径，并加上 additional_prefix
    matching_folders = []
    for prefix in folder_prefixes:
        matching_folders.extend(
            os.path.join(additional_prefix, folder) for folder in data.keys() if folder.startswith(prefix)
        )

    return matching_folders


if __name__ == "__main__":
    base_path = "/Volumes/My Passport/UrbanBIS/Mesh/Qingdao/"
    json_file_path = '/Volumes/My Passport/UrbanBIS/Mesh/Qingdao/filter.json'  # 替换为你的 JSON 文件路径

    # Lihu
    # folder_prefixes = [
    #     # "1.1",
    #     # "1.3",
    #     # "1.4",
    #     # "10.2",
    #     # "11.1",
    #     # "12.1",
    #     # "12.2",
    #     # "2.1",
    #     # "2.2",
    #     # "2.3",
    #     # "3.3",
    #     # "4.1",
    #     # "4.2",
    #     # "4.3",
    #     # "5.2",
    #     # "5.5",
    #     # "6.1",
    #     # "6.2",
    #     # "6.3",
    #     # "8.1",
    #     # "8.2",
    #     # "8.3",
    #     # "9.1",
    #     # "9.2",
    #     "9.3"
    # ]

    # QingDao
    folder_prefixes = [
        # "0005-0019",
        # "0020-0026",
        # "0027-0034",
        # "0035-0050",
        # "0089-0100",
        # "0101-0110",
        # "1-0051-0069-6",
        # "1-0051-0069-71",
        # "1-0051-0075-2",
        # "1-0051-0080-4",
        # "1-0058-0140-12",
        # "1-0062-0088-7.2",
        # "1-0066-0088-5",
        # "1-0073-0088-3",
        # "1-0111-0129-8.2",
        # "1-0111-0131-10",
        # "1-0111-0140-11",
        # "1-0124-0140-8.1",
        # "1-0131-0140-9",
        # "1.3-0185-0192-14",
        # "2-0068-0184-9",
        # "2-0070-0165-11",
        # "2-0071-0174-8",
        # "2-0141-0163-2",
        # "2-0141-0168-6.2",
        # "2-0141-0170-6.1",
        # "2-0141-0184-4",
        # "2-0146-0172-7.2",
        # "2-0159-0184-5",
        # "2-0160-0184-71",
        # "2-0162-0184-10",
        # "2-0164-0184-4",
        # "2.3-0189-0204-15",
        # "2.3-0197-0231-13",
        # "3-0185-0196-6",
        # "3-0185-0201-4",
        # "3-0185-0202-9",
        # "3-0185-0206-1",
        # "3-0185-0206-3",
        # "3-0185-0217-2",
        # "3-0185-212-10",
        # "3-0191-0221-7",
        # "3-0194-0207-5",
        # "3-0205-0228-8",
        # "3-0205-218-16",
        # "3-0213-224-11",
        # "3-0219-237-17",
        "3-0228-251-18",
        "3-0230-256-12"
    ]

    additional_prefix = '/Volumes/My Passport/UrbanBIS/Mesh/Qingdao/'  # 要添加的路径前缀

    # 获取所有匹配的文件夹
    matching_folders = get_matching_folders_with_prefix(json_file_path, folder_prefixes, additional_prefix)

    # 启用 Metal GPU 渲染
    enable_metal_gpu_rendering()

    # 渲染设置
    render_settings = {
        'engine': 'CYCLES',
        'resolution_x': 1920,
        'resolution_y': 1080,
        'view_transform': 'Standard',
        'look': 'None',
        'exposure': 0,
        'gamma': 1,
        'max_bounces': 4,  # 降低光线反弹数量
        'background_color': (1, 1, 1, 1)  # White background
    }

    # 处理选定的文件夹
    process_selected_folders(matching_folders,
                             "/Volumes/My Passport/UrbanBIS/Mesh/Qingdao/0005-0019/building/building1",
                             render_settings)

    # test
    # file_path = '/Users/wuchengyu/Downloads/Code/Buildiffusion/data/processed_object_0.obj'
    # output_path = '/Users/wuchengyu/Downloads/Code/Buildiffusion/data/'
    # import_and_process_obj(file_path, output_path, render_settings=render_settings, focal_length=35.0, sensor_width=32.0)