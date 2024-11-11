import bpy
import os
import math
import numpy as np
import json
from mathutils import Matrix

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def get_RT_from_blender(cam: bpy.types.Object):
    """Returns the R and T matrices from the given camera."""
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()  # 旋转矩阵 R
    T_world2bcam = -1 * R_world2bcam @ location  # 平移向量 T
    return R_world2bcam, T_world2bcam

def import_and_process_obj(file_path, output_base_path, target_scale=40.0, render_settings=None, focal_length=50.0, sensor_width=32.0):
    clear_scene()

    dic_name = 'blender'
    obj_name = os.path.basename(os.path.dirname(file_path))
    output_path = os.path.join(output_base_path, dic_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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

    camera_positions = [
        ("_1", (100, 0, 25), (math.radians(75), math.radians(0), math.radians(90))),
        ("_2", (-100, 0, 25), (math.radians(255), math.radians(180), math.radians(90))),
        ("_3", (0, 100, 25), (math.radians(75), math.radians(0), math.radians(180))),
        ("_4", (0, -100, 25), (math.radians(255), math.radians(180), math.radians(180))),
        ("_0", (0, 0, 150), (0, 0, 0))
    ]

    if render_settings is not None:
        bpy.context.scene.render.engine = render_settings.get('engine', 'CYCLES')
        bpy.context.scene.render.resolution_x = render_settings.get('resolution_x', 1920)
        bpy.context.scene.render.resolution_y = render_settings.get('resolution_y', 1080)
        bpy.context.scene.view_settings.view_transform = render_settings.get('view_transform', 'Standard')
        bpy.context.scene.view_settings.look = render_settings.get('look', 'None')
        bpy.context.scene.view_settings.exposure = render_settings.get('exposure', 0)
        bpy.context.scene.view_settings.gamma = render_settings.get('gamma', 1)

        bpy.context.scene.world.use_nodes = True
        bg = bpy.context.scene.world.node_tree.nodes['Background']
        bg.inputs['Color'].default_value = render_settings.get('background_color', (1, 1, 1, 1))
    else:
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080
        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.view_settings.look = 'None'
        bpy.context.scene.view_settings.exposure = 0
        bpy.context.scene.view_settings.gamma = 1
        bpy.context.scene.world.use_nodes = True
        bg = bpy.context.scene.world.node_tree.nodes['Background']
        bg.inputs['Color'].default_value = (1, 1, 1, 1)

    camera_intrinsics = {}

    for name, location, rotation in camera_positions:
        clear_existing_cameras()

        bpy.ops.object.camera_add(location=location, rotation=rotation)
        camera = bpy.context.object
        camera.name = name
        bpy.context.scene.camera = camera

        # Set focal length and sensor width
        camera.data.lens = focal_length
        camera.data.sensor_width = sensor_width
        camera.data.clip_start = 0.1
        camera.data.clip_end = 1000.0

        output_file_path = os.path.join(output_path, f"{obj_name}{name}.png")
        bpy.context.scene.render.filepath = output_file_path

        # Render the image
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)

        # Save camera intrinsics
        focal_length_px = (focal_length / sensor_width) * bpy.context.scene.render.resolution_x
        intrinsics = {
            "focal_length_px": focal_length_px,
            "sensor_width": sensor_width,
            "resolution_x": bpy.context.scene.render.resolution_x,
            "resolution_y": bpy.context.scene.render.resolution_y
        }

        # Get and save the R and T matrices
        R, T = get_RT_from_blender(camera)
        intrinsics["R"] = [list(row) for row in R]  # 转为列表形式以便存储
        intrinsics["T"] = list(T)

        camera_intrinsics[name] = intrinsics

    # Save camera intrinsics and extrinsics to a JSON file
    intrinsics_file_path = os.path.join(output_path, f"{obj_name}_camera_intrinsics.json")
    with open(intrinsics_file_path, 'w') as json_file:
        json.dump(camera_intrinsics, json_file, indent=4)

    print(f"Processing and rendering completed for {file_path}")

def clear_existing_cameras():
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

if __name__ == "__main__":
    render_settings = {
        'engine': 'CYCLES',
        'resolution_x': 1920,
        'resolution_y': 1080,
        'view_transform': 'Standard',
        'look': 'None',
        'exposure': 0,
        'gamma': 1,
        'background_color': (1, 1, 1, 1)
    }

    file_path = '/Users/wuchengyu/Downloads/Code/Buildiffusion/data/processed_object_0.obj'
    output_path = '/Users/wuchengyu/Downloads/Code/Buildiffusion/data/'
    import_and_process_obj(file_path, output_path, render_settings=render_settings, focal_length=35.0, sensor_width=32.0)