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


def import_and_process_obj(file_path, output_base_path, target_scale=40.0, render_settings=None, focal_length=50.0):
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

    camera_positions = [
        ("_1", (100, 0, 0), (math.radians(90), math.radians(0), math.radians(90))),
        ("_2", (-100, 0, 0), (math.radians(270), math.radians(180), math.radians(90))),
        ("_3", (0, 100, 0), (math.radians(90), math.radians(0), math.radians(180))),
        ("_4", (0, -100, 0), (math.radians(270), math.radians(180), math.radians(180))),
        ("_0", (0, 0, 150), (0, 0, 0))
    ]

    clip_start = 0.1
    clip_end = 1000.0

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

    for name, location, rotation in camera_positions:
        clear_existing_cameras()

        bpy.ops.object.camera_add(location=location, rotation=rotation)
        camera = bpy.context.object
        camera.name = name
        bpy.context.scene.camera = camera

        # Set focal length instead of FOV
        camera.data.lens = focal_length

        # Set Clip Start and Clip End for the camera
        camera.data.clip_start = clip_start
        camera.data.clip_end = clip_end

        output_file_path = os.path.join(output_path, f"{obj_name}{name}.png")
        bpy.context.scene.render.filepath = output_file_path

        # Render the image
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)

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
    import_and_process_obj(file_path, output_path, render_settings=render_settings, focal_length=35.0)