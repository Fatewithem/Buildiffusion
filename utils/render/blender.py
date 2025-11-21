import bpy
import os
import math
import mathutils
import struct
import time

# === 设置参数 ===
ply_path = os.path.abspath("/home/code/Buildiffusion/sample_7_color.ply")  # 替换为你的路径
output_path = os.path.abspath("/home/code/Buildiffusion/color.png")

# 清空场景
bpy.ops.wm.read_factory_settings(use_empty=True)

print(">> Begin loading PLY")
# 读取原始 PLY 文件的 RGB 点数据（仅适用于 binary_little_endian 格式）
def load_ply_vertices(ply_path):
    with open(ply_path, 'rb') as f:
        lines = []
        while True:
            line = f.readline()
            lines.append(line)
            if line.startswith(b'end_header'):
                break
        header = b''.join(lines).decode()
        num_vertices = int([l for l in header.split('\n') if l.startswith('element vertex')][0].split()[-1])
        # 定位数据起始
        data = f.read()
        vertices = []
        for i in range(num_vertices):
            offset = i * (8*3 + 1*3)
            x, y, z = struct.unpack('<ddd', data[offset:offset+24])
            r, g, b = struct.unpack('<BBB', data[offset+24:offset+27])
            vertices.append(((x, y, z), (r/255.0, g/255.0, b/255.0)))
        return vertices

# 加载点数据
vertices = load_ply_vertices(ply_path)
print(">> Done loading PLY")

# 限制点云数量为 500 个用于测试
vertices = vertices[:10000]

# 可选：在加载后打印数量日志
print(f"Loaded {len(vertices)} points for rendering.")

# 计算点云中心和尺寸
if vertices:
    coords = [mathutils.Vector(v[0]) for v in vertices]
    center = sum(coords, mathutils.Vector()) / len(coords)
    size = max((v - center).length for v in coords)
else:
    center = mathutils.Vector((0,0,0))
    size = 0

print(">> Begin instancing sphere mesh")

# 创建一个基础球体 mesh 和 object
radius = 0.01 * size if size > 0 else 0.02
bpy.ops.mesh.primitive_uv_sphere_add(radius=radius)
template_sphere = bpy.context.object
template_sphere.name = "BaseSphere"

# 使用 collection 组织所有实例
pc_collection = bpy.data.collections.new("PointCloud")
bpy.context.scene.collection.children.link(pc_collection)


# 设置基础球体的渲染属性
template_sphere.cycles.is_shadow_catcher = False
template_sphere.cycles.use_camera_cull = False
template_sphere.cycles.use_adaptive_subdivision = False
template_sphere.cycles.use_motion_blur = False

# 确保将 template_sphere 对象加入主集合中
# bpy.context.scene.collection.objects.link(template_sphere)

# 创建共享材质
shared_mat = bpy.data.materials.new(name="SharedColorMaterial")
shared_mat.use_nodes = True

# 清空默认节点
nodes = shared_mat.node_tree.nodes
nodes.clear()

# 创建 Emission 材质
emission = nodes.new(type="ShaderNodeEmission")
output = nodes.new(type="ShaderNodeOutputMaterial")
shared_mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

# 实例化所有点位对象
for i, ((x, y, z), (r, g, b)) in enumerate(vertices):
    inst = template_sphere.copy()
    inst.data = template_sphere.data.copy()
    inst.location = (x, y, z)
    inst.name = f"pt_{i}"

    # 设置材质
    mat = shared_mat.copy()
    mat.node_tree.nodes["Emission"].inputs["Color"].default_value = (r, g, b, 1)
    inst.data.materials.append(mat)

    pc_collection.objects.link(inst)

print(">> Done instancing spheres")

# 设置渲染引擎
bpy.context.scene.render.engine = 'CYCLES'
# 修改渲染设备为 CPU，并禁用 GPU 加速设置
bpy.context.scene.cycles.device = 'CPU'

# 设置背景颜色为白色
if bpy.context.scene.world is None:
    bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes['Background']
bg.inputs[0].default_value = (1, 1, 1, 1)  # 白色背景


# 添加相机并手动控制位置与朝向
bpy.ops.object.camera_add()
cam = bpy.context.object
bpy.context.scene.camera = cam

# 相机在 XY 平面远处
cam.location = mathutils.Vector((10.0, 10.0, 0.8))

# 看向点云中心
direction = center - cam.location
cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# 例如：绕Z轴微调 +15°
cam.rotation_euler.y -= math.radians(120)

# 设置光源
bpy.ops.object.light_add(type='AREA', location=(5, 5, 5))

# 设置分辨率
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.image_settings.file_format = 'PNG'

# 输出文件路径
bpy.context.scene.render.filepath = output_path

print(">> Begin rendering...")
# （可选）如果使用Eevee渲染器，可设置点大小较大
# bpy.context.object.show_transparent = True

# 渲染图像
bpy.ops.render.render(write_still=True)
print(">> Rendering complete")