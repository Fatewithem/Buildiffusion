import numpy as np
import open3d as o3d


def fit_plane_svd(points: np.ndarray):
    """
    使用 SVD 对三维点进行平面拟合。
    返回: (plane_center, normal)
      - plane_center: 拟合平面的某个参考点（这里选择点云质心）
      - normal: 单位法向量
    """
    # 计算质心
    centroid = np.mean(points, axis=0)
    # 去质心化
    points_centered = points - centroid

    # 对去质心化后的点做 SVD
    u, s, vh = np.linalg.svd(points_centered, full_matrices=False)

    # V 的最后一列（vh[-1]）对应最小特征值，代表法向量方向
    normal = vh[-1]

    # 确保法向量是单位向量
    normal = normal / np.linalg.norm(normal)

    return centroid, normal


def create_uniform_grid_on_plane(centroid, normal, points, step=0.01):
    """
    在拟合平面上，根据原始点云投影到平面的最小外包矩形范围，
    生成均匀分布的点，并返回这些点的 ndarray。

    参数:
    - centroid: 拟合平面的中心点（质心）
    - normal:   平面法向量（单位）
    - points:   原始点云点 (N,3)
    - step:     网格点间距

    返回:
    - plane_points: 在平面上均匀分布的点 (M,3)
    """

    # 1) 构造一个局部坐标系: 以 normal 为 z 方向，其余两个正交方向为 x、y
    #    这样后续才能方便地把点投影到 plane_x, plane_y 坐标
    # normal 作为 z 轴
    z_axis = normal

    # 任选一个向量与 z_axis 不平行，用来求 x_axis
    # 例如直接取全局 X 轴(1,0,0)，若与 normal 平行则换一个即可
    arbitrary_axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary_axis, z_axis)) > 0.99:
        # 若平行则改用Y轴
        arbitrary_axis = np.array([0.0, 1.0, 0.0])

    # 用叉乘求出平面 x 方向
    x_axis = np.cross(arbitrary_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)  # 归一化

    # 再用叉乘得到 y 方向
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 2) 将所有点投影到该平面，找出 plane_x 和 plane_y 的最小、最大值
    #    点投影到平面坐标系 (plane_x, plane_y)
    def project_to_plane_coordinate(p):
        # 先转为中心坐标系
        pc = p - centroid
        return np.array([np.dot(pc, x_axis), np.dot(pc, y_axis)])

    projected_2d = np.array([project_to_plane_coordinate(p) for p in points])

    min_x, min_y = projected_2d.min(axis=0)
    max_x, max_y = projected_2d.max(axis=0)

    # 3) 在 [min_x, max_x] x [min_y, max_y] 的 2D 区域中，根据 step 生成网格
    xs = np.arange(min_x, max_x, step)
    ys = np.arange(min_y, max_y, step)

    # 4) 将网格上的每个 (u, v) 映射回 3D 坐标
    plane_points = []
    for u in xs:
        for v in ys:
            # 平面坐标 (u, v) -> 3D
            # 3D 点 = centroid + u * x_axis + v * y_axis
            point_3d = centroid + u * x_axis + v * y_axis
            plane_points.append(point_3d)

    plane_points = np.array(plane_points)
    return plane_points


def main():
    # ========== 1. 读取本地 .ply 点云 ==========
    ply_file = "/home/code/Buildiffusion/test/split_7.ply"  # 替换为你的点云文件路径
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    print(f"原始点云包含 {points.shape[0]} 个点。")

    # ========== 2. 使用 SVD 拟合平面 ==========
    centroid, normal = fit_plane_svd(points)
    print("平面质心:", centroid)
    print("平面法向量:", normal)

    # ========== 3. 在平面上生成均匀分布的点 ==========
    # step 可以根据需要调整，这里默认设置为 0.01
    uniform_points = create_uniform_grid_on_plane(centroid, normal, points, step=0.01)
    print(f"生成的均匀平面点数量: {uniform_points.shape[0]}")

    # ========== 4. 转为 open3d 点云并保存 ==========
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(uniform_points)

    # 这里演示保存为 plane_uniform.ply
    o3d.io.write_point_cloud("/home/code/Buildiffusion/test/split_7_plane.ply", plane_pcd)
    print("均匀点云已保存为 plane_uniform.ply")


if __name__ == "__main__":
    main()