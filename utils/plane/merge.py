import numpy as np
import open3d as o3d


# ---------- 工具函数 ----------
def fit_plane_svd(points):
    """
    使用 SVD 对三维点进行平面拟合，返回 (normal, d) 形式的平面方程：
      plane: n . x + d = 0
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    _, _, vh = np.linalg.svd(points_centered, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    # 平面方程: normal . x + d = 0
    # 把质心代入，可求 d = -n . centroid
    d = - np.dot(normal, centroid)

    return normal, d


def plane_intersection_line(n1, d1, n2, d2):
    """
    给定两个平面方程:
        n1 . x + d1 = 0
        n2 . x + d2 = 0
    返回 (p0, v):
        p0: 交线上任一点
        v:  交线的方向向量(单位向量)
    若平面平行/几乎平行，则返回 None。
    """
    # 方向向量 = n1 x n2
    v = np.cross(n1, n2)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        # 认为平行或同一平面
        return None
    v = v / norm_v  # 归一化

    # 求交线上一个点 p0
    # 解 n1.(p0) + d1=0 与 n2.(p0)+ d2=0
    # 可用线性代数方法:
    #   n1.x = -d1,  n2.x = -d2
    #   设 A = [n1; n2] (2x3),  b = [-d1, -d2]
    #   不过二维方程不足以唯一确定 x，需要再与 v 正交来构造。
    #   做法之一：用“混合积”的方法。
    #
    #   令 p0 = ( d1*(n2 x v) + d2*(v x n1 ) ) / ( n1 . (n2 x v) )
    #   参考常见平面交线公式。
    denom = np.dot(n1, np.cross(n2, v))
    if abs(denom) < 1e-12:
        return None

    p0 = (-d1 * np.cross(n2, v) - d2 * np.cross(v, n1)) / denom
    return (p0, v)


def get_plane_local_axes(normal):
    """
    给定平面法向量 normal，构造与之正交的 (x_axis, y_axis, z_axis=normal)。
    """
    z_axis = normal
    # 任选一个向量不与 normal 平行用作 x_axis 的初始参考
    arbitrary_axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary_axis, z_axis)) > 0.99:
        arbitrary_axis = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(arbitrary_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return x_axis, y_axis, z_axis


def project_point_to_plane_coords(pt, plane_origin, x_axis, y_axis):
    """
    将 3D 点 pt 投影到平面坐标系:
      (u, v) = [ (pt-plane_origin).x_axis, (pt-plane_origin).y_axis ]
    """
    vec = pt - plane_origin
    u = np.dot(vec, x_axis)
    v = np.dot(vec, y_axis)
    return np.array([u, v])


def unproject_point_from_plane_coords(u, v, plane_origin, x_axis, y_axis):
    """
    将平面坐标 (u, v) 映射回 3D 坐标
      Pt = plane_origin + u*x_axis + v*y_axis
    """
    return plane_origin + u * x_axis + v * y_axis


def create_uniform_sampling_2d(points_2d, step=0.01, use_convex_hull=True):
    """
    给定平面上的 2D 投影点集, 用最小外包矩形或凸包来生成 2D 网格采样点。
    返回采样点 (N,2) 数组。
    """
    import shapely
    from shapely.geometry import MultiPoint, Point

    if use_convex_hull:
        hull_polygon = MultiPoint(points_2d).convex_hull
    else:
        # 或者用 bounding box
        hull_polygon = MultiPoint(points_2d).envelope  # 这是矩形

    min_x, min_y, max_x, max_y = hull_polygon.bounds
    xs = np.arange(min_x, max_x, step)
    ys = np.arange(min_y, max_y, step)

    sampled = []
    for xx in xs:
        for yy in ys:
            p = Point(xx, yy)
            if p.within(hull_polygon):
                sampled.append([xx, yy])
    return np.array(sampled)


# ---------- 主要示例流程 ----------
def main():
    # ---------- 1. 读取两个平面点云 ----------
    pcd1 = o3d.io.read_point_cloud("/home/code/Buildiffusion/test/split_1.ply")
    pcd2 = o3d.io.read_point_cloud("/home/code/Buildiffusion/test/split_2.ply")

    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    print(f"Plane1 点数: {len(points1)}, Plane2 点数: {len(points2)}")

    # ---------- 2. 分别对两个点云拟合平面 ----------
    n1, d1 = fit_plane_svd(points1)
    n2, d2 = fit_plane_svd(points2)
    print("平面1方程: n1 . x + d1 = 0 =>", n1, d1)
    print("平面2方程: n2 . x + d2 = 0 =>", n2, d2)

    # 如果平面非常接近平行，需要特殊处理
    cross_n1n2 = np.cross(n1, n2)
    if np.linalg.norm(cross_n1n2) < 1e-10:
        print("两个平面平行或几乎平行，示例中暂不做处理")
        return

    # ---------- 3. 计算两平面的交线(若存在) ----------
    res = plane_intersection_line(n1, d1, n2, d2)
    if res is None:
        print("无法求交线，平面平行或数值不稳定")
        return
    p0, line_dir = res
    print("平面交线: 过点 p0 =", p0, ", 方向 =", line_dir)

    # 这条交线在两平面中都存在；后面我们要在各自平面的局部坐标中描述它

    # ---------- 4. 分别对平面1 & 平面2 构造局部坐标系 ----------
    x1, y1, z1 = get_plane_local_axes(n1)
    x2, y2, z2 = get_plane_local_axes(n2)

    # 为了让拼合后是连续的，需要在“交线”处的采样相匹配。
    # 因此我们要在每个平面坐标系下都找到交线的投影，并让投影到平面坐标系的那条线
    # 采用相同的离散点。

    # ---------- 5. 求平面1的“原始点投影”，以及交线投影 ----------
    # 5.1 找一个 plane1 上的“参考点”，可以用质心或者点云中心
    centroid1 = points1.mean(axis=0)

    # 5.2 将 plane1 的所有点投影到 2D (u1,v1)
    proj1_2d = []
    for pt in points1:
        uv = project_point_to_plane_coords(pt, centroid1, x1, y1)
        proj1_2d.append(uv)
    proj1_2d = np.array(proj1_2d)

    # 5.3 交线的投影: param form: L(t)= p0 + t*line_dir
    #     我们要在 plane1 的坐标系中描述 L(t) => (u(t), v(t))
    #     先看看这条线在 plane1 的坐标系是什么样的方程
    #     令 param = t,  X(t)= project_to_plane_coords(L(t))
    #     L(t)= p0 + t*line_dir
    #     => 2D: (u1(t), v1(t))= projection( p0 + t*line_dir )

    def line_param_to_plane1_uv(t):
        pt = p0 + t * line_dir
        return project_point_to_plane_coords(pt, centroid1, x1, y1)

    # 为了确定 t 的合理范围，我们可以让交线在 plane1 的投影之内(或边界附近)，
    # 做一个简单做法：枚举 t in [-10,10] 之类，然后只保留落在边界附近的点。

    # 先得到 plane1 在 2D 的凸包(或者外包矩形)
    # => shapely 多边形
    import shapely
    from shapely.geometry import MultiPoint, Point, LineString
    poly1 = MultiPoint(proj1_2d).convex_hull

    # 我们可以粗略在一个范围里采样 t，看看投影是否落在 poly1 内
    T_vals = np.linspace(-100, 100, 10001)  # 看需求调整
    valid_ts = []
    for t in T_vals:
        uv = line_param_to_plane1_uv(t)
        p_2d = Point(uv[0], uv[1])
        if p_2d.within(poly1):
            valid_ts.append(t)
    if not valid_ts:
        print("交线在 plane1 的投影与该平面点云无公共区域。")
        return
    t_min, t_max = min(valid_ts), max(valid_ts)
    # 这样大约就是 plane1 边界上那段交线的参数范围

    # ---------- 6. 同理，对 plane2 做类似处理 ----------
    centroid2 = points2.mean(axis=0)

    proj2_2d = []
    for pt in points2:
        uv = project_point_to_plane_coords(pt, centroid2, x2, y2)
        proj2_2d.append(uv)
    proj2_2d = np.array(proj2_2d)

    poly2 = MultiPoint(proj2_2d).convex_hull

    def line_param_to_plane2_uv(t):
        pt = p0 + t * line_dir
        return project_point_to_plane_coords(pt, centroid2, x2, y2)

    valid_ts_2 = []
    for t in T_vals:
        uv = line_param_to_plane2_uv(t)
        p_2d = Point(uv[0], uv[1])
        if p_2d.within(poly2):
            valid_ts_2.append(t)
    if not valid_ts_2:
        print("交线在 plane2 的投影与该平面点云无公共区域。")
        return
    t2_min, t2_max = min(valid_ts_2), max(valid_ts_2)

    # 交线上真正的公共段是 [max(t_min, t2_min), min(t_max, t2_max)]
    common_min = max(t_min, t2_min)
    common_max = min(t_max, t2_max)
    if common_min >= common_max:
        print("两个平面在 3D 中理论上相交，但点云范围并无公共段。")
        return

    # ---------- 7. 在公共段上取一些离散点(确保两个平面共享这些点) ----------
    # 例如把它离散为 N 段
    N_edge = 50
    t_vals_common = np.linspace(common_min, common_max, N_edge)

    # 7.1 生成 plane1 的完整采样(2D)
    #     先对 plane1 的 2D 投影做一个均匀采样:
    sampling1_2d = create_uniform_sampling_2d(proj1_2d, step=0.02, use_convex_hull=True)

    # 但是要把“交线那条边”替换成我们刚才离散好的那 N_edge 个点
    # 简单方式：把 sampling1_2d 里落在交线附近的点都去掉，改为使用 t_vals_common。
    # 这里示例中，为了简化，就不做“删除原有边缘点”的操作了。
    # 而是**额外**把这 N_edge 个点“强行”加入 plane1 的采样，让 plane2 那边也一致。

    sampling1_2d = list(sampling1_2d)
    for t in t_vals_common:
        uv_line = line_param_to_plane1_uv(t)
        sampling1_2d.append(uv_line)
    sampling1_2d = np.array(sampling1_2d)

    # 将 plane1_2d -> 3D
    plane1_pcd_pts = []
    for (u, v) in sampling1_2d:
        pt3d = unproject_point_from_plane_coords(u, v, centroid1, x1, y1)
        plane1_pcd_pts.append(pt3d)

    # 7.2 生成 plane2 的完整采样(2D)，同理
    sampling2_2d = create_uniform_sampling_2d(proj2_2d, step=0.02, use_convex_hull=True)
    sampling2_2d = list(sampling2_2d)
    for t in t_vals_common:
        uv_line = line_param_to_plane2_uv(t)
        sampling2_2d.append(uv_line)
    sampling2_2d = np.array(sampling2_2d)

    plane2_pcd_pts = []
    for (u, v) in sampling2_2d:
        pt3d = unproject_point_from_plane_coords(u, v, centroid2, x2, y2)
        plane2_pcd_pts.append(pt3d)

    # ---------- 8. 合并两个平面的采样点云 ----------
    all_pts = np.vstack([plane1_pcd_pts, plane2_pcd_pts])

    # 转为 Open3D 点云
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(all_pts)

    # 保存一下
    o3d.io.write_point_cloud("/home/code/Buildiffusion/test/merged_planes.ply", merged_pcd)
    print("已保存拼合后的点云: merged_planes.ply")

    # 可视化看看
    o3d.visualization.draw_geometries([merged_pcd])


if __name__ == "__main__":
    main()