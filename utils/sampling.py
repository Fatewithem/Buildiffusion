import numpy as np
import open3d as o3d


def farthest_point_sampling(points, num_samples):
    """
    Perform Farthest Point Sampling (FPS) on a set of points.

    :param points: Input point cloud as a numpy array of shape (N, 3).
    :param num_samples: Number of points to sample.
    :return: Sampled points as a numpy array of shape (num_samples, 3), and sampled indices.
    """
    N = points.shape[0]
    sampled_points = np.zeros((num_samples, 3))
    sampled_indices = []

    # Initialize: randomly select the first point
    farthest_index = np.random.randint(0, N)
    sampled_points[0] = points[farthest_index]
    sampled_indices.append(farthest_index)

    # Initialize distances
    distances = np.full(N, np.inf)

    for i in range(1, num_samples):
        # Update distances: min distance to any sampled point
        dist_to_new_point = np.linalg.norm(points - points[farthest_index], axis=1)
        distances = np.minimum(distances, dist_to_new_point)

        # Find the farthest point
        farthest_index = np.argmax(distances)
        sampled_points[i] = points[farthest_index]
        sampled_indices.append(farthest_index)

    return sampled_points, np.array(sampled_indices)


def downsample_ply_with_fps(input_ply_path, output_ply_path, num_samples):
    """
    Downsample a PLY file using Farthest Point Sampling (FPS).

    :param input_ply_path: Path to the input PLY file.
    :param output_ply_path: Path to save the downsampled PLY file.
    :param num_samples: Number of points to sample.
    """
    try:
        # Load the PLY file
        pcd = o3d.io.read_point_cloud(input_ply_path)
        points = np.asarray(pcd.points)

        if len(points) == 0:
            print("The PLY file contains no points.")
            return None

        print(f"Original number of points: {len(points)}")

        # Perform FPS
        sampled_points = farthest_point_sampling(points, num_samples)

        # Create a new point cloud with sampled points
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

        # Save the downsampled point cloud
        o3d.io.write_point_cloud(output_ply_path, downsampled_pcd)
        print(f"Number of points after FPS: {num_samples}")
        print(f"Downsampled PLY file saved to: {output_ply_path}")

        return downsampled_pcd

    except Exception as e:
        print(f"Error processing PLY file: {e}")
        return None


# Example usage
# input_ply = "/home/code/Blender/untitled.ply"  # Replace with your input PLY file path
# output_ply = "/home/code/Blender/untitled_fps.ply"  # Replace with the desired output PLY file path
# num_samples = 2048  # Adjust the number of points to sample
# downsample_ply_with_fps(input_ply, output_ply, num_samples)