# utils/downsampling.py
import numpy as np
import open3d as o3d

def random_downsample(points, num_points):
    indices = np.random.choice(points.shape[0], num_points)
    return points[indices]

def voxel_downsample(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down_pcd.points)