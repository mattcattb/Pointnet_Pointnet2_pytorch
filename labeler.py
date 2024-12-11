import os
import numpy as np
import open3d as o3d

def load_point_cloud(filepath):
    """Load point cloud data from a .ply file."""
    pcd = o3d.io.read_point_cloud(filepath)
    data = np.asarray(pcd.points)
    return data

def label_scene_points(scene_points, plant_points):
    """Label points with RGB values: Red for background, Green for plant."""
    labels = np.zeros((scene_points.shape[0], 3))  # Initialize an array for RGB labels

    # Use KDTree for efficient point lookup
    kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plant_points)))

    for i, point in enumerate(scene_points):
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        if (plant_points[idx[0]] == point).all():
            labels[i] = [0, 1, 0]  # Green for plant
        else:
            labels[i] = [1, 0, 0]  # Red for background

    return labels

def save_labeled_point_cloud(data, labels, filename):
    """Save labeled point cloud to a .ply file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.colors = o3d.utility.Vector3dVector(labels)
    o3d.io.write_point_cloud(filename, pcd)

def process_scenes(input_folder, plant_points_file, output_folder):
    """Process all .ply files in the input folder and save labeled point clouds to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load plant points
    plant_points = load_point_cloud(plant_points_file)

    for filename in os.listdir(input_folder):
        if filename.endswith('.ply'):
            filepath = os.path.join(input_folder, filename)
            scene_points = load_point_cloud(filepath)
            
            if scene_points.shape[0] == 0:
                print(f"File {filepath} has no points, skipping.")
                continue

            labels = label_scene_points(scene_points, plant_points)
            output_filepath = os.path.join(output_folder, filename)
            save_labeled_point_cloud(scene_points, labels, output_filepath)
            print(f"Processed {filename} and saved to {output_filepath}")

if __name__ == "__main__":
    input_folder = "/home/mattyb/Desktop/3D-Pointcloud-Paper/data/raw-scenes"  # The folder containing scene .ply files
    plant_points_file = "/home/mattyb/Desktop/3D-Pointcloud-Paper/data/output/plant.ply"  # The .ply file containing plant points
    output_folder = "labeled_scene"  # Output folder for labeled point clouds

    process_scenes(input_folder, plant_points_file, output_folder)