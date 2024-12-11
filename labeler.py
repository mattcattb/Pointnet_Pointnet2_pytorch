import os
import numpy as np
import open3d as o3d

def load_point_cloud(filepath):
    """Load point cloud data from a .ply file."""
    pcd = o3d.io.read_point_cloud(filepath)
    data = np.asarray(pcd.points)
    return data

def label_scene_points(scene_points, plant_points, threshold=1e-3):
    """Label points with RGB values: Green for plant points, Red for background."""
    labels = np.zeros((scene_points.shape[0], 3))  # Initialize an array for RGB labels

    # Using KDTree for efficient point lookup
    plant_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plant_points)))
    
    for i, point in enumerate(scene_points):
        [_, idx, d] = plant_tree.search_knn_vector_3d(point, 1)
        if d[0] <= threshold:
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

def process_scenes(scene_files, label_files, output_folder):
    """Process all scene files and their corresponding label files, then save labeled point clouds to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, (scene_file, label_file) in enumerate(zip(scene_files, label_files)):
        # Load plant points
        plant_points = load_point_cloud(label_file)
        
        # Load scene points
        scene_points = load_point_cloud(scene_file)
        
        if scene_points.shape[0] == 0:
            print(f"File {scene_file} has no points, skipping.")
            continue

        # Label scene points
        labels = label_scene_points(scene_points, plant_points)

        # Save the labeled point cloud with sequential filename (e.g., scene0.ply, scene1.ply, ...)
        output_filename = os.path.join(output_folder, f"scene{idx}.ply")
        save_labeled_point_cloud(scene_points, labels, output_filename)
        print(f"Processed {scene_file} and saved to {output_filename}")

if __name__ == "__main__":
    # List of scene files and their corresponding label files
    scene_files = [
        "/home/mattyb/Desktop/3D-Pointcloud-Paper/data/raw-scenes/large_scene - Cloud.ply",
        "/home/mattyb/Desktop/3D-Pointcloud-Paper/data/raw-scenes/med_scene - Cloud.ply",
        "/home/mattyb/Desktop/3D-Pointcloud-Paper/data/raw-scenes/small_scene - Cloud.ply"
    ]
    
    label_files = [
        "/home/mattyb/Desktop/3D-Pointcloud-Paper/data/label_ply_only/large_scene_plant-Cloud.ply",
        "/home/mattyb/Desktop/3D-Pointcloud-Paper/data/label_ply_only/med_scene - Cloud - Cloud.ply",
        "/home/mattyb/Desktop/3D-Pointcloud-Paper/data/label_ply_only/small_scene - Cloud - Cloud.ply"
    ]    

    output_folder = "labeled_scenes"  # Output folder for labeled point clouds

    process_scenes(scene_files, label_files, output_folder)