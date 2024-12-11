import open3d as o3d
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os


def load_point_cloud(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    data = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255  # Convert the color range from [0, 1] to [0, 255]
    return data, colors, pcd

def save_point_cloud(points, labels, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = labels.astype(np.int32)
    colors = plt.get_cmap("tab20")(labels % 20)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)

def create_experiment_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "outputs"), exist_ok=True)
    return experiment_dir

def log_results(experiment_dir, log_type, message):
    log_filename = os.path.join(experiment_dir, "logs", f"{log_type}.log")
    with open(log_filename, 'a') as f:
        f.write(message + '\n')

def save_metadata(experiment_dir, metadata):
    metadata_filename = os.path.join(experiment_dir, "metadata.txt")
    with open(metadata_filename, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")