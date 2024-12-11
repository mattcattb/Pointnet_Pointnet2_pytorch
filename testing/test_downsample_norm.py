import os
import sys
import numpy as np
import open3d as o3d
import json
# Assuming that main_script.py is in the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import functions from the main script
from run import random_downsample, normalize_point_cloud
from utils.io import load_point_cloud, save_point_cloud

def test_downsample(ply_filepath, output_dir, num_points):
    data, colors, _ = load_point_cloud(ply_filepath)
    downsampled_points, downsampled_colors = random_downsample(data, colors, num_points)
    downsampled_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(ply_filepath))[0]}_downsampled.ply")
    save_point_cloud(downsampled_points, downsampled_colors, downsampled_filename)
    print(f"Saved downsampled point cloud to '{downsampled_filename}'")

def test_normalize(ply_filepath, output_dir):
    data, colors, _ = load_point_cloud(ply_filepath)
    normalized_points = normalize_point_cloud(data)
    normalized_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(ply_filepath))[0]}_normalized.ply")
    save_point_cloud(normalized_points, colors, normalized_filename)
    print(f"Saved normalized point cloud to '{normalized_filename}'")

def main(config):
    data_path = config['data_path']
    output_dir = "output"
    num_points = config['points_list'][0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(data_path):
        if filename.endswith('.ply'):
            filepath = os.path.join(data_path, filename)
            
            # Perform downsampling
            test_downsample(filepath, output_dir, num_points)
            
            # Perform normalization
            test_normalize(filepath, output_dir)

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), '..', 'config.json'), 'r') as f:
        config = json.load(f)
    main(config)