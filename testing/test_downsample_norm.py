import os
import sys
import numpy as np
import open3d as o3d
import json

# Assuming that point_cloud_processing.py is in the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import functions from point_cloud_processing module
from utils.downsampling import random_downsample, focused_downsample
from utils.normalization import normalize_point_cloud

from utils.io import load_point_cloud, save_point_cloud

def test_random_downsample(ply_filepath, output_dir, num_points):
    points, colors, _ = load_point_cloud(ply_filepath)
    downsampled_points = random_downsample(points, num_points)
    downsampled_colors = colors[:len(downsampled_points)]  # Ensure colors match the downsampled points
    downsampled_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(ply_filepath))[0]}_random_downsampled.ply")
    save_point_cloud(downsampled_points, downsampled_colors, downsampled_filename)
    print(f"Saved random downsampled point cloud to '{downsampled_filename}'")

def test_focused_downsample(ply_filepath, output_dir, num_points, plant_ratio):
    points, colors, _ = load_point_cloud(ply_filepath)
    downsampled_points, downsampled_colors = focused_downsample(points, colors, num_points, plant_ratio)
    focused_downsampled_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(ply_filepath))[0]}_focused_downsampled.ply")
    save_point_cloud(downsampled_points, downsampled_colors, focused_downsampled_filename)
    print(f"Saved focused downsampled point cloud to '{focused_downsampled_filename}'")

def test_normalize(ply_filepath, output_dir):
    points, colors, _ = load_point_cloud(ply_filepath)
    normalized_points, centroid, furthest_distance = normalize_point_cloud(points)
    normalized_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(ply_filepath))[0]}_normalized.ply")
    save_point_cloud(normalized_points, colors, normalized_filename)
    print(f"Saved normalized point cloud to '{normalized_filename}'")

def main(config):
    data_path = config['data_path']
    output_dir = 'testing/output_dir'
    num_points = config['points_list'][0]
    plant_ratio = config['plant_ratio']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(data_path):
        if filename.endswith('.ply'):
            filepath = os.path.join(data_path, filename)
            
            # Perform random downsampling
            test_random_downsample(filepath, output_dir, num_points)
            
            # Perform focused downsampling
            test_focused_downsample(filepath, output_dir, num_points, plant_ratio)
            
            # Perform normalization
            test_normalize(filepath, output_dir)

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), '..', 'config.json'), 'r') as f:
        config = json.load(f)
    main(config)