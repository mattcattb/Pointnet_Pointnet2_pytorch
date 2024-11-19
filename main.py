# main.py
import argparse
from data_processing import load_ply
from utils.downsampling import random_downsample, voxel_downsample
from utils.normalization import normalize_points
from utils.prepare_for_pytorch import prepare_for_pytorch

def process_pointcloud(file_path, batch_size, downsample_method="random", num_points=2048, voxel_size=0.05):
    points = load_ply(file_path)
    if downsample_method == "random":
        points = random_downsample(points, num_points)
    elif downsample_method == "voxel":
        points = voxel_downsample(points, voxel_size)
    points = normalize_points(points)
    points_tensor = prepare_for_pytorch(points, batch_size=batch_size)
    
    return points_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Point Cloud Data.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the .ply file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for PyTorch tensor')
    parser.add_argument('--downsample_method', type=str, default="random", choices=["random", "voxel"], help='Downsample method')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points for random downsampling')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size for voxel downsampling')
    
    args = parser.parse_args()
    
    points_tensor = process_pointcloud(args.file_path, args.batch_size, args.downsample_method, args.num_points, args.voxel_size)
    print(points_tensor.shape)