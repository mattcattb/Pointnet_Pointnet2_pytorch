# data_processing.py
import numpy as np
from plyfile import PlyData

def load_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
    return points

# Test loading function
if __name__ == "__main__":
    points = load_ply("example.ply")
    print(points.shape)