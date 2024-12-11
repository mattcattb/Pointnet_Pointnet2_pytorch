# utils/normalization.py
import numpy as np

def normalize_point_cloud(points):
    centroid = np.mean(points[:, :3], axis=0)
    points[:, :3] -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(points[:, :3]**2, axis=1)))
    points[:, :3] /= furthest_distance
    return points, centroid, furthest_distance