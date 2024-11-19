# utils/normalization.py
import numpy as np

def normalize_points(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points /= max_dist
    return points