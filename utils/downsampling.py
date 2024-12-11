import numpy as np

"""
    Random Downsample will randomly select points until num_points are selected.

    Focused Downsample will select points to fit a ratio specified. 

"""

def random_downsample(xyz, num_points):
    num_points_available = xyz.shape[0]
    if num_points_available < num_points:
        raise ValueError(f"The point cloud has only {num_points_available} points, fewer than the requested {num_points} points.")
    indices = np.random.choice(num_points_available, num_points, replace=False)
    return xyz[indices, :]

def focused_downsample(xyz, colors, num_points, plant_ratio):
    plant_indices = np.where(np.all(colors == [0, 255, 0], axis=1))[0]
    background_indices = np.setdiff1d(np.arange(len(xyz)), plant_indices)
    num_from_labeled = int(num_points * plant_ratio)
    num_from_scene = num_points - num_from_labeled
    if len(plant_indices) < num_from_labeled or len(background_indices) < num_from_scene:
        raise ValueError("Not enough points to sample the specified ratio of labeled to unlabeled points")
    labeled_selection = np.random.choice(plant_indices, num_from_labeled, replace=False)
    scene_selection = np.random.choice(background_indices, num_from_scene, replace=False)
    selected_indices = np.concatenate((labeled_selection, scene_selection))
    np.random.shuffle(selected_indices)
    downsampled_xyz = xyz[selected_indices, :]
    downsampled_colors = colors[selected_indices, :]
    return downsampled_xyz, downsampled_colors