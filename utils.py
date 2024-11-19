import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from path import Path
import plotly.graph_objects as go
import plotly.express as px

from plyfile import PlyData
import numpy as np

def read_ply(file_path):
    # Read the PLY file
    ply_data = PlyData.read(file_path)

    # Extract vertex coordinates
    vertices = np.vstack([ply_data['vertex'].data['x'],
                          ply_data['vertex'].data['y'],
                          ply_data['vertex'].data['z']]).T

    return vertices




def random_downsample(vertices, num_points=1024):
    if vertices.shape[0] > num_points:
        indices = np.random.choice(vertices.shape[0], num_points, replace=False)
        vertices = vertices[indices]
    return vertices

# Downsample and prepare for PointCNN
def normalize(pointcloud):
  norm_pc = pointcloud - np.mean(pointcloud, axis=0)
  norm_pc /= np.max(np.linalg.norm(norm_pc,axis=1))
  return norm_pc

def normalize_pair(pointcloud, subset_pointclouds):
  # normalize pointcloud and list of subsets of segementation pointclouds
  if (subset_pointclouds is not list):
    subset_pointclouds = [subset_pointclouds]

  norm_pc = pointcloud - np.mean(pointcloud, axis=0)
  max = np.max(np.linalg.norm(norm_pc,axis=1))
  norm_pc /= max
  normed_subset = []
  for subset in subset_pointclouds:
    x = subset - np.mean(pointcloud, axis=0)
    x /= max
    normed_subset.append(x)

  return (norm_pc, normed_subset)