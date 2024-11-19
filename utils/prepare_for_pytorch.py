# utils/prepare_for_pytorch.py
import torch
import numpy as np

def prepare_for_pytorch(points, batch_size=1):
    points = points.T  # Transpose to (3, num_points)
    points = np.expand_dims(points, axis=0)  # Add batch dimension
    points = torch.tensor(points, dtype=torch.float32)
    points = points.repeat(batch_size, 1, 1)  # Repeat for batch size
    return points.to('cuda')  # Send to GPU if available