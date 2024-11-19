import torch
import torch.nn.functional as F
from models.pointnet2_sem_seg_msg import get_model as get_model_msg
from models.pointnet2_sem_seg import get_model as get_model_pointnet2
from models.pointnet_sem_seg import get_model as get_model_pointnet

def test_model(model, input_shape):
    try:
        xyz = torch.rand(*input_shape)
        model.eval()
        with torch.no_grad():
            output, _ = model(xyz)
        return True
    except Exception as e:
        print(f"Error with input shape {input_shape}: {e}")
        return False

def find_max_points_with_no_memory_limit(model, batch_size, input_channels, step=512, max_points=100000):
    points = step
    while points <= max_points:
        input_shape = (batch_size, input_channels, points)
        if not test_model(model, input_shape):
            return points - step
        points += step
    return max_points

def main():
    num_classes = 13  # Number of semantic segmentation classes
    batch_size = 6    # Adjust according to your typical batch size
    input_channels = 9  # Usually 3 for (XYZ) or up to 9 if including additional features
    
    models = [
        {'name': 'PointNet', 'model': get_model_pointnet(num_classes)},
        {'name': 'PointNet2', 'model': get_model_pointnet2(num_classes)},
        {'name': 'PointNet2 MSG', 'model': get_model_msg(num_classes)}
    ]
    
    max_points_list = []
    for model_info in models:
        name = model_info['name']
        model = model_info['model']
        max_points = find_max_points_with_no_memory_limit(model, batch_size, input_channels)
        print(f"{name}: Maximum points = {max_points}")
        max_points_list.append((name, max_points))
    
    print(max_points_list)

if __name__ == "__main__":
    main()