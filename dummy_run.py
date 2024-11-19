# run_sem_seg.py
from models import pointnet2_sem_seg_msg, pointnet2_sem_seg, pointnet_sem_seg
import torch

def get_models(num_classes):
    # Initialize the models, and include identifiable names
    pointnet2_msg_model = {'name': 'PointNet2 MSG', 'model': pointnet2_sem_seg_msg.get_model(num_classes)}
    pointnet2_model = {'name': 'PointNet2', 'model': pointnet2_sem_seg.get_model(num_classes)}
    pointnet_model = {'name': 'PointNet', 'model': pointnet_sem_seg.get_model(num_classes)}
    
    # Return them as an array
    return [pointnet2_msg_model, pointnet2_model, pointnet_model]

def run_models_on_input(models, xyz):
    for model_info in models:
        model = model_info['model']
        name  = model_info['name']
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output, _ = model(xyz)
            print(f"Model: {name}, Output Shape: {output.shape}")

def main():
    import torch
    num_classes = 13  # Number of semantic segmentation classes

    # Get the models
    models = get_models(num_classes)

    # Create random input data
    xyz = torch.rand(6, 9, 2048)  # Batch size 6, 9 channels, 2048 points

    # Run the models on the input data and print the output shapes
    run_models_on_input(models, xyz)

if __name__ == "__main__":
    main()