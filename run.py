import os
import json
import torch
import numpy as np
from models import pointnet2_sem_seg_msg, pointnet2_sem_seg, pointnet_sem_seg
from utils.io import load_point_cloud, save_metadata, save_point_cloud, log_results, create_experiment_dir
from utils.downsampling import focused_downsample
from utils.normalization import normalize_point_cloud

def get_models(config):
    pointnet2_msg_model = {'name': 'PointNet2 MSG', 'model': pointnet2_sem_seg_msg.get_model(config['num_classes'])}
    pointnet2_model = {'name': 'PointNet2', 'model': pointnet2_sem_seg.get_model(config['num_classes'])}
    pointnet_model = {'name': 'PointNet', 'model': pointnet_sem_seg.get_model(config['num_classes'])}
    return [pointnet2_msg_model, pointnet2_model, pointnet_model]

def prepare_input_data(data, required_channels=9):
    N, C = data.shape
    if C < required_channels:
        padding = np.zeros((N, required_channels - C))
        padded_data = np.hstack((data, padding))
    else:
        padded_data = data[:, :required_channels]
    return padded_data

def run_models_on_input(models, xyz, filename, centroid, scale, num_points):
    results = {'predictions': {}, 'downsampled_points': xyz[0, :3, :].T.tolist(), 'downsampled_labels': None}
    
    xyz[:, :3, :] = (xyz[:, :3, :] - centroid.reshape(-1, 1)) / scale  # Normalize input samples for the batch

    for model_info in models:
        model = model_info['model']
        name = model_info['name']
        model.eval()
        with torch.no_grad():
            output, _ = model(torch.from_numpy(xyz).float())
            pred_labels = output.max(dim=2)[1].cpu().numpy()[0]  # Max log-probability from output
            results['predictions'][name] = pred_labels.tolist()

    return results

def evaluate_accuracy(predictions, labels):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if predictions.shape != labels.shape:
        print(f"Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")
        raise ValueError("Shapes of predictions and labels do not match.")
    total = labels.shape[0]
    correct = np.sum(predictions == labels)
    return correct / total if total > 0 else 0.0


def process_scenes(data_path, points_list, batch_size, plant_ratio, models, experiment_dir):
    results_dict = {}
    for filename in os.listdir(data_path):
        if filename.endswith('.ply'):
            filepath = os.path.join(data_path, filename)
            data, colors, _ = load_point_cloud(filepath)
            if data.shape[0] == 0:
                log_results(experiment_dir, "errors", f"File {filepath} has no points, skipping.")
                continue
            labels = np.all(colors == [0, 255, 0], axis=1).astype(int) # save green points as proper segmentations
            data, centroid, scale = normalize_point_cloud(data)
            data = prepare_input_data(data).T
            data = np.expand_dims(data, axis=0)
            batch_data = np.tile(data, (batch_size, 1, 1))
            results_dict[filename] = {}
            for num_points in points_list:
                print("batch_data: ", batch_data, batch_data.shape)
                if batch_data.shape[2] < num_points:
                    log_results(experiment_dir, "errors", f"File {filepath} has fewer points than {num_points}, skipping downsampling.")
                    continue
                downsampled_data, downsampled_labels, selected_indices = focused_downsample(batch_data, colors, num_points, plant_ratio)
                print("downsampled_data: ", downsampled_data, downsampled_data.shape)
                model_outputs = run_models_on_input(models, downsampled_data, filepath, centroid, scale, num_points)
                
                # Ensure downsampled labels are converted to list
                model_outputs['downsampled_labels'] = downsampled_labels.tolist()
                
                # Store all relevant data in the results dictionary
                results_dict[filename][num_points] = model_outputs

    return results_dict

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    data_path = config['data_path']
    points_list = config['points_list']
    batch_size = config['batch_size']
    plant_ratio = config['plant_ratio']
    base_dir = os.getcwd()
    experiment_dir = create_experiment_dir(base_dir)
    save_metadata(experiment_dir, config)
    models = get_models(config)
    results_dict = process_scenes(data_path, points_list, batch_size, plant_ratio, models, experiment_dir)
    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    main()