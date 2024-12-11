import os
import json
import torch
import numpy as np
from models import pointnet2_sem_seg_msg, pointnet2_sem_seg, pointnet_sem_seg
from utils.io import load_point_cloud, save_metadata, save_point_cloud, log_results, create_experiment_dir

def get_models(config):
    pointnet2_msg_model = {'name': 'PointNet2 MSG', 'model': pointnet2_sem_seg_msg.get_model(config['num_classes'])}
    pointnet2_model = {'name': 'PointNet2', 'model': pointnet2_sem_seg.get_model(config['num_classes'])}
    pointnet_model = {'name': 'PointNet', 'model': pointnet_sem_seg.get_model(config['num_classes'])}
    return [pointnet2_msg_model, pointnet2_model, pointnet_model]

def normalize_point_cloud(points, required_channels=9):
    centroid = np.mean(points[:, :3], axis=0)
    points[:, :3] -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(points[:, :3]**2, axis=1)))
    points[:, :3] /= furthest_distance
    if points.shape[1] < required_channels:
        paddings = np.zeros((points.shape[0], required_channels - points.shape[1]))
        points = np.hstack((points, paddings))
    return points, centroid, furthest_distance

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

def random_downsample(xyz, colors, num_points, plant_ratio):
    B, C, N = xyz.shape
    plant_indices = np.where(np.all(colors == [0, 255, 0], axis=1))[0]
    background_indices = np.setdiff1d(np.arange(N), plant_indices)
    num_from_labeled = int(num_points * plant_ratio)
    num_from_scene = num_points - num_from_labeled
    if len(plant_indices) < num_from_labeled or len(background_indices) < num_from_scene:
        raise ValueError("Not enough points to sample the specified ratio of labeled to unlabeled points")
    labeled_selection = np.random.choice(plant_indices, num_from_labeled, replace=False)
    scene_selection = np.random.choice(background_indices, num_from_scene, replace=False)
    selected_indices = np.concatenate((labeled_selection, scene_selection))
    np.random.shuffle(selected_indices)
    downsampled_xyz = xyz[:, :, selected_indices]
    downsampled_labels = np.concatenate((np.ones(num_from_labeled), np.zeros(num_from_scene)))
    return downsampled_xyz, downsampled_labels, selected_indices

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
            data = prepare_input_data(data, required_channels=9).T
            data = np.expand_dims(data, axis=0)
            batch_data = np.tile(data, (batch_size, 1, 1))
            results_dict[filename] = {}
            for num_points in points_list:
                print("batch_data: ", batch_data, batch_data.shape)
                if batch_data.shape[2] < num_points:
                    log_results(experiment_dir, "errors", f"File {filepath} has fewer points than {num_points}, skipping downsampling.")
                    continue
                downsampled_data, downsampled_labels, selected_indices = random_downsample(batch_data, colors, num_points, plant_ratio)
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