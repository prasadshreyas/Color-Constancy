# Shreyas Prasad
# 10/18/23
# CS 7180: Advanced Perception

import torch
from torch.utils.data import DataLoader
from image_dataset import ImageDataset
from colornet import ColorNet
from train import CustomTransform

def angular_error(ground_truth, corrected, measurement_type='mean'):
    if measurement_type == 'mean':
        e_gt = torch.mean(ground_truth, dim=[2, 3])  # Assuming [N, C, H, W] format
        e_est = torch.mean(corrected, dim=[2, 3])
    elif measurement_type == 'median':
        e_gt = torch.median(ground_truth, dim=[2, 3]).values
        e_est = torch.median(corrected, dim=[2, 3]).values

    error_cos = torch.sum(e_gt * e_est, dim=1) / (torch.norm(e_gt, dim=1) * torch.norm(e_est, dim=1))
    e_angular = torch.rad2deg(torch.acos(error_cos))
    return e_angular

def euclidean_error(ground_truth, corrected, measurement_type='mean'):
    if measurement_type == 'mean':
        e_gt = torch.mean(ground_truth, dim=[2, 3])  # Assuming [N, C, H, W] format
        e_est = torch.mean(corrected, dim=[2, 3])
    elif measurement_type == 'median':
        e_gt = torch.median(ground_truth, dim=[2, 3]).values
        e_est = torch.median(corrected, dim=[2, 3]).values

    e_euclidean = torch.norm(e_gt - e_est, dim=1)
    return e_euclidean

def test_model(model, test_loader):
    """
    Function to test a model's performance on a given test dataset.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            e_angular = angular_error(labels, outputs)
            e_euclidean = euclidean_error(labels, outputs)
            print(f'Angular error: {e_angular}')
            print(f'Euclidean error: {e_euclidean}')


    print('Test completed')

def main():

    transform = CustomTransform().transform
    
    test_dataset = ImageDataset(
        'data/test/',
        'data/ground_truth.mat',
        'illum',
        (32, 32),
        transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=160, shuffle=False)

    # Load model
    model = ColorNet()
    model.load_state_dict(torch.load('colornet_model.pth'))

    # Test the model
    test_model(model, test_loader)

if __name__ == '__main__':
    main()
