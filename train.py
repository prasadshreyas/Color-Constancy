# Shreyas Prasad
# 10/18/23
# CS 7180: Advanced Perception

from torchvision import transforms
from clahe_transform import CLAHETransform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from image_dataset import ImageDataset
from colornet import ColorNet

class CustomTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
            transforms.Resize(32, 32), # Resize to 32x32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def train_model(model, criterion, optimizer, train_loader, num_epochs=20):
    """
    Train the model
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    print('Training complete')

def main():
    """
    Train the model
    """
    transform = CustomTransform()

    train_dataset = ImageDataset(
        dataset_path='data/train/',
        mat_file='data/ground_truth.mat',
        key='illum',
        patch_size=(32, 32),
        transform=transform.transform
    )
    train_loader = DataLoader(train_dataset, batch_size=160, shuffle=True)

    # Model
    model = ColorNet()

    # Loss and Optimizer
    criterion = nn.CosineEmbeddingLoss()  # Adjust as needed
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, num_epochs=20)

    # Save the model
    torch.save(model.state_dict(), 'colornet_model.pth')

if __name__ == '__main__':
    main()