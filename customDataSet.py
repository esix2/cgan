import os
import torch
from torch.utils.data import Dataset 
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, simulated_map_dir, measured_map_dir, transform=None):
        self.simulated_map_dir = simulated_map_dir
        self.measured_map_dir = measured_map_dir
        self.transform = transform
        self.simulated_map_files = os.listdir(simulated_map_dir)
        self.measured_map_files = os.listdir(measured_map_dir)

    def __len__(self):
        return min(len(self.simulated_map_files), len(self.measured_map_files))

    def __getitem__(self, idx):
        simulated_map_name = os.path.join(self.simulated_map_dir, self.simulated_map_files[idx])
        measured_map_name = os.path.join(self.measured_map_dir, self.measured_map_files[idx])
        simulated_map = Image.open(simulated_map_name).convert("L")  # Convert to grayscale
        measured_map = Image.open(measured_map_name).convert("L")  # Convert to grayscale
        if self.transform:
            simulated_map = self.transform(simulated_map)
            measured_map = self.transform(measured_map)
        return simulated_map, measured_map

def get_data_loaders(dpm_dir, irt_dir, batch_size, test_split=0.2, seed=42):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = CustomDataset(dpm_dir, irt_dir, transform=transform)
    
    # Determine the size of training and test datasets
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    
    # Split the dataset into training and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


