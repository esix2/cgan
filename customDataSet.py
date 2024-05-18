import os
from torch.utils.data import Dataset 
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

