import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DistanceMatrixDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load distance matrix from CSV
        file_path = os.path.join(self.data_dir, self.files[idx])
        distance_matrix = np.loadtxt(file_path, delimiter=",")

        # Convert to torch tensor and add a channel dimension (1, H, W) for convolutional layers
        distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32).unsqueeze(
            0
        )

        # Normalize values to range [-1, 1]
        distance_matrix = (distance_matrix - distance_matrix.min()) / (
            distance_matrix.max() - distance_matrix.min()
        )
        distance_matrix = distance_matrix * 2 - 1  # Scale to [-1, 1]

        if self.transform:
            distance_matrix = self.transform(distance_matrix)

        return distance_matrix
