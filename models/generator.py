import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, 128, kernel_size=4, stride=1, padding=0
            ),  # Shape: (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # Shape: (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # Shape: (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            ),  # Shape: (1, output_size, output_size)
            nn.Tanh(),
        )
        self.output_size = output_size

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)  # Reshape to (batch_size, latent_dim, 1, 1)
        img = self.model(z)
        img = nn.functional.interpolate(
            img, size=(self.output_size, self.output_size), mode="bilinear"
        )
        return img
