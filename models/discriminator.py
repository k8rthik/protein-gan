import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=4, stride=2, padding=1
            ),  # Shape: (32, input_size/2, input_size/2)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                32, 64, kernel_size=4, stride=2, padding=1
            ),  # Shape: (64, input_size/4, input_size/4)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # Shape: (128, input_size/8, input_size/8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),  # Shape: (1, 1, 1)
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1).squeeze(1)  # Flatten for binary output
