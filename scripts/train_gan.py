import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matrix_dataloader import DistanceMatrixDataset
from torch.utils.data import DataLoader

from models.discriminator import Discriminator
from models.generator import Generator

# Hyperparameters
latent_dim = 100
output_size = 64  # Adjust based on your distance matrix size
lr = 0.0002
epochs = 10000
batch_size = 32

# Initialize models and optimizers
generator = Generator(latent_dim, output_size)
discriminator = Discriminator(output_size)
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


# Load data (Assume you have a DataLoader for your matrices)
def load_data(batch_size, data_dir="./data/processed/"):
    dataset = DistanceMatrixDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Training loop
for epoch in range(epochs):
    for i, real_imgs in enumerate(load_data(batch_size)):
        # Ground truths
        valid = torch.ones(real_imgs.size(0), 1)
        fake = torch.zeros(real_imgs.size(0), 1)

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise and generate fake images
        z = torch.randn(real_imgs.size(0), latent_dim)
        generated_imgs = generator(z)

        # Generator loss
        g_loss = adversarial_loss(discriminator(generated_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real and fake images
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Print progress
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] [Batch {i}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            )
