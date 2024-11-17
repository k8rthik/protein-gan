import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.discriminator import Discriminator
from models.generator import Generator
from scripts.matrix_dataloader import DistanceMatrixDataset

# Hyperparameters
latent_dim = 100
batch_size = 32
epochs = 10000
learning_rate = 0.0002
target_size = 64  # Matrix size (e.g., 64x64)

# Device configuration (use GPU if available)
device = torch.device("mps")

# Initialize Generator and Discriminator
generator = Generator(latent_dim, target_size).to(device)
discriminator = Discriminator(target_size).to(device)

# Loss function (Binary Cross-Entropy)
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(
    discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
)

# Load preprocessed data
data_dir = "./data/processed/"
dataset = DistanceMatrixDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training Loop
for epoch in range(epochs):
    for i, real_matrices in enumerate(dataloader):
        # Transfer real matrices to device
        real_matrices = real_matrices.to(device)

        # Ground truths
        valid = torch.ones((real_matrices.size(0), 1), device=device, dtype=torch.float)
        fake = torch.zeros((real_matrices.size(0), 1), device=device, dtype=torch.float)

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate fake matrices
        z = torch.randn((real_matrices.size(0), latent_dim), device=device)
        generated_matrices = generator(z)

        # Generator loss (fool the discriminator)
        g_loss = adversarial_loss(discriminator(generated_matrices))
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real loss
        real_loss = adversarial_loss(discriminator(real_matrices), valid)

        # Fake loss
        fake_loss = adversarial_loss(discriminator(generated_matrices.detach()), fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Print progress
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] [Batch {i}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]"
            )

    # Save generated samples periodically
    if epoch % 100 == 0:
        os.makedirs("./data/generated_samples/", exist_ok=True)
        torch.save(
            generator.state_dict(),
            f"./data/generated_samples/generator_epoch_{epoch}.pth",
        )
        torch.save(
            discriminator.state_dict(),
            f"./data/generated_samples/discriminator_epoch_{epoch}.pth",
        )
        print(f"Model saved at epoch {epoch}")
