"""DCGAN generator: latent vector -> one-channel distance-matrix image.

Kept from the original: a plain transposed-convolution DCGAN stack with
BatchNorm + ReLU and a Tanh output, matching the [-1, 1] data range.

Fixed: the original stack produced 32x32 regardless of `output_size` and then
called `F.interpolate(..., mode="bilinear")` to reach the requested size, so
every sample was a blurred upsample and the 64x64 structure the discriminator
saw was not something the generator could actually control. The stack now
reaches `output_size` exactly, by construction.
"""

from __future__ import annotations

import torch
from torch import nn

from folduzz import config
from folduzz.errors import InvalidInputError
from folduzz.models.ops import BASE_RESOLUTION, stage_count


class Generator(nn.Module):
    """Maps `(batch, latent_dim)` noise to `(batch, 1, size, size)` in [-1, 1]."""

    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        output_size: int = config.MATRIX_SIZE,
        base_channels: int = config.GENERATOR_BASE_CHANNELS,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise InvalidInputError(f"latent_dim must be >= 1, got {latent_dim}")
        if base_channels < 1:
            raise InvalidInputError(f"base_channels must be >= 1, got {base_channels}")

        stages = stage_count(output_size)
        self.latent_dim = latent_dim
        self.output_size = output_size

        channels = [max(base_channels >> i, 8) for i in range(stages)]
        layers: list[nn.Module] = [
            # latent -> BASE_RESOLUTION x BASE_RESOLUTION
            nn.ConvTranspose2d(latent_dim, channels[0], kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        ]
        for index in range(stages - 1):
            layers += [
                nn.ConvTranspose2d(
                    channels[index], channels[index + 1], kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(channels[index + 1]),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.ConvTranspose2d(channels[-1], 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    @property
    def native_size(self) -> int:
        """Spatial size the convolution stack produces without any resizing."""
        return BASE_RESOLUTION * 2 ** stage_count(self.output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            latent = z.view(z.size(0), z.size(1), 1, 1)
        elif z.ndim == 4:
            latent = z
        else:
            raise InvalidInputError(
                f"latent must be (batch, latent_dim) or (batch, latent_dim, 1, 1), "
                f"got {tuple(z.shape)}"
            )
        if latent.shape[1] != self.latent_dim:
            raise InvalidInputError(
                f"expected latent_dim {self.latent_dim}, got {latent.shape[1]}"
            )
        return self.model(latent)
