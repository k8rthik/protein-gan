"""DCGAN discriminator: one-channel distance matrix -> one logit per sample.

Kept from the original: strided convolutions, LeakyReLU(0.2), BatchNorm on the
inner blocks, no normalisation on the first block.

Fixed two things:

1. Shape. The original always used three stride-2 blocks and then a 4x4
   stride-1 convolution, so a 64x64 input arrived at that last layer as 8x8 and
   left as 5x5. `view(-1, 1).squeeze(1)` turned that into 25 numbers per sample
   which BCELoss then broadcast against a batch-sized target: the loss was
   computed, but not the loss anyone intended. The number of stride-2 blocks now
   depends on `input_size`, so the final layer always sees 4x4 and emits 1x1.
2. Sigmoid. It is gone from the model; training uses `BCEWithLogitsLoss`, which
   is the numerically stable formulation of the same objective.
"""

from __future__ import annotations

import torch
from torch import nn

from folduzz import config
from folduzz.errors import InvalidInputError
from folduzz.models.ops import stage_count


class Discriminator(nn.Module):
    """Maps `(batch, 1, size, size)` to `(batch,)` real-vs-fake logits."""

    def __init__(
        self,
        input_size: int = config.MATRIX_SIZE,
        base_channels: int = config.DISCRIMINATOR_BASE_CHANNELS,
    ) -> None:
        super().__init__()
        if base_channels < 1:
            raise InvalidInputError(f"base_channels must be >= 1, got {base_channels}")

        stages = stage_count(input_size)
        self.input_size = input_size

        channels = [base_channels << i for i in range(stages)]
        layers: list[nn.Module] = [
            nn.Conv2d(1, channels[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(config.LEAKY_RELU_SLOPE, inplace=True),
        ]
        for index in range(stages - 1):
            layers += [
                nn.Conv2d(
                    channels[index], channels[index + 1], kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(channels[index + 1]),
                nn.LeakyReLU(config.LEAKY_RELU_SLOPE, inplace=True),
            ]
        # Final 4x4 -> 1x1. No Sigmoid: see module docstring.
        layers += [nn.Conv2d(channels[-1], 1, kernel_size=4, stride=1, padding=0)]
        self.model = nn.Sequential(*layers)

    def forward(self, matrices: torch.Tensor) -> torch.Tensor:
        if matrices.ndim != 4:
            raise InvalidInputError(
                f"expected (batch, 1, size, size), got {tuple(matrices.shape)}"
            )
        if matrices.shape[1] != 1:
            raise InvalidInputError(f"expected 1 channel, got {matrices.shape[1]}")
        if matrices.shape[-1] != self.input_size or matrices.shape[-2] != self.input_size:
            raise InvalidInputError(
                f"expected {self.input_size}x{self.input_size} matrices, "
                f"got {matrices.shape[-2]}x{matrices.shape[-1]}"
            )
        return self.model(matrices).reshape(matrices.shape[0])
