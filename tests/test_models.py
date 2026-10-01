"""Tests for the DCGAN generator and discriminator shapes and behaviour."""

from __future__ import annotations

import pytest
import torch

from folduzz import config
from folduzz.errors import InvalidInputError
from folduzz.models import Discriminator, Generator
from folduzz.models.ops import symmetrize


class TestGenerator:
    def test_output_shape_is_exactly_the_matrix_size(self):
        generator = Generator(latent_dim=16, output_size=64)
        out = generator(torch.randn(4, 16))
        assert out.shape == (4, 1, 64, 64)

    def test_accepts_already_reshaped_latents(self):
        generator = Generator(latent_dim=16, output_size=64)
        assert generator(torch.randn(2, 16, 1, 1)).shape == (2, 1, 64, 64)

    def test_output_is_in_tanh_range(self):
        generator = Generator(latent_dim=16, output_size=64)
        out = generator(torch.randn(8, 16))
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_no_interpolation_resize_is_needed(self):
        """The original generator built a 32x32 stack and then bilinearly
        upsampled to 64x64, which blurs every sample. The fixed stack reaches
        the target size by transposed convolution alone."""
        generator = Generator(latent_dim=16, output_size=64)
        assert generator.native_size == 64

    def test_rejects_unsupported_size(self):
        with pytest.raises(InvalidInputError):
            Generator(latent_dim=16, output_size=63)

    def test_rejects_bad_latent_dim(self):
        with pytest.raises(InvalidInputError):
            Generator(latent_dim=0, output_size=64)

    def test_rejects_wrong_latent_width(self):
        generator = Generator(latent_dim=16, output_size=64)
        with pytest.raises(InvalidInputError):
            generator(torch.randn(4, 17))

    def test_supports_32_and_128(self):
        for size in (32, 128):
            out = Generator(latent_dim=8, output_size=size)(torch.randn(2, 8))
            assert out.shape == (2, 1, size, size)

    def test_is_deterministic_in_eval_mode(self):
        generator = Generator(latent_dim=16, output_size=64).eval()
        z = torch.randn(3, 16)
        with torch.no_grad():
            torch.testing.assert_close(generator(z), generator(z))


class TestDiscriminator:
    def test_returns_one_logit_per_sample(self):
        discriminator = Discriminator(input_size=64)
        out = discriminator(torch.randn(5, 1, 64, 64))
        assert out.shape == (5,)

    def test_original_shape_bug_is_fixed(self):
        """With the original 3-stride stack, a 64x64 input reached the final
        4x4-kernel conv as 8x8, producing a 5x5 map; `view(-1, 1).squeeze(1)`
        then returned 25 values per sample and the BCE loss silently broadcast
        against a batch-sized target."""
        discriminator = Discriminator(input_size=64)
        assert discriminator(torch.randn(3, 1, 64, 64)).numel() == 3

    def test_outputs_logits_not_probabilities(self):
        """Sigmoid lives in BCEWithLogitsLoss, not in the model."""
        discriminator = Discriminator(input_size=64)
        out = discriminator(torch.randn(64, 1, 64, 64) * 20)
        assert out.min() < 0.0 or out.max() > 1.0

    def test_rejects_wrong_spatial_size(self):
        discriminator = Discriminator(input_size=64)
        with pytest.raises(InvalidInputError):
            discriminator(torch.randn(2, 1, 32, 32))

    def test_rejects_wrong_channel_count(self):
        discriminator = Discriminator(input_size=64)
        with pytest.raises(InvalidInputError):
            discriminator(torch.randn(2, 3, 64, 64))

    def test_rejects_unsupported_size(self):
        with pytest.raises(InvalidInputError):
            Discriminator(input_size=100)


class TestGeneratorDiscriminatorTogether:
    def test_gradients_flow_from_discriminator_into_generator(self):
        generator = Generator(latent_dim=16, output_size=64)
        discriminator = Discriminator(input_size=64)
        loss = discriminator(generator(torch.randn(4, 16))).mean()
        loss.backward()
        grads = [p.grad for p in generator.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_default_config_sizes_are_compatible(self):
        generator = Generator(config.LATENT_DIM, config.MATRIX_SIZE)
        discriminator = Discriminator(config.MATRIX_SIZE)
        out = discriminator(generator(torch.randn(2, config.LATENT_DIM)))
        assert out.shape == (2,)


class TestSymmetrize:
    def test_makes_symmetric_with_zero_diagonal(self):
        matrix = torch.randn(2, 1, 8, 8)
        out = symmetrize(matrix, diagonal_value=-1.0)
        torch.testing.assert_close(out, out.transpose(-1, -2))
        assert torch.allclose(out[..., range(8), range(8)], torch.full((2, 1, 8), -1.0))

    def test_does_not_mutate_input(self):
        matrix = torch.randn(1, 1, 4, 4)
        before = matrix.clone()
        symmetrize(matrix)
        torch.testing.assert_close(matrix, before)

    def test_rejects_non_square(self):
        with pytest.raises(InvalidInputError):
            symmetrize(torch.randn(1, 1, 4, 5))
