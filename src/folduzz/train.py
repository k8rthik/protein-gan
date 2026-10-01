"""DCGAN training loop with checkpointing, JSONL logging and resume.

What the original `train_gan.py` did and this does not:

* `adversarial_loss(discriminator(generated_matrices))` was called with one
  argument, so the script raised `TypeError` on its first batch. It had never
  been run.
* The generator was updated *before* the discriminator each step, using a
  discriminator from the previous iteration, and the same forward pass was then
  reused for the discriminator's fake loss.
* Targets were shaped `(batch, 1)` while the discriminator returned `(batch,)`
  (really `(batch * 25,)`; see models/discriminator.py), so BCE broadcast into a
  `(batch, batch)` loss surface.
* `torch.device("mps")` was unconditional, `epochs = 10000` was hard-coded, and
  "save generated samples periodically" actually saved weights into
  `data/generated_samples/` with no way to resume from them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from folduzz import config
from folduzz.checkpoints import (
    DataContract,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)
from folduzz.dataset import DistanceMatrixDataset, load_manifest
from folduzz.errors import CheckpointError, DatasetError
from folduzz.models import Discriminator, Generator

LAST_CHECKPOINT_NAME = "last.pt"
LOG_NAME = "log.jsonl"
CONFIG_NAME = "train_config.json"


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    d_loss: float
    g_loss: float
    d_real_mean: float
    d_fake_mean: float
    seconds: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "epoch": self.epoch,
            "d_loss": self.d_loss,
            "g_loss": self.g_loss,
            "d_real_prob_mean": self.d_real_mean,
            "d_fake_prob_mean": self.d_fake_mean,
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class TrainResult:
    epochs_completed: int
    seconds: float
    device: str
    final_d_loss: float
    final_g_loss: float
    checkpoint_path: Path
    history: tuple[EpochStats, ...]


def _init_weights(module: nn.Module) -> None:
    """DCGAN paper initialisation: N(0, 0.02) on convs, N(1, 0.02) on BatchNorm."""
    name = module.__class__.__name__
    if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0.0)


def _append_log(path: Path, stats: EpochStats) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stats.as_dict()) + "\n")


def _run_epoch(
    epoch: int,
    loader: DataLoader,
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: config.TrainConfig,
) -> EpochStats:
    started = time.monotonic()
    generator.train()
    discriminator.train()
    totals = {"d": 0.0, "g": 0.0, "real": 0.0, "fake": 0.0}
    batches = 0

    for real in loader:
        real = real.to(device)
        batch_size = real.shape[0]
        real_targets = torch.full((batch_size,), cfg.real_label, device=device)
        fake_targets = torch.full((batch_size,), cfg.fake_label, device=device)

        # --- discriminator: real + fake -------------------------------------
        optimizer_d.zero_grad(set_to_none=True)
        real_logits = discriminator(real)
        d_real_loss = criterion(real_logits, real_targets)

        noise = torch.randn(batch_size, cfg.latent_dim, device=device)
        fake = generator(noise)
        fake_logits = discriminator(fake.detach())
        d_fake_loss = criterion(fake_logits, fake_targets)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_d.step()

        # --- generator: fool the just-updated discriminator ------------------
        optimizer_g.zero_grad(set_to_none=True)
        g_loss = criterion(discriminator(fake), torch.ones(batch_size, device=device))
        g_loss.backward()
        optimizer_g.step()

        totals["d"] += float(d_loss.detach())
        totals["g"] += float(g_loss.detach())
        totals["real"] += float(torch.sigmoid(real_logits.detach()).mean())
        totals["fake"] += float(torch.sigmoid(fake_logits.detach()).mean())
        batches += 1

    divisor = max(batches, 1)
    return EpochStats(
        epoch=epoch,
        d_loss=totals["d"] / divisor,
        g_loss=totals["g"] / divisor,
        d_real_mean=totals["real"] / divisor,
        d_fake_mean=totals["fake"] / divisor,
        seconds=time.monotonic() - started,
    )


def train(
    data_dir: Path | str = config.PROCESSED_DIR,
    run_dir: Path | str = config.RUNS_DIR / "default",
    cfg: config.TrainConfig = config.DEFAULT_TRAIN_CONFIG,
    device_name: str | None = None,
    resume: bool = False,
    split: str = "train",
) -> TrainResult:
    """Train the DCGAN, writing checkpoints and a JSONL log into `run_dir`."""
    manifest = load_manifest(data_dir)
    contract = DataContract.from_manifest(manifest)
    dataset = DistanceMatrixDataset.from_directory(data_dir, split)
    if dataset.matrix_size != contract.matrix_size:
        raise DatasetError(
            f"manifest says matrix_size={contract.matrix_size} but arrays are "
            f"{dataset.matrix_size}x{dataset.matrix_size}"
        )

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_path / LAST_CHECKPOINT_NAME
    device = resolve_device(device_name)
    torch.manual_seed(cfg.seed)

    effective = cfg.evolve(matrix_size=contract.matrix_size)
    generator = Generator(effective.latent_dim, contract.matrix_size).to(device)
    discriminator = Discriminator(contract.matrix_size).to(device)
    generator.apply(_init_weights)
    discriminator.apply(_init_weights)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=effective.learning_rate_g,
        betas=(effective.beta1, effective.beta2),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=effective.learning_rate_d,
        betas=(effective.beta1, effective.beta2),
    )

    start_epoch = 0
    if resume:
        if not checkpoint_path.is_file():
            raise CheckpointError(f"--resume given but {checkpoint_path} does not exist")
        previous = load_checkpoint(checkpoint_path)
        mismatch = previous.contract.describe_mismatch(contract)
        if mismatch is not None:
            raise CheckpointError(
                f"{checkpoint_path} was trained on different data ({mismatch}); "
                "train a fresh run instead of resuming"
            )
        generator.load_state_dict(previous.generator_state)
        discriminator.load_state_dict(previous.discriminator_state)
        if previous.optimizer_g_state is not None:
            optimizer_g.load_state_dict(previous.optimizer_g_state)
        if previous.optimizer_d_state is not None:
            optimizer_d.load_state_dict(previous.optimizer_d_state)
        start_epoch = previous.epoch
        print(f"resumed from {checkpoint_path} at epoch {start_epoch}")

    loader = DataLoader(
        dataset,
        batch_size=effective.batch_size,
        shuffle=True,
        drop_last=len(dataset) > effective.batch_size,
        num_workers=effective.num_workers,
    )
    criterion = nn.BCEWithLogitsLoss()
    log_path = run_path / LOG_NAME
    (run_path / CONFIG_NAME).write_text(
        json.dumps(
            {
                "train_config": effective.__dict__,
                "contract": contract.__dict__,
                "device": str(device),
                "dataset_size": len(dataset),
                "split": split,
            },
            indent=2,
        )
        + "\n"
    )

    print(
        f"training on {len(dataset)} matrices ({contract.matrix_size}x"
        f"{contract.matrix_size}) for {effective.epochs} epochs on {device}"
    )
    history: list[EpochStats] = []
    started = time.monotonic()
    for offset in range(effective.epochs):
        epoch = start_epoch + offset + 1
        stats = _run_epoch(
            epoch,
            loader,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            criterion,
            device,
            effective,
        )
        history.append(stats)
        _append_log(log_path, stats)
        if epoch % effective.log_every == 0 or offset == effective.epochs - 1:
            print(
                f"  epoch {epoch:4d}  D {stats.d_loss:.4f}  G {stats.g_loss:.4f}  "
                f"D(real) {stats.d_real_mean:.3f}  D(fake) {stats.d_fake_mean:.3f}  "
                f"{stats.seconds:.1f}s"
            )
        if epoch % effective.checkpoint_every == 0:
            save_checkpoint(
                run_path / "checkpoints" / f"epoch_{epoch:05d}.pt",
                generator,
                discriminator,
                effective,
                contract,
                epoch,
            )
            save_checkpoint(
                checkpoint_path,
                generator,
                discriminator,
                effective,
                contract,
                epoch,
                optimizer_g,
                optimizer_d,
            )

    final_epoch = start_epoch + effective.epochs
    save_checkpoint(
        checkpoint_path,
        generator,
        discriminator,
        effective,
        contract,
        final_epoch,
        optimizer_g,
        optimizer_d,
    )
    last = history[-1]
    return TrainResult(
        epochs_completed=effective.epochs,
        seconds=time.monotonic() - started,
        device=str(device),
        final_d_loss=last.d_loss,
        final_g_loss=last.g_loss,
        checkpoint_path=checkpoint_path,
        history=tuple(history),
    )
