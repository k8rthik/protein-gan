"""`folduzz` command line: fetch, preprocess, train, generate, evaluate.

Every argument is validated here, at the boundary, so the library modules can
assume their inputs are sane and raise `FolduzzError` only for genuine data
problems. Errors are printed as one line on stderr with a non-zero exit code --
no tracebacks in the user's face unless `--traceback` is passed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from folduzz import config
from folduzz.errors import FolduzzError, InvalidInputError

EXIT_OK = 0
EXIT_ERROR = 1


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def _unit_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a fraction in [0, 1], got {value}")
    return value


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="folduzz",
        description=(
            "Experimental DCGAN over 64x64 CA-CA distance matrices from PDB structures."
        ),
    )
    parser.add_argument(
        "--traceback", action="store_true", help="show full tracebacks on error"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="download PDB structures from RCSB")
    fetch.add_argument("--ids", type=Path, default=config.PDB_ID_FILE, help="PDB ID list file")
    fetch.add_argument("--out", type=Path, default=config.RAW_DIR, help="output directory")
    fetch.add_argument("--limit", type=_positive_int, default=None, help="stop after N entries")
    fetch.add_argument(
        "--delay",
        type=_positive_float,
        default=config.FETCH_DELAY_SECONDS,
        help="seconds between requests (politeness)",
    )

    pre = subparsers.add_parser(
        "preprocess", help="turn raw PDB files into normalised distance matrices"
    )
    pre.add_argument("--raw", type=Path, default=config.RAW_DIR)
    pre.add_argument("--out", type=Path, default=config.PROCESSED_DIR)
    pre.add_argument("--matrix-size", type=_positive_int, default=config.MATRIX_SIZE)
    pre.add_argument("--stride", type=_positive_int, default=config.WINDOW_STRIDE)
    pre.add_argument(
        "--max-distance", type=_positive_float, default=config.MAX_DISTANCE_ANGSTROM
    )
    pre.add_argument("--val-fraction", type=_unit_float, default=config.VAL_FRACTION)

    train = subparsers.add_parser("train", help="train the DCGAN")
    train.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    train.add_argument("--run-dir", type=Path, default=config.RUNS_DIR / "default")
    train.add_argument("--epochs", type=_positive_int, default=config.DEFAULT_TRAIN_CONFIG.epochs)
    train.add_argument(
        "--batch-size", type=_positive_int, default=config.DEFAULT_TRAIN_CONFIG.batch_size
    )
    train.add_argument(
        "--learning-rate", type=_positive_float, default=config.DEFAULT_TRAIN_CONFIG.learning_rate_g
    )
    train.add_argument("--seed", type=int, default=config.DEFAULT_TRAIN_CONFIG.seed)
    train.add_argument("--device", default=None, help="mps, cpu or cuda (default: auto)")
    train.add_argument("--resume", action="store_true", help="resume from last checkpoint")

    generate = subparsers.add_parser("generate", help="sample matrices from a checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--out", type=Path, default=config.GENERATED_DIR / "samples.npy")
    generate.add_argument("--count", type=_positive_int, default=config.DEFAULT_NUM_SAMPLES)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--device", default=None)
    generate.add_argument(
        "--symmetrize",
        action="store_true",
        help="post-process samples with (M + M^T)/2 and a zeroed diagonal",
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="score generated matrices against real ones and baselines"
    )
    evaluate.add_argument("--samples", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    evaluate.add_argument("--split", choices=("train", "val"), default="val")
    evaluate.add_argument("--out", type=Path, default=config.REPORTS_DIR)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument(
        "--embed-count",
        type=_positive_int,
        default=128,
        help="how many matrices to run through 3D embedding (the slow metric)",
    )
    return parser


def _run_fetch(args: argparse.Namespace) -> int:
    from folduzz.fetch import fetch_all, parse_id_file, summarize

    pdb_ids = parse_id_file(args.ids)
    total = len(pdb_ids) if args.limit is None else min(args.limit, len(pdb_ids))
    print(f"fetching {total} structures into {args.out} (delay {args.delay}s)")

    seen = {"n": 0}

    def report(result) -> None:  # noqa: ANN001 - FetchResult
        seen["n"] += 1
        if result.status == "failed":
            print(f"  [{seen['n']}/{total}] {result.pdb_id} FAILED: {result.message}")
        elif seen["n"] % 25 == 0 or seen["n"] == total:
            print(f"  [{seen['n']}/{total}] {result.pdb_id} {result.status}")

    results = fetch_all(
        pdb_ids, args.out, delay=args.delay, limit=args.limit, on_result=report
    )
    counts = summarize(results)
    print(
        f"done: {counts['downloaded']} downloaded, {counts['cached']} cached, "
        f"{counts['failed']} failed"
    )
    return EXIT_OK


def _run_preprocess(args: argparse.Namespace) -> int:
    from folduzz.preprocess import PreprocessOptions, preprocess_directory

    options = PreprocessOptions(
        matrix_size=args.matrix_size,
        stride=args.stride,
        max_distance=args.max_distance,
        min_chain_length=max(args.matrix_size, config.MIN_CHAIN_LENGTH),
        val_fraction=args.val_fraction,
    )
    report = preprocess_directory(args.raw, args.out, options)
    print(
        f"{report.structures_used} structures -> {report.window_count} windows "
        f"({report.train_count} train / {report.val_count} val); "
        f"{report.structures_skipped} skipped; "
        f"{100 * report.clipped_fraction:.2f}% of distances clipped at "
        f"{args.max_distance} A"
    )
    print(f"wrote {args.out}/train.npy, {args.out}/val.npy, {args.out}/{config.MANIFEST_NAME}")
    return EXIT_OK


def _run_train(args: argparse.Namespace) -> int:
    from folduzz.train import train

    cfg = config.DEFAULT_TRAIN_CONFIG.evolve(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate_g=args.learning_rate,
        learning_rate_d=args.learning_rate,
        seed=args.seed,
    )
    result = train(
        data_dir=args.data,
        run_dir=args.run_dir,
        cfg=cfg,
        device_name=args.device,
        resume=args.resume,
    )
    print(
        f"trained {result.epochs_completed} epochs in {result.seconds:.0f}s on "
        f"{result.device}; final D loss {result.final_d_loss:.4f}, "
        f"G loss {result.final_g_loss:.4f}"
    )
    print(f"checkpoint: {result.checkpoint_path}")
    return EXIT_OK


def _run_generate(args: argparse.Namespace) -> int:
    from folduzz.generate import generate_to_file

    summary = generate_to_file(
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        count=args.count,
        seed=args.seed,
        device_name=args.device,
        symmetrize=args.symmetrize,
    )
    print(
        f"wrote {summary.count} samples of shape "
        f"{summary.matrix_size}x{summary.matrix_size} to {summary.path}"
        + (" (symmetrized)" if args.symmetrize else "")
    )
    return EXIT_OK


def _run_evaluate(args: argparse.Namespace) -> int:
    from folduzz.evaluate import evaluate_to_files

    paths = evaluate_to_files(
        samples_path=args.samples,
        data_dir=args.data,
        split=args.split,
        out_dir=args.out,
        seed=args.seed,
        embed_count=args.embed_count,
    )
    print(f"wrote {paths.json_path}")
    print(f"wrote {paths.markdown_path}")
    if paths.preview_path is not None:
        print(f"wrote {paths.preview_path} (top row real, bottom row generated)")
    print()
    print(paths.markdown_path.read_text())
    return EXIT_OK


_HANDLERS = {
    "fetch": _run_fetch,
    "preprocess": _run_preprocess,
    "train": _run_train,
    "generate": _run_generate,
    "evaluate": _run_evaluate,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:  # argparse already enforces this
        raise InvalidInputError(f"unknown command {args.command!r}")
    try:
        return handler(args)
    except FolduzzError as exc:
        if args.traceback:
            raise
        print(f"folduzz {args.command}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
