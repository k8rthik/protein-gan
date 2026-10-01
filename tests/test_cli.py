"""Tests for argument validation and error handling at the CLI boundary."""

from __future__ import annotations

import json

import numpy as np
import pytest

from folduzz import config
from folduzz.cli import EXIT_ERROR, EXIT_OK, build_parser, main
from folduzz.distance import normalize
from tests.test_metrics_validity import real_matrices

VALID_PDB = (
    b"HEADER    TEST\n"
    + b"ATOM      1  CA  ALA A   1      0.000   0.000   0.000\n" * 40
    + b"END\n"
)


class TestParserValidation:
    def test_requires_a_command(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_rejects_unknown_command(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["frobnicate"])

    @pytest.mark.parametrize(
        "argv",
        [
            ["fetch", "--limit", "0"],
            ["fetch", "--limit", "-3"],
            ["fetch", "--limit", "many"],
            ["fetch", "--delay", "0"],
            ["preprocess", "--matrix-size", "0"],
            ["preprocess", "--stride", "-1"],
            ["preprocess", "--max-distance", "0"],
            ["preprocess", "--val-fraction", "1.5"],
            ["preprocess", "--val-fraction", "-0.1"],
            ["train", "--epochs", "0"],
            ["train", "--batch-size", "0"],
            ["train", "--learning-rate", "0"],
            ["generate", "--checkpoint", "x.pt", "--count", "0"],
            ["evaluate", "--samples", "x.npy", "--split", "nope"],
            ["evaluate", "--samples", "x.npy", "--embed-count", "0"],
        ],
    )
    def test_rejects_invalid_values(self, argv):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)

    def test_generate_requires_a_checkpoint(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["generate"])

    def test_evaluate_requires_samples(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["evaluate"])

    def test_defaults_come_from_config(self):
        args = build_parser().parse_args(["preprocess"])
        assert args.matrix_size == config.MATRIX_SIZE
        assert args.max_distance == config.MAX_DISTANCE_ANGSTROM


class TestErrorReporting:
    def test_missing_id_file_is_a_clean_error(self, tmp_path, capsys):
        code = main(["fetch", "--ids", str(tmp_path / "nope.txt"), "--out", str(tmp_path)])
        assert code == EXIT_ERROR
        assert "error:" in capsys.readouterr().err

    def test_traceback_flag_reraises(self, tmp_path):
        from folduzz.errors import InvalidInputError

        with pytest.raises(InvalidInputError):
            main(["--traceback", "fetch", "--ids", str(tmp_path / "nope.txt")])

    def test_missing_raw_dir_is_a_clean_error(self, tmp_path, capsys):
        code = main(
            ["preprocess", "--raw", str(tmp_path / "nope"), "--out", str(tmp_path / "out")]
        )
        assert code == EXIT_ERROR
        assert "does not exist" in capsys.readouterr().err

    def test_missing_checkpoint_is_a_clean_error(self, tmp_path, capsys):
        code = main(["generate", "--checkpoint", str(tmp_path / "nope.pt")])
        assert code == EXIT_ERROR
        assert "error:" in capsys.readouterr().err

    def test_missing_samples_is_a_clean_error(self, tmp_path, capsys):
        code = main(["evaluate", "--samples", str(tmp_path / "nope.npy")])
        assert code == EXIT_ERROR
        assert "error:" in capsys.readouterr().err


class TestEndToEndThroughTheCli:
    def test_preprocess_then_train_then_generate_then_evaluate(self, tmp_path, capsys):
        raw = config.FIXTURE_DIR / "raw"
        processed = tmp_path / "processed"
        run = tmp_path / "run"
        samples = tmp_path / "gen" / "samples.npy"
        reports = tmp_path / "reports"

        assert main(["preprocess", "--raw", str(raw), "--out", str(processed)]) == EXIT_OK
        assert (processed / "train.npy").is_file()

        assert (
            main(
                [
                    "train",
                    "--data", str(processed),
                    "--run-dir", str(run),
                    "--epochs", "2",
                    "--batch-size", "4",
                    "--device", "cpu",
                ]
            )
            == EXIT_OK
        )
        checkpoint = run / "last.pt"
        assert checkpoint.is_file()

        assert (
            main(
                [
                    "generate",
                    "--checkpoint", str(checkpoint),
                    "--out", str(samples),
                    "--count", "8",
                    "--device", "cpu",
                ]
            )
            == EXIT_OK
        )
        assert np.load(samples).shape == (8, config.MATRIX_SIZE, config.MATRIX_SIZE)

        assert (
            main(
                [
                    "evaluate",
                    "--samples", str(samples),
                    "--data", str(processed),
                    "--split", "val",
                    "--out", str(reports),
                    "--embed-count", "4",
                ]
            )
            == EXIT_OK
        )
        blob = json.loads((reports / "evaluation.json").read_text())
        assert "dcgan_raw" in blob["sources"]
        assert "MDS stress-1" in capsys.readouterr().out

    def test_fetch_reports_progress(self, tmp_path, monkeypatch, capsys):
        ids = tmp_path / "ids.txt"
        ids.write_text("1ABC\n2DEF\n")
        monkeypatch.setattr("folduzz.fetch.http_get", lambda url, **_: VALID_PDB)
        monkeypatch.setattr("folduzz.fetch.time.sleep", lambda _: None)
        assert main(["fetch", "--ids", str(ids), "--out", str(tmp_path / "raw")]) == EXIT_OK
        assert "2 downloaded" in capsys.readouterr().out

    def test_generate_symmetrize_flag(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()
        matrices = real_matrices(count=8, size=32, seed=2)
        np.save(processed / "train.npy", normalize(matrices, max_distance=50.0))
        np.save(processed / "val.npy", normalize(matrices[:2], max_distance=50.0))
        (processed / config.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "matrix_size": 32,
                    "max_distance_angstrom": 50.0,
                    "window_stride": 16,
                    "windows": [],
                }
            )
        )
        run = tmp_path / "run"
        main(
            [
                "train", "--data", str(processed), "--run-dir", str(run),
                "--epochs", "1", "--batch-size", "4", "--device", "cpu",
            ]
        )
        out = tmp_path / "s.npy"
        assert (
            main(
                [
                    "generate", "--checkpoint", str(run / "last.pt"), "--out", str(out),
                    "--count", "4", "--device", "cpu", "--symmetrize",
                ]
            )
            == EXIT_OK
        )
        samples = np.load(out)
        np.testing.assert_allclose(samples, np.transpose(samples, (0, 2, 1)), atol=1e-6)
