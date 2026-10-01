"""Tests for the resumable, polite RCSB fetcher (no network access)."""

from __future__ import annotations

import pytest

from folduzz import config
from folduzz.errors import FetchError, InvalidInputError
from folduzz.fetch import (
    FetchResult,
    fetch_all,
    fetch_structure,
    http_get,
    parse_id_file,
    validate_pdb_id,
)

VALID_PDB = (b"HEADER    TEST\n" + b"ATOM      1  CA  ALA A   1      0.000   0.000   0.000\n" * 40
             + b"END\n")


def fake_downloader(payload: bytes = VALID_PDB, fail_ids: frozenset[str] = frozenset()):
    calls: list[str] = []

    def download(url: str) -> bytes:
        calls.append(url)
        if any(bad in url for bad in fail_ids):
            raise FetchError(f"404 for {url}")
        return payload

    download.calls = calls  # type: ignore[attr-defined]
    return download


class TestValidatePdbId:
    @pytest.mark.parametrize("raw", ["1abc", "1ABC", " 1abc ", "1abc,train", "1ABC\n"])
    def test_accepts_and_normalises(self, raw):
        assert validate_pdb_id(raw) == "1ABC"

    @pytest.mark.parametrize("raw", ["", "abc", "12345", "1a!c", "../etc", "1ab"])
    def test_rejects_malformed(self, raw):
        with pytest.raises(InvalidInputError):
            validate_pdb_id(raw)

    def test_rejects_non_string(self):
        with pytest.raises(InvalidInputError):
            validate_pdb_id(None)  # type: ignore[arg-type]


class TestParseIdFile:
    def test_reads_plain_ids(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("1ABC\n2DEF\n")
        assert parse_id_file(path) == ("1ABC", "2DEF")

    def test_strips_legacy_split_column_and_comments(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("# comment\n1A0S,train\n\n2ABH,test\n")
        assert parse_id_file(path) == ("1A0S", "2ABH")

    def test_deduplicates_preserving_order(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("1ABC\n2DEF\n1abc\n")
        assert parse_id_file(path) == ("1ABC", "2DEF")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(InvalidInputError):
            parse_id_file(tmp_path / "nope.txt")

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("# only a comment\n")
        with pytest.raises(InvalidInputError):
            parse_id_file(path)


class TestFetchStructure:
    def test_downloads_and_writes(self, tmp_path):
        download = fake_downloader()
        result = fetch_structure("1ABC", tmp_path, downloader=download)
        assert result.status == "downloaded"
        assert result.path == tmp_path / "1abc.pdb"
        assert result.path.read_bytes() == VALID_PDB
        assert download.calls == ["https://files.rcsb.org/download/1ABC.pdb"]

    def test_existing_file_is_not_redownloaded(self, tmp_path):
        (tmp_path / "1abc.pdb").write_bytes(VALID_PDB)
        download = fake_downloader()
        result = fetch_structure("1ABC", tmp_path, downloader=download)
        assert result.status == "cached"
        assert download.calls == []

    def test_truncated_existing_file_is_refetched(self, tmp_path):
        (tmp_path / "1abc.pdb").write_bytes(b"tiny")
        download = fake_downloader()
        result = fetch_structure("1ABC", tmp_path, downloader=download)
        assert result.status == "downloaded"
        assert result.path.read_bytes() == VALID_PDB

    def test_short_payload_is_rejected(self, tmp_path):
        result = fetch_structure("1ABC", tmp_path, downloader=fake_downloader(b"nope"))
        assert result.status == "failed"
        assert "too small" in result.message
        assert not (tmp_path / "1abc.pdb").exists()

    def test_payload_without_atoms_is_rejected(self, tmp_path):
        payload = b"HEADER\n" + b"X" * 2000
        result = fetch_structure("1ABC", tmp_path, downloader=fake_downloader(payload))
        assert result.status == "failed"
        assert "ATOM" in result.message

    def test_download_failure_is_reported_not_raised(self, tmp_path):
        download = fake_downloader(fail_ids=frozenset({"1ABC"}))
        result = fetch_structure("1ABC", tmp_path, downloader=download)
        assert result.status == "failed"
        assert "404" in result.message

    def test_no_partial_file_left_behind_on_failure(self, tmp_path):
        fetch_structure("1ABC", tmp_path, downloader=fake_downloader(b"x"))
        assert list(tmp_path.iterdir()) == []

    def test_invalid_id_raises(self, tmp_path):
        with pytest.raises(InvalidInputError):
            fetch_structure("not-an-id", tmp_path, downloader=fake_downloader())


class TestFetchAll:
    def test_fetches_each_id_once(self, tmp_path):
        download = fake_downloader()
        results = fetch_all(("1ABC", "2DEF"), tmp_path, downloader=download, sleeper=lambda _: None)
        assert [r.status for r in results] == ["downloaded", "downloaded"]
        assert len(download.calls) == 2

    def test_sleeps_between_network_calls_only(self, tmp_path):
        (tmp_path / "1abc.pdb").write_bytes(VALID_PDB)
        delays: list[float] = []
        fetch_all(
            ("1ABC", "2DEF", "3GHI"),
            tmp_path,
            downloader=fake_downloader(),
            sleeper=delays.append,
        )
        # 1ABC was cached (no sleep); two real downloads => two polite delays.
        assert delays == [config.FETCH_DELAY_SECONDS] * 2

    def test_is_resumable_after_partial_run(self, tmp_path):
        first = fetch_all(("1ABC",), tmp_path, downloader=fake_downloader(), sleeper=lambda _: None)
        assert first[0].status == "downloaded"
        download = fake_downloader()
        second = fetch_all(("1ABC", "2DEF"), tmp_path, downloader=download, sleeper=lambda _: None)
        assert [r.status for r in second] == ["cached", "downloaded"]
        assert len(download.calls) == 1

    def test_continues_past_failures(self, tmp_path):
        download = fake_downloader(fail_ids=frozenset({"2DEF"}))
        results = fetch_all(
            ("1ABC", "2DEF", "3GHI"), tmp_path, downloader=download, sleeper=lambda _: None
        )
        assert [r.status for r in results] == ["downloaded", "failed", "downloaded"]

    def test_empty_id_list_raises(self, tmp_path):
        with pytest.raises(InvalidInputError):
            fetch_all((), tmp_path, downloader=fake_downloader(), sleeper=lambda _: None)

    def test_limit_truncates_work(self, tmp_path):
        download = fake_downloader()
        results = fetch_all(
            ("1ABC", "2DEF", "3GHI"),
            tmp_path,
            downloader=download,
            sleeper=lambda _: None,
            limit=2,
        )
        assert len(results) == 2


class TestHttpGetRetries:
    def test_retries_then_succeeds(self, monkeypatch):
        attempts = {"n": 0}

        def flaky(url, timeout):  # noqa: ARG001
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise OSError("connection reset")
            return VALID_PDB

        monkeypatch.setattr("folduzz.fetch._urlopen_bytes", flaky)
        assert http_get("http://x", sleeper=lambda _: None) == VALID_PDB
        assert attempts["n"] == 3

    def test_raises_after_exhausting_retries(self, monkeypatch):
        def always_fail(url, timeout):  # noqa: ARG001
            raise OSError("down")

        monkeypatch.setattr("folduzz.fetch._urlopen_bytes", always_fail)
        with pytest.raises(FetchError):
            http_get("http://x", sleeper=lambda _: None)

    def test_backoff_grows(self, monkeypatch):
        waits: list[float] = []

        def always_fail(url, timeout):  # noqa: ARG001
            raise OSError("down")

        monkeypatch.setattr("folduzz.fetch._urlopen_bytes", always_fail)
        with pytest.raises(FetchError):
            http_get("http://x", sleeper=waits.append)
        assert waits == sorted(waits) and len(waits) == config.FETCH_MAX_RETRIES - 1


class TestFetchResult:
    def test_is_frozen(self):
        result = FetchResult(pdb_id="1ABC", path=None, status="failed", message="x")
        with pytest.raises(Exception):
            result.status = "downloaded"  # type: ignore[misc]
