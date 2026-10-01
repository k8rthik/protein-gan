"""Download PDB structures from RCSB: polite, resumable, atomic.

The original `get_pdbs.py` used `Bio.PDB.PDBList`, mutated the ID list while
iterating over it (so it silently skipped entries), and renamed `.ent` files
afterwards. This version talks to `files.rcsb.org` directly so the rate limit,
the retry policy and the resume behaviour are all visible and testable:

* one request at a time with a fixed delay between them (~3 req/s ceiling),
* 3 attempts with exponential backoff per structure,
* payload validated before it is written (size, presence of ATOM records),
* written to `<id>.pdb.part` then renamed, so an interrupted run never leaves a
  half file that a later run would treat as cached,
* a structure already on disk is never re-downloaded.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from folduzz import config
from folduzz.errors import FetchError, InvalidInputError

Downloader = Callable[[str], bytes]
Sleeper = Callable[[float], None]
FetchStatus = Literal["downloaded", "cached", "failed"]

_ID_LENGTH = 4


@dataclass(frozen=True)
class FetchResult:
    pdb_id: str
    path: Path | None
    status: FetchStatus
    message: str = ""


def validate_pdb_id(raw: str) -> str:
    """Normalise one PDB ID, rejecting anything that is not 4 alphanumerics.

    Also guards the filesystem: these IDs become file names, so `../etc` must
    never make it through.
    """
    if not isinstance(raw, str):
        raise InvalidInputError(f"PDB ID must be a string, got {type(raw).__name__}")
    token = raw.strip().split(",")[0].strip()
    if len(token) != _ID_LENGTH or not token.isalnum():
        raise InvalidInputError(
            f"invalid PDB ID {raw!r}: expected 4 alphanumeric characters"
        )
    return token.upper()


def parse_id_file(path: Path | str) -> tuple[str, ...]:
    """Read a PDB ID list. Blank lines and `#` comments are ignored.

    Accepts the legacy `1A0S,train` format by dropping the second column; the
    train/validation split is now derived by hashing the ID (see `splits.py`).
    """
    id_path = Path(path)
    if not id_path.is_file():
        raise InvalidInputError(f"PDB ID file does not exist: {id_path}")

    seen: dict[str, None] = {}
    for line_number, line in enumerate(id_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            seen.setdefault(validate_pdb_id(stripped), None)
        except InvalidInputError as exc:
            raise InvalidInputError(f"{id_path}:{line_number}: {exc}") from exc

    if not seen:
        raise InvalidInputError(f"no PDB IDs found in {id_path}")
    return tuple(seen)


def _urlopen_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": config.FETCH_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return bytes(response.read())


def http_get(
    url: str,
    timeout: float = config.FETCH_TIMEOUT_SECONDS,
    retries: int = config.FETCH_MAX_RETRIES,
    sleeper: Sleeper = time.sleep,
) -> bytes:
    """GET `url` with retries and exponential backoff. Raises `FetchError`."""
    if retries < 1:
        raise InvalidInputError(f"retries must be >= 1, got {retries}")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return _urlopen_bytes(url, timeout)
        except urllib.error.HTTPError as exc:
            # 404 means the entry does not exist; retrying cannot help.
            if exc.code == 404:
                raise FetchError(f"{url}: HTTP 404 (no such entry)") from exc
            last_error = exc
        except Exception as exc:  # timeouts, DNS, resets
            last_error = exc
        if attempt < retries - 1:
            sleeper(config.FETCH_BACKOFF_SECONDS * (2**attempt))
    raise FetchError(f"{url}: giving up after {retries} attempts ({last_error})")


def _reject_reason(payload: bytes) -> str | None:
    if len(payload) < config.MIN_PDB_BYTES:
        return f"payload too small ({len(payload)} bytes)"
    if b"ATOM" not in payload:
        return "payload contains no ATOM records"
    return None


def fetch_structure(
    pdb_id: str,
    dest_dir: Path | str,
    downloader: Downloader | None = None,
) -> FetchResult:
    """Ensure `<dest_dir>/<id>.pdb` exists. Never raises on network failure."""
    normalized = validate_pdb_id(pdb_id)
    directory = Path(dest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{normalized.lower()}.pdb"

    if target.is_file() and target.stat().st_size >= config.MIN_PDB_BYTES:
        return FetchResult(pdb_id=normalized, path=target, status="cached")

    get = downloader if downloader is not None else http_get
    url = config.RCSB_URL_TEMPLATE.format(pdb_id=normalized)
    try:
        payload = get(url)
    except FetchError as exc:
        return FetchResult(pdb_id=normalized, path=None, status="failed", message=str(exc))

    reason = _reject_reason(payload)
    if reason is not None:
        return FetchResult(pdb_id=normalized, path=None, status="failed", message=reason)

    partial = target.with_suffix(".pdb.part")
    partial.write_bytes(payload)
    partial.replace(target)
    return FetchResult(pdb_id=normalized, path=target, status="downloaded")


def fetch_all(
    pdb_ids: Sequence[str] | Iterable[str],
    dest_dir: Path | str = config.RAW_DIR,
    downloader: Downloader | None = None,
    sleeper: Sleeper = time.sleep,
    delay: float = config.FETCH_DELAY_SECONDS,
    limit: int | None = None,
    on_result: Callable[[FetchResult], None] | None = None,
) -> tuple[FetchResult, ...]:
    """Fetch every ID in order, sleeping only after a real network request."""
    ids = tuple(pdb_ids)
    if not ids:
        raise InvalidInputError("no PDB IDs to fetch")
    if limit is not None:
        if limit < 1:
            raise InvalidInputError(f"limit must be >= 1, got {limit}")
        ids = ids[:limit]

    results: list[FetchResult] = []
    for pdb_id in ids:
        result = fetch_structure(pdb_id, dest_dir, downloader=downloader)
        results.append(result)
        if on_result is not None:
            on_result(result)
        if result.status != "cached":
            sleeper(delay)
    return tuple(results)


def summarize(results: Sequence[FetchResult]) -> dict[str, int]:
    """Count results by status, for CLI output."""
    counts = {"downloaded": 0, "cached": 0, "failed": 0}
    for result in results:
        counts[result.status] += 1
    return counts
