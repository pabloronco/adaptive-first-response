from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DRYAD_API_BASE = "https://datadryad.org/api/v2"
DRYAD_FILE_STREAM_BASES = (
    "https://datadryad.org/downloads/file_stream/{file_id}",
    "https://datadryad.org/stash/downloads/file_stream/{file_id}",
)
USER_AGENT = "adaptive-first-response-r0-data-audit/0.1"


@dataclass(frozen=True)
class DryadTarget:
    doi: str
    filename: str
    destination: Path


R0_TARGETS: tuple[DryadTarget, ...] = (
    DryadTarget(
        doi="doi:10.5061/dryad.0rxwdbsdt",
        filename="all_abundance_wide_format.csv",
        destination=Path("data/raw/wsg_cama/all_abundance_wide_format.csv"),
    ),
    DryadTarget(
        doi="doi:10.5061/dryad.0rxwdbsdt",
        filename="DailyMaxTemperature.csv",
        destination=Path("data/raw/wsg_cama/DailyMaxTemperature.csv"),
    ),
    DryadTarget(
        doi="doi:10.5061/dryad.sqv9s4ndp",
        filename="PAMA.Month.CPUE.csv",
        destination=Path("data/raw/wsg_effort_geo/PAMA.Month.CPUE.csv"),
    ),
    DryadTarget(
        doi="doi:10.5061/dryad.sqv9s4ndp",
        filename="Network_PAMA_CPUE.Map.csv",
        destination=Path("data/raw/wsg_effort_geo/Network_PAMA_CPUE.Map.csv"),
    ),
)


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _embedded_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return HAL-style embedded object lists without depending on Dryad link names."""
    embedded = payload.get("_embedded", {})
    if isinstance(embedded, dict):
        for value in embedded.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    for key in ("items", "versions", "files"):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    if isinstance(payload, list):  # pragma: no cover - defensive, type kept for API convenience
        return payload
    return []


def _latest_version_id(doi: str) -> int:
    encoded = urllib.parse.quote(doi, safe="")
    payload = _request_json(f"{DRYAD_API_BASE}/datasets/{encoded}/versions")
    versions = _embedded_items(payload)
    if not versions:
        raise RuntimeError(f"Dryad returned no versions for {doi}")

    def version_key(item: dict[str, Any]) -> tuple[int, int]:
        number = item.get("versionNumber")
        item_id = item.get("id")
        try:
            number_i = int(number)
        except (TypeError, ValueError):
            number_i = -1
        try:
            id_i = int(item_id)
        except (TypeError, ValueError):
            id_i = -1
        return number_i, id_i

    latest = max(versions, key=version_key)
    try:
        return int(latest["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Dryad version metadata for {doi} has no usable id") from exc


def list_dataset_files(doi: str) -> list[dict[str, Any]]:
    version_id = _latest_version_id(doi)
    payload = _request_json(f"{DRYAD_API_BASE}/versions/{version_id}/files")
    files = _embedded_items(payload)
    if not files:
        raise RuntimeError(f"Dryad returned no files for {doi} version {version_id}")
    return files


def select_file_metadata(files: Iterable[dict[str, Any]], filename: str) -> dict[str, Any]:
    matches = [item for item in files if Path(str(item.get("path", ""))).name == filename]
    if len(matches) != 1:
        available = sorted(Path(str(item.get("path", ""))).name for item in files)
        raise RuntimeError(
            f"Expected exactly one Dryad file named {filename!r}; found {len(matches)}. "
            f"Available files: {available}"
        )
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file_stream(file_id: int, destination: Path) -> None:
    errors: list[str] = []
    for template in DRYAD_FILE_STREAM_BASES:
        url = template.format(file_id=file_id)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
                shutil.copyfileobj(response, out)
            return
        except urllib.error.HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
        except urllib.error.URLError as exc:
            errors.append(f"{url} -> {exc.reason}")
    raise RuntimeError("Unable to download Dryad file. " + "; ".join(errors))


def fetch_targets(
    targets: Iterable[DryadTarget] = R0_TARGETS,
    *,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    by_doi: dict[str, list[dict[str, Any]]] = {}
    manifest: list[dict[str, Any]] = []

    for target in targets:
        target.destination.parent.mkdir(parents=True, exist_ok=True)
        if target.destination.exists() and not overwrite:
            manifest.append(
                {
                    "doi": target.doi,
                    "filename": target.filename,
                    "destination": str(target.destination),
                    "status": "already_present",
                    "size": target.destination.stat().st_size,
                    "sha256": _sha256(target.destination),
                }
            )
            continue

        files = by_doi.setdefault(target.doi, list_dataset_files(target.doi))
        metadata = select_file_metadata(files, target.filename)
        try:
            file_id = int(metadata["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Dryad metadata for {target.filename} has no usable file id") from exc

        expected_size_raw = metadata.get("size")
        try:
            expected_size = int(expected_size_raw) if expected_size_raw is not None else None
        except (TypeError, ValueError):
            expected_size = None

        with tempfile.TemporaryDirectory(prefix="r0_dryad_") as temp_dir:
            temp_path = Path(temp_dir) / target.filename
            _download_file_stream(file_id, temp_path)
            actual_size = temp_path.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch for {target.filename}: expected {expected_size}, got {actual_size}"
                )
            shutil.move(str(temp_path), str(target.destination))

        manifest.append(
            {
                "doi": target.doi,
                "filename": target.filename,
                "dryad_file_id": file_id,
                "destination": str(target.destination),
                "status": "downloaded",
                "size": target.destination.stat().st_size,
                "sha256": _sha256(target.destination),
            }
        )

    return manifest
