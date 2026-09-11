from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_response.data_fetch import DryadTarget, _embedded_items, select_file_metadata


def test_embedded_items_accepts_hal_payload() -> None:
    payload = {
        "_embedded": {
            "stash:files": [
                {"id": 1, "path": "a.csv"},
                {"id": 2, "path": "b.csv"},
            ]
        }
    }
    assert [item["id"] for item in _embedded_items(payload)] == [1, 2]


def test_select_file_metadata_matches_exact_basename() -> None:
    files = [
        {"id": 1, "path": "folder/all_abundance_wide_format.csv"},
        {"id": 2, "path": "folder/DailyMaxTemperature.csv"},
    ]
    selected = select_file_metadata(files, "DailyMaxTemperature.csv")
    assert selected["id"] == 2


def test_select_file_metadata_rejects_missing_or_ambiguous() -> None:
    with pytest.raises(RuntimeError, match="exactly one"):
        select_file_metadata([], "x.csv")
    with pytest.raises(RuntimeError, match="exactly one"):
        select_file_metadata(
            [{"id": 1, "path": "x.csv"}, {"id": 2, "path": "nested/x.csv"}],
            "x.csv",
        )


def test_dryad_target_is_explicit_about_destination() -> None:
    target = DryadTarget("doi:example", "data.csv", Path("data/raw/data.csv"))
    assert target.filename == "data.csv"
    assert target.destination.as_posix() == "data/raw/data.csv"
