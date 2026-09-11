from __future__ import annotations

import csv
from pathlib import Path

import pytest

from adaptive_response.data_audit import audit


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_audit_reports_exact_effort_and_coordinate_coverage(tmp_path: Path) -> None:
    cama = tmp_path / "cama.csv"
    effort = tmp_path / "effort.csv"
    coords = tmp_path / "coords.csv"

    _write_csv(
        cama,
        ["Year", "SiteID", "Month", "habtype", "CAMA"],
        [
            {"Year": 2023, "SiteID": 101, "Month": 4, "habtype": "lagoon", "CAMA": 0},
            {"Year": 2023, "SiteID": 101, "Month": 5, "habtype": "lagoon", "CAMA": 2},
        ],
    )
    _write_csv(
        effort,
        ["SiteID", "month", "Year", "trap.sets"],
        [
            {"SiteID": 101, "month": 4, "Year": 2023, "trap.sets": 6},
            {"SiteID": 101, "month": 5, "Year": 2023, "trap.sets": 5},
        ],
    )
    _write_csv(
        coords,
        ["SiteID", "Year", "LatitudeDD", "LongitudeDD"],
        [
            {"SiteID": 101, "Year": 2023, "LatitudeDD": 48.1, "LongitudeDD": -122.5},
        ],
    )

    result = audit(cama, effort, coords)

    assert result["cama"]["rows"] == 2
    assert result["cama"]["detection_rows"] == 1
    assert result["effort_join"]["coverage_fraction"] == 1.0
    assert result["coordinate_join"]["coverage_fraction"] == 1.0
    assert result["effort_join"]["trap_sets_median"] == 5.5


def test_audit_normalizes_integer_like_join_keys(tmp_path: Path) -> None:
    cama = tmp_path / "cama.csv"
    effort = tmp_path / "effort.csv"
    coords = tmp_path / "coords.csv"

    _write_csv(
        cama,
        ["Year", "SiteID", "Month", "habtype", "CAMA"],
        [{"Year": "2023", "SiteID": "101", "Month": "4", "habtype": "lagoon", "CAMA": 1}],
    )
    _write_csv(
        effort,
        ["SiteID", "month", "Year", "trap.sets"],
        [{"SiteID": "101.0", "month": "4.0", "Year": "2023.0", "trap.sets": 6}],
    )
    _write_csv(
        coords,
        ["SiteID", "Year", "LatitudeDD", "LongitudeDD"],
        [{"SiteID": "101.0", "Year": "2023.0", "LatitudeDD": 48.1, "LongitudeDD": -122.5}],
    )

    result = audit(cama, effort, coords)

    assert result["effort_join"]["coverage_fraction"] == 1.0
    assert result["coordinate_join"]["coverage_fraction"] == 1.0


def test_audit_surfaces_incomplete_join_without_imputing(tmp_path: Path) -> None:
    cama = tmp_path / "cama.csv"
    effort = tmp_path / "effort.csv"
    coords = tmp_path / "coords.csv"

    _write_csv(
        cama,
        ["Year", "SiteID", "Month", "habtype", "CAMA"],
        [
            {"Year": 2022, "SiteID": 1, "Month": 4, "habtype": "channel", "CAMA": 0},
            {"Year": 2022, "SiteID": 2, "Month": 4, "habtype": "tideflat", "CAMA": 0},
        ],
    )
    _write_csv(
        effort,
        ["SiteID", "month", "Year", "trap.sets"],
        [{"SiteID": 1, "month": 4, "Year": 2022, "trap.sets": 6}],
    )
    _write_csv(
        coords,
        ["SiteID", "Year", "LatitudeDD", "LongitudeDD"],
        [{"SiteID": 1, "Year": 2022, "LatitudeDD": 48.0, "LongitudeDD": -122.0}],
    )

    result = audit(cama, effort, coords)

    assert result["effort_join"]["coverage_fraction"] == 0.5
    assert result["coordinate_join"]["coverage_fraction"] == 0.5
    assert result["effort_join"]["unmatched_example_keys"]
    assert result["coordinate_join"]["unmatched_example_keys"]


def test_audit_rejects_missing_required_columns(tmp_path: Path) -> None:
    cama = tmp_path / "cama.csv"
    effort = tmp_path / "effort.csv"
    coords = tmp_path / "coords.csv"

    _write_csv(cama, ["Year", "SiteID", "Month", "habtype"], [])
    _write_csv(effort, ["SiteID", "month", "Year", "trap.sets"], [])
    _write_csv(coords, ["SiteID", "Year", "LatitudeDD", "LongitudeDD"], [])

    with pytest.raises(ValueError, match="missing expected columns: CAMA"):
        audit(cama, effort, coords)
