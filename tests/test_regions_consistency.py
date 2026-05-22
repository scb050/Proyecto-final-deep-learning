"""Tests de consistencia entre config/regions.yaml y config/stations.yaml."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# `regions.py` lee paths relativos a la raiz del repo; aseguramos cwd correcto.
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    # Limpiar caches lru_cache entre tests para que respeten el chdir.
    from src.utils import regions
    for fn in [
        regions.load_regions, regions.load_stations, regions.stations_by_region,
        regions.region_of, regions.region_color, regions.all_regions,
        regions.region_color_map,
    ]:
        fn.cache_clear()
    yield


def test_assert_consistency_does_not_raise():
    from src.utils.regions import assert_consistency
    assert_consistency()


def test_total_stations_is_40():
    from src.utils.regions import load_regions, load_stations
    total = sum(r["n_stations"] for r in load_regions().values())
    expected = len(list(load_stations()))
    assert total == expected, f"Esperado {expected} estaciones (stations.yaml), se encontraron {total} (regions.yaml)"


def test_expected_regions_present():
    from src.utils.regions import all_regions
    expected = {"Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"}
    assert set(all_regions()) == expected


def test_colors_are_valid_hex():
    from src.utils.regions import all_regions, region_color
    pat = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for r in all_regions():
        c = region_color(r)
        assert pat.match(c), f"Color invalido en {r!r}: {c!r}"


def test_each_station_belongs_to_exactly_one_region():
    from src.utils.regions import load_regions, load_stations
    seen: dict[str, str] = {}
    for name, info in load_regions().items():
        for code in info["station_codes"]:
            assert code.upper() not in seen, f"{code} aparece duplicado"
            seen[code.upper()] = name
    declared = {s["code"].upper(): s["region"] for s in load_stations()}
    assert declared.keys() == seen.keys()
    assert all(declared[k] == seen[k] for k in declared)


def test_region_color_map_complete():
    from src.utils.regions import all_regions, region_color_map
    m = region_color_map()
    assert set(m) == set(all_regions())
    assert all(v.startswith("#") and len(v) == 7 for v in m.values())
