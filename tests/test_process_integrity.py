"""Verifica integridad de los parquets generados por ``src.data.process``.

Se ejecuta sobre ``data/processed/<wmo>.parquet`` ya escritos. Si los archivos
no existen (por ejemplo en CI sin datos), el test se *salta* gracefully.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import load_yaml

_PROCESSED = Path("data/processed")
_STATIONS_YAML = Path("config/stations.yaml")
_EXPECTED_RANGE = (pd.Timestamp("2018-01-01 00:00:00"), pd.Timestamp("2025-12-31 23:00:00"))
_TEMP_BOUNDS = (-10.0, 50.0)
_CYCLIC_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]


def _expected_codes() -> list[str]:
    cfg = load_yaml(_STATIONS_YAML)
    return sorted(s["code"].upper() for s in cfg.get("stations", []))


def _processed_paths() -> list[Path]:
    if not _PROCESSED.exists():
        return []
    return sorted(_PROCESSED.glob("*.parquet"))


@pytest.fixture(scope="module")
def parquet_paths() -> list[Path]:
    paths = _processed_paths()
    if not paths:
        pytest.skip("No hay parquets en data/processed/ — corre `make process` antes.")
    return paths


def test_count_matches_stations_yaml(parquet_paths):
    """Hay un parquet por cada estación declarada en ``config/stations.yaml``."""
    expected = set(_expected_codes())
    found = {p.stem.upper() for p in parquet_paths}
    assert expected == found, f"Faltan {expected - found}; sobran {found - expected}"
    assert len(parquet_paths) == len(expected)


def test_each_parquet_has_monotonic_unique_index(parquet_paths):
    for path in parquet_paths:
        df = pd.read_parquet(path)
        assert isinstance(df.index, pd.DatetimeIndex), f"{path.name}: índice no datetime"
        assert df.index.tz is None, f"{path.name}: el índice debe ser tz-naive"
        assert df.index.is_monotonic_increasing, f"{path.name}: índice no monótono"
        assert df.index.is_unique, f"{path.name}: timestamps duplicados"


def test_temporal_range_is_2018_to_2025(parquet_paths):
    lo, hi = _EXPECTED_RANGE
    for path in parquet_paths:
        df = pd.read_parquet(path)
        # Tolerancia: la estación puede arrancar/terminar dentro del rango global.
        assert df.index.min() >= lo, f"{path.name}: empieza antes de {lo}"
        assert df.index.max() <= hi, f"{path.name}: termina después de {hi}"


def test_target_column_in_physical_range(parquet_paths):
    """``temp_c`` (target) está dentro del rango físico [-10, 50] °C, salvo NaN."""
    lo, hi = _TEMP_BOUNDS
    for path in parquet_paths:
        df = pd.read_parquet(path)
        assert "temp_c" in df.columns, f"{path.name}: falta target temp_c"
        valid = df["temp_c"].dropna()
        if len(valid) == 0:
            continue
        assert valid.min() >= lo, f"{path.name}: temp_c min={valid.min()} < {lo}"
        assert valid.max() <= hi, f"{path.name}: temp_c max={valid.max()} > {hi}"


def test_cyclic_features_in_unit_interval(parquet_paths):
    for path in parquet_paths:
        df = pd.read_parquet(path)
        for c in _CYCLIC_COLS:
            assert c in df.columns, f"{path.name}: falta {c}"
            arr = df[c].to_numpy()
            assert np.all((arr >= -1.0 - 1e-9) & (arr <= 1.0 + 1e-9)), (
                f"{path.name}: {c} fuera de [-1, 1]"
            )


def test_station_id_consistent_with_filename(parquet_paths):
    """``station_id`` es constante por archivo y consistente con el código WMO ordenado."""
    expected_ids = {code: i for i, code in enumerate(_expected_codes())}
    for path in parquet_paths:
        df = pd.read_parquet(path)
        assert "station_id" in df.columns
        ids = df["station_id"].unique()
        assert len(ids) == 1, f"{path.name}: station_id no es constante"
        assert int(ids[0]) == expected_ids[path.stem.upper()], (
            f"{path.name}: station_id={ids[0]} no coincide con el orden de stations.yaml"
        )
