"""EDA específico de series de tiempo:
- Descomposición STL
- Test ADF (estacionariedad)
- ACF / PACF
- Detección de gaps temporales y outliers
- Espectro (FFT) para frecuencias dominantes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from src.utils import get_logger, load_parquet, load_yaml, save_json

log = get_logger(__name__)


# ------------------------------------------------------------------ ADF

def adf_test(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 50:
        return {"n": len(s), "skipped": True}
    stat, pval, lags, n, crit, _ = adfuller(s, maxlag=48)
    return {
        "n": int(n),
        "lags": int(lags),
        "stat": float(stat),
        "pvalue": float(pval),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "stationary_at_5pct": bool(pval < 0.05),
    }


# ------------------------------------------------------------------ STL

def stl_plot(series: pd.Series, period: int, out_path: Path) -> None:
    s = series.dropna()
    if len(s) < 2 * period:
        return
    res = STL(s, period=period, robust=True).fit()
    fig = res.plot()
    fig.set_size_inches(10, 8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ ACF/PACF

def acf_pacf_plot(series: pd.Series, lags: int, out_path: Path) -> None:
    s = series.dropna()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(s, lags=lags, ax=axes[0])
    plot_pacf(s, lags=lags, ax=axes[1], method="ywm")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Gaps

def gap_summary(df: pd.DataFrame, freq: str) -> dict:
    expected = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    missing = expected.difference(df.index)
    return {
        "expected_len": int(len(expected)),
        "actual_len": int(len(df)),
        "missing_ts": int(len(missing)),
        "longest_gap_hours": int(_longest_gap(df.index, expected)),
    }


def _longest_gap(actual: pd.DatetimeIndex, expected: pd.DatetimeIndex) -> int:
    present = expected.isin(actual)
    if present.all():
        return 0
    longest = curr = 0
    for p in present:
        curr = 0 if p else curr + 1
        longest = max(longest, curr)
    return longest


# ------------------------------------------------------------------ FFT

def fft_dominant_periods(series: pd.Series, top_k: int = 5) -> list[dict]:
    s = series.dropna().to_numpy()
    if len(s) < 512:
        return []
    s = s - s.mean()
    spec = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(len(s), d=1.0)
    spec[0] = 0
    idx = np.argsort(spec)[::-1][:top_k]
    return [
        {"period_hours": float(1 / f) if f > 0 else None, "amplitude": float(spec[i])}
        for i, f in zip(idx, freqs[idx])
    ]


# ------------------------------------------------------------------ Driver

def run(config: dict) -> None:
    in_dir = Path(config["paths"]["data_processed"]) / "resampled"
    fig_dir = Path(config["paths"]["figures"]) / "eda_ts"
    tab_dir = Path(config["paths"]["tables"]) / "eda_ts"
    target = config["task"]["target"]
    freq = config["task"]["freq"]
    period = 24 if freq.upper().startswith("H") else 7

    for path in sorted(in_dir.glob("*.parquet")):
        df = load_parquet(path)
        if target not in df:
            log.warning("Target %s ausente en %s", target, path.stem)
            continue
        s = df[target]

        report = {
            "adf": adf_test(s),
            "gaps": gap_summary(df, freq),
            "fft_top": fft_dominant_periods(s),
        }
        save_json(report, tab_dir / f"{path.stem}.json")

        stl_plot(s, period=period, out_path=fig_dir / path.stem / f"stl_{target}.png")
        acf_pacf_plot(
            s.diff().dropna(), lags=min(72, len(s) // 4),
            out_path=fig_dir / path.stem / f"acf_pacf_{target}.png",
        )
        log.info("EDA TS %s — ADF p=%.4g", path.stem, report["adf"].get("pvalue", float("nan")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
