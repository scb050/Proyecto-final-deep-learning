"""Runner multi-seed: corre N entrenamientos por modelo y guarda métricas/preds.

Genera por cada (modelo, seed) un directorio en
``experiments/<model>/seed=<s>/`` con:

- ``checkpoint.pt`` — mejor estado por val loss
- ``history.json``  — curva train/val
- ``predictions.npz``  — y_true, y_pred, timestamps
- ``metrics.json`` — métricas por horizonte
- ``env.json`` — captura de entorno
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.features import build_features
from src.data.scalers import FeatureScaler
from src.data.split import _apply_sampling, split_dataframe
from src.data.windowing import make_windows
from src.evaluation.metrics import compute_metrics
from src.utils import (
    capture_environment,
    get_logger,
    load_parquet,
    load_yaml,
    save_json,
    set_seed,
)
from src.utils.io import save_yaml

from .trainer import Trainer

log = get_logger(__name__)


def _load_model_class(dotted: str):
    module_name, _, cls = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), cls)


def _build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _resolve_feature_cols(df_columns: list[str], target_cols: list[str], exog: list[str]) -> list[str]:
    # Todas las columnas numéricas excepto los targets => entrada al modelo.
    return [c for c in df_columns if c not in target_cols] + [c for c in target_cols if c in exog]


def run_one(
    model_cfg: dict,
    cfg: dict,
    seed: int,
    station: str,
) -> dict:
    set_seed(seed)
    proc = Path(cfg["paths"]["data_processed"])
    # `data/processed/<station>.parquet` es la fuente única (un archivo por
    # estación con todos los años). El split temporal se aplica en memoria
    # con `split_dataframe`, honrando `config.sampling.train_years` si está
    # activo.
    presplit_dir = proc / "train" / f"{station}.parquet"
    if presplit_dir.exists():
        df_tr = load_parquet(proc / "train" / f"{station}.parquet")
        df_va = load_parquet(proc / "val"   / f"{station}.parquet")
        df_te = load_parquet(proc / "test"  / f"{station}.parquet")
    else:
        df_full = load_parquet(proc / f"{station}.parquet")
        # Las columnas categóricas/strings (region, biome, koppen_class) son
        # metadatos estáticos: el modelo no las consume como features numéricas.
        df_full = df_full.select_dtypes(include=["number"])
        # Aplicamos en memoria el mismo feature engineering (lags + rolling +
        # fillna time-interpolate) que el pipeline de `src.data.features`
        # escribiría en `data/processed/features/<station>.parquet`.
        df_full = build_features(df_full, cfg)
        # `_apply_sampling` SOLO modifica `split_cfg.by_year.train_years`. El
        # DataFrame se pasa COMPLETO a `split_dataframe`, que parte por años.
        split_cfg, _ = _apply_sampling(cfg)
        # Validar cobertura: si la estación no tiene datos en val_years o
        # test_years, `split_dataframe` reventaría en `assert_no_leakage`.
        years_in_df = set(df_full.index.year.unique().tolist())
        by_year = split_cfg.get("by_year", {})
        needed = set(by_year.get("val_years", [])) | set(by_year.get("test_years", []))
        if not needed.issubset(years_in_df):
            missing = sorted(needed - years_in_df)
            log.warning(
                "Estación %s sin cobertura completa (faltan años %s) — saltando.",
                station, missing,
            )
            return {}
        splits = split_dataframe(df_full, split_cfg)
        df_tr, df_va, df_te = splits.train, splits.val, splits.test

    targets = cfg["task"].get("multi_target") or [cfg["task"]["target"]]
    feature_cols = _resolve_feature_cols(list(df_tr.columns), targets, cfg["task"]["exog"])
    lookback = cfg["task"]["lookback"]
    horizon = cfg["task"]["horizon"]

    # Escalado: fit SOLO con datos raw de train (evita materializar
    # (N_windows × lookback, F) en float64 → OOM con dataset completo en CPU).
    # Las estadísticas son equivalentes a fitear sobre windows (misma distribución).
    fx = FeatureScaler(name=cfg["scaling"]["method"]).fit(
        df_tr[feature_cols].to_numpy(dtype=np.float32), source="train"
    )
    fy = FeatureScaler(name=cfg["scaling"]["method"]).fit(
        df_tr[targets].to_numpy(dtype=np.float32), source="train"
    )

    def _prescale(df):
        out = df.copy()
        out[feature_cols] = fx.transform(df[feature_cols].to_numpy(dtype=np.float32))
        out[targets]      = fy.transform(df[targets].to_numpy(dtype=np.float32))
        return out

    w_tr = make_windows(_prescale(df_tr), feature_cols, targets, lookback, horizon)
    w_va = make_windows(_prescale(df_va), feature_cols, targets, lookback, horizon)
    w_te = make_windows(_prescale(df_te), feature_cols, targets, lookback, horizon)

    Xtr, Ytr = w_tr.X, w_tr.y
    Xva, Yva = w_va.X, w_va.y
    Xte, Yte = w_te.X, w_te.y

    # Modelo
    cls = _load_model_class(model_cfg["model"]["class"])
    model = cls(
        n_features=Xtr.shape[-1],
        n_targets=Ytr.shape[-1],
        lookback=lookback,
        horizon=horizon,
        **model_cfg.get("architecture", {}),
    )

    bs = model_cfg["training"]["batch_size"]
    tr_loader = _build_loader(Xtr, Ytr, bs, shuffle=True)
    va_loader = _build_loader(Xva, Yva, bs, shuffle=False)
    te_loader = _build_loader(Xte, Yte, bs, shuffle=False)

    out_dir = Path(cfg["paths"]["experiments"]) / model_cfg["model"]["name"] / station / f"seed={seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip si ya existe un run completo (resume logic)
    if (out_dir / "metrics.json").exists() and (out_dir / "predictions.npz").exists():
        log.info("Skip %s/%s seed=%d — ya existe, omitiendo.", model_cfg["model"]["name"], station, seed)
        from src.utils.io import load_json
        return load_json(out_dir / "metrics.json")

    trainer = Trainer(
        model,
        lr=model_cfg["training"]["lr"],
        weight_decay=model_cfg["training"].get("weight_decay", 0.0),
        device=cfg["project"]["device"],
        grad_clip=model_cfg["training"].get("grad_clip", 1.0),
    )
    info = trainer.fit(
        tr_loader, va_loader,
        epochs=model_cfg["training"]["epochs"],
        patience=model_cfg["training"]["early_stopping_patience"],
        checkpoint_path=out_dir / "checkpoint.pt",
    )

    # Mejor checkpoint para evaluación final.
    model.load_state_dict(torch.load(out_dir / "checkpoint.pt", map_location=trainer.device))
    y_pred_scaled, y_true_scaled = trainer.predict(te_loader)
    y_pred = fy.inverse_transform(y_pred_scaled)
    y_true = fy.inverse_transform(y_true_scaled)

    np.savez(
        out_dir / "predictions.npz",
        y_pred=y_pred, y_true=y_true, timestamps=w_te.timestamps,
        target_names=np.array(targets),
    )
    save_json(info["history"], out_dir / "history.json")
    metrics = compute_metrics(y_true, y_pred, per_horizon=cfg["evaluation"]["per_horizon"])
    save_json(metrics, out_dir / "metrics.json")
    save_json(capture_environment(), out_dir / "env.json")
    save_yaml(model_cfg, out_dir / "config_used.yaml")
    fx.save(out_dir / "scaler_x.joblib")
    fy.save(out_dir / "scaler_y.joblib")

    log.info("Run %s/%s seed=%d — RMSE total=%.4f", model_cfg["model"]["name"], station, seed, metrics["rmse_total"])
    return metrics


def _apply_sampling_overrides(cfg: dict, model_cfg: dict, seeds: int,
                               stations: list[str]) -> tuple[dict, int, list[str]]:
    """Aplica los overrides de ``config.sampling`` sobre model_cfg, seeds y stations.

    No modifica la lógica del modelo: solo recorta volumen y compute.
    """
    sampling = cfg.get("sampling", {}) or {}
    if not sampling.get("enabled", False):
        return model_cfg, seeds, stations

    epochs_override = int(sampling.get("max_epochs_override", model_cfg["training"]["epochs"]))
    seeds_override = int(sampling.get("n_seeds_override", seeds))
    n_per_region = int(sampling.get("n_stations_per_region", 1))

    model_cfg = {**model_cfg, "training": {**model_cfg["training"], "epochs": epochs_override}}

    from src.utils.regions import all_regions, stations_by_region
    sampled = []
    for region in all_regions():
        sampled.extend(stations_by_region(region)[:n_per_region])
    sampled_set = {s.upper() for s in sampled}
    filtered_stations = [s for s in stations if s.upper() in sampled_set] or sampled

    log.warning(
        "Sampling activo: epochs=%d, seeds=%d, stations=%d (%s)",
        epochs_override, seeds_override, len(filtered_stations), filtered_stations,
    )
    return model_cfg, seeds_override, filtered_stations


def run(config_path: str, model_name: str, seeds: int) -> None:
    cfg = load_yaml(config_path)
    model_cfg_path = Path(cfg["paths"]["models_config_dir"]) / f"{model_name}.yaml"
    model_cfg = load_yaml(model_cfg_path)

    stations_cfg = load_yaml(cfg["paths"]["stations_config"])
    stations = [s["code"] for s in stations_cfg.get("stations", [])] or ["all"]

    model_cfg, seeds, stations = _apply_sampling_overrides(cfg, model_cfg, seeds, stations)

    base_seed = cfg["project"]["seed"]
    for station in stations:
        for i in range(seeds):
            seed = base_seed + i
            run_one(model_cfg, cfg, seed=seed, station=station)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()
    if args.seeds is None:
        _cfg = load_yaml(args.config)
        seeds = int(_cfg["project"]["seeds_per_model"])
    else:
        seeds = args.seeds
    run(args.config, args.model, seeds)


if __name__ == "__main__":
    main()
