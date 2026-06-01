"""
FastAPI — TCN Weather Forecaster
=================================
Sirve el mejor modelo del benchmark (TCN) para predecir temperatura horaria
a 24 h, 72 h y 168 h sobre estaciones de la red INMET Brasil.

Uso local:
    uvicorn api:app --reload --port 8000
    # Documentación interactiva: http://localhost:8000/docs

Endpoints:
    GET  /              → info del servicio y modelo
    GET  /health        → estado del servicio
    GET  /stations      → estaciones disponibles con metadata
    POST /predict       → predicción dado estación + timestamp
    GET  /predict/latest/{station} → predicción desde último dato disponible
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Setup ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.features import build_features
from src.data.scalers import FeatureScaler
from src.models.model_tcn import TCNForecaster
from src.utils import load_yaml

CFG      = load_yaml(ROOT / "config" / "config.yaml")
EXP_DIR  = ROOT / Path(CFG["paths"]["experiments"]) / "tcn"
PROC_DIR = ROOT / Path(CFG["paths"]["data_processed"])
LOOKBACK = int(CFG["task"]["lookback"])   # 168
HORIZON  = int(CFG["task"]["horizon"])    # 168
TARGET   = CFG["task"]["target"]          # "temp_c"
EXOG     = CFG["task"]["exog"]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET] + [c for c in [TARGET] if c in EXOG]


# ── Modelo cargado por estación ───────────────────────────────────────────────
class StationPredictor:
    def __init__(self, station: str):
        seed_dir = EXP_DIR / station / "seed=42"
        if not seed_dir.exists():
            raise FileNotFoundError(f"No hay modelo TCN para estación '{station}'")

        model_cfg = load_yaml(seed_dir / "config_used.yaml")
        arch = model_cfg.get("architecture", {})

        # Determinar n_features a partir del scaler guardado
        self.scaler_x = FeatureScaler.load(seed_dir / "scaler_x.joblib")
        self.scaler_y = FeatureScaler.load(seed_dir / "scaler_y.joblib")
        n_features = self.scaler_x._scaler.n_features_in_  # 145

        self.model = TCNForecaster(
            n_features=n_features,
            n_targets=1,
            lookback=LOOKBACK,
            horizon=HORIZON,
            **arch,
        )
        self.model.load_state_dict(
            torch.load(seed_dir / "checkpoint.pt", map_location="cpu")
        )
        self.model.eval()
        self.station = station
        self.metrics = _load_metrics(seed_dir)

    def predict_from_window(self, window_df: pd.DataFrame) -> dict:
        """Recibe DataFrame (168, F) ya con features engineered."""
        feat_cols = _feature_cols(window_df)
        X = window_df[feat_cols].to_numpy(dtype=np.float32)
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.tensor(X_scaled[None], dtype=torch.float32)

        with torch.no_grad():
            y_scaled = self.model(X_tensor).numpy()  # (1, 168, 1)

        y_pred = self.scaler_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).ravel()

        return {
            "h24":  round(float(y_pred[23]),  2),
            "h72":  round(float(y_pred[71]),  2),
            "h168": round(float(y_pred[167]), 2),
            "full_168h": [round(float(v), 2) for v in y_pred],
        }


def _load_metrics(seed_dir: Path) -> dict:
    import json
    p = seed_dir / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ── Cache de modelos y datos procesados ──────────────────────────────────────
_model_cache: dict[str, StationPredictor] = {}
_data_cache:  dict[str, pd.DataFrame]     = {}


def _get_predictor(station: str) -> StationPredictor:
    station = station.upper()
    if station not in _model_cache:
        try:
            _model_cache[station] = StationPredictor(station)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return _model_cache[station]


def _get_station_data(station: str) -> pd.DataFrame:
    """Carga y cachea el parquet procesado con features engineered."""
    station = station.upper()
    if station not in _data_cache:
        # Busca en train/val/test o en raíz de processed
        for subdir in ["", "train/", "val/", "test/"]:
            p = PROC_DIR / f"{subdir}{station}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df = df.select_dtypes(include=["number"])
                df = build_features(df, CFG)
                _data_cache[station] = df
                return df
        raise HTTPException(
            status_code=404,
            detail=f"Datos procesados no encontrados para estación '{station}'"
        )
    return _data_cache[station]


def _extract_window(df: pd.DataFrame, end_dt: Optional[datetime]) -> pd.DataFrame:
    """Extrae ventana de LOOKBACK horas terminando en end_dt (o al final del df)."""
    if end_dt is not None:
        ts = pd.Timestamp(end_dt)
        if ts not in df.index:
            # Buscar el timestamp más cercano
            idx = df.index.searchsorted(ts)
            if idx == 0 or idx > len(df):
                raise HTTPException(
                    status_code=400,
                    detail=f"Timestamp {end_dt} fuera del rango de datos de esta estación."
                )
            ts = df.index[min(idx, len(df) - 1)]
        end_pos = df.index.get_loc(ts)
    else:
        end_pos = len(df) - 1

    start_pos = end_pos - LOOKBACK + 1
    if start_pos < 0:
        raise HTTPException(
            status_code=400,
            detail=f"No hay suficientes datos históricos antes del timestamp indicado (mínimo {LOOKBACK} horas)."
        )
    return df.iloc[start_pos: end_pos + 1]


def _available_stations() -> list[str]:
    if not EXP_DIR.exists():
        return []
    return sorted(p.name for p in EXP_DIR.iterdir() if p.is_dir())


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    station: str = Field(..., example="A701",
                         description="Código de estación INMET (ej. A701, A001, A801)")
    timestamp: Optional[str] = Field(
        None,
        example="2023-06-15T12:00:00",
        description=(
            "Timestamp ISO 8601 del último dato conocido. "
            "El modelo usará las 168 horas previas como contexto. "
            "Si es None, usa el último dato disponible en los datos de test."
        )
    )


class PredictResponse(BaseModel):
    station:   str
    model:     str = "TCN"
    context_end:  str
    h24:       float = Field(..., description="Temperatura predicha a 24 h (°C)")
    h72:       float = Field(..., description="Temperatura predicha a 72 h (°C)")
    h168:      float = Field(..., description="Temperatura predicha a 168 h (°C)")
    full_168h: list[float] = Field(..., description="Serie completa de 168 predicciones horarias (°C)")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TCN Weather Forecaster API",
    description=(
        "## Forecasting Meteorológico — Red INMET Brasil\n\n"
        "Sirve el **mejor modelo del benchmark** (TCN — Temporal Convolutional Network) "
        "para predecir temperatura del aire a **24 h, 72 h y 168 h** sobre 38 estaciones "
        "meteorológicas de Brasil.\n\n"
        "**RMSE promedio:** 2.51 °C sobre estaciones comunes (test 2023).\n\n"
        "**Repositorio:** [Proyecto-final-Deep-learning](https://github.com/)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
async def preload_common_stations():
    """Pre-carga modelos de las 4 estaciones del ranking justo."""
    for station in ["A001", "A101", "A701", "A801"]:
        try:
            _get_predictor(station)
            print(f"  ✅ TCN/{station} listo")
        except HTTPException:
            print(f"  ⚠️  TCN/{station} no disponible")


@app.get("/", tags=["Info"], summary="Información del servicio")
def root():
    return {
        "service":   "TCN Weather Forecaster API",
        "model":     "TCN (Temporal Convolutional Network) — Bai et al. 2018",
        "task":      "Forecasting de temperatura horaria — Red INMET Brasil",
        "rmse":      "2.51 °C (promedio sobre 4 estaciones comunes, test 2023)",
        "horizons":  ["24 h (1 día)", "72 h (3 días)", "168 h (7 días)"],
        "stations":  len(_available_stations()),
        "docs":      "/docs",
    }


@app.get("/health", tags=["Info"], summary="Estado del servicio")
def health():
    return {
        "status":          "ok",
        "models_in_cache": len(_model_cache),
        "stations_avail":  len(_available_stations()),
    }


@app.get("/stations", tags=["Info"], summary="Estaciones disponibles")
def stations():
    """Lista todas las estaciones con modelo TCN entrenado."""
    result = []
    for s in _available_stations():
        metrics_path = EXP_DIR / s / "seed=42" / "metrics.json"
        rmse = None
        if metrics_path.exists():
            import json
            m = json.loads(metrics_path.read_text())
            rmse = m.get("rmse_total")
        result.append({"station": s, "rmse_total": rmse})
    return {"count": len(result), "stations": result}


@app.post("/predict", response_model=PredictResponse, tags=["Predicción"],
          summary="Predicción de temperatura")
def predict(req: PredictRequest):
    """
    Predice temperatura del aire a **24 h, 72 h y 168 h** para una estación INMET.

    - Si se proporciona `timestamp`, usa las 168 horas previas como contexto histórico.
    - Si `timestamp` es `null`, usa el último dato disponible en los datos de test (2023).
    """
    predictor = _get_predictor(req.station)
    df        = _get_station_data(req.station)

    end_dt = datetime.fromisoformat(req.timestamp) if req.timestamp else None
    window = _extract_window(df, end_dt)

    result       = predictor.predict_from_window(window)
    context_end  = str(window.index[-1])

    return PredictResponse(
        station=req.station.upper(),
        context_end=context_end,
        **result,
    )


@app.get("/predict/latest/{station}", response_model=PredictResponse,
         tags=["Predicción"], summary="Predicción desde el último dato disponible")
def predict_latest(station: str):
    """
    Predicción usando las últimas 168 horas de datos disponibles
    para la estación indicada. Equivale a POST /predict sin timestamp.
    """
    predictor = _get_predictor(station)
    df        = _get_station_data(station)
    window    = _extract_window(df, None)
    result    = predictor.predict_from_window(window)

    return PredictResponse(
        station=station.upper(),
        context_end=str(window.index[-1]),
        **result,
    )
