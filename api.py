"""
FastAPI — TCN Weather Forecaster
"""
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.features import build_features
from src.data.scalers import FeatureScaler
from src.models.model_tcn import TCNForecaster
from src.utils import load_yaml

CFG      = load_yaml(ROOT / "config" / "config.yaml")
EXP_DIR  = ROOT / Path(CFG["paths"]["experiments"]) / "tcn"
PROC_DIR = ROOT / Path(CFG["paths"]["data_processed"])
LOOKBACK = int(CFG["task"]["lookback"])
HORIZON  = int(CFG["task"]["horizon"])
TARGET   = CFG["task"]["target"]
EXOG     = CFG["task"]["exog"]

def _feature_cols(df):
    return [c for c in df.columns if c != TARGET] + [c for c in [TARGET] if c in EXOG]

class StationPredictor:
    def __init__(self, station: str):
        seed_dir = EXP_DIR / station / "seed=42"
        if not seed_dir.exists():
            raise FileNotFoundError(f"No hay modelo TCN para estación '{station}'")
        model_cfg = load_yaml(seed_dir / "config_used.yaml")
        arch = model_cfg.get("architecture", {})
        self.scaler_x = FeatureScaler.load(seed_dir / "scaler_x.joblib")
        self.scaler_y = FeatureScaler.load(seed_dir / "scaler_y.joblib")
        n_features = self.scaler_x._scaler.n_features_in_
        self.model = TCNForecaster(n_features=n_features, n_targets=1,
                                   lookback=LOOKBACK, horizon=HORIZON, **arch)
        self.model.load_state_dict(torch.load(seed_dir / "checkpoint.pt", map_location="cpu"))
        self.model.eval()
        self.station = station

    def predict_from_window(self, window_df: pd.DataFrame) -> dict:
        feat_cols = _feature_cols(window_df)
        X = window_df[feat_cols].to_numpy(dtype=np.float32)
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.tensor(X_scaled[None], dtype=torch.float32)
        with torch.no_grad():
            y_scaled = self.model(X_tensor).numpy()
        y_pred = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        return {
            "h24":  round(float(y_pred[23]),  2),
            "h72":  round(float(y_pred[71]),  2),
            "h168": round(float(y_pred[167]), 2),
            "full_168h": [round(float(v), 2) for v in y_pred],
        }

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
    station = station.upper()
    if station not in _data_cache:
        for subdir in ["", "train/", "val/", "test/"]:
            p = PROC_DIR / f"{subdir}{station}.parquet"
            if p.exists():
                df = pd.read_parquet(p).select_dtypes(include=["number"])
                _data_cache[station] = build_features(df, CFG)
                return _data_cache[station]
        raise HTTPException(status_code=404, detail=f"Datos no encontrados para '{station}'")
    return _data_cache[station]

def _extract_window(df, end_dt):
    if end_dt is not None:
        ts = pd.Timestamp(end_dt)
        idx = df.index.searchsorted(ts)
        if idx == 0 or idx > len(df):
            raise HTTPException(status_code=400, detail=f"Timestamp {end_dt} fuera del rango.")
        end_pos = min(idx, len(df) - 1)
    else:
        end_pos = len(df) - 1
    start_pos = end_pos - LOOKBACK + 1
    if start_pos < 0:
        raise HTTPException(status_code=400, detail=f"Datos insuficientes (mínimo {LOOKBACK} horas).")
    return df.iloc[start_pos: end_pos + 1]

def _available_stations():
    if not EXP_DIR.exists(): return []
    return sorted(p.name for p in EXP_DIR.iterdir() if p.is_dir())

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    station: str = Field(..., example="A701", description="Código INMET de la estación (ej: A701, A001, A801)")
    timestamp: Optional[str] = Field(None, example="2023-06-15T12:00:00",
        description="Timestamp ISO 8601 del último dato conocido. Si es null, usa el último dato disponible.")

class PredictResponse(BaseModel):
    station: str
    model: str = "TCN"
    context_end: str
    h24:  float = Field(..., description="Temperatura predicha a 24 h (°C)")
    h72:  float = Field(..., description="Temperatura predicha a 72 h (°C)")
    h168: float = Field(..., description="Temperatura predicha a 168 h (°C)")
    full_168h: list[float] = Field(..., description="168 predicciones horarias completas (°C)")

# ── HTML Landing Page ─────────────────────────────────────────────────────────
LANDING_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>TCN Weather Forecaster API</title>
<style>
  :root {
    --blue:    #2563eb; --blue-light: #dbeafe; --blue-dark: #1e3a8a;
    --green:   #16a34a; --green-light: #dcfce7;
    --orange:  #ea580c; --orange-light: #ffedd5;
    --gray:    #6b7280; --gray-light: #f9fafb; --gray-border: #e5e7eb;
    --text:    #111827; --radius: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f1f5f9; color: var(--text); }

  /* NAV */
  nav { background: var(--blue-dark); color: white; padding: 0 2rem;
        display: flex; align-items: center; gap: 1rem; height: 60px; }
  nav .logo { font-size: 1.1rem; font-weight: 700; letter-spacing: -0.3px; }
  nav .badge { background: rgba(255,255,255,0.15); font-size: .72rem;
               padding: 2px 8px; border-radius: 20px; font-weight: 600; }
  nav a { color: rgba(255,255,255,0.8); text-decoration: none; font-size: .88rem;
          margin-left: auto; padding: 6px 14px; border: 1px solid rgba(255,255,255,0.3);
          border-radius: 6px; transition: background .2s; }
  nav a:hover { background: rgba(255,255,255,0.15); }

  /* HERO */
  .hero { background: linear-gradient(135deg, var(--blue-dark) 0%, #1d4ed8 60%, #2563eb 100%);
          color: white; padding: 3.5rem 2rem 3rem; text-align: center; }
  .hero h1 { font-size: 2.2rem; font-weight: 800; margin-bottom: .5rem; }
  .hero p  { font-size: 1.05rem; opacity: .85; max-width: 600px; margin: 0 auto 1.8rem; }
  .hero .pills { display: flex; justify-content: center; gap: .7rem; flex-wrap: wrap; }
  .pill { background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25);
          padding: 5px 14px; border-radius: 20px; font-size: .82rem; font-weight: 500; }

  /* STATS */
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
           gap: 1rem; max-width: 900px; margin: -1.5rem auto 0; padding: 0 1.5rem; position: relative; z-index:2; }
  .stat-card { background: white; border-radius: var(--radius); padding: 1.2rem 1rem;
               box-shadow: 0 4px 20px rgba(0,0,0,.08); text-align: center; }
  .stat-card .val { font-size: 1.8rem; font-weight: 800; color: var(--blue); line-height: 1; }
  .stat-card .lbl { font-size: .78rem; color: var(--gray); margin-top: 4px; }

  /* MAIN */
  main { max-width: 900px; margin: 2.5rem auto; padding: 0 1.5rem 4rem; }
  h2 { font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem; color: var(--blue-dark);
       border-left: 4px solid var(--blue); padding-left: .7rem; }

  /* ENDPOINTS */
  .endpoint { background: white; border-radius: var(--radius); margin-bottom: 1rem;
              box-shadow: 0 2px 10px rgba(0,0,0,.05); overflow: hidden; }
  .ep-header { display: flex; align-items: center; gap: .9rem; padding: 1rem 1.3rem;
               cursor: pointer; user-select: none; }
  .ep-header:hover { background: var(--gray-light); }
  .method { font-size: .75rem; font-weight: 700; padding: 4px 10px; border-radius: 6px;
            min-width: 50px; text-align: center; letter-spacing: .5px; }
  .GET  { background: var(--blue-light);  color: var(--blue); }
  .POST { background: var(--green-light); color: var(--green); }
  .ep-path { font-family: 'Courier New', monospace; font-size: .95rem; font-weight: 600; }
  .ep-desc { font-size: .85rem; color: var(--gray); margin-left: auto; }
  .ep-body { border-top: 1px solid var(--gray-border); padding: 1.2rem 1.3rem;
             display: none; background: var(--gray-light); }
  .ep-body.open { display: block; }
  .ep-body p { font-size: .88rem; color: #374151; margin-bottom: .8rem; line-height: 1.6; }
  .ep-body table { width: 100%; border-collapse: collapse; font-size: .83rem; margin-bottom: .8rem; }
  .ep-body th { background: #e2e8f0; padding: 6px 10px; text-align: left; font-weight: 600; }
  .ep-body td { padding: 6px 10px; border-bottom: 1px solid var(--gray-border); }
  .ep-body td code { background: #e2e8f0; padding: 1px 5px; border-radius: 4px; font-size: .82rem; }

  /* CODE BLOCK */
  pre { background: #1e293b; color: #e2e8f0; border-radius: 8px; padding: 1rem 1.2rem;
        font-size: .8rem; line-height: 1.6; overflow-x: auto; margin-top: .5rem; }
  pre .kw  { color: #93c5fd; }
  pre .str { color: #86efac; }
  pre .num { color: #fcd34d; }
  pre .key { color: #f9a8d4; }

  /* EXAMPLE BOX */
  .ex-title { font-size: .8rem; font-weight: 600; color: var(--gray); margin-bottom: .4rem; text-transform: uppercase; letter-spacing: .5px; }
  .try-btn { display: inline-block; margin-top: .8rem; padding: 7px 16px; background: var(--blue);
             color: white; border-radius: 7px; font-size: .82rem; text-decoration: none;
             font-weight: 600; transition: background .2s; }
  .try-btn:hover { background: var(--blue-dark); }

  /* ARCHITECTURE CARD */
  .arch-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }
  .arch-card { background: white; border-radius: var(--radius); padding: 1.1rem 1.2rem;
               box-shadow: 0 2px 10px rgba(0,0,0,.05); }
  .arch-card h3 { font-size: .9rem; font-weight: 700; color: var(--blue-dark); margin-bottom: .5rem; }
  .arch-card ul { font-size: .83rem; color: #374151; padding-left: 1.2rem; line-height: 1.8; }

  /* FOOTER */
  footer { text-align: center; font-size: .8rem; color: var(--gray); padding: 2rem;
           border-top: 1px solid var(--gray-border); background: white; }
  footer strong { color: var(--blue-dark); }

  @media(max-width:600px) {
    .hero h1 { font-size: 1.6rem; }
    .arch-grid { grid-template-columns: 1fr; }
    .ep-desc { display: none; }
  }
</style>
</head>
<body>

<nav>
  <span class="logo">🌡️ TCN Weather Forecaster</span>
  <span class="badge">v1.0.0</span>
  <a href="/docs" target="_blank">Swagger UI →</a>
</nav>

<div class="hero">
  <h1>TCN Weather Forecaster API</h1>
  <p>Predicción de temperatura horaria a múltiples horizontes sobre la red INMET de Brasil, usando el mejor modelo del benchmark: <strong>Temporal Convolutional Network</strong>.</p>
  <div class="pills">
    <span class="pill">🏆 Mejor modelo del benchmark</span>
    <span class="pill">📍 38 estaciones INMET</span>
    <span class="pill">⏱ Horizontes: 24h · 72h · 168h</span>
    <span class="pill">🧠 Deep Learning — PyTorch</span>
  </div>
</div>

<div class="stats">
  <div class="stat-card"><div class="val">2.51°C</div><div class="lbl">RMSE promedio (test 2023)</div></div>
  <div class="stat-card"><div class="val">38</div><div class="lbl">Estaciones disponibles</div></div>
  <div class="stat-card"><div class="val">168h</div><div class="lbl">Ventana de contexto (7 días)</div></div>
  <div class="stat-card"><div class="val">TCN</div><div class="lbl">Mejor modelo — Ranking #1</div></div>
  <div class="stat-card"><div class="val">24M</div><div class="lbl">Registros horarios procesados</div></div>
</div>

<main>

  <br/><br/>
  <h2>Endpoints</h2>

  <!-- GET / -->
  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method GET">GET</span>
      <span class="ep-path">/</span>
      <span class="ep-desc">Información general del servicio y modelo</span>
    </div>
    <div class="ep-body">
      <p>Devuelve metadata del servicio: modelo usado, RMSE, horizontes, número de estaciones disponibles.</p>
      <div class="ex-title">Ejemplo de respuesta</div>
      <pre>{
  <span class="key">"service"</span>:  <span class="str">"TCN Weather Forecaster API"</span>,
  <span class="key">"model"</span>:    <span class="str">"TCN (Temporal Convolutional Network) — Bai et al. 2018"</span>,
  <span class="key">"rmse"</span>:    <span class="str">"2.51 °C (test 2023)"</span>,
  <span class="key">"horizons"</span>: [<span class="str">"24 h"</span>, <span class="str">"72 h"</span>, <span class="str">"168 h"</span>],
  <span class="key">"stations"</span>: <span class="num">38</span>
}</pre>
      <a class="try-btn" href="/docs#/Info/root__get" target="_blank">Probar en Swagger →</a>
    </div>
  </div>

  <!-- GET /health -->
  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method GET">GET</span>
      <span class="ep-path">/health</span>
      <span class="ep-desc">Estado del servicio y modelos cargados en memoria</span>
    </div>
    <div class="ep-body">
      <p>Devuelve el estado operacional del servicio, cuántos modelos están cargados en caché y cuántas estaciones están disponibles.</p>
      <pre>{
  <span class="key">"status"</span>:           <span class="str">"ok"</span>,
  <span class="key">"models_in_cache"</span>:  <span class="num">4</span>,
  <span class="key">"stations_avail"</span>:   <span class="num">38</span>
}</pre>
      <a class="try-btn" href="/docs#/Info/health_health_get" target="_blank">Probar en Swagger →</a>
    </div>
  </div>

  <!-- GET /stations -->
  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method GET">GET</span>
      <span class="ep-path">/stations</span>
      <span class="ep-desc">Lista de estaciones disponibles con su RMSE individual</span>
    </div>
    <div class="ep-body">
      <p>Devuelve todas las estaciones INMET con modelo TCN entrenado y su RMSE total en el conjunto de test (2023).</p>
      <pre>{
  <span class="key">"count"</span>: <span class="num">38</span>,
  <span class="key">"stations"</span>: [
    { <span class="key">"station"</span>: <span class="str">"A001"</span>, <span class="key">"rmse_total"</span>: <span class="num">2.34</span> },
    { <span class="key">"station"</span>: <span class="str">"A101"</span>, <span class="key">"rmse_total"</span>: <span class="num">2.71</span> },
    ...
  ]
}</pre>
      <a class="try-btn" href="/docs#/Info/stations_stations_get" target="_blank">Probar en Swagger →</a>
    </div>
  </div>

  <!-- GET /predict/latest/{station} -->
  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method GET">GET</span>
      <span class="ep-path">/predict/latest/{station}</span>
      <span class="ep-desc">Predicción inmediata desde el último dato disponible</span>
    </div>
    <div class="ep-body">
      <p>El endpoint más sencillo. Solo necesitas el código de estación — el modelo usa automáticamente las últimas 168 horas de datos procesados como contexto histórico.</p>
      <div class="ex-title">Ejemplo</div>
      <pre><span class="kw">GET</span> /predict/latest/A701</pre>
      <pre>{
  <span class="key">"station"</span>:      <span class="str">"A701"</span>,
  <span class="key">"model"</span>:        <span class="str">"TCN"</span>,
  <span class="key">"context_end"</span>:  <span class="str">"2023-12-31 23:00:00"</span>,
  <span class="key">"h24"</span>:          <span class="num">21.43</span>,
  <span class="key">"h72"</span>:          <span class="num">20.87</span>,
  <span class="key">"h168"</span>:         <span class="num">19.64</span>,
  <span class="key">"full_168h"</span>:    [<span class="num">21.8</span>, <span class="num">21.6</span>, <span class="num">21.2</span>, <span class="str">...</span>]
}</pre>
      <a class="try-btn" href="/docs#/Predicci%C3%B3n/predict_latest_predict_latest__station__get" target="_blank">Probar en Swagger →</a>
    </div>
  </div>

  <!-- POST /predict -->
  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method POST">POST</span>
      <span class="ep-path">/predict</span>
      <span class="ep-desc">Predicción con timestamp específico</span>
    </div>
    <div class="ep-body">
      <p>Predicción de temperatura a 24h, 72h y 168h para una estación y timestamp específicos. El modelo usa las 168 horas previas al timestamp como contexto histórico.</p>
      <table>
        <tr><th>Campo</th><th>Tipo</th><th>Requerido</th><th>Descripción</th></tr>
        <tr><td><code>station</code></td><td>string</td><td>✅ Sí</td><td>Código INMET (ej: <code>A701</code>)</td></tr>
        <tr><td><code>timestamp</code></td><td>ISO 8601</td><td>❌ No</td><td>Último dato conocido. Si es null → usa el último disponible</td></tr>
      </table>
      <div class="ex-title">Request</div>
      <pre>{
  <span class="key">"station"</span>:   <span class="str">"A701"</span>,
  <span class="key">"timestamp"</span>: <span class="str">"2023-06-15T12:00:00"</span>
}</pre>
      <div class="ex-title">Response</div>
      <pre>{
  <span class="key">"station"</span>:      <span class="str">"A701"</span>,
  <span class="key">"model"</span>:        <span class="str">"TCN"</span>,
  <span class="key">"context_end"</span>:  <span class="str">"2023-06-15T12:00:00"</span>,
  <span class="key">"h24"</span>:          <span class="num">24.15</span>,
  <span class="key">"h72"</span>:          <span class="num">23.80</span>,
  <span class="key">"h168"</span>:         <span class="num">22.40</span>,
  <span class="key">"full_168h"</span>:    [<span class="num">24.1</span>, <span class="num">23.9</span>, <span class="str">...</span>]
}</pre>
      <a class="try-btn" href="/docs#/Predicci%C3%B3n/predict_predict_post" target="_blank">Probar en Swagger →</a>
    </div>
  </div>

  <br/>
  <h2>Arquitectura del Modelo</h2>
  <div class="arch-grid">
    <div class="arch-card">
      <h3>🧠 TCN — Temporal Convolutional Network</h3>
      <ul>
        <li>4 bloques de convoluciones causales dilatadas</li>
        <li>64 canales por bloque · kernel_size = 3</li>
        <li>Dropout = 0.2 · Weight normalization</li>
        <li>Lookback: 168 h · Horizon: 168 h</li>
        <li>~400K parámetros entrenables</li>
      </ul>
    </div>
    <div class="arch-card">
      <h3>📊 Evaluación en Test (2023)</h3>
      <ul>
        <li>RMSE: <strong>2.51 °C</strong> (4 estaciones comunes)</li>
        <li>Ranking #1 sobre 7 arquitecturas</li>
        <li>Entrenado en 38 estaciones · 2 seeds</li>
        <li>Significativo: Friedman p = 2×10⁻²⁶</li>
        <li>Datos: ~2.63M registros INMET</li>
      </ul>
    </div>
    <div class="arch-card">
      <h3>⚙️ Pipeline de Features</h3>
      <ul>
        <li>145 features engineered por estación</li>
        <li>Lags temporales + rolling windows</li>
        <li>Variables cíclicas (hora, día del año)</li>
        <li>StandardScaler fitteado solo en train</li>
      </ul>
    </div>
    <div class="arch-card">
      <h3>🚀 Cómo usar la API</h3>
      <ul>
        <li><code>uvicorn api:app --reload --port 8000</code></li>
        <li>Swagger UI: <a href="/docs">/docs</a></li>
        <li>ReDoc: <a href="/redoc">/redoc</a></li>
        <li>OpenAPI JSON: <a href="/openapi.json">/openapi.json</a></li>
      </ul>
    </div>
  </div>

</main>

<footer>
  <strong>TCN Weather Forecaster API</strong> · Proyecto Final Deep Learning · INMET Brasil ·
  Modelo: TCN (Bai et al., 2018) · PyTorch + FastAPI
</footer>

<script>
function toggle(header) {
  const body = header.nextElementSibling;
  body.classList.toggle('open');
}
// Abrir el primer endpoint por defecto
document.querySelector('.ep-header').click();
</script>
</body>
</html>
"""

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TCN Weather Forecaster API",
    description=(
        "## Forecasting Meteorológico — Red INMET Brasil\n\n"
        "Sirve el **mejor modelo del benchmark** (TCN) para predecir temperatura del aire "
        "a **24 h, 72 h y 168 h** sobre 38 estaciones meteorológicas de Brasil.\n\n"
        "**RMSE promedio:** 2.51 °C (test 2023) · **Ranking:** #1 sobre 7 arquitecturas"
    ),
    version="1.0.0",
)

@app.on_event("startup")
async def preload():
    for s in ["A001", "A101", "A701", "A801"]:
        try:
            _get_predictor(s); print(f"  ✅ TCN/{s} listo")
        except HTTPException:
            print(f"  ⚠️  TCN/{s} no disponible")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing(): return LANDING_HTML

@app.get("/health", tags=["Info"], summary="Estado del servicio")
def health():
    return {"status": "ok", "models_in_cache": len(_model_cache),
            "stations_avail": len(_available_stations())}

@app.get("/stations", tags=["Info"], summary="Estaciones disponibles")
def stations():
    result = []
    for s in _available_stations():
        p = EXP_DIR / s / "seed=42" / "metrics.json"
        rmse = json.loads(p.read_text()).get("rmse_total") if p.exists() else None
        result.append({"station": s, "rmse_total": rmse})
    return {"count": len(result), "stations": result}

@app.post("/predict", response_model=PredictResponse, tags=["Predicción"],
          summary="Predicción por estación y timestamp")
def predict(req: PredictRequest):
    """Predice temperatura a **24h, 72h y 168h** dada una estación y timestamp opcional."""
    pred = _get_predictor(req.station)
    df   = _get_station_data(req.station)
    end  = datetime.fromisoformat(req.timestamp) if req.timestamp else None
    win  = _extract_window(df, end)
    res  = pred.predict_from_window(win)
    return PredictResponse(station=req.station.upper(), context_end=str(win.index[-1]), **res)

@app.get("/predict/latest/{station}", response_model=PredictResponse,
         tags=["Predicción"], summary="Predicción desde el último dato disponible")
def predict_latest(station: str):
    """Predicción usando las **últimas 168 horas** disponibles para la estación."""
    pred = _get_predictor(station)
    df   = _get_station_data(station)
    win  = _extract_window(df, None)
    res  = pred.predict_from_window(win)
    return PredictResponse(station=station.upper(), context_end=str(win.index[-1]), **res)
