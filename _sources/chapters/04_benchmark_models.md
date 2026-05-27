# Capítulo 4 — Implementación y entrenamiento

```{note}
Notebook fuente: [`04_benchmark_models.ipynb`](../../notebooks/04_benchmark_models.ipynb).
```

Se entrenan cinco arquitecturas (LSTM, GRU, TCN, Transformer y N-BEATS) con
**N semillas** distintas para obtener media ± desviación estándar.

Cada run produce en `experiments/<modelo>/<estación>/seed=<s>/`:

- `checkpoint.pt` (mejor estado por validación),
- `history.json` (curvas train/val),
- `predictions.npz` (`y_true`, `y_pred`, `timestamps`),
- `metrics.json` (RMSE / MAE / R² / MAPE / sMAPE, globales y por horizonte),
- `env.json` (versiones de libs y commit de git).
