# Capítulo 7 — Reproducibilidad

## Garantías incorporadas

| Riesgo                         | Garantía                                 | Verificación                              |
|--------------------------------|------------------------------------------|-------------------------------------------|
| Aleatoriedad no controlada     | `src.utils.set_seed`                     | `tests/test_seed.py`                      |
| Leakage entre splits           | Split temporal estricto                  | `tests/test_split_no_leakage.py`          |
| Leakage por ventaneo           | Ventanas dentro de un único split        | `tests/test_windowing_no_leakage.py`      |
| Leakage estadístico (scaling)  | `FeatureScaler.fit(train)` solamente     | `tests/test_scaler_fit_train_only.py`     |
| Métricas mal calculadas        | Sanity tests con valores conocidos       | `tests/test_metrics.py`                   |
| Drift de dependencias          | `capture_environment()` por run          | `experiments/.../env.json`                |
| Cambios silenciosos del dato   | `hash_dataframe`, `hash_file`            | `src.utils.reproducibility`               |

## Cómo reproducir el estudio

```bash
git clone <repo>
cd proyecto-final-deep
make setup
# coloca los ZIP anuales INMET en data/raw/ (ver data/raw/README.md)
make all   # ingest + process + eda + benchmark + book
```
