# Capítulo 1 — Problema y dataset

```{note}
Este capítulo se nutre del notebook
[`notebooks/01_problem_dataset.ipynb`](../../notebooks/01_problem_dataset.ipynb).
Ejecutarlo regenera tablas y figuras citadas aquí.
```

## Problema

Forecasting multistep de variables meteorológicas a frecuencia horaria. Dado
un *lookback* de $L$ horas, predecir el *horizonte* $H$ siguiente.

## Dataset

Series anuales del [INMET](https://portal.inmet.gov.br/dadoshistoricos)
para 2016–2023 (8 años) en una o más estaciones automáticas.

Detalles operativos —encoding `latin-1`, separador `;`, decimal coma,
header de 8 líneas, faltantes `-9999`— se manejan en
`src/data/clean.py` y se documentan en `data/raw/README.md`.
