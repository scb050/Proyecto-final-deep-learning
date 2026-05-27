# Capítulo 6 — Benchmark final y tests estadísticos

```{note}
Notebook fuente: [`06_benchmark_final.ipynb`](../../notebooks/06_benchmark_final.ipynb).
```

Tabla agregada (media ± std + IC95% bootstrap), RMSE por paso del horizonte,
y los siguientes tests:

- **Diebold-Mariano** con corrección Harvey-Leybourne-Newbold,
- **Friedman** con post-hoc **Nemenyi / Bonferroni / Holm**,
- **Wilcoxon signed-rank** sobre errores pareados,
- **Ljung-Box** y **BDS** sobre residuos del mejor modelo.

Resultados en `results/tables/benchmark/` y `results/stats/`.
