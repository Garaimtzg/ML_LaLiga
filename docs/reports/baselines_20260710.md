# Baselines 1X2 — evaluación walk-forward (F2)

Generado: 2026-07-10T17:42:54+00:00

Métricas: menor es mejor salvo accuracy. Las cuotas de cierre son el
baseline exigente (SPEC §6.1): todo modelo de F3 se compara contra esto.

| Baseline | Temporada | N | Log-loss | Brier | RPS | Accuracy |
|----------|-----------|---|----------|-------|-----|----------|
| frecuencias | 2023-24 | 380 | 1.0744 | 0.6498 | 0.2238 | 0.439 |
| elo_logistico | 2023-24 | 380 | 0.9612 | 0.5733 | 0.1872 | 0.521 |
| cuotas_cierre | 2023-24 | 380 | 0.9498 | 0.5649 | 0.1826 | 0.555 |
| frecuencias | 2024-25 | 380 | 1.0718 | 0.6480 | 0.2287 | 0.445 |
| elo_logistico | 2024-25 | 380 | 0.9734 | 0.5776 | 0.1953 | 0.542 |
| cuotas_cierre | 2024-25 | 380 | 0.9462 | 0.5594 | 0.1875 | 0.555 |
| frecuencias | 2025-26 | 380 | 1.0506 | 0.6330 | 0.2236 | 0.489 |
| elo_logistico | 2025-26 | 380 | 0.9772 | 0.5774 | 0.1988 | 0.526 |
| cuotas_cierre | 2025-26 | 380 | 0.9650 | 0.5717 | 0.1958 | 0.545 |

## Media por baseline

| Baseline | Log-loss | Brier | RPS | Accuracy |
|---|---|---|---|---|
| frecuencias | 1.0656 | 0.6436 | 0.2254 | 0.458 |
| elo_logistico | 0.9706 | 0.5761 | 0.1938 | 0.530 |
| cuotas_cierre | 0.9537 | 0.5653 | 0.1886 | 0.552 |
