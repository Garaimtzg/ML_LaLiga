# ADR-018 — Backtest jornada a jornada, registro de modelos y regla anti-sorpresa

- **Fecha**: 2026-07-12 (Fase 3)
- **Estado**: aceptada

## Contexto

SPEC §6.5 exige backtesting walk-forward "jornada a jornada (re-simulando el
ciclo real de reentrenamiento)" sobre las 3 últimas temporadas, con informe
comparado contra los baselines. SPEC §6.4 y §3.2 exigen versionado de modelos
con métricas y la regla anti-sorpresa del 10 %.

## Opciones consideradas

1. **Reentrenar solo una vez por temporada de test**: mucho más rápido, pero
   no re-simula el ciclo real (el modelo de la jornada 30 sabría lo mismo que
   el de la 1) e infraestima el rendimiento del sistema. Descartado.
2. **Calibrar dentro de la propia temporada de test**: fuga temporal directa.
   Descartado.

## Decisión

**Backtest** (`evaluation/backtest.py`):

- Para cada temporada de test T: calibradores y peso del ensemble se ajustan
  con el pool walk-forward de temporadas ANTERIORES a T (ADR-017); después T
  se predice jornada a jornada, reentrenando Dixon-Coles y LightGBM antes de
  cada jornada con todo lo jugado hasta la víspera (incluidas las jornadas
  previas de la propia T). El DC arranca en caliente desde la jornada anterior
  (~230 ajustes de LightGBM y ~120 de DC para 3 temporadas: minutos, no horas).
- Se evalúan cinco modelos: `dixon_coles`, `lgbm_±cuotas` (calibrados) y
  `ensemble_±cuotas`, con las métricas de SPEC §6.5 más la accuracy específica
  en partidos del Alavés.
- El informe `docs/reports/backtest_<fecha>.md` incluye: tabla por temporada,
  baselines de F2 en las mismas temporadas, medias, la tabla de fiabilidad del
  ensemble sin cuotas y el veredicto de los criterios de aceptación de SPEC
  §12.1 (sin cuotas < baseline Elo; con cuotas ≤ cuotas de cierre + 0.01).

**Registro** (`models/train.py`):

- Cada `alaves train` serializa el artefacto completo (DC + variantes LightGBM
  + calibradores + pesos) en `models/registry/<versión>/model.pkl` (pickle:
  artefacto local generado y consumido por este mismo código) junto a
  `metrics.json` y `config.json`, y escribe la fila en `model_registry`.
  Versión = `<feature_set>-<fecha>-<hora>`.
- **Regla anti-sorpresa**: la métrica de referencia es el log-loss de
  validación del `ensemble_sin_cuotas` (existe siempre, con y sin `--no-odds`,
  lo que hace homogénea la comparación entre versiones). Si empeora más del
  10 % (`[models].max_logloss_regression`) respecto a la última versión
  promocionada, la nueva se registra igualmente (auditoría) pero con
  `promoted=false`, y `alaves predict` la ignora.

## Consecuencias

- El backtest reproduce el ciclo real de la temporada 2026-27; sus números son
  una estimación honesta del rendimiento futuro del sistema.
- Un entrenamiento accidentalmente malo (datos corruptos, bug) no puede
  colarse en producción de forma silenciosa.
- Reproducibilidad (SPEC §12.4): match_id + model_version + feature_set_version
  recuperan las mismas probabilidades (artefacto congelado + features en BD).
