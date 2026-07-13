# ADR-017 — Calibración isotónica sobre folds temporales y ensemble ponderado

- **Fecha**: 2026-07-12 (Fase 3)
- **Estado**: aceptada

## Contexto

SPEC §6.3: calibración posterior del LightGBM con regresión isotónica por
clase "sobre folds de validación temporal", verificada con reliability
diagrams; y ensemble final por media ponderada con peso ajustado por log-loss
en validación. Hay que concretar de dónde salen esos folds y qué se calibra.

## Opciones consideradas

1. **Calibrar también el Dixon-Coles**: descartado; es un modelo probabilístico
   generativo razonablemente calibrado de serie, y SPEC liga la calibración al
   modelo B. Si el reliability del ensemble mostrara sesgo del DC, se revisará.
2. **Platt (sigmoide) en vez de isotónica**: la isotónica es no paramétrica y
   corrige formas de descalibración que la sigmoide no puede; con >2.000
   predicciones out-of-fold hay muestra de sobra para ella. Platt queda como
   alternativa si algún día el pool es pequeño.

## Decisión

- **Folds temporales = temporadas walk-forward** (`train.py`): cada temporada
  desde la segunda se predice con modelos entrenados solo con las anteriores.
  El pool de esas predicciones out-of-fold (nunca vistas en entrenamiento)
  alimenta: (a) los calibradores isotónicos por clase del LightGBM y (b) la
  búsqueda en rejilla del peso del ensemble (paso 0.05, `[models.ensemble]`).
- **Métricas de validación honestas**: se calculan sobre la ÚLTIMA temporada,
  con calibradores y peso ajustados SOLO con las temporadas previas a ella.
  Los calibradores definitivos del artefacto usan el pool completo.
- **Suelo del 1 %** tras calibrar: la isotónica es escalonada y puede devolver
  0 exacto; en fútbol ningún resultado 1X2 baja en la práctica del 1 %, y un 0
  exacto revienta el log-loss con un solo "imposible" que ocurra. Tras el
  suelo se renormaliza a suma 1.
- **Ensemble** = `w·DC + (1−w)·LightGBM_calibrado`, un peso por variante.
  Con w∈{0,1} la rejilla recupera el mejor modelo individual, así que el
  ensemble nunca es peor que sus componentes en validación (test).
- **Reliability**: en F3 la verificación es numérica — tabla de fiabilidad por
  clase y bin (predicho medio vs frecuencia observada) incluida en el informe
  de backtest. El diagrama gráfico llega con el dashboard (F6), que es donde
  SPEC §9.5 lo sitúa de cara al usuario.

## Consecuencias

- Ninguna probabilidad publicada sale sin calibrar y sin haber sido validada
  fuera de muestra.
- El peso del ensemble queda registrado por variante en el artefacto y es
  auditable en `model_registry.config_json`.
