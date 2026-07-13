# ADR-024 — Explicabilidad: TreeSHAP nativo de LightGBM y ablation study

- **Fecha**: 2026-07-13 (Fase 5)
- **Estado**: aceptada

## Contexto

SPEC §7 exige un análisis de importancia de variables: SHAP global (beeswarm +
bar), dependencia parcial de las top, ablation study por bloques e informe
interpretativo, todo sobre la variante **sin cuotas** (la interpretable: con
cuotas el mercado tapa qué variables futbolísticas importan).

## Decisión

### SHAP sin la librería `shap`

La librería `shap` arrastra `numba`, que en este entorno no resuelve (choca con
numpy 2.x y solo soporta Python <3.10 en las versiones que uv selecciona). En
vez de degradar numpy o el intérprete, se usa el **TreeSHAP nativo de
LightGBM**: `Booster.predict(X, pred_contrib=True)` devuelve los valores SHAP
exactos (mismo algoritmo TreeSHAP, implementado en el propio LightGBM). Para
multiclase da, por clase, bloques de (n_features + 1) columnas (contribuciones
+ valor base); se reordena a (muestras, variables, clases).

Ventajas: cero dependencias nuevas problemáticas, valores exactos, y un test
verifica la propiedad de aditividad (SHAP + base = raw score del modelo).

**Dependencia nueva**: solo `matplotlib` (gráficos PNG del informe); es parte
del stack aprobado en CLAUDE.md §2.

### Alcance del análisis (`explain/`)

- `importance.py`: SHAP global (media de |SHAP| por variable, agregada sobre
  clases), beeswarm de la clase "victoria local" (qué empuja hacia el 1) y
  dependencia parcial de las variables top (efecto marginal barriendo cada una
  con el resto fijo). Backend `Agg` (sin ventana, solo escribe PNG).
- `ablation.py`: bloques de features por patrón de nombre (elo, xg, forma,
  descanso, contexto). Mide el log-loss walk-forward de la variante sin cuotas
  quitando cada bloque; un bloque aporta si quitarlo empeora el log-loss. Se
  usa el LightGBM sin cuotas (no el ensemble completo) para aislar la
  contribución de las features al clasificador y acotar el coste.
- `report.py`: orquesta todo y escribe `docs/reports/feature_importance.md` con
  las figuras y las tablas, en lenguaje claro.

**CLI**: `alaves report --importance` (SPEC §10). El beeswarm por partido con
waterfall (SPEC §7.2) es del dashboard (F6); el motor ya expone los valores
SHAP por muestra para construirlo.

## Consecuencias

- El proyecto responde empíricamente "¿qué variables mueven las predicciones?"
  y "¿cuánto aporta cada bloque?" sin depender de librerías frágiles.
- El informe se regenera con un comando tras cada reentrenamiento; sus figuras
  (PNG) acompañan al Markdown para que sea legible en GitHub.
- Limitación honesta anotada en el informe: deltas de ablation de pocas
  milésimas están dentro del ruido; importan el signo y el orden de magnitud
  relativo entre bloques, no la cifra exacta.
