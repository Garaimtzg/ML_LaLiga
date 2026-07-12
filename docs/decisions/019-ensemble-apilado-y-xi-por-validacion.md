# ADR-019 — Ensemble apilado de 3 componentes y ξ elegido por validación

- **Fecha**: 2026-07-12 (Fase 3)
- **Estado**: aceptada

## Contexto

El primer backtest real de F3 (informe `backtest_20260712.md`) no pasó los
criterios de SPEC §12.1 por poco:

- ensemble sin cuotas 0.9770 vs baseline Elo 0.9706 (faltaban 0.006)
- ensemble con cuotas 0.9752 vs cuotas de cierre + 0.01 = 0.9637

Diagnóstico: el LightGBM es el componente débil (log-loss ~0.99-1.01 con
~2.500 filas y ~70 features), el Dixon-Coles se queda a 0.004 del Elo, y el
ensemble de 2 componentes no usaba dos señales fuertes que ya están en el
sistema: las probabilidades implícitas del mercado (apertura) y el propio Elo
logístico.

## Opciones consideradas

1. **Ajustar hiperparámetros del LightGBM** (rejilla u optuna): posible, pero
   el margen realista de un GBM con estos datos es limitado y el coste de
   validación alto. Queda como siguiente palanca si esto no basta.
2. **Meta-modelo logístico sobre los componentes** (stacking clásico con
   aprendiz): más flexible, pero menos transparente y con más riesgo de
   sobreajuste sobre un pool de ~2.000 predicciones que una media ponderada.
3. **Apilado por media ponderada de 3 componentes** (elegida).

## Decisión

**Ensemble apilado** (`ensemble.optimal_weights`, rejilla sobre el símplex de
paso 0.05, minimizando log-loss sobre el pool walk-forward — igual de honesto
que antes: nada del test influye en sus propios pesos):

- `con_cuotas`: Dixon-Coles + LightGBM calibrado + **mercado (apertura)**.
  Las cuotas de apertura ya eran feature del GBM, pero como componente directo
  no se diluyen entre otras 69 columnas. Si un partido no tiene cuotas, ese
  componente cae al Dixon-Coles para esa fila.
- `sin_cuotas`: Dixon-Coles + LightGBM calibrado + **Elo logístico** (el mismo
  modelo del baseline, ajustado walk-forward). Es stacking estándar: el
  criterio "batir al baseline" exige que el sistema final produzca mejores
  probabilidades que el Elo solo, y la forma más directa es incorporar esa
  señal y dejar que la validación decida su peso. Con peso 1 en él, el
  ensemble empata con el baseline; cualquier otra cosa que la validación
  elija lo mejora.

**ξ por validación** (SPEC §6.2 ya pedía "ajustar por validación"): la rejilla
`xi_grid = [0.0005, 0.001, 0.0019, 0.0035]` se evalúa en el pool walk-forward
y se elige el de mejor log-loss medio. En el backtest, el ξ de cada temporada
de test se elige solo con temporadas anteriores; en el entrenamiento final,
con todo el pool. Configurable en `[models.dixon_coles]`; con la lista vacía
se usa el `xi` fijo.

## Consecuencias

- El baseline Elo ya no puede ganar al ensemble sin cuotas en validación por
  construcción (es un componente); la pregunta pasa a ser cuánto se le gana
  en test, que es lo que mide el backtest.
- El informe de backtest y `alaves train` muestran los pesos por componente:
  si el LightGBM recibe peso ~0, es la señal empírica de que hay que mejorarlo
  (siguiente palanca: hiperparámetros/selección de features, con su ADR).
- Coste extra de cómputo moderado: 4 ajustes de Dixon-Coles por temporada del
  pool (con arranque en caliente) y una logística por jornada en el backtest.
