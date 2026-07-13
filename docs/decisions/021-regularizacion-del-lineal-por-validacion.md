# ADR-021 — Regularización del componente lineal elegida por validación

- **Fecha**: 2026-07-12 (Fase 3)
- **Estado**: aceptada

## Contexto

Tras ADR-020 el ensemble sin cuotas seguía sin batir al baseline Elo por un
margen mínimo (0.9740 vs 0.9706 en el backtest de 3 temporadas — dentro del
ruido muestral, pero SPEC §12.1 pide `<` estricto). El diagnóstico por los
pesos del apilado: el componente lineal recibía solo 0.10 de peso, señal de
que rendía peor que el propio Dixon-Coles y, por tanto, peor que el baseline
Elo del que debería ser un superconjunto.

## Causa

El baseline Elo logístico ajusta una logística sobre `elo_clubelo_diff` **en
crudo** (valores de ±cientos): su coeficiente es minúsculo y la regularización
L2 con `C=1.0` apenas lo toca, así que ajusta la señal Elo a fondo. El
componente lineal (ADR-020) **estandariza** las features, con lo que ese mismo
coeficiente pasa a ser de orden 1 y la L2 con `C=1.0` lo **encoge**,
debilitando la señal Elo justo en el modelo que debía explotarla mejor.

## Decisión

Elegir `C` (inverso de la fuerza de regularización L2 del componente lineal)
**por validación walk-forward**, igual que el ξ del Dixon-Coles (ADR-019):

- Rejilla `c_grid = [0.3, 1.0, 3.0, 10.0, 30.0]` en `[models.linear]`; se
  evalúa el log-loss medio sobre el pool out-of-fold y se elige el mejor. En
  el backtest, la C de cada temporada de test se elige solo con las anteriores;
  en el entrenamiento final, con todo el pool.
- Con `c_grid` vacío se usa `c` fijo (por defecto 1.0), sin selección.

**Instrumentación**: el backtest reporta ahora el componente `lineal_elo_forma`
como fila propia, además de `dixon_coles`, `lgbm_*` y los ensembles. Así el
log-loss de cada componente es visible y la elección de pesos deja de ser una
caja negra: si un componente fuerte recibe poco peso, se ve en la tabla.

## Consecuencias

- El componente lineal puede ahora ajustar la señal Elo con la regularización
  adecuada y sumarle forma y xG, en vez de quedar por debajo del baseline.
- `alaves train` imprime el `C` elegido junto al `xi`; ambos quedan en
  `model_registry.config_json` (reproducibilidad, SPEC §12.4).
- Nota de honestidad estadística: sobre ~1.140 partidos, diferencias de
  log-loss del orden de 0.003 están dentro del error muestral (~0.02). El
  objetivo de superar el baseline se persigue sin sobreajustar al criterio:
  todas las elecciones (ξ, C, calibración, pesos) son walk-forward, nunca
  sobre las temporadas de test.
