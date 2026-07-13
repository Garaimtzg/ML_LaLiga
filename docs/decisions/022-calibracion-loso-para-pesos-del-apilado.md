# ADR-022 — Selección de pesos del apilado con calibración leave-one-season-out

- **Fecha**: 2026-07-13 (Fase 3)
- **Estado**: aceptada

## Contexto

Tras ADR-021, la instrumentación del backtest reveló el problema de fondo: el
componente lineal Elo+forma ya batía al baseline Elo por sí solo (log-loss
medio 0.9703 vs 0.9706), pero el ensemble sin cuotas quedaba en 0.9740 —
**peor que su propio mejor componente**. Los pesos lo confirmaban: DC 0.70,
LightGBM 0.20, lineal solo 0.10, es decir, la mayor parte del peso en los dos
componentes peores (DC 0.974 y LightGBM ~1.00).

## Causa

Los pesos del apilado se elegían minimizando el log-loss del pool con los
componentes ya calibrados **sobre ese mismo pool** (in-sample). La isotónica
del Dixon-Coles y del LightGBM, evaluada sobre los datos con los que se
ajustó, parecía mejor de lo que generaliza; el componente lineal, en cambio,
entraba sin calibrar (honesto). Resultado: la comparación estaba sesgada a
favor de DC y LightGBM, y el optimizador les daba un peso que no merecían,
diluyendo al componente que de verdad predecía mejor. Un ensemble no debería
quedar por debajo de su mejor componente fuera de muestra; esto ocurría por
ese sesgo.

## Decisión

Elegir los pesos con los tres componentes calibrados **leave-one-season-out**
(`_loso_calibrated`): para cada temporada del pool, sus tres componentes se
calibran con isotónicas ajustadas SOLO con las demás temporadas. Así los tres
—Dixon-Coles, LightGBM y el tercero (mercado o lineal)— se comparan con
probabilidades calibradas fuera de muestra, sin la ventaja artificial del
in-sample, y el optimizador reparte el peso según lo que cada uno predice de
verdad.

Los calibradores que se guardan en el artefacto se ajustan, como antes, con el
pool completo (los que usará el modelo final); solo la *elección de pesos* usa
la versión LOSO. Cada componente pasa a tener su propio calibrador
(`VariantModel.component_calibrators = [dc, lightgbm, tercero]`), en lugar del
esquema anterior donde solo el LightGBM y el DC se calibraban.

## Consecuencias

- El apilado deja de estar sesgado hacia los componentes que se ven a sí
  mismos calibrados; el componente honestamente mejor recibe el peso que le
  corresponde (verificado con un test de mecanismo: un componente genuinamente
  mejor se lleva >0.5 del peso).
- Coste extra moderado: por cada temporada del pool se ajustan calibradores
  con las demás (isotónica, milisegundos).
- Sigue siendo walk-forward puro: ninguna elección (ξ, C, calibración, pesos)
  ve las temporadas de test. La honestidad estadística de ADR-021 se mantiene:
  diferencias de log-loss de milésimas sobre ~1.140 partidos están dentro del
  ruido muestral; el objetivo de superar el baseline se persigue sin
  sobreajustar al criterio.
