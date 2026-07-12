# ADR-015 — Dixon-Coles: implementación propia y decisiones de parametrización

- **Fecha**: 2026-07-12 (Fase 3)
- **Estado**: aceptada

## Contexto

SPEC §6.2 fija el modelo A: Poisson bivariante con corrección de marcadores
bajos (Dixon & Coles, 1997), ponderación temporal exponencial y MLE con
`scipy.optimize` en código propio. Quedan por decidir los detalles que el
paper deja abiertos y que afectan a resultados y reproducibilidad.

## Opciones consideradas

1. **Librerías existentes** (`penaltyblog`, implementaciones sueltas de GitHub):
   descartadas por SPEC explícito ("no depender de librerías abandonadas") y
   porque el código propio es condición de transparencia del proyecto.
2. **Poisson simple sin corrección**: descartado; infraestima sistemáticamente
   los 0-0/1-1, que son ~19 % de los partidos de LaLiga.

## Decisión

Implementación en `models/dixon_coles.py` con estas elecciones concretas:

- **Parametrización**: `lambda = exp(atk_h − def_a + gamma)`,
  `mu = exp(atk_a − def_h)`. Ambos parámetros son "mayor = mejor", lo que hace
  los rankings de ataque/defensa legibles en el dashboard (F6).
- **Identificabilidad**: sumar una constante a todos los ataques y defensas no
  cambia el modelo; se elimina el grado de libertad fijando media(ataques)=0
  (el ataque del primer equipo = −suma del resto en el vector optimizado).
- **Ponderación temporal**: `peso = exp(−xi·días)` respecto a la fecha de
  referencia del ajuste. `xi = 0.0019/día` por defecto (SPEC; un partido de
  hace un año pesa ~0.5), configurable en `[models.dixon_coles]`.
- **rho acotado** a ±0.2 en la optimización (con `lambda·mu` grande, un rho
  desbocado haría tau ≤ 0; además se protege el log con clip). El valor
  ajustado real ronda −0.05.
- **Matriz de marcadores truncada en 10 goles** por equipo y renormalizada
  (la masa perdida es ~1e-9; la corrección tau conserva la suma exactamente,
  verificado por test).
- **Equipos no vistos en entrenamiento** (recién ascendidos sin historia en
  la BD, p. ej. Racing y Dépor en 2026-27): heredan la media de parámetros de
  los 3 equipos más débiles del ajuste. Tratarlos como equipo medio (parámetros
  0) los sobrevaloraría; la evidencia empírica es que el ascendido rinde como
  un colista. El LightGBM del ensemble además ve su `promoted_flag` y su Elo
  de ClubElo (que sí cubre Segunda), así que el ensemble no queda ciego.
- **Optimización**: L-BFGS-B con gradiente numérico (≈60 parámetros y objetivo
  vectorizado: converge en segundos; un gradiente analítico añadiría código
  denso sin necesidad). Soporta `warm_start` desde un modelo previo para los
  reajustes jornada a jornada del backtest.
- **Verificación** (SPEC §11): caso resuelto a mano en `tests/test_dixon_coles.py`
  (matriz con lambda=1.2, mu=0.8, rho=−0.1) + recuperación de parámetros sobre
  una liga sintética generada con parámetros conocidos.

## Consecuencias

- Los parámetros ataque/defensa por equipo quedan disponibles como ranking
  interpretable, y la matriz de marcadores da el "marcador más probable" que
  exige la salida de SPEC §2.
- `xi`, `max_goals` y la cota de rho son configurables sin tocar código;
  el ajuste fino de `xi` por validación walk-forward queda para cuando el
  backtest esté operativo (esta fase lo deja en el valor del SPEC).
