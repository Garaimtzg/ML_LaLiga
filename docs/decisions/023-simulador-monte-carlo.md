# ADR-023 — Simulador Monte Carlo de la clasificación

- **Fecha**: 2026-07-13 (Fase 4)
- **Estado**: aceptada

## Contexto

SPEC §8 exige proyectar la clasificación final por simulación Monte Carlo
(N=10.000) sobre las probabilidades 1X2 del ensemble, con salidas por equipo
(posición esperada, distribución de posiciones, P(título/Champions/Europa/
descenso), puntos esperados). Hay que fijar el diseño concreto.

## Opciones consideradas

1. **Sumar los resultados más probables** de cada partido: descartado por SPEC
   — sesga sistemáticamente (ignora la varianza y los empates) y no da
   distribuciones, solo un punto.
2. **Monte Carlo muestreando cada partido** (elegida): da distribuciones
   completas y probabilidades honestas de cada zona.

## Decisión

`simulation/monte_carlo.py`:

- **Resultado 1X2 de cada partido pendiente**: se muestrea de la distribución
  del ensemble (no el más probable). Los puntos se reparten según el resultado.
- **Diferencia de goles para el desempate**: se muestrea de la matriz de
  marcadores del Dixon-Coles CONDICIONADA al signo del resultado ya muestreado
  (victoria local → celdas con dg>0, renormalizadas). Así la DG es coherente
  con el 1X2 del ensemble y con la distribución de marcadores del DC.
- **Desempate**: puntos y luego diferencia de goles; el head-to-head
  reglamentario de LaLiga se aproxima por DG (limitación de SPEC §8.2). Los
  empates exactos (mismos puntos y DG) se rompen con ruido aleatorio ínfimo
  para no favorecer a un índice de equipo.
- **Vectorización**: por cada partido se muestrean las N simulaciones de una
  vez con numpy; N=10.000 sobre una liga entera corre en segundos. Semilla
  fija por defecto (reproducibilidad, CLAUDE.md §2), parametrizable.
- **Zonas parametrizadas** en `config/settings.toml` (`[league.zones]`), no
  hardcodeadas: título, Champions, Europa, Conference, descenso como rangos de
  posición.

**CLI `alaves simulate`** con dos modos:

- *Temporada en curso* (por defecto): partidos jugados = clasificación de
  partida; programados = por simular (los ingiere la F7). Hasta entonces avisa
  honestamente de que no hay nada que simular, como `predict`.
- *Demo/validación* (`--season S --from-matchday N`): proyecta una temporada
  histórica desde la jornada N tomando las anteriores como reales. Permite
  contrastar la tabla proyectada con la que ocurrió de verdad, con datos reales
  ya disponibles (sin esperar a la F7).

## Consecuencias

- El motor no depende de la fuente de las probabilidades: recibe P(1X2) por
  partido, así que sirve igual para el ensemble actual que para versiones
  futuras del modelo.
- **Limitación del modo demo** (documentada): al proyectar una temporada
  histórica, las features de forma de las jornadas tardías ven resultados
  reales de las jornadas intermedias que "deberían" estar por jugar, lo que
  hace las probabilidades algo optimistas. No afecta al uso real (F7), donde
  los partidos futuros no tienen resultado. El modo demo es para validar el
  pipeline, no para medir precisión.
- El gráfico de evolución jornada a jornada de P(descenso)/P(Europa) del Alavés
  (SPEC §8.4) se deja para el dashboard (F6), su ubicación natural; el motor ya
  expone todo lo necesario para construirlo.
