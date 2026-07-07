# ADR-006 — Cálculo aproximado de la jornada (matchday)

- **Fecha**: 2026-07-07 (Fase 1)
- **Estado**: aceptada

## Contexto

La tabla `matches` tiene columna `matchday` (SPEC §3.2) y las features de F2
usan la jornada como "fase de la temporada". Pero football-data y Understat no
publican la jornada oficial de LaLiga, y los aplazamientos hacen que la
jornada oficial no coincida con el orden cronológico.

## Opciones consideradas

1. Dejar `matchday` a NULL hasta tener una fuente oficial (API-Football, F7).
2. Scrapear una fuente adicional solo para la jornada oficial.
3. **Aproximarla por conteo**: ordenados los partidos por fecha, el partido N
   de un equipo pertenece a su "jornada N"; para un partido concreto se toma
   `max(nº de partido del local, nº del visitante)`.

## Decisión

Opción 3 (`assign_matchdays` en `etl/ingest.py`). El uso real de `matchday` es
como indicador ordinal de fase de temporada (jornada 3 ≈ inicio, jornada 35 ≈
final), no como etiqueta reglamentaria; la aproximación por conteo es exacta
salvo aplazamientos, y con aplazamientos se desvía como mucho unas pocas
jornadas en partidos aislados — irrelevante para ese uso. El `max()` absorbe
el caso típico (un partido aplazado no puede ser "jornada 2" si ambos equipos
ya jugaron 5 partidos).

Opción 1 descartada porque F2 la necesita ya; opción 2 descartada por añadir
un scraper (fragilidad) para un dato de valor marginal.

## Consecuencias

- `matchday` ∈ [1, 38] garantizado por validación, pero puede diferir de la
  jornada oficial de LaLiga en partidos aplazados.
- Cuando API-Football entre en F7 con el calendario oficial 2026-27, la
  temporada en curso tendrá jornada oficial; si se quisiera corregir el
  histórico, sería una mejora localizada en `assign_matchdays`.
- Test dedicado que documenta el comportamiento con aplazamientos.

## Actualización (2026-07-07, ADR-008)

Con el cambio del xG a FBref, su columna **Wk (jornada oficial)** sobreescribe
esta aproximación para todos los partidos que FBref cubre. El conteo del
`assign_matchdays` se mantiene como base/fallback para partidos sin Wk.
