# ADR-013 — Elo interno: fórmula y parámetros v1

- **Fecha**: 2026-07-10 (Fase 2)
- **Estado**: aceptada

## Contexto

SPEC §3.1 exige mantener un Elo propio recalculable junto al de ClubElo, para
no depender de la fuente y poder ajustar el factor K. Hay que fijar la
variante concreta.

## Opciones consideradas

1. **Elo clásico** (esperado logístico a escala 400, K fijo, ventaja de campo
   aditiva).
2. Elo con multiplicador por margen de goles (estilo World Football Elo).
3. K variable por fase de temporada o por antigüedad del equipo.

## Decisión

Opción 1 para v1 — la más simple y legible:

    E_local = 1 / (1 + 10^(-((R_local + ventaja) - R_visitante) / 400))
    R' = R + K · (S - E)        S ∈ {1, 0.5, 0}

- Parámetros en `config/settings.toml` ([features.elo_internal]): `K=20`,
  `ventaja=60` puntos Elo, `rating_inicial=1500` para todos.
- Sin multiplicador por margen: añade un hiperparámetro más a calibrar y su
  aportación se medirá en F3 comparando contra esta base (el margen de goles
  ya entra al modelo vía las features de forma con xG).
- Cálculo secuencial en orden cronológico: el rating PRE-partido de cada fila
  solo ve partidos anteriores (sin fugas por construcción). El POST-partido
  se persiste en `elo.elo_internal` (fecha = día del partido) para dashboard
  y comparación con ClubElo en el análisis de variables (F5).
- Limitación aceptada y documentada: al no incluir partidos de Segunda, un
  equipo ascendido reaparece con su último rating de Primera (o 1500 si es
  nuevo). El `promoted_flag` y el Elo de ClubElo (que sí cubre Segunda)
  compensan — precisamente por esto SPEC mantiene ambos Elo.

## Consecuencias

- Verificado contra un caso resuelto a mano en tests (SPEC §11).
- K y ventaja son ajustables por validación walk-forward en F3 sin tocar
  código; cualquier cambio de fórmula requerirá revisar este ADR.
