# ADR-026 — Modo temporada (F7): ciclo post-jornada y fuente del calendario

- **Fecha**: 2026-07-13 (Fase 7)
- **Estado**: aceptada

## Contexto

SPEC §3.3 define el ciclo `alaves ingest --matchday`: tras cada jornada,
descargar resultados y estadísticas, validar, insertar, recalcular features,
evaluar las predicciones que ya tienen resultado, reentrenar y predecir la
siguiente jornada, y simular la clasificación. Falta decidir de dónde sale el
**calendario** de la temporada en curso.

## Opciones consideradas para el calendario

1. **API-Football (SPEC §3.1)**: da calendario oficial, alineaciones, lesiones
   y árbitro, pero requiere API key, tiene límite de 100 req/día en el tier
   gratuito y añade un adaptador nuevo con su autenticación.
2. **fixtures.csv de football-data.co.uk** (elegida): archivo único con los
   próximos partidos de todas las ligas (columna `Div`), sin API key, con el
   mismo formato y adaptador que ya usamos para los resultados.

## Decisión

- **Calendario desde football-data (`fixtures.csv`)**. Lo único que el sistema
  necesita del calendario es la lista de próximos encuentros (para predecir y
  simular); las alineaciones/lesiones/árbitro de API-Football **no están en el
  feature set v1** (SPEC §4.1 las dejaba condicionadas a "si API-Football lo
  permite"), así que su complejidad y su límite de peticiones no compensan hoy.
  Se mantiene la filosofía de "football-data como columna vertebral" (ADR-003).
  API-Football queda diferida a una posible v2 del feature set.
- **`ingest_fixtures`**: inserta los próximos partidos de la división como
  `status='scheduled'` con sus cuotas de apertura; nunca pisa un partido ya
  `finished` (el resultado manda sobre el calendario); los equipos del archivo
  sin alias en `config/teams.toml` se saltan con aviso (el archivo trae todas
  las ligas).
- **`ingest_matchday`**: refresca todo lo temporal de la temporada en curso
  —resultados (el CSV crece cada jornada), xG (FBref/Understat), calendario y
  Elo reciente (`force=True`)— y cada fuente que falle **degrada con aviso**,
  nunca aborta el ciclo entero (la BD manda, la red es el medio).
- **`evaluate_season`** (`evaluation/season.py`): cruza las predicciones
  persistidas con los resultados ya conocidos y da log-loss/Brier/RPS/acierto
  acumulados de la temporada. Es la auditoría honesta del rendimiento REAL
  (CLAUDE.md §5.5), distinta del backtest sobre el pasado: juzga lo que el
  modelo predijo *antes* de conocer el resultado. Si un partido tiene varias
  predicciones (reentrenos), se queda la más reciente.
- **Ciclo completo en el CLI** (`alaves ingest --matchday`): ingesta →
  evaluación de predicciones pasadas → reentrenamiento con registro →
  predicción de la próxima jornada (persistida) → simulación de la
  clasificación. Cada paso es robusto: si falta un prerrequisito (sin
  resultados nuevos, sin modelo, sin calendario) avisa y continúa.

## Consecuencias

- El sistema queda listo para operar en vivo durante la 2026-27 con un solo
  comando semanal, sin API keys.
- La ingesta en vivo depende de la red y de que football-data ya publique la
  temporada; se prueba con fixtures congelados (mini-liga) y se ejecuta de
  verdad en la máquina del usuario cuando arranque la temporada.
- Si en el futuro se quieren lesiones/alineaciones, se añadirá API-Football con
  su propio ADR y su bloque de features, sin tocar este ciclo.
