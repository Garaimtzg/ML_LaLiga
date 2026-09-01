# ADR-027 — Tratamiento de la temporada en curso cuando está a medias

Fecha: 2026-09-01 · Estado: aceptada · Fase: F7

## Contexto

La 2026-27 ya ha empezado: el Alavés lleva 3 jornadas jugadas y quedan 35. Es
la primera vez que la base de datos contiene una temporada **empezada pero sin
terminar**, y eso rompe supuestos que hasta ahora nunca habían fallado porque
todas las temporadas de la BD estaban cerradas:

1. **`train_models` validaba sobre la última temporada del pool walk-forward.**
   Con la 2026-27 dentro, esa "última temporada" pasaba a tener 30 partidos en
   vez de 380. Las métricas de validación se volvían ruido, y como de ese
   log-loss depende la **regla anti-sorpresa del registry** (SPEC §6.4: no se
   promociona una versión que empeore >10 %), la promoción de modelos habría
   quedado a merced del azar de tres jornadas.
2. **La elección de ξ (Dixon-Coles) y C (lineal) promediaba el log-loss por
   temporada.** Media de medias: la temporada de 30 partidos habría votado
   tanto como una de 380.
3. **`run_backtest` testea las 3 últimas temporadas.** Con la 2026-27 dentro,
   una de esas tres habría sido de 30 partidos, ensuciando las medias con las
   que se comprueban los criterios de aceptación de SPEC §12.1.
4. **`validate` solo recorría `historical_seasons`**, así que la temporada en
   curso no se validaba en absoluto: ni sus goles, ni su xG, ni sus jornadas.
   Aplicarle los chequeos de siempre tampoco valía: exigen 380 partidos y 38
   por equipo, que una temporada a medias nunca cumple.
5. **El calendario local (`data/fixtures.csv`) pisaba al remoto.** Se sembró a
   mano cuando football-data aún no publicaba la 2026-27; sus fechas envejecen
   en cuanto sí la publica, y una fecha vieja cambia la jornada asignada al
   agrupar por proximidad de fechas.
6. **La ficha "El Alavés en detalle" dibujaba las 38 jornadas**, incluidas las
   no jugadas: líneas de Elo y de forma que se prolongaban hacia el futuro
   como si se supiera algo de él.

## Opciones consideradas

**A. No tocar nada y esperar.** Los problemas se diluyen solos según avanza la
temporada (en mayo la 2026-27 tendrá 380 partidos). Descartada: durante los
ocho meses intermedios el sistema mide mal y promociona modelos a ciegas —
justo la temporada para la que se construyó.

**B. Excluir la temporada en curso de todo.** Simple, pero tira a la basura los
datos más recientes y más relevantes: los partidos de esta misma temporada son
los que mejor describen a las plantillas de esta misma temporada.

**C. (elegida) Entrena, pero no juzga.** La temporada a medias entra al
entrenamiento y al pool de calibración como cualquier otra, pero nunca se usa
como temporada de validación ni de test, y la BD se valida con reglas propias.

## Decisión

Un único concepto compartido, `features.build.in_progress_seasons(features,
settings)`, que detecta las temporadas empezadas y sin terminar por dos vías
(el frame trae partidos sin jugar, o la temporada actual tiene menos partidos
jugados que una entera). Con él:

- **`train_models`** valida sobre la última temporada **completa**, y toma como
  `prior` solo los folds anteriores a ella (nunca los posteriores, que
  contaminarían la validación). Lo jugado de la temporada en curso sí entrena
  los modelos finales y sí entra al pool de calibración y de pesos. La lista
  de temporadas a medias queda escrita en `metrics.json`.
- **`choose_xi` / `choose_c`** miden el log-loss sobre todos los partidos del
  pool a la vez, no como media de medias. Con temporadas del mismo tamaño la
  cuenta es idéntica a la anterior, así que no altera nada de lo ya validado.
- **`run_backtest`** excluye las temporadas a medias de sus temporadas de test.
- **`validate`** añade un bloque propio para la temporada en curso: partidos
  jugados + programados dentro del total de la liga, 20 equipos, goles sin
  nulos, jornada asignada a **todos** los partidos (también a los programados,
  o `predict --next` no sabría cuál es la próxima), jornadas jugadas
  correlativas sin huecos, xG y cuotas de lo jugado, y — nuevo — que ninguna
  predicción guardada tenga fecha posterior a su partido (CLAUDE.md §5.5).
- **`ingest_fixtures`**: si un encuentro aparece en el calendario remoto y en
  la siembra local, manda el **remoto**; el local solo rellena lo que el remoto
  no trae.
- **`focus_timeline`** devuelve solo partidos jugados, y añade resultado (V/E/D),
  goles y puntos acumulados.
- **`alaves status`** separa jugados de programados y resume la temporada en
  curso (jornadas jugadas, próxima jornada, predicciones guardadas).

## Consecuencias

- Las métricas de validación siguen midiéndose sobre 380 partidos durante toda
  la temporada, así que la regla anti-sorpresa del registry sigue teniendo
  sentido en cada reentrenamiento semanal.
- El modelo aprende de las jornadas nuevas desde la primera: no hay que esperar
  a que acabe la temporada para que los resultados de 2026-27 cuenten.
- El coste: la validación se mide sobre una temporada cada vez más lejana en el
  tiempo (la 2025-26 durante todo el curso). Es el precio de tener una métrica
  estable; la foto del rendimiento *real* de esta temporada la da
  `evaluate_season`, que cruza las predicciones persistidas con los resultados
  ya conocidos y se ve en la página "Rendimiento del modelo".
- Cuando la 2026-27 termine, dejará de detectarse como "en curso" y volverá a
  ser una temporada normal sin cambiar una línea de código.
