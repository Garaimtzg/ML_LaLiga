# ADR-031 — La próxima jornada sobrevive a los aplazados, y cada predicción guarda su huella

Fecha: 2026-11-05 · Estado: aceptada · Fase: F7 (operación)

## Contexto

Se pidió comprobar el proyecto entero "con los datos que hay". El entorno
remoto de desarrollo no llega a ninguna fuente (ADR-007: football-data, FBref
y Understat responden 403 desde el proxy, y ClubElo solo sirve HTTP), así que la
comprobación se hizo con una **liga sintética servida en el formato exacto de
cada fuente**: los 20 equipos reales de cada temporada 2018-19 → 2026-27,
~3.400 partidos generados con un Poisson, y rarezas reales de las fuentes
(cabeceras repetidas en FBref, FBref sin xG desde 2024-25, BOM en el
`fixtures.csv`, un partido de la 2026-27 aplazado). Solo se sustituyó el
transporte HTTP; cache, parsers, BD, features, modelos, simulación y dashboard
fueron el código de producción. Todo el CLI (`ingest --historical`,
`validate`, `status`, `features`, `baselines`, `train`, `backtest`,
`predict`, `simulate`, `report --importance`, `ingest --matchday` tres
semanas seguidas) y las seis páginas del dashboard funcionaron, salvo estos
fallos, que los fixtures de 12 partidos no podían destapar:

1. **Un aplazado secuestraba "la próxima jornada".** El ciclo semanal, `alaves
   predict --next` y `alaves status` tomaban como próxima la jornada de número
   más bajo con algún partido pendiente. Un aplazado conserva su jornada
   original (FBref publica la Wk de siempre), así que con un partido de la J9
   movido a diciembre la "próxima" era la J9: se predecía **solo ese partido** y
   la J13, la que venía de verdad, se quedaba sin predicción guardada. En LaLiga
   pasa todas las temporadas: la Supercopa de enero aplaza partidos de cuatro
   equipos. Lo mismo ocurría con un partido ya jugado cuyo resultado aún no ha
   publicado football-data: se predecía después de jugarse y `alaves validate`
   fallaba ("predicción posterior al partido").
2. **El dashboard abría "Próxima jornada" en la J1**, ya jugada hace meses
   (`index=0` del selector).
3. **La proyección de la temporada predecía media liga a ciegas.** Las medias
   móviles de forma contaban los partidos programados como huecos: a 6 partidos
   vista la ventana de 5 ya no tenía ningún dato. En la proyección a 5-nov, desde
   la J18 el 45 % de las features de forma iban vacías y desde la J23 el 80 %
   (en el histórico, el 0,2 %). El LightGBM las mandaba por la rama de "dato
   ausente" que aprendió de los primeros partidos de 2018 y el lineal las
   imputaba con la mediana (equipos "iguales"). Las rachas ya resolvían esto
   heredando la última conocida; las medias no. En la prueba sintética el
   efecto era de 1-2,4 puntos de probabilidad por partido y ~0,5 puestos en la
   posición esperada del Alavés.
4. **La predicción no guardaba con qué features se hizo** (CLAUDE.md §5.5 pide
   "hash de features") y `alaves predict` **sobrescribía el Parquet** del
   feature store completo con las diez filas de la jornada predicha.
5. **La evaluación de la temporada contaba predicciones tardías**: tomaba la
   última de cada partido aunque se hubiera guardado después de jugarse.
6. **El parser de FBref no saltaba las cabeceras repetidas** (`tr.thead`) que su
   comentario decía saltar: una sola en el tbody tumbaba la ingesta entera por
   el marcador "Score".
7. `uv run mypy src` fallaba: `python_version = "3.11"` fijo en la config de mypy
   choca con los stubs de numpy 2.5 / scipy 1.18 (solo existen para 3.12+, y uv
   los instala ahí).

## Opciones consideradas (próxima jornada)

1. **Jornada mínima con algo pendiente** (la de antes): rota por los aplazados.
2. **La jornada del partido con fecha más cercana**: arregla el caso de un
   aplazado lejano, pero la semana en que el aplazado se juega (miércoles) se
   predeciría solo ese y no la jornada del fin de semana.
3. **La jornada más baja con al menos la mitad de sus partidos por jugar, más
   los pendientes anteriores que se jueguen antes de que acabe** (elegida).

## Decisión

Nuevo módulo `models/predict.py`, único dueño de qué se predice y cómo se
guarda; lo usan `alaves predict`, el ciclo semanal, `alaves status` y el
dashboard (antes cada uno tenía su versión, y la del CLI estaba duplicada):

- **`next_round`**: solo cuentan los programados con fecha de hoy (UTC) en
  adelante; la próxima jornada es la de número más bajo con al menos la mitad
  de sus partidos por jugar (una jornada empezada sigue siendo la próxima); los
  pendientes de jornadas anteriores entran si se juegan antes de que termine.
  Un aplazado a un mes vista se predice cuando le toca. `status` avisa si hay
  programados con fecha pasada.
- **`persist_predictions`**: guarda el snapshot de features en la tabla
  `features` (no en el Parquet) y una fila por predicción con
  `features_hash` (sha256 del vector de features, 16 hex). La columna se añade
  sola a las BDs existentes (`init_schema` hace un `ALTER TABLE` idempotente).
- **`evaluate_season`** solo cuenta predicciones con fecha no posterior al
  partido (el mismo criterio que `alaves validate`).
- **Forma**: la media móvil se calcula sobre los partidos con dato y un partido
  programado hereda la última media conocida (`features/form.py`). Con todo
  jugado el resultado es idéntico al bit al de antes (verificado sobre el
  histórico completo), así que el feature set sigue siendo `v1` y los modelos
  entrenados siguen valiendo.
- El dashboard abre "Próxima jornada" en la próxima de verdad.
- FBref: las filas `thead` se saltan en los dos parsers.
- mypy analiza con la versión del intérprete del entorno.
- La liga sintética y el recorrido completo quedan como test
  (`tests/e2e/`, ~20 s, dentro de `uv run pytest`; `-m "not e2e"` lo salta).

## Consecuencias

- Las predicciones guardadas antes de este cambio no tienen `features_hash`
  (NULL); las nuevas sí.
- El modo demo de `simulate` (temporadas cerradas) no cambia: ahí todo está
  jugado, y su limitación conocida sigue siendo la de ADR-023.
- El test punta a punta comprueba que el sistema encaja, no la calidad del
  modelo: sus métricas son de datos sintéticos y no dicen nada del rendimiento
  real. Lo real sigue saliendo de la ingesta en local (ADR-007).
