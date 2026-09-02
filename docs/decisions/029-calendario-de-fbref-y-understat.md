# ADR-029 — El calendario se obtiene solo: FBref + Understat, fusionados

Fecha: 2026-09-02 · Estado: aceptada · Fase: F7

## Contexto

Con la 2026-27 en marcha, el ciclo semanal ingería bien los resultados pero
dejaba la BD **sin un solo partido programado**, así que `predict --next` y
`simulate` no tenían nada que hacer. El calendario oficial de LaLiga existe y es
público desde antes de empezar la temporada; el problema era de dónde leerlo.

ADR-026 eligió el `fixtures.csv` de football-data. En la práctica ese archivo
**solo lista los encuentros inminentes** (una o dos jornadas), no la temporada:
en una ejecución real aportó 1 partido, y en otra 0 porque los que traía ya
constaban jugados. Con eso se puede predecir la próxima jornada a duras penas,
pero no se puede proyectar la clasificación final, que es medio proyecto.

El remedio previsto era sembrar `data/fixtures.csv` a mano. Funciona, pero es
trabajo manual recurrente para un dato que ya está publicado, y envejece.

Además había un fallo tapando la mejor fuente: **FBref respondía 404 para la
2026-27**. La causa no era un bloqueo anti-bot sino la URL: `schedule_url()`
construye siempre la versionada
(`/comps/12/2026-2027/schedule/2026-2027-La-Liga-...`), y **FBref solo versiona
las temporadas ya cerradas** — la vigente vive en la URL sin año
(`/comps/12/schedule/La-Liga-Scores-and-Fixtures`). Por eso el mensaje decía
"la URL ya no existe" y la Wayback tampoco tenía nada que archivar.

## Opciones consideradas

**A. Seguir sembrando `data/fixtures.csv` a mano.** Cero código, pero deja el
sistema dependiendo de que alguien copie 380 filas y las mantenga cuando LaLiga
mueve horarios. Es justo el trabajo que el proyecto debería quitar.

**B. Añadir API-Football.** Tiene el calendario completo y estructurado, pero
exige API key y su plan gratuito son 100 peticiones/día. ADR-026 ya lo descartó
por eso, y nada ha cambiado.

**C. Una sola fuente nueva.** Ninguna las tiene todas: football-data tiene las
**cuotas** (que alimentan la variante `con_cuotas`) pero pocos partidos; FBref
tiene la **jornada oficial** pero bloquea bots a ratos; Understat tiene la
**temporada entera** pero ni cuotas ni jornada.

**D. (elegida) Fusionar los tres, campo a campo.** Cada uno aporta lo que tiene
y ninguno es imprescindible.

## Decisión

`ingest_fixtures` deja de ser "el primero que responda" y pasa a **fusionar**.
Los orígenes se procesan en orden de preferencia y cada uno solo rellena lo que
el anterior dejó vacío (la precedencia se lee de arriba abajo en el código):

| Orden | Origen | Qué aporta que los demás no |
|-------|--------|------------------------------|
| 1 | football-data `fixtures.csv` | **cuotas de apertura** |
| 2 | FBref *Scores & Fixtures* | **jornada oficial (Wk)** |
| 3 | Understat `getLeagueData` | **la temporada entera** |
| 4 | `data/fixtures.csv` local | último recurso manual |

La fecha la fija el primer origen que traiga el partido; la jornada y las
cuotas, el primero que las tenga. Un partido que solo conoce Understat entra
igual; uno que solo conoce football-data entra con sus cuotas.

Piezas nuevas:

- `fbref.current_schedule_url()` y una cascada de URLs candidatas: para la
  temporada en curso se prueba **primero la URL sin año** y después la
  versionada (porque `current_season` es config, no una verdad de FBref). Esto
  arregla el 404 y, de paso, devuelve la jornada oficial también para los
  partidos ya jugados.
- `fbref.parse_fixtures()` y `understat.parse_league_fixtures()`: la
  contrapartida de los parsers existentes, quedándose con lo **no jugado**. La
  misma página y la misma llamada que ya se hacían para el xG traen el
  calendario, así que no cuestan peticiones nuevas — `ingest_fixtures` lee la
  cache que el paso de xG acaba de refrescar.
- `assign_scheduled_matchdays` **no deduce si no hace falta**: si todos los
  programados traen jornada oficial, no toca nada. La deducción por proximidad
  de fechas falla justo donde más duele (jornadas entre semana, aplazamientos),
  así que el dato oficial siempre gana.

## Consecuencias

- El calendario se obtiene solo. `data/fixtures.csv` queda como red de
  seguridad, no como requisito.
- Tres fuentes independientes para el mismo dato: que FBref bloquee bots o que
  football-data no publique deja de dejar al sistema sin nada que predecir.
- Con FBref accesible, las jornadas dejan de ser una deducción: la proyección
  Monte Carlo agrupa por la jornada real, no por fechas cercanas.
- El coste: `ingest_fixtures` es más largo y hace más peticiones (aunque
  reutiliza cache). A cambio deja de haber un paso manual en la rutina semanal.
- Understat sube de categoría: era "relleno de xG" y ahora es también la única
  fuente que garantiza los 380 partidos. Si cambiara su endpoint interno (ya
  pasó una vez, ADR-008/011), quedarían FBref y la siembra local.
