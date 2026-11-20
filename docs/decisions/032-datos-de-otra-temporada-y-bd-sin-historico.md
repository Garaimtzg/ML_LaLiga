# ADR-032 — Ningún dato de otra temporada entra en la BD, y una BD sin histórico lo dice

Fecha: 2026-11-20 · Estado: aceptada · Fase: F7 (operación)

## Contexto

El ciclo semanal (`alaves ingest --matchday`) acabó así en una ejecución real:

```
2026-27: 119 jugados (0 con xG), 161 programados.
AVISO: Understat no disponible para 2026-27: getLeagueData de Understat no devolvió
       JSON válido: Expecting value: line 1 column 1 (char 0)
AVISO: FBref no disponible para 2026-27 y SÍ hace falta (...): FBref tiene el partido
       girona vs rayo-vallecano (2025-08-15, temporada 2026-27) pero football-data no.
AVISO: ClubElo no responde para los 31 equipos (...). Motivo: HTTP 502 (...)
       ATENCIÓN: alaves, almeria, athletic-club, ... (31 nombres) no tiene(n) NINGÚN Elo
MergeError: incompatible merge keys [0] <StringDtype(na_value=nan)> and dtype('O')
```

Detrás hay cuatro problemas distintos:

1. **FBref sirvió la página de la temporada anterior.** La temporada en curso
   vive en la URL sin año (ADR-029) y, con FBref bloqueado, la cascada pide a la
   Wayback Machine capturas de esa URL (ADR-030): las que hay son de cuando esa
   URL mostraba la 2025-26 (Girona–Rayo, 15-ago-2025, fue su primer partido). Se
   aceptaba la más reciente sin mirar las fechas. El cruce con football-data
   abortó el xG, pero el calendario reutiliza esa misma página desde la cache:
   sus partidos aún por jugar (febrero–mayo de 2026) entraron como
   **programados de la 2026-27**. Eso explica los 161 programados con Understat
   caído (football-data solo publica los inminentes). Si se quedan, la proyección los
   simula como partidos reales, y su jornada podía pegarse a partidos de verdad.
2. **`build_features` revienta sin ningún Elo de ClubElo.** Con la tabla `elo`
   vacía, `merge_asof` recibe un `team_id` de tipo `object` frente a ids `str`
   (pandas 3) y aborta el ciclo entero. Pasa siempre que ClubElo está caído en
   una BD que nunca tuvo Elo.
3. **La BD no tenía histórico** —ni un rating Elo, ni siquiera el interno— y
   nada lo decía: el ciclo semanal solo trae la temporada en curso, y sin el
   histórico no hay nada con qué entrenar. Los avisos de las fuentes tapaban la
   causa de fondo.
4. **Los avisos no eran accionables**: el de Understat no decía qué había
   respondido ("Expecting value" vale igual para una página de bloqueo que para
   un error del servidor) y el de ClubElo listaba 31 nombres para decir "todos".

## Opciones consideradas (datos de otra temporada)

1. **Fiarse de la URL**: es lo que fallaba; una URL sin año no garantiza año.
2. **Comprobar los equipos** (los de la página = los de la temporada): no sirve
   entre temporadas consecutivas, que comparten 17 de 20 equipos.
3. **Comprobar las fechas contra una ventana de temporada** (elegida): una
   temporada "AAAA-BB" solo puede tener partidos entre el 1-jul de AAAA y el
   31-jul del año siguiente (la 2019-20 acabó el 19-jul-2020). Es barato, no
   tiene falsos positivos y vale para cualquier fuente.

## Decisión

- **Ventana de temporada en `config/settings.toml`** (`[league]
  season_start = "07-01"`, `season_end = "07-31"`; `LeagueConfig.season_bounds`).
- **FBref**: en toda la cascada (cache, directo, Wayback, página guardada a mano)
  una página con algún partido fuera de la ventana se rechaza como "de otra
  temporada" y se prueba el siguiente candidato. Lo mismo para el calendario
  que se lee de esa página.
- **Calendario**: cualquier partido de cualquier origen con fecha fuera de la
  ventana se descarta antes de fusionarse, con un aviso que dice cuántos y de
  dónde.
- **Limpieza**: al empezar, `ingest --matchday` borra los programados de la
  temporada en curso con fecha fuera de la ventana (y lo que cuelga de ellos),
  y lo avisa. Arregla las BDs a las que ya se les coló una página vieja. Nunca
  toca un partido jugado.
- **`alaves validate`** comprueba que ningún partido de la temporada en curso
  tenga fecha de otra.
- **Sin Elo**: las features de Elo quedan vacías y el resto sigue (el LightGBM y
  el lineal ya toleran huecos).
- **Sin histórico**: el ciclo semanal avisa al empezar, con la ruta de la BD,
  de que hay que ejecutar `alaves ingest --historical`.
- **Avisos**: Understat dice qué tipo de respuesta llegó y cómo empieza;
  ClubElo dice "ningún equipo tiene Elo" en vez de listar 31.

## Consecuencias

- Una fuente que sirva otra temporada degrada como una caída (ADR-030), pero
  sin meter nada en la BD.
- La ventana es configurable: si LaLiga cambiara su calendario (una temporada
  que acabe en agosto), se ajusta en `settings.toml`.
- Los tests de calendario usan ahora la 2026-27 como temporada en curso, que es
  la de sus ficheros de prueba: antes declaraban "2019-20" con calendarios de
  2026.
- `tests/e2e` reproduce esta ejecución (BD sin histórico, FBref de otra
  temporada, Understat con HTML, ClubElo con 502) y falla con el código
  anterior.
