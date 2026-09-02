# ADR-028 — La ingesta semanal falla a la cara, no en silencio

Fecha: 2026-09-02 · Estado: aceptada · Fase: F7

## Contexto

La primera ejecución real de `alaves ingest --matchday` con la 2026-27 en marcha
devolvió esto:

```
  2026-27: 0 jugados (0 con xG), 0 programados.
  AVISO: football-data aún no publica resultados de 2026-27 (La fuente
  'football_data' usa el nombre de equipo 'Malaga', no registrado(s) ...)
```

Dos problemas distintos, y el mensaje no dejaba claro ninguno:

1. **El aviso mentía sobre la causa.** football-data *sí* publicaba los
   resultados; lo que pasaba es que el CSV traía un equipo (`Malaga`, ascendido)
   sin alias en `config/teams.toml`, y eso abortaba la ingesta **entera** de la
   temporada: ni resultados, ni xG, ni cuotas. El texto "aún no publica
   resultados" apuntaba al sitio equivocado.
2. **`ingest_football_data_season` resolvía los nombres dentro del bucle**, así
   que abortaba en el **primer** desconocido. Con varios ascendidos, el usuario
   tenía que arreglar un alias, relanzar la ingesta (varios minutos de red),
   descubrir el siguiente, y así uno a uno.
3. **El calendario devolvió `0 programados` sin un solo aviso.** `ingest_fixtures`
   podía no encontrar nada —remoto sin partidos de la división, archivo local
   ausente o con otro formato— y devolver `0` como si fuera un resultado normal.
   Sin calendario no hay próxima jornada que predecir, así que ese `0` es
   justo lo que hay que explicar.

## Opciones consideradas

**A. Registrar los equipos que faltan y ya.** Arregla hoy y vuelve a romper con
el próximo ascenso. Descartada: el ascenso de equipos es un evento anual
garantizado, no una anomalía.

**B. Aceptar nombres desconocidos creando un equipo sobre la marcha.** Rompe
ADR-005 (alias explícitos, fallo ruidoso): un `team_id` inventado por el
pipeline se cuela en la BD y ensucia el histórico sin que nadie lo revise.

**C. (elegida) Mantener el fallo ruidoso, pero que sea útil a la primera.**
Informar de todos los nombres desconocidos de golpe, decir con precisión qué se
ha dejado de ingerir, y no devolver nunca un calendario vacío sin motivo.

## Decisión

- **`ingest_football_data_season` recoge todos los nombres desconocidos antes de
  insertar nada** y lanza un único `UnknownTeamError` con la lista completa. Es
  el patrón que ya usaba el adaptador de FBref; football-data era la excepción.
  Además garantiza que no entren partidos a medias antes de abortar.
- **El aviso del ciclo distingue la causa**: si el fallo es `UnknownTeamError`,
  el mensaje dice que **no se ha ingerido ningún resultado** de la temporada y
  que el arreglo está en `config/teams.toml`. El "aún no publica resultados" se
  reserva para las descargas que de verdad fallan.
- **`ingest_fixtures` devuelve un `FixturesReport`** (en vez de una tupla) con
  lo insertado, los equipos sin alias, cuántos encuentros aportó cada origen,
  cuántos se saltaron por estar ya jugados, y los motivos por los que un origen
  no aportó nada. El ciclo avisa cuando no hay calendario **y explica por qué**,
  distinguiendo "no hay de dónde tirar" de "ya está todo jugado" (que no es un
  problema y no se avisa).
- **`config/teams.toml` incorpora al Málaga CF**, ascendido a la 2026-27.

## Consecuencias

- Un ascenso múltiple se resuelve en una sola pasada: la ingesta dice de una vez
  todos los alias que faltan.
- El coste de mantenimiento anual sigue existiendo (hay que añadir a mano los
  ascendidos), pero es explícito y está donde debe: en config, revisado por una
  persona. Es el precio de ADR-005 y se paga a gusto.
- `0 programados` deja de ser un resultado ambiguo. Si el sistema no puede
  predecir la próxima jornada, la salida dice por qué.
