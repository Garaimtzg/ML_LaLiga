# ADR-005 — Mapeo de nombres de equipos entre fuentes

- **Fecha**: 2026-07-07 (Fase 1)
- **Estado**: aceptada

## Contexto

Cada fuente escribe los equipos a su manera: "Alaves" (football-data),
"Ath Bilbao" vs "Athletic Club", "Sociedad" vs "Real Sociedad"... Cruzar
fuentes exige un id canónico común. SPEC §3.2 ya prevé la tabla
`teams(team_id, name, aliases_json)`.

## Opciones consideradas

1. **Matching difuso** (normalizar + distancia de edición) en tiempo de ingesta.
2. **Tabla explícita de alias** en `config/teams.toml`, con fallo ruidoso ante
   nombres desconocidos.

## Decisión

Opción 2. El matching difuso es cómodo hasta que falla en silencio (¿"Real
Madrid" ↔ "Real Sociedad" al 60 %?); con ~28 clubes en 8 temporadas, mantener
los alias a mano cuesta minutos y es 100 % auditable — coherente con el
principio rector ("el usuario entiende y controla todo").

- `config/teams.toml`: una entrada por club con `team_id` canónico (slug),
  nombre para mostrar y alias exactos por fuente (`football_data`, `understat`,
  `clubelo` — este último es el componente de URL de la API).
- `TeamRegistry.resolve(fuente, nombre)` devuelve el id canónico o lanza
  `UnknownTeamError` con un mensaje que dice exactamente qué alias añadir y
  dónde. Nunca se inserta un equipo "adivinado".
- La tabla `teams` de la BD se siembra desde este archivo en cada ingesta.
- Los alias de ClubElo son los de mejor esfuerzo (la API no publica un listado
  oficial de nombres); si alguno falla, el error del adaptador (`clubelo.py`)
  remite a este archivo para corregirlo. Un test verifica que no haya alias
  duplicados por fuente.

## Consecuencias

- Incorporar un club nuevo (ascendido en 2026-27) = añadir una entrada al TOML.
- Un cambio de nomenclatura en una fuente rompe la ingesta de forma visible e
  inmediata, con instrucciones de arreglo en el propio error (fail-fast).
