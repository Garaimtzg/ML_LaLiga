# ADR-003 — Fuentes de datos de la Fase 1 (y cuáles se difieren)

- **Fecha**: 2026-07-07 (Fase 1)
- **Estado**: aceptada

## Contexto

SPEC §3.1 lista seis fuentes. La F1 exige "BD poblada y validada" con las
temporadas 2018-19 → 2025-26, pero no todas las fuentes aportan a ese
entregable con el mismo coste/beneficio.

## Opciones consideradas

1. Implementar los seis adaptadores en F1.
2. Implementar en F1 solo las fuentes que pueblan el histórico core y diferir
   el resto a la fase que las consume.

## Decisión

Opción 2. En F1 entran:

- **football-data.co.uk** (columna vertebral): resultados, estadísticas básicas
  (tiros, córners, faltas, tarjetas) y cuotas de apertura y cierre de bet365,
  Pinnacle, máximo y media de mercado. Un CSV por temporada, formato estable.
- **Understat**: xG por partido de ambos equipos, parseando el JSON
  `matchesData` embebido en la página de liga (1 página por temporada, mucho
  más eficiente y respetuoso que scrapear partido a partido). El bloque
  `teamsData` de la misma página (npxG, PPDA...) se incorporará en F2, cuando
  las features lo consuman.
- **ClubElo**: histórico Elo completo por club vía API CSV (1 petición por
  club, 28 en total).

Se difieren:

- **FBref** (stats técnico-tácticas detalladas) → F2. Es el scraping más
  pesado (~1 req/6s, decenas de páginas por temporada) y sus columnas solo las
  consumen las features del bloque técnico-táctico (SPEC §4.1), que se
  seleccionan por forward selection en F2/F3. La tabla `match_stats` ya tiene
  todas sus columnas creadas y el xG core está cubierto por Understat.
- **Transfermarkt** (valores de plantilla) → F2, con las features de "estatus".
- **API-Football** (calendario 2026-27, lesiones, alineaciones) → F7; requiere
  API key y su valor es para la temporada en curso, no para el histórico.

## Consecuencias

- La F1 puebla `matches`, `match_stats` (parcial: básicas + xG), `odds` y `elo`.
- Los baselines de F2 (frecuencias, Elo, cuotas) ya tienen todos sus datos.
- Riesgo aceptado: si FBref cambia de formato antes de F2, se descubrirá
  entonces (mitigado porque el diseño de adaptadores + fixtures es uniforme).
- La cobertura de xG se valida por temporada (`alaves validate`); si Understat
  no cubriera algún partido, la validación lo señala en vez de ocultarlo.
