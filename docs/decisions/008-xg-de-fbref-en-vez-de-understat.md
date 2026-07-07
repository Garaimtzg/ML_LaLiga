# ADR-008 — El xG histórico se obtiene de FBref; Understat queda en pausa

- **Fecha**: 2026-07-07 (Fase 1, tras la primera ingesta real)
- **Estado**: aceptada
- **Modifica**: ADR-003 (fuentes de F1), ADR-006 (jornada aproximada)

## Contexto

Al ejecutar la ingesta real, la página de liga de Understat resultó no
contener ya el JSON embebido (`datesData`) del que dependía el adaptador:
**Understat rediseñó su web en diciembre de 2025** y ahora carga los datos
por JavaScript en el cliente. La verificación fue triple: (1) la página
descargada por el usuario es la real (título correcto) pero pesa 18 KB y no
contiene ningún `JSON.parse`; (2) el issue #71 de ScraperFC (dic-2025)
reporta exactamente esta rotura; (3) el código fuente de las librerías
`understatapi` 0.7.1 y `ScraperFC` 4.5.0 (descargado de PyPI) sigue buscando
`datesData`, es decir, todo el ecosistema de scraping de Understat está roto.

## Opciones consideradas

1. **Ingeniería inversa del nuevo Understat**: descubrir sus endpoints
   internos de datos. Frágil (acaban de demostrar que cambian el sitio),
   efímero y costoso de mantener.
2. **Automatizar un navegador** (Playwright) para ejecutar su JavaScript.
   Dependencia estructural pesadísima para un solo dato.
3. **Obtener el xG de FBref**: la página "Scores & Fixtures" de cada
   temporada lista los 380 partidos con fecha, marcador, xG de ambos equipos
   y **jornada oficial** (columna Wk). Una página por temporada.

## Decisión

Opción 3. SPEC §3.1 ya designaba FBref como fuente principal de estadísticas
avanzadas y Understat como "alternativa/contraste"; este ADR solo adelanta a
F1 el papel de FBref como fuente de xG. Detalles:

- Nuevo adaptador `etl/sources/fbref.py` que parsea la tabla de calendario
  por atributos `data-stat` (la interfaz más estable de FBref). Rate limit
  conservador de 6 s/petición (FBref pide moderación a los bots) — con cache,
  el histórico completo son 8 peticiones una sola vez.
- **Bonus**: la columna Wk da la jornada oficial, que sobreescribe la
  aproximación por conteo del ADR-006 (que queda como base/fallback).
- Dependencia nueva: `beautifulsoup4` (ya prevista en el stack de CLAUDE.md
  §2); parsear HTML real con regex sería frágil.
- El adaptador de Understat **queda en pausa**, no se borra: su parser del
  formato antiguo sigue testeado, y en F2 se evaluará si el nuevo Understat
  merece ingeniería inversa como fuente de contraste de xG (SPEC §3.1) o si
  se elimina definitivamente (requeriría actualizar SPEC).
- En F2, cuando se scrapee FBref a nivel de partido para el bloque
  técnico-táctico, este mismo adaptador se ampliará.

## Consecuencias

- El pipeline de F1 vuelve a poder poblar el xG histórico completo.
- El xG de FBref proviene del proveedor Opta (Understat usaba modelo propio);
  para el modelo es indiferente mientras la fuente sea consistente en todo el
  histórico, que lo es (FBref tiene xG de La Liga desde 2017-18).
- `matches.matchday` pasa a ser la jornada oficial de LaLiga, no una
  aproximación (mejora para las features de F2).
- Riesgo: FBref también protege su web (Cloudflare); si el User-Agent
  identificable resultara bloqueado en local, se aplicaría el mismo ajuste de
  cabeceras que en ADR-004-actualización.
