# ADR-004 — Cache local de descargas y rate limiting

- **Fecha**: 2026-07-07 (Fase 1)
- **Estado**: aceptada

## Contexto

CLAUDE.md §6 obliga a respetar rate limits y a no re-descargar lo ya guardado.
Hay que decidir el mecanismo concreto.

## Opciones consideradas

1. Librería de cache HTTP (`requests-cache`, `hishel`).
2. Cache manual de archivos en `data/raw/<fuente>/` + rate limit por host.

## Decisión

Opción 2: `etl/http_cache.py` implementa `fetch_text(url, cache_path, ...)`:

- Si el archivo de cache existe → se lee del disco, sin tocar la red.
- Si no → espera lo que falte del rate limit del host (registro del último
  acceso por host en memoria de proceso), descarga con `httpx` (timeout 30 s,
  User-Agent identificable), valida que la respuesta no esté vacía y persiste.
- `--force` en el CLI re-descarga ignorando la cache (p. ej. cuando
  football-data corrige un dato histórico).

Motivo frente a la opción 1: una dependencia menos, control total y
transparente de dónde queda cada archivo (auditable a mano, CLAUDE.md
"nada de magia"), y el patrón de acceso es trivial (GETs idempotentes de
archivos estáticos). Los datos históricos no cambian: la cache no necesita
expiración.

Rate limits configurados en `config/settings.toml`: football-data 1 s
(descarga directa de CSV), Understat 3 s, ClubElo 2 s.

## Consecuencias

- Re-ejecutar la ingesta completa tras la primera vez cuesta ~0 peticiones.
- Los datos crudos quedan en `data/raw/` (gitignored) como copia exacta de lo
  descargado: reproducibilidad y debugging de parsers sin red.
- El rate limit es por proceso; si algún día hay paralelismo habrá que
  revisarlo (no aplica en F1).

## Actualización (2026-07-07, tras la primera ingesta real)

Dos ajustes surgidos del primer contacto con las fuentes reales:

1. **Cabeceras por fuente**: el User-Agent identificable
   (`alaves-predictor/0.1`) funciona con football-data y ClubElo, pero
   Understat sirve una página sin datos y FBref devuelve 403 a clientes que
   no parecen navegador. `fetch_text` acepta `headers` opcionales y
   `http_cache.BROWSER_HEADERS` (compartidas) se usan para esas dos fuentes
   (los datos son públicos y el acceso es mínimo: 1 página por temporada,
   cacheada, mismo patrón que las librerías públicas de scraping).
2. **Auto-recuperación de cache envenenada**: si un HTML cacheado no parsea
   (p. ej. una página de bloqueo guardada por una ejecución anterior), la
   ingesta lo re-descarga UNA vez antes de fallar, en lugar de exigir borrado
   manual de `data/raw/`.
3. **Errores de descarga limpios**: los fallos HTTP (403/404/429, timeouts)
   se convierten en `SourceDownloadError` con la URL y una pista de arreglo,
   nunca un traceback crudo (CLAUDE.md §6).
