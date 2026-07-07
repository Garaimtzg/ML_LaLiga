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
