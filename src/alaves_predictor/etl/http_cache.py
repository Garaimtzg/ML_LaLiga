"""Descarga HTTP con cache local en disco y rate limit (ADR-004).

Reglas de datos (CLAUDE.md §6): respetar rate limits y no re-descargar lo ya
guardado. Cada respuesta se guarda en data/raw/<fuente>/<archivo>; si el
archivo existe, se reutiliza sin tocar la red (salvo force=True).
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from alaves_predictor.etl.errors import SourceDownloadError, SourceFormatError

# User-Agent identificable y honesto: proyecto personal, no un bot anónimo.
_HEADERS = {"User-Agent": "alaves-predictor/0.1 (proyecto educativo personal; contacto via GitHub)"}

# Cabeceras de navegador para las fuentes que rechazan clientes no-navegador
# (Understat y FBref devuelven 403/página vacía al UA identificable; ADR-004).
# Los datos son públicos y el acceso mínimo: 1 página por temporada, cacheada.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

# Instante de la última descarga real por host, para aplicar rate limit global
# del proceso aunque se llame desde bucles distintos.
_last_request_at: dict[str, float] = {}


def fetch_text(
    url: str,
    cache_path: Path,
    *,
    rate_limit_seconds: float = 1.0,
    force: bool = False,
    encoding: str | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    """Devuelve el cuerpo de `url` como texto, usando cache local.

    - Si `cache_path` existe y force=False: lee del disco, sin petición HTTP.
    - Si descarga: espera lo que falte del rate limit del host, valida que la
      respuesta no esté vacía y la persiste en `cache_path`.
    - `headers` permite a un adaptador sobreescribir las cabeceras por defecto
      (p. ej. Understat exige parecer un navegador; ver understat.py).
    """
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding=encoding or "utf-8")

    host = httpx.URL(url).host
    elapsed = time.monotonic() - _last_request_at.get(host, 0.0)
    if elapsed < rate_limit_seconds:
        time.sleep(rate_limit_seconds - elapsed)

    try:
        response = httpx.get(url, headers=headers or _HEADERS, timeout=30.0, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise SourceDownloadError(f"Fallo de red descargando {url}: {exc}") from exc
    _last_request_at[host] = time.monotonic()
    if response.status_code != 200:
        hints = {
            403: "la fuente rechaza al cliente (bloqueo anti-bot); puede requerir "
            "cabeceras de navegador o esperar un rato",
            404: "la URL ya no existe; la fuente puede haber cambiado de estructura",
            429: "rate limit superado; espera unos minutos y relanza (la cache "
            "conserva lo ya descargado)",
        }
        hint = hints.get(response.status_code, "revisa la URL y el estado de la fuente")
        raise SourceDownloadError(f"HTTP {response.status_code} al descargar {url}: {hint}.")
    if encoding:
        response.encoding = encoding
    text = response.text
    if not text.strip():
        raise SourceFormatError(f"Respuesta vacía de {url}: la fuente puede haber cambiado.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding=encoding or "utf-8")
    return text
