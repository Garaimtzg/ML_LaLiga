"""Descarga HTTP con cache local en disco y rate limit (ADR-004).

Reglas de datos (CLAUDE.md §6): respetar rate limits y no re-descargar lo ya
guardado. Cada respuesta se guarda en data/raw/<fuente>/<archivo>; si el
archivo existe, se reutiliza sin tocar la red (salvo force=True).
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from alaves_predictor.etl.errors import SourceFormatError

# User-Agent identificable y honesto: proyecto personal, no un bot anónimo.
_HEADERS = {"User-Agent": "alaves-predictor/0.1 (proyecto educativo personal; contacto via GitHub)"}

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
) -> str:
    """Devuelve el cuerpo de `url` como texto, usando cache local.

    - Si `cache_path` existe y force=False: lee del disco, sin petición HTTP.
    - Si descarga: espera lo que falte del rate limit del host, valida que la
      respuesta no esté vacía y la persiste en `cache_path`.
    """
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding=encoding or "utf-8")

    host = httpx.URL(url).host
    elapsed = time.monotonic() - _last_request_at.get(host, 0.0)
    if elapsed < rate_limit_seconds:
        time.sleep(rate_limit_seconds - elapsed)

    response = httpx.get(url, headers=_HEADERS, timeout=30.0, follow_redirects=True)
    _last_request_at[host] = time.monotonic()
    response.raise_for_status()
    if encoding:
        response.encoding = encoding
    text = response.text
    if not text.strip():
        raise SourceFormatError(f"Respuesta vacía de {url}: la fuente puede haber cambiado.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding=encoding or "utf-8")
    return text
