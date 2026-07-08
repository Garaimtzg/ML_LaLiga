"""Descarga HTTP con cache local en disco y rate limit (ADR-004).

Reglas de datos (CLAUDE.md §6): respetar rate limits y no re-descargar lo ya
guardado. Cada respuesta se guarda en data/raw/<fuente>/<archivo>; si el
archivo existe, se reutiliza sin tocar la red (salvo force=True).

Dos transportes (ADR-009):
- httpx con User-Agent identificable y honesto (por defecto).
- curl_cffi imitando la huella TLS de Chrome (`impersonate=True`), solo para
  fuentes tras Cloudflare que rechazan cualquier cliente no-navegador aunque
  las cabeceras sean de navegador (FBref). Los datos son públicos y el acceso
  mínimo: 1 página por temporada, cacheada y con rate limit.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from curl_cffi import CurlError
from curl_cffi import requests as cf_requests

from alaves_predictor.etl.errors import SourceDownloadError, SourceFormatError

# User-Agent identificable y honesto: proyecto personal, no un bot anónimo.
_HEADERS = {"User-Agent": "alaves-predictor/0.1 (proyecto educativo personal; contacto via GitHub)"}

# Cabeceras de navegador para fuentes que filtran por User-Agent pero no por
# huella TLS (Understat, ADR-004-actualización). Si además validan la huella
# TLS (FBref), usar impersonate=True.
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

_STATUS_HINTS = {
    403: "la fuente rechaza al cliente (bloqueo anti-bot); puede requerir "
    "impersonate=True o esperar un rato",
    404: "la URL ya no existe; la fuente puede haber cambiado de estructura",
    429: "rate limit superado; espera unos minutos y relanza (la cache conserva lo ya descargado)",
}


# Reintentos ante fallos transitorios de red (timeouts, cortes de conexión):
# api.clubelo.com, p. ej., responde lento a veces. Backoff: 2 s, 4 s.
_TIMEOUT_S = 60.0
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0


def _download(url: str, headers: dict[str, str] | None, impersonate: bool) -> tuple[int, bytes]:
    """Descarga cruda con el transporte que toque, con reintentos ante fallos de red."""
    last_error: Exception | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            if impersonate:
                response = cf_requests.get(url, impersonate="chrome", timeout=_TIMEOUT_S)
                return response.status_code, response.content
            http_response = httpx.get(
                url, headers=headers or _HEADERS, timeout=_TIMEOUT_S, follow_redirects=True
            )
            return http_response.status_code, http_response.content
        except (httpx.HTTPError, CurlError) as exc:
            last_error = exc
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S * attempt)
    raise SourceDownloadError(
        f"Fallo de red descargando {url} tras {_RETRY_ATTEMPTS} intentos: {last_error}"
    ) from last_error


def fetch_text(
    url: str,
    cache_path: Path,
    *,
    rate_limit_seconds: float = 1.0,
    force: bool = False,
    encoding: str | None = None,
    headers: dict[str, str] | None = None,
    impersonate: bool = False,
) -> str:
    """Devuelve el cuerpo de `url` como texto, usando cache local.

    - Si `cache_path` existe y force=False: lee del disco, sin petición HTTP.
    - Si descarga: espera lo que falte del rate limit del host, valida que la
      respuesta no esté vacía y la persiste en `cache_path`.
    - `headers` sobreescribe las cabeceras por defecto (p. ej. Understat).
    - `impersonate=True` usa curl_cffi con huella TLS de Chrome (FBref, ADR-009).
    """
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding=encoding or "utf-8")

    host = httpx.URL(url).host
    elapsed = time.monotonic() - _last_request_at.get(host, 0.0)
    if elapsed < rate_limit_seconds:
        time.sleep(rate_limit_seconds - elapsed)

    status_code, body = _download(url, headers, impersonate)
    _last_request_at[host] = time.monotonic()
    if status_code != 200:
        hint = _STATUS_HINTS.get(status_code, "revisa la URL y el estado de la fuente")
        raise SourceDownloadError(f"HTTP {status_code} al descargar {url}: {hint}.")

    text = body.decode(encoding or "utf-8", errors="strict" if encoding else "replace")
    if not text.strip():
        raise SourceFormatError(f"Respuesta vacía de {url}: la fuente puede haber cambiado.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding=encoding or "utf-8")
    return text
