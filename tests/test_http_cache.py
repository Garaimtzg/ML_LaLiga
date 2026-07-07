"""Tests de la capa de descarga: cache, y errores HTTP convertidos en ETLError
con mensaje claro (CLAUDE.md §6: nunca un traceback crudo al usuario)."""

from pathlib import Path

import httpx
import pytest

from alaves_predictor.etl.errors import SourceDownloadError, SourceFormatError
from alaves_predictor.etl.http_cache import fetch_text


def _fake_get(status_code: int, text: str = "contenido"):
    def fake(url, **kwargs):
        return httpx.Response(status_code, text=text, request=httpx.Request("GET", url))

    return fake


def test_cache_evita_la_red(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "f.txt"
    cache.write_text("cacheado")

    def explota(*args, **kwargs):
        raise AssertionError("no debería tocar la red con cache presente")

    monkeypatch.setattr(httpx, "get", explota)
    assert fetch_text("https://x.test/f", cache) == "cacheado"


def test_descarga_correcta_persiste_en_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", _fake_get(200, "datos"))
    cache = tmp_path / "sub" / "f.txt"
    assert fetch_text("https://x.test/f", cache, rate_limit_seconds=0.0) == "datos"
    assert cache.read_text() == "datos"


@pytest.mark.parametrize(("status", "pista"), [(403, "anti-bot"), (429, "rate limit")])
def test_http_error_da_mensaje_claro(tmp_path: Path, monkeypatch, status, pista) -> None:
    monkeypatch.setattr(httpx, "get", _fake_get(status))
    with pytest.raises(SourceDownloadError, match=pista) as exc_info:
        fetch_text("https://x.test/f", tmp_path / "f.txt", rate_limit_seconds=0.0)
    assert "x.test" in str(exc_info.value)  # la URL siempre en el mensaje
    assert not (tmp_path / "f.txt").exists()  # nada corrupto en cache


def test_fallo_de_red_da_mensaje_claro(tmp_path: Path, monkeypatch) -> None:
    def timeout(url, **kwargs):
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(httpx, "get", timeout)
    with pytest.raises(SourceDownloadError, match="Fallo de red"):
        fetch_text("https://x.test/f", tmp_path / "f.txt", rate_limit_seconds=0.0)


def test_respuesta_vacia_falla(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", _fake_get(200, "   "))
    with pytest.raises(SourceFormatError, match="vacía"):
        fetch_text("https://x.test/f", tmp_path / "f.txt", rate_limit_seconds=0.0)
