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


class _FakeCurlResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.content = text.encode("utf-8")


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


def test_impersonate_usa_curl_cffi(tmp_path: Path, monkeypatch) -> None:
    """Con impersonate=True la descarga va por curl_cffi con huella de Chrome (ADR-009)."""
    seen = {}

    def fake_cf_get(url, **kwargs):
        seen["impersonate"] = kwargs.get("impersonate")
        return _FakeCurlResponse(200, "datos fbref")

    def httpx_prohibido(*args, **kwargs):
        raise AssertionError("con impersonate=True no debe usarse httpx")

    monkeypatch.setattr("alaves_predictor.etl.http_cache.cf_requests.get", fake_cf_get)
    monkeypatch.setattr(httpx, "get", httpx_prohibido)
    cache = tmp_path / "f.html"
    text = fetch_text("https://x.test/f", cache, rate_limit_seconds=0.0, impersonate=True)
    assert text == "datos fbref"
    assert seen["impersonate"] == "chrome"
    assert cache.read_text() == "datos fbref"


def test_impersonate_403_da_mensaje_claro(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "alaves_predictor.etl.http_cache.cf_requests.get",
        lambda url, **kwargs: _FakeCurlResponse(403, "blocked"),
    )
    with pytest.raises(SourceDownloadError, match="anti-bot"):
        fetch_text(
            "https://x.test/f", tmp_path / "f.html", rate_limit_seconds=0.0, impersonate=True
        )


@pytest.mark.parametrize(("status", "pista"), [(403, "anti-bot"), (429, "rate limit")])
def test_http_error_da_mensaje_claro(tmp_path: Path, monkeypatch, status, pista) -> None:
    monkeypatch.setattr(httpx, "get", _fake_get(status))
    with pytest.raises(SourceDownloadError, match=pista) as exc_info:
        fetch_text("https://x.test/f", tmp_path / "f.txt", rate_limit_seconds=0.0)
    assert "x.test" in str(exc_info.value)  # la URL siempre en el mensaje
    assert not (tmp_path / "f.txt").exists()  # nada corrupto en cache


def test_fallo_de_red_reintenta_y_da_mensaje_claro(tmp_path: Path, monkeypatch) -> None:
    calls = {"n": 0}

    def timeout(url, **kwargs):
        calls["n"] += 1
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(httpx, "get", timeout)
    monkeypatch.setattr("alaves_predictor.etl.http_cache._RETRY_BACKOFF_S", 0.0)
    with pytest.raises(SourceDownloadError, match="tras 3 intentos"):
        fetch_text("https://x.test/f", tmp_path / "f.txt", rate_limit_seconds=0.0)
    assert calls["n"] == 3


def test_fallo_transitorio_se_recupera_al_reintentar(tmp_path: Path, monkeypatch) -> None:
    """Un timeout puntual (ClubElo lento) no debe tumbar la ingesta entera."""
    calls = {"n": 0}

    def flaky(url, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectTimeout("timeout")
        return httpx.Response(200, text="datos", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", flaky)
    monkeypatch.setattr("alaves_predictor.etl.http_cache._RETRY_BACKOFF_S", 0.0)
    text = fetch_text("https://x.test/f", tmp_path / "f.txt", rate_limit_seconds=0.0)
    assert text == "datos"
    assert calls["n"] == 3


def test_respuesta_vacia_falla(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", _fake_get(200, "   "))
    with pytest.raises(SourceFormatError, match="vacía"):
        fetch_text("https://x.test/f", tmp_path / "f.txt", rate_limit_seconds=0.0)
