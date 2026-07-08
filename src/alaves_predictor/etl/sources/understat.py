"""Adaptador de Understat (SPEC §3.1): xG por partido, vía su API interna (ADR-011).

El rediseño de understat.com (dic-2025) eliminó el JSON embebido en el HTML
(`datesData`), pero la web carga los mismos datos desde un endpoint JSON
interno y sin bloqueo anti-bot:

    GET https://understat.com/getLeagueData/La%20liga/<año>

(año = inicio de temporada; descubierto inspeccionando las peticiones del
navegador). El parseo es tolerante con el sobre exterior — lista directa o
diccionario con la lista bajo datesData/matchesData/... — porque es un
endpoint interno sin contrato público y puede variar.

Papel en el pipeline: fuente de RELLENO de xG para partidos donde FBref no lo
aporta (su versión 2026 quitó el xG de las páginas de calendario), y fuente
prevista para el modo temporada (F7).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from urllib.parse import quote

from pydantic import BaseModel, ValidationError

from alaves_predictor.config import UnderstatConfig
from alaves_predictor.etl.errors import SourceFormatError
from alaves_predictor.etl.http_cache import BROWSER_HEADERS

SOURCE_NAME = "understat"

# Claves bajo las que el endpoint puede envolver la lista de partidos.
_MATCH_LIST_KEYS = ("datesData", "matchesData", "matches", "dates")


class UnderstatMatch(BaseModel):
    """Partido terminado con marcador y xG de ambos equipos."""

    understat_id: str
    match_date: date
    home_team: str  # campo "title" de la fuente (se mapea después)
    away_team: str
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float


def season_year(season: str) -> int:
    """Convierte "2018-19" en 2018 (convención de URL de Understat)."""
    return int(season.split("-")[0])


def league_data_url(season: str, cfg: UnderstatConfig) -> str:
    return f"{cfg.base_url}/getLeagueData/{quote(cfg.league)}/{season_year(season)}"


def league_page_url(season: str, cfg: UnderstatConfig) -> str:
    """Página de liga (espacios como guion bajo): sirve de Referer del endpoint."""
    return f"{cfg.base_url}/league/{cfg.league.replace(' ', '_')}/{season_year(season)}"


def api_headers(season: str, cfg: UnderstatConfig) -> dict[str, str]:
    """Cabeceras que espera getLeagueData: sin ellas el servidor responde 404.

    El endpoint es interno (lo llama el JavaScript de la página): exige
    parecer una petición AJAX del propio sitio — Referer de la página de liga
    y X-Requested-With — además de las cabeceras de navegador habituales.
    """
    return {
        **BROWSER_HEADERS,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": league_page_url(season, cfg),
        "X-Requested-With": "XMLHttpRequest",
    }


def _entries_from(data: object) -> list[dict]:
    """Localiza la lista de partidos dentro de la respuesta, sea cual sea el sobre."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in _MATCH_LIST_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return value
        raise SourceFormatError(
            "La respuesta de getLeagueData no contiene una lista de partidos bajo "
            f"{'/'.join(_MATCH_LIST_KEYS)}; claves presentes: {sorted(data)[:10]}."
        )
    raise SourceFormatError("La respuesta de getLeagueData no es JSON de lista ni objeto.")


def parse_league_data(text: str) -> list[UnderstatMatch]:
    """Extrae los partidos ya jugados (isResult=true) de la respuesta JSON."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SourceFormatError(
            f"getLeagueData de Understat no devolvió JSON válido: {exc}"
        ) from exc

    matches: list[UnderstatMatch] = []
    for entry in _entries_from(data):
        if not entry.get("isResult"):
            continue  # partido aún no jugado
        try:
            matches.append(
                UnderstatMatch(
                    understat_id=str(entry["id"]),
                    match_date=datetime.strptime(
                        str(entry["datetime"]), "%Y-%m-%d %H:%M:%S"
                    ).date(),
                    home_team=entry["h"]["title"],
                    away_team=entry["a"]["title"],
                    home_goals=int(entry["goals"]["h"]),
                    away_goals=int(entry["goals"]["a"]),
                    home_xg=float(entry["xG"]["h"]),
                    away_xg=float(entry["xG"]["a"]),
                )
            )
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise SourceFormatError(
                f"Partido de Understat con formato inesperado ({entry.get('id')}): {exc}"
            ) from exc

    if not matches:
        raise SourceFormatError("getLeagueData de Understat no contiene partidos jugados.")
    return matches
