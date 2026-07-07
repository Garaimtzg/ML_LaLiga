"""Adaptador de Understat (SPEC §3.1): xG por partido.

Understat no ofrece API: la página https://understat.com/league/La_liga/<año>
(año = inicio de temporada, p. ej. 2018 para la 2018-19) lleva embebido un
`var matchesData = JSON.parse('...')` con todos los partidos de la temporada,
con los caracteres especiales escapados como \\xNN. Aquí se extrae ese bloque,
se desescapa y se valida con pydantic.

Nota (ADR-003): la página también incluye `teamsData` con npxG y PPDA por
partido; se incorporará en F2 cuando las features lo necesiten.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime

from pydantic import BaseModel, ValidationError

from alaves_predictor.config import UnderstatConfig
from alaves_predictor.etl.errors import SourceFormatError

SOURCE_NAME = "understat"

_MATCHES_DATA_RE = re.compile(r"var\s+matchesData\s*=\s*JSON\.parse\('(.*?)'\)", re.DOTALL)


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


def league_url(season: str, cfg: UnderstatConfig) -> str:
    return f"{cfg.base_url}/{season_year(season)}"


def decode_embedded_json(escaped: str) -> list[dict]:
    """Desescapa el string de JSON.parse (secuencias \\xNN) y lo carga como JSON.

    unicode_escape convierte \\xNN a caracteres latin-1; el paso
    latin-1 -> utf-8 recupera los caracteres multibyte originales (acentos).
    """
    decoded = escaped.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
    data = json.loads(decoded)
    if not isinstance(data, list):
        raise SourceFormatError("matchesData de Understat no es una lista.")
    return data


def parse_league_page(html: str) -> list[UnderstatMatch]:
    """Extrae los partidos ya jugados (isResult=true) de la página de liga."""
    found = _MATCHES_DATA_RE.search(html)
    if not found:
        raise SourceFormatError(
            "No se encuentra 'matchesData' en la página de Understat; "
            "la web puede haber cambiado de estructura."
        )
    raw_matches = decode_embedded_json(found.group(1))

    matches: list[UnderstatMatch] = []
    for entry in raw_matches:
        if not entry.get("isResult"):
            continue  # partido aún no jugado
        try:
            matches.append(
                UnderstatMatch(
                    understat_id=str(entry["id"]),
                    match_date=datetime.strptime(entry["datetime"], "%Y-%m-%d %H:%M:%S").date(),
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
        raise SourceFormatError("La página de Understat no contiene partidos jugados.")
    return matches
