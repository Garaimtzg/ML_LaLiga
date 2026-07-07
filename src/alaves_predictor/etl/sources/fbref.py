"""Adaptador de FBref (SPEC §3.1): xG por partido y jornada oficial (ADR-008).

La página "Scores & Fixtures" de una temporada lista los ~380 partidos con
jornada oficial (Wk), fecha, equipos, marcador y xG de ambos equipos:
https://fbref.com/en/comps/12/2018-2019/schedule/2018-2019-La-Liga-Scores-and-Fixtures

Una sola página por temporada (mismo coste que tenía Understat) y con rate
limit conservador (FBref pide moderación a los bots; 6 s entre peticiones).
El parseo se apoya en los atributos `data-stat` de las celdas, que son la
interfaz más estable de FBref (los usan todas las librerías del ecosistema).

En F2 este adaptador se ampliará a las páginas de estadísticas detalladas
por partido (pases, presión, portería...) para completar `match_stats`.
"""

from __future__ import annotations

import re
from datetime import date

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, ValidationError

from alaves_predictor.config import FBrefConfig
from alaves_predictor.etl.errors import SourceFormatError

SOURCE_NAME = "fbref"

# Marcador tipo "1–2" (FBref usa guion en dash U+2013; se acepta también "-").
_SCORE_RE = re.compile(r"^(\d+)\s*[–-]\s*(\d+)$")


class FBrefMatch(BaseModel):
    """Partido terminado del calendario de FBref."""

    matchday: int | None  # columna Wk (jornada oficial); None si falta
    match_date: date
    home_team: str  # nombre tal como lo escribe FBref (se mapea después)
    away_team: str
    home_goals: int
    away_goals: int
    home_xg: float | None  # puede faltar en partidos muy antiguos
    away_xg: float | None


def season_slug(season: str) -> str:
    """Convierte "2018-19" en "2018-2019" (convención de URL de FBref)."""
    start = int(season.split("-")[0])
    return f"{start}-{start + 1}"


def schedule_url(season: str, cfg: FBrefConfig) -> str:
    slug = season_slug(season)
    return (
        f"{cfg.base_url}/{cfg.competition_id}/{slug}/schedule/"
        f"{slug}-{cfg.competition_slug}-Scores-and-Fixtures"
    )


def wayback_url(season: str, cfg: FBrefConfig) -> str:
    """URL del snapshot de la Wayback Machine para una temporada (ADR-010).

    El sufijo `id_` en el timestamp pide el HTML original archivado, sin la
    barra de herramientas que inyecta archive.org. Se pide el snapshot más
    cercano al 1 de agosto posterior al fin de temporada, cuando la página ya
    contiene la temporada completa; si el snapshot fuera anterior/incompleto,
    la validación de cobertura de xG lo detectaría.
    """
    end_year = int(season.split("-")[0]) + 1
    timestamp = f"{end_year}0801000000"
    return f"{cfg.wayback_base}/{timestamp}id_/{schedule_url(season, cfg)}"


def _row_cells(row: Tag) -> dict[str, str]:
    """Mapea data-stat -> texto de la celda para una fila de la tabla."""
    return {
        str(cell.get("data-stat") or ""): cell.get_text(strip=True)
        for cell in row.find_all(["td", "th"])
        if isinstance(cell, Tag)
    }


def parse_schedule(html: str) -> list[FBrefMatch]:
    """Extrae los partidos jugados de la página Scores & Fixtures."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id=re.compile(r"^sched"))
    if table is None or not isinstance(table, Tag):
        raise SourceFormatError(
            "No se encuentra la tabla de calendario (id 'sched...') en la página de FBref; "
            "la web puede haber cambiado de estructura o servido una página de bloqueo."
        )
    body = table.find("tbody")
    if body is None or not isinstance(body, Tag):
        raise SourceFormatError("La tabla de calendario de FBref no tiene tbody.")

    matches: list[FBrefMatch] = []
    for row in body.find_all("tr"):
        if not isinstance(row, Tag):
            continue
        cells = _row_cells(row)
        score_text = cells.get("score", "")
        if not cells.get("home_team") or not score_text:
            continue  # fila espaciadora, cabecera repetida o partido sin jugar
        score = _SCORE_RE.match(score_text)
        if score is None:
            raise SourceFormatError(
                f"Marcador de FBref con formato inesperado: '{score_text}' "
                f"({cells.get('home_team')} vs {cells.get('away_team')})."
            )
        try:
            matches.append(
                FBrefMatch(
                    matchday=int(cells["gameweek"]) if cells.get("gameweek") else None,
                    match_date=date.fromisoformat(cells["date"]),
                    home_team=cells["home_team"],
                    away_team=cells["away_team"],
                    home_goals=int(score.group(1)),
                    away_goals=int(score.group(2)),
                    home_xg=float(cells["home_xg"]) if cells.get("home_xg") else None,
                    away_xg=float(cells["away_xg"]) if cells.get("away_xg") else None,
                )
            )
        except (ValidationError, KeyError, ValueError) as exc:
            raise SourceFormatError(
                f"Fila del calendario de FBref inválida "
                f"({cells.get('home_team')} vs {cells.get('away_team')}): {exc}"
            ) from exc

    if not matches:
        raise SourceFormatError("El calendario de FBref no contiene partidos jugados.")
    return matches
