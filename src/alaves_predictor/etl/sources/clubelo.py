"""Adaptador de ClubElo (SPEC §3.1): rating Elo histórico por club.

API CSV pública: http://api.clubelo.com/<Club> devuelve el histórico completo
del club con columnas Rank,Club,Country,Level,Elo,From,To (una fila por rango
de fechas de vigencia del rating). Se guarda una fila en la tabla `elo` por
fecha de inicio de vigencia (From).
"""

from __future__ import annotations

import csv
import io
from datetime import date

from pydantic import BaseModel, ValidationError

from alaves_predictor.config import ClubEloConfig
from alaves_predictor.etl.errors import SourceFormatError

SOURCE_NAME = "clubelo"


class EloRating(BaseModel):
    club: str
    elo: float
    valid_from: date
    valid_to: date


def club_url(club_alias: str, cfg: ClubEloConfig) -> str:
    return f"{cfg.base_url}/{club_alias}"


def parse_csv(text: str, club_alias: str) -> list[EloRating]:
    """Parsea el histórico de un club. Vacío -> error con pista de configuración."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "Elo" not in reader.fieldnames:
        raise SourceFormatError(
            f"Respuesta de ClubElo para '{club_alias}' sin columna 'Elo'; "
            "la API puede haber cambiado."
        )
    ratings: list[EloRating] = []
    for line_no, row in enumerate(reader, start=2):
        try:
            ratings.append(
                EloRating(
                    club=row["Club"],
                    elo=float(row["Elo"]),
                    valid_from=date.fromisoformat(row["From"]),
                    valid_to=date.fromisoformat(row["To"]),
                )
            )
        except (ValidationError, KeyError, ValueError) as exc:
            raise SourceFormatError(
                f"Fila {line_no} del CSV de ClubElo ('{club_alias}') inválida: {exc}"
            ) from exc
    if not ratings:
        raise SourceFormatError(
            f"ClubElo no devolvió historial para '{club_alias}'. Suele significar que el "
            "alias es incorrecto: ajusta la clave 'clubelo' de ese equipo en config/teams.toml."
        )
    return ratings
