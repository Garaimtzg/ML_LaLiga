"""Adaptador de football-data.co.uk (SPEC §3.1): resultados, stats básicas y cuotas.

Formato: CSV por temporada y división en
https://www.football-data.co.uk/mmz4281/<código-temporada>/SP1.csv
donde el código de "2018-19" es "1819". Columnas documentadas en
https://www.football-data.co.uk/notes.txt

Notas de formato manejadas aquí:
- Fechas dd/mm/yyyy (y dd/mm/yy en archivos antiguos).
- Las cuotas de cierre (B365CH...) existen desde 2019-20; antes, NULL.
- Los CSV a veces traen filas vacías o columnas de cola sin nombre.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime

from pydantic import BaseModel, ValidationError

from alaves_predictor.config import FootballDataConfig
from alaves_predictor.etl.errors import SourceFormatError

SOURCE_NAME = "football-data"

# Casas de apuestas que persistimos en la tabla `odds` (ADR-003):
# bet365 y Pinnacle (referencia de cuota "sharp"), más el máximo y la media de
# mercado que calcula la propia fuente. Prefijo de columna -> nombre en BD.
BOOKMAKER_PREFIXES: dict[str, str] = {
    "B365": "bet365",
    "PS": "pinnacle",
    "Max": "market_max",
    "Avg": "market_avg",
}


class FootballDataMatch(BaseModel):
    """Una fila del CSV ya validada y con tipos correctos."""

    match_date: date
    home_team: str  # nombre tal como lo escribe la fuente (se mapea después)
    away_team: str
    home_goals: int
    away_goals: int
    full_time_result: str  # H | D | A
    # Estadísticas básicas (pueden faltar en archivos muy antiguos)
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_fouls: int | None = None
    away_fouls: int | None = None
    home_corners: int | None = None
    away_corners: int | None = None
    home_yellow: int | None = None
    away_yellow: int | None = None
    home_red: int | None = None
    away_red: int | None = None
    # Cuotas: bookmaker -> (h, d, a); apertura y cierre por separado
    odds_open: dict[str, tuple[float, float, float]] = {}
    odds_close: dict[str, tuple[float, float, float]] = {}


class FootballDataFixture(BaseModel):
    """Un partido PROGRAMADO (aún sin jugar) del archivo de fixtures."""

    match_date: date
    home_team: str
    away_team: str
    odds_open: dict[str, tuple[float, float, float]] = {}


def season_code(season: str) -> str:
    """Convierte "2018-19" al código de URL "1819"."""
    start, end = season.split("-")
    return start[-2:] + end


def csv_url(season: str, cfg: FootballDataConfig) -> str:
    return f"{cfg.base_url}/{season_code(season)}/{cfg.division}.csv"


def fixtures_url(cfg: FootballDataConfig) -> str:
    """URL del archivo único de próximos partidos de todas las ligas (F7)."""
    return cfg.fixtures_url


def parse_fixtures(text: str, division: str) -> list[FootballDataFixture]:
    """Partidos programados de una división del fixtures.csv global (F7).

    El archivo lista los próximos encuentros de todas las ligas (columna Div);
    se filtra por la división pedida. Trae cuotas de apertura pero, obviamente,
    ni marcador ni estadísticas (aún no se ha jugado).
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise SourceFormatError("fixtures.csv de football-data vacío o sin cabecera.")
    required = {"Div", "Date", "HomeTeam", "AwayTeam"}
    missing = required - set(reader.fieldnames)
    if missing:
        raise SourceFormatError(
            f"El fixtures.csv no tiene las columnas {sorted(missing)}; "
            "la fuente puede haber cambiado de formato."
        )

    fixtures: list[FootballDataFixture] = []
    for line_no, row in enumerate(reader, start=2):
        if (row.get("Div") or "").strip() != division:
            continue
        if not (row.get("HomeTeam") or "").strip():
            continue
        odds_open = {}
        for prefix, bookmaker in BOOKMAKER_PREFIXES.items():
            if triplet := _odds_triplet(row, prefix):
                odds_open[bookmaker] = triplet
        try:
            fixtures.append(
                FootballDataFixture(
                    match_date=_parse_date(row["Date"].strip()),
                    home_team=row["HomeTeam"].strip(),
                    away_team=row["AwayTeam"].strip(),
                    odds_open=odds_open,
                )
            )
        except (ValidationError, ValueError, KeyError) as exc:
            raise SourceFormatError(f"Fila {line_no} del fixtures.csv inválida: {exc}") from exc
    return fixtures


def _parse_date(raw: str) -> date:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise SourceFormatError(f"Fecha no reconocida en football-data: '{raw}'")


def _opt_int(row: dict[str, str], col: str) -> int | None:
    value = (row.get(col) or "").strip()
    return int(float(value)) if value else None


def _odds_triplet(row: dict[str, str], prefix: str) -> tuple[float, float, float] | None:
    """Lee las columnas <prefix>H/D/A; devuelve None si falta alguna."""
    try:
        h = float(row[f"{prefix}H"])
        d = float(row[f"{prefix}D"])
        a = float(row[f"{prefix}A"])
    except (KeyError, ValueError, TypeError):
        return None
    return (h, d, a)


def parse_csv(text: str) -> list[FootballDataMatch]:
    """Parsea el CSV de una temporada. Falla ruidosamente ante formato inesperado."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise SourceFormatError("CSV de football-data vacío o sin cabecera.")
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required - set(reader.fieldnames)
    if missing:
        raise SourceFormatError(
            f"El CSV de football-data no tiene las columnas {sorted(missing)}; "
            "la fuente puede haber cambiado de formato."
        )

    matches: list[FootballDataMatch] = []
    for line_no, row in enumerate(reader, start=2):
        if not (row.get("HomeTeam") or "").strip():
            continue  # fila vacía de cola, habitual en estos CSV
        odds_open = {}
        odds_close = {}
        for prefix, bookmaker in BOOKMAKER_PREFIXES.items():
            if triplet := _odds_triplet(row, prefix):
                odds_open[bookmaker] = triplet
            # Las de cierre insertan una C tras el prefijo: B365CH, PSCH, MaxCH, AvgCH.
            if triplet := _odds_triplet(row, f"{prefix}C"):
                odds_close[bookmaker] = triplet
        try:
            matches.append(
                FootballDataMatch(
                    match_date=_parse_date(row["Date"].strip()),
                    home_team=row["HomeTeam"].strip(),
                    away_team=row["AwayTeam"].strip(),
                    home_goals=int(float(row["FTHG"])),
                    away_goals=int(float(row["FTAG"])),
                    full_time_result=row["FTR"].strip(),
                    home_shots=_opt_int(row, "HS"),
                    away_shots=_opt_int(row, "AS"),
                    home_shots_on_target=_opt_int(row, "HST"),
                    away_shots_on_target=_opt_int(row, "AST"),
                    home_fouls=_opt_int(row, "HF"),
                    away_fouls=_opt_int(row, "AF"),
                    home_corners=_opt_int(row, "HC"),
                    away_corners=_opt_int(row, "AC"),
                    home_yellow=_opt_int(row, "HY"),
                    away_yellow=_opt_int(row, "AY"),
                    home_red=_opt_int(row, "HR"),
                    away_red=_opt_int(row, "AR"),
                    odds_open=odds_open,
                    odds_close=odds_close,
                )
            )
        except (ValidationError, ValueError, KeyError) as exc:
            raise SourceFormatError(
                f"Fila {line_no} del CSV de football-data inválida: {exc}"
            ) from exc

    if not matches:
        raise SourceFormatError("El CSV de football-data no contiene partidos.")
    return matches
