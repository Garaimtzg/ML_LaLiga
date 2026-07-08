"""Carga y validación de la configuración del proyecto.

Toda la parametrización (temporadas, fuentes, equipos) vive en config/*.toml
(CLAUDE.md §8: nada hardcodeado en código). Este módulo la valida con pydantic
y la expone como objetos tipados.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class LeagueConfig(BaseModel):
    teams_per_season: int = 20

    @property
    def rounds(self) -> int:
        """Jornadas de una liga a doble vuelta: 2·(n−1)."""
        return 2 * (self.teams_per_season - 1)

    @property
    def matches_per_season(self) -> int:
        """Partidos totales: n·(n−1) (cada par se enfrenta ida y vuelta)."""
        return self.teams_per_season * (self.teams_per_season - 1)


class DataConfig(BaseModel):
    db_path: Path
    raw_dir: Path


class FootballDataConfig(BaseModel):
    base_url: str
    division: str = "SP1"
    rate_limit_seconds: float = 1.0


class FBrefConfig(BaseModel):
    base_url: str
    competition_id: int = 12  # La Liga en FBref
    competition_slug: str = "La-Liga"
    rate_limit_seconds: float = 6.0
    # Fallback para temporadas pasadas si FBref bloquea la descarga directa
    # con un desafío JavaScript (ADR-010).
    wayback_base: str = "https://web.archive.org/web"
    wayback_cdx_base: str = "https://web.archive.org/cdx/search/cdx"


class UnderstatConfig(BaseModel):
    base_url: str
    rate_limit_seconds: float = 3.0


class ClubEloConfig(BaseModel):
    base_url: str
    rate_limit_seconds: float = 2.0
    history_start: str = "2018-07-01"


class SourcesConfig(BaseModel):
    football_data: FootballDataConfig
    fbref: FBrefConfig
    understat: UnderstatConfig  # en pausa: rediseño de dic-2025 (ADR-008)
    clubelo: ClubEloConfig


class TeamConfig(BaseModel):
    """Nombre canónico de un equipo y sus alias en cada fuente.

    Cada fuente admite un alias o una lista de alias: FBref, por ejemplo, ha
    cambiado su nomenclatura con los años ("Real Betis" en páginas antiguas,
    "Betis" en las modernas) y los snapshots de la Wayback Machine mezclan
    épocas (ADR-010).
    """

    name: str
    football_data: str | list[str]
    fbref: str | list[str]
    understat: str | list[str]
    clubelo: str | list[str]

    def aliases_for(self, source: str) -> list[str]:
        value = getattr(self, source)
        return [value] if isinstance(value, str) else list(value)


class Settings(BaseModel):
    focus_team: str
    current_season: str
    league: LeagueConfig = Field(default_factory=LeagueConfig)
    historical_seasons: list[str]
    data: DataConfig
    sources: SourcesConfig
    teams: dict[str, TeamConfig]


def load_settings(config_dir: Path = Path("config")) -> Settings:
    """Lee config/settings.toml y config/teams.toml y devuelve Settings validado."""
    settings_path = config_dir / "settings.toml"
    teams_path = config_dir / "teams.toml"
    if not settings_path.exists():
        raise FileNotFoundError(
            f"No se encuentra {settings_path}. Ejecuta los comandos desde la raíz del repo."
        )
    with settings_path.open("rb") as fh:
        raw = tomllib.load(fh)
    with teams_path.open("rb") as fh:
        teams_raw = tomllib.load(fh)

    return Settings(
        focus_team=raw["project"]["focus_team"],
        current_season=raw["project"]["current_season"],
        league=LeagueConfig(**raw.get("league", {})),
        historical_seasons=raw["seasons"]["historical"],
        data=DataConfig(**raw["data"]),
        sources=SourcesConfig(**raw["sources"]),
        teams={team_id: TeamConfig(**cfg) for team_id, cfg in teams_raw.items()},
    )
