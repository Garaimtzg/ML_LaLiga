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
    # Zonas de la clasificación por rango de posición [desde, hasta] (1 = líder).
    # Parametrizadas (CLAUDE.md §8); por defecto, el reparto europeo de LaLiga.
    zones: dict[str, list[int]] = Field(
        default_factory=lambda: {
            "titulo": [1, 1],
            "champions": [1, 4],
            "europa": [5, 6],
            "conference": [7, 7],
            "descenso": [18, 20],
        }
    )

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
    features_dir: Path = Path("data/features")


class EloInternalConfig(BaseModel):
    """Parámetros del Elo interno recalculable (ADR-013)."""

    k: float = 20.0
    home_advantage: float = 60.0  # puntos Elo sumados al local en el esperado
    initial_rating: float = 1500.0


class FeaturesConfig(BaseModel):
    """Parámetros del feature store (F2, SPEC §4)."""

    feature_set_version: str = "v1"
    form_windows: list[int] = Field(default_factory=lambda: [5, 10])
    no_crowd_seasons: list[str] = Field(default_factory=list)
    derbies: list[list[str]] = Field(default_factory=list)
    elo_internal: EloInternalConfig = Field(default_factory=EloInternalConfig)


class DixonColesConfig(BaseModel):
    """Parámetros del modelo Dixon-Coles (SPEC §6.2, ADR-015/019)."""

    xi: float = 0.0019  # ponderación temporal: peso = exp(-xi · días)
    max_goals: int = 10  # truncamiento de la matriz de marcadores
    rho_bound: float = 0.2  # cota de |rho| para que tau se mantenga > 0
    # candidatos de xi evaluados en validación walk-forward (ADR-019);
    # vacío o con un único valor => se usa `xi` sin selección
    xi_grid: list[float] = Field(default_factory=list)

    def xi_candidates(self) -> list[float]:
        return list(self.xi_grid) if self.xi_grid else [self.xi]


class LightGBMConfig(BaseModel):
    """Hiperparámetros v1 del clasificador (SPEC §6.3, ADR-016)."""

    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 50
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0


class LinearConfig(BaseModel):
    """Componente lineal Elo+forma del ensemble sin cuotas (ADR-020/021)."""

    c: float = 1.0  # inverso de la regularización L2 por defecto
    # candidatos de C evaluados en validación walk-forward (ADR-021); con las
    # features estandarizadas, C bajo encoge la señal Elo — hay que buscarlo
    c_grid: list[float] = Field(default_factory=lambda: [0.3, 1.0, 3.0, 10.0, 30.0])

    def c_candidates(self) -> list[float]:
        return list(self.c_grid) if self.c_grid else [self.c]


class EnsembleConfig(BaseModel):
    weight_grid_step: float = 0.05


class ModelsConfig(BaseModel):
    """Parámetros de la fase de modelado (F3)."""

    registry_dir: Path = Path("models/registry")
    max_logloss_regression: float = 0.10  # regla anti-sorpresa (SPEC §6.4)
    dixon_coles: DixonColesConfig = Field(default_factory=DixonColesConfig)
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    linear: LinearConfig = Field(default_factory=LinearConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)


class FootballDataConfig(BaseModel):
    base_url: str
    division: str = "SP1"
    rate_limit_seconds: float = 1.0
    # Archivo único de próximos partidos de todas las ligas (F7, ADR-026).
    fixtures_url: str = "https://www.football-data.co.uk/fixtures.csv"
    # Calendario local opcional (mismo formato) para sembrar los fixtures a mano
    # hasta que football-data publique la temporada (F7, ADR-026).
    local_fixtures_file: str = "data/fixtures.csv"


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
    base_url: str  # raíz del sitio; el endpoint es <base>/getLeagueData/<liga>/<año>
    league: str = "La liga"
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
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
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
        features=FeaturesConfig(**raw.get("features", {})),
        models=ModelsConfig(**raw.get("models", {})),
        teams={team_id: TeamConfig(**cfg) for team_id, cfg in teams_raw.items()},
    )
