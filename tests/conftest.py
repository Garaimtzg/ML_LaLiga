"""Fixtures compartidos: mini-liga sintética de 4 equipos (12 partidos, 6 jornadas).

Los tests nunca tocan la red: `fake_fetch` sirve los fixtures congelados de
tests/fixtures/ según la URL pedida, imitando a http_cache.fetch_text.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alaves_predictor.config import (
    ClubEloConfig,
    DataConfig,
    FBrefConfig,
    FeaturesConfig,
    FootballDataConfig,
    LeagueConfig,
    Settings,
    SourcesConfig,
    TeamConfig,
    UnderstatConfig,
)
from alaves_predictor.etl import db

FIXTURES = Path(__file__).parent / "fixtures"

MINI_TEAMS = {
    "alaves": TeamConfig(
        name="Deportivo Alavés",
        football_data="Alaves",
        fbref="Alavés",
        understat="Alaves",
        clubelo="Alaves",
    ),
    "barcelona": TeamConfig(
        name="FC Barcelona",
        football_data="Barcelona",
        fbref="Barcelona",
        understat="Barcelona",
        clubelo="Barcelona",
    ),
    "real-sociedad": TeamConfig(
        name="Real Sociedad",
        football_data="Sociedad",
        fbref="Real Sociedad",
        understat="Real Sociedad",
        clubelo="Sociedad",
    ),
    "getafe": TeamConfig(
        name="Getafe CF",
        football_data="Getafe",
        fbref="Getafe",
        understat="Getafe",
        clubelo="Getafe",
    ),
}


@pytest.fixture()
def mini_settings(tmp_path: Path) -> Settings:
    """Settings de una mini-liga de 4 equipos con rutas en tmp."""
    return Settings(
        focus_team="alaves",
        current_season="2019-20",
        league=LeagueConfig(teams_per_season=4),
        historical_seasons=["2018-19"],
        data=DataConfig(
            db_path=tmp_path / "test.db",
            raw_dir=tmp_path / "raw",
            features_dir=tmp_path / "features",
        ),
        features=FeaturesConfig(derbies=[["alaves", "real-sociedad"]]),
        sources=SourcesConfig(
            football_data=FootballDataConfig(
                base_url="https://fd.test/mmz4281", rate_limit_seconds=0.0
            ),
            fbref=FBrefConfig(base_url="https://fbref.test/en/comps", rate_limit_seconds=0.0),
            understat=UnderstatConfig(
                base_url="https://us.test", league="Mini liga", rate_limit_seconds=0.0
            ),
            clubelo=ClubEloConfig(
                base_url="http://elo.test", rate_limit_seconds=0.0, history_start="2018-07-01"
            ),
        ),
        teams=MINI_TEAMS,
    )


@pytest.fixture()
def fake_fetch(monkeypatch: pytest.MonkeyPatch):
    """Sustituye fetch_text en el orquestador por un lector de fixtures según URL."""

    def _fetch(url: str, cache_path: Path, **kwargs) -> str:
        if "fd.test" in url:
            return (FIXTURES / "football_data_mini.csv").read_text()
        if "fbref.test" in url:
            return (FIXTURES / "fbref_schedule_mini.html").read_text()
        if "us.test" in url:
            assert "getLeagueData/Mini%20liga/2018" in url  # endpoint nuevo (ADR-011)
            return (FIXTURES / "understat_league_mini.json").read_text()
        if "elo.test" in url:
            club = url.rsplit("/", 1)[-1]
            path = FIXTURES / f"clubelo_{club}.csv"
            if not path.exists():
                raise AssertionError(f"El test pidió un club sin fixture: {club}")
            return path.read_text()
        raise AssertionError(f"URL inesperada en test: {url}")

    monkeypatch.setattr("alaves_predictor.etl.ingest.fetch_text", _fetch)
    return _fetch


@pytest.fixture()
def mini_db(mini_settings: Settings) -> sqlite3.Connection:
    conn = db.connect(mini_settings.data.db_path)
    db.init_schema(conn)
    yield conn
    conn.close()


def make_synthetic_features(n_seasons: int = 4, seed: int = 42):
    """Frame de features sintético multi-temporada para los tests de modelos (F3).

    Liga de 6 equipos con fuerzas conocidas; los goles se generan con el
    proceso Dixon-Coles sin correlación. Incluye una feature informativa
    (strength_diff), una de ruido y cuotas implícitas derivadas de las
    probabilidades reales — suficiente para que LightGBM aprenda algo.
    """
    import numpy as np
    import pandas as pd

    from alaves_predictor.models import dixon_coles as dc

    rng = np.random.default_rng(seed)
    teams = ["t1", "t2", "t3", "t4", "t5", "t6"]
    # diferencias grandes a propósito: los tests de aprendizaje necesitan señal
    attack = dict(zip(teams, [0.8, 0.4, 0.0, 0.0, -0.4, -0.8], strict=True))
    defense = dict(zip(teams, [0.6, 0.3, 0.0, -0.3, 0.0, -0.6], strict=True))
    gamma = 0.25
    seasons = [f"{2018 + i}-{(19 + i) % 100:02d}" for i in range(n_seasons)]

    rows = []
    for s_idx, season in enumerate(seasons):
        date = pd.Timestamp(f"{2018 + s_idx}-09-01")
        # dos vueltas dobles por temporada: 60 partidos (más datos para aprender)
        pairs = [(h, a) for h in teams for a in teams if h != a] * 2
        rng.shuffle(pairs)
        for i, (home, away) in enumerate(pairs):
            lam = np.exp(attack[home] - defense[away] + gamma)
            mu = np.exp(attack[away] - defense[home])
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            true_probs = dc.outcome_probs(dc.score_matrix(lam, mu, 0.0, 8))
            noisy = np.clip(true_probs + rng.normal(0, 0.02, 3), 0.02, None)
            noisy = noisy / noisy.sum()
            rows.append(
                {
                    "match_id": f"{season}_{home}_{away}",
                    "season": season,
                    "matchday": i // 3 + 1,
                    "date": str((date + pd.Timedelta(days=i)).date()),
                    "home_id": home,
                    "away_id": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "home_xg": lam,
                    "away_xg": mu,
                    "result": "H" if hg > ag else ("D" if hg == ag else "A"),
                    "as_of_date": str((date + pd.Timedelta(days=i - 1)).date()),
                    # features del "modelo": una informativa, una de ruido, cuotas
                    "strength_diff": (attack[home] + defense[home])
                    - (attack[away] + defense[away]),
                    # proxy de Elo: fuerza real a escala Elo con algo de ruido
                    # (la usa el componente elo_logistico del ensemble apilado)
                    "elo_clubelo_diff": 200.0
                    * ((attack[home] + defense[home]) - (attack[away] + defense[away]))
                    + rng.normal(0, 20),
                    "noise": rng.normal(),
                    "imp_home": noisy[0],
                    "imp_draw": noisy[1],
                    "imp_away": noisy[2],
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_features():
    return make_synthetic_features()


@pytest.fixture()
def model_settings(mini_settings: Settings) -> Settings:
    """mini_settings con hiperparámetros adaptados al tamaño del dataset sintético."""
    mini_settings.models.lightgbm.n_estimators = 80
    mini_settings.models.lightgbm.min_child_samples = 5
    # el registry de tests nunca debe escribir en el repo
    mini_settings.models.registry_dir = mini_settings.data.db_path.parent / "registry"
    return mini_settings
