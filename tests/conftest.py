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
        data=DataConfig(db_path=tmp_path / "test.db", raw_dir=tmp_path / "raw"),
        sources=SourcesConfig(
            football_data=FootballDataConfig(
                base_url="https://fd.test/mmz4281", rate_limit_seconds=0.0
            ),
            fbref=FBrefConfig(base_url="https://fbref.test/en/comps", rate_limit_seconds=0.0),
            understat=UnderstatConfig(
                base_url="https://us.test/league/La_liga", rate_limit_seconds=0.0
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
            return (FIXTURES / "understat_mini.html").read_text()
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
