"""Fuentes que fallan o sirven datos de otra temporada (ADR-032).

Casos sacados de una ejecución real del ciclo semanal sobre una BD nueva:
ClubElo caído (502), Understat respondiendo algo que no es JSON y FBref
sirviendo la página de la temporada ANTERIOR (una captura vieja de la URL sin
año). Antes esto acababa en un crash y con partidos de otra temporada metidos
en el calendario.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alaves_predictor import cli
from alaves_predictor.etl import db
from alaves_predictor.etl.errors import SourceDownloadError, SourceFormatError
from alaves_predictor.etl.ingest import (
    _fetch_fbref_schedule,
    _without_elo,
    ingest_fixtures,
    make_match_id,
    purge_out_of_season_scheduled,
)
from alaves_predictor.etl.sources import understat
from alaves_predictor.etl.teams import TeamRegistry
from alaves_predictor.etl.validate import validate_db
from alaves_predictor.features.build import build_features

FIXTURES = Path(__file__).parent / "fixtures"
# Página de FBref de la 2018-19 (con un partido aún por jugar, del 30-may-2019):
# para la temporada en curso de los tests (2026-27) es "de otra temporada".
STALE_FBREF = (FIXTURES / "fbref_schedule_mini.html").read_text(encoding="utf-8")


def _match(conn, settings, home, away, day, status="scheduled", goals=(None, None)):
    season = settings.current_season
    match_id = make_match_id(season, home, away)
    db.upsert(
        conn,
        "matches",
        {
            "match_id": match_id,
            "season": season,
            "matchday": 1,
            "date": day,
            "home_id": home,
            "away_id": away,
            "home_goals": goals[0],
            "away_goals": goals[1],
            "status": status,
            "source": "test",
            "fetched_at": datetime.now(UTC).isoformat(),
        },
        key_cols=["match_id"],
    )
    conn.commit()
    return match_id


def test_la_ventana_de_temporada_cubre_la_2019_20() -> None:
    from alaves_predictor.config import LeagueConfig

    first, last = LeagueConfig().season_bounds("2019-20")
    assert str(first) == "2019-07-01"
    assert str(last) == "2020-07-31"  # la 2019-20 acabó el 19-jul-2020


def test_features_sin_ningun_elo_de_clubelo(mini_db, mini_settings) -> None:
    """BD nueva con ClubElo caído: antes, MergeError en merge_asof y ciclo abortado."""
    TeamRegistry(mini_settings.teams).seed_db(mini_db)
    _match(mini_db, mini_settings, "alaves", "getafe", "2026-08-15", "finished", (1, 0))
    _match(mini_db, mini_settings, "getafe", "alaves", "2026-12-01")
    df = build_features(mini_db, mini_settings, include_scheduled=True)
    assert len(df) == 2
    assert df["elo_clubelo_diff"].isna().all()  # vacías, no un crash


def test_la_cascada_de_fbref_rechaza_una_pagina_de_otra_temporada(
    mini_settings, tmp_path, monkeypatch
) -> None:
    """Directo, Wayback y cache sirven la misma página vieja: ninguna se acepta."""
    mini_settings.data.raw_dir = tmp_path / "raw"

    def _stale(url, cache_path, **kwargs):
        if "cdx" in url:
            raise SourceDownloadError("CDX no disponible")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(STALE_FBREF, encoding="utf-8")
        return STALE_FBREF

    monkeypatch.setattr("alaves_predictor.etl.ingest.fetch_text", _stale)
    with pytest.raises(SourceDownloadError, match="otra temporada"):
        _fetch_fbref_schedule(mini_settings.current_season, mini_settings, force=True)


def test_el_calendario_descarta_partidos_de_otra_temporada(
    mini_db, mini_settings, tmp_path, monkeypatch
) -> None:
    """El caso real: los pendientes de la página vieja entraban como programados."""
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    mini_settings.data.raw_dir = tmp_path / "raw"
    mini_settings.sources.football_data.local_fixtures_file = str(tmp_path / "no.csv")
    fd_csv = "Div,Date,Time,HomeTeam,AwayTeam\nSP1,22/11/2026,21:00,Alaves,Getafe\n"

    def _fetch(url, cache_path, **kwargs):
        if "fixtures.csv" in url:
            return fd_csv
        if "fbref.test" in url:
            return STALE_FBREF
        raise SourceDownloadError("caído")

    monkeypatch.setattr("alaves_predictor.etl.ingest.fetch_text", _fetch)
    report = ingest_fixtures(mini_db, mini_settings, registry)

    assert report.inserted == 1  # solo el de football-data
    rows = mini_db.execute("SELECT date, matchday FROM matches").fetchall()
    assert [(r["date"], r["matchday"]) for r in rows] == [("2026-11-22", None)]
    # el pendiente de la página vieja (2019) no entra ni presta su jornada:
    # FBref se descarta entero, sin aportar ni un partido
    assert report.by_source == {"football-data": 1}


def test_cualquier_origen_con_fechas_de_otra_temporada_se_descarta(
    mini_db, mini_settings, tmp_path, monkeypatch
) -> None:
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    local = tmp_path / "fixtures.csv"
    local.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam\n"
        "SP1,14/02/2026,21:00,Alaves,Getafe\n"
        "SP1,13/12/2026,18:30,Getafe,Alaves\n",
        encoding="utf-8",
    )
    mini_settings.sources.football_data.local_fixtures_file = str(local)

    def _down(url, cache_path, **kwargs):
        raise SourceDownloadError("caído")

    monkeypatch.setattr("alaves_predictor.etl.ingest.fetch_text", _down)
    report = ingest_fixtures(mini_db, mini_settings, registry)
    assert report.inserted == 1
    assert report.out_of_season == {"local": 1}  # el de febrero de 2026 es de la 2025-26


def test_purga_programados_de_otra_temporada_y_lo_que_cuelga(mini_db, mini_settings) -> None:
    TeamRegistry(mini_settings.teams).seed_db(mini_db)
    stale = _match(mini_db, mini_settings, "alaves", "getafe", "2026-02-14")  # de la 2025-26
    good = _match(mini_db, mini_settings, "getafe", "alaves", "2026-12-01")
    played = _match(mini_db, mini_settings, "barcelona", "alaves", "2026-03-01", "finished", (2, 1))
    mini_db.execute(
        "INSERT INTO odds (match_id, bookmaker, open_h, open_d, open_a, source, fetched_at) "
        "VALUES (?, 'bet365', 2.0, 3.0, 4.0, 'test', 'x')",
        (stale,),
    )
    mini_db.commit()
    assert any(
        not r.passed and "fechas dentro de la temporada" in r.name
        for r in validate_db(mini_db, mini_settings)
    )

    assert purge_out_of_season_scheduled(mini_db, mini_settings.current_season, mini_settings) == 1
    ids = {r["match_id"] for r in mini_db.execute("SELECT match_id FROM matches")}
    assert ids == {good, played}  # un jugado nunca se borra, aunque su fecha sea rara
    assert mini_db.execute("SELECT COUNT(*) AS n FROM odds").fetchone()["n"] == 0


def test_understat_que_no_responde_json_dice_que_llego() -> None:
    with pytest.raises(SourceFormatError) as exc:
        understat.parse_league_data("<!DOCTYPE html>\n<html><title>Just a moment...</title>")
    message = str(exc.value)
    assert "página HTML" in message
    assert "Just a moment" in message  # el principio de la respuesta, para diagnosticar
    with pytest.raises(SourceFormatError, match="no es JSON"):
        understat.parse_league_fixtures("Internal Server Error")
    assert understat.parse_league_fixtures(json.dumps({"dates": []})) == []


def test_sin_elo_para_nadie_no_lista_31_nombres() -> None:
    teams = [f"t{i}" for i in range(31)]
    assert _without_elo(teams, 31) == "NINGÚN equipo tiene Elo en la BD"
    assert _without_elo(["malaga"], 31) == "malaga no tiene(n) NINGÚN Elo en la BD"


def test_el_ciclo_avisa_si_la_bd_no_tiene_historico(mini_db, mini_settings, capsys) -> None:
    cli._warn_if_no_history(mini_db, mini_settings)
    out = capsys.readouterr().out
    assert "alaves ingest --historical" in out
    assert str(mini_settings.data.db_path.resolve()) in out
