"""Tests del modo temporada (F7): fixtures, ingesta post-jornada y evaluación."""

from __future__ import annotations

from datetime import UTC, datetime

from alaves_predictor.etl import db
from alaves_predictor.etl.ingest import (
    assign_scheduled_matchdays,
    ingest_fixtures,
    ingest_matchday,
    make_match_id,
)
from alaves_predictor.etl.sources import football_data as fd
from alaves_predictor.etl.teams import TeamRegistry
from alaves_predictor.evaluation.season import evaluate_season, resolved_predictions

FIXTURES_CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
    "SP1,15/08/2026,21:00,Alaves,Getafe,2.10,3.20,3.60\n"
    "E0,15/08/2026,16:00,Arsenal,Chelsea,2.00,3.40,3.80\n"
    "SP1,16/08/2026,18:30,Barcelona,Sociedad,1.55,4.10,5.50\n"
)


def test_parse_fixtures_filtra_division_y_lee_cuotas():
    fixtures = fd.parse_fixtures(FIXTURES_CSV, "SP1")
    assert len(fixtures) == 2  # la fila E0 (Premier) se descarta
    first = fixtures[0]
    assert first.home_team == "Alaves" and first.away_team == "Getafe"
    assert first.match_date.year == 2026
    assert first.odds_open["bet365"] == (2.10, 3.20, 3.60)


def test_parse_fixtures_tolera_bom():
    """football-data a veces sirve el CSV con BOM; no debe romper el parseo."""
    with_bom = "﻿" + FIXTURES_CSV  # BOM Unicode al inicio
    assert len(fd.parse_fixtures(with_bom, "SP1")) == 2
    latin1_bom = "\xef\xbb\xbf" + FIXTURES_CSV  # BOM leído como latin-1
    assert len(fd.parse_fixtures(latin1_bom, "SP1")) == 2


def test_ingest_fixtures_inserta_programados(mini_db, mini_settings, fake_fetch):
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    inserted, unknown = ingest_fixtures(mini_db, mini_settings, registry)
    assert inserted == 3  # Alaves-Getafe, Barcelona-Sociedad, Sociedad-Alaves (del fixture mini)
    assert unknown == []
    rows = mini_db.execute(
        "SELECT status, home_goals FROM matches WHERE status = 'scheduled'"
    ).fetchall()
    assert len(rows) == 3
    assert all(r["home_goals"] is None for r in rows)
    # las cuotas de apertura del programado se guardan
    season = mini_settings.current_season
    mid = make_match_id(season, "alaves", "getafe")
    odds = mini_db.execute("SELECT open_h FROM odds WHERE match_id = ?", (mid,)).fetchone()
    assert odds["open_h"] == 2.10


def test_ingest_fixtures_desde_archivo_local(mini_db, mini_settings, monkeypatch, tmp_path):
    """Sin remoto disponible, el calendario local siembra los programados (F7)."""
    from alaves_predictor.etl.errors import SourceDownloadError

    # el remoto falla (como a principio de temporada); solo hay archivo local
    def _boom(*args, **kwargs):
        raise SourceDownloadError("remoto no disponible aún")

    monkeypatch.setattr("alaves_predictor.etl.ingest.fetch_text", _boom)
    local = tmp_path / "fixtures.csv"
    local.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam\nSP1,16/08/2026,21:00,Alaves,Getafe\n",
        encoding="utf-8",
    )
    mini_settings.sources.football_data.local_fixtures_file = str(local)

    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    inserted, unknown = ingest_fixtures(mini_db, mini_settings, registry)
    assert inserted == 1 and unknown == []
    mid = make_match_id(mini_settings.current_season, "alaves", "getafe")
    row = mini_db.execute("SELECT status FROM matches WHERE match_id = ?", (mid,)).fetchone()
    assert row["status"] == "scheduled"


def test_ingest_fixtures_no_pisa_un_partido_jugado(mini_db, mini_settings, fake_fetch):
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    season = mini_settings.current_season
    mid = make_match_id(season, "alaves", "getafe")
    now = datetime.now(UTC).isoformat()
    db.upsert(
        mini_db,
        "matches",
        {
            "match_id": mid,
            "season": season,
            "matchday": 1,
            "date": "2026-08-15",
            "home_id": "alaves",
            "away_id": "getafe",
            "home_goals": 2,
            "away_goals": 0,
            "status": "finished",
            "source": "test",
            "fetched_at": now,
        },
        key_cols=["match_id"],
    )
    mini_db.commit()
    ingest_fixtures(mini_db, mini_settings, registry)
    # el jugado sigue 'finished' con su marcador, el calendario no lo revierte
    row = mini_db.execute(
        "SELECT status, home_goals FROM matches WHERE match_id = ?", (mid,)
    ).fetchone()
    assert row["status"] == "finished" and row["home_goals"] == 2


def test_ingest_matchday_orquesta_todo(mini_db, mini_settings, fake_fetch):
    report = ingest_matchday(mini_db, mini_settings)
    assert report.season == mini_settings.current_season
    assert report.finished == 12  # los 12 partidos del mini CSV
    assert report.xg_coverage == 12  # FBref aporta xG a todos
    # los fixtures reusan pares ya jugados en la mini-liga: se saltan por estar
    # 'finished' (el resultado manda sobre el calendario). Cero programados nuevos.
    assert report.scheduled == 0
    assert not report.warnings  # todas las fuentes respondieron por fixtures
    counts = dict(
        mini_db.execute("SELECT status, COUNT(*) n FROM matches GROUP BY status").fetchall()
    )
    assert counts["finished"] == 12


def _seed_prediction(conn, settings, home, away, hg, ag, probs, status="finished"):
    season = settings.current_season
    mid = make_match_id(season, home, away)
    now = datetime.now(UTC).isoformat()
    db.upsert(
        conn,
        "matches",
        {
            "match_id": mid,
            "season": season,
            "matchday": 1,
            "date": "2026-08-15",
            "home_id": home,
            "away_id": away,
            "home_goals": hg,
            "away_goals": ag,
            "status": status,
            "source": "test",
            "fetched_at": now,
        },
        key_cols=["match_id"],
    )
    conn.execute(
        "INSERT INTO predictions (match_id, model_version, created_at, p_home, p_draw, p_away, "
        "pred_result) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (mid, "v-test", now, probs[0], probs[1], probs[2], "H"),
    )
    conn.commit()


def test_evaluate_season_sobre_predicciones_resueltas(mini_db, mini_settings):
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    # dos partidos jugados con predicción + uno programado con predicción
    _seed_prediction(mini_db, mini_settings, "alaves", "getafe", 2, 0, (0.6, 0.25, 0.15))
    _seed_prediction(mini_db, mini_settings, "barcelona", "real-sociedad", 1, 1, (0.5, 0.3, 0.2))
    _seed_prediction(
        mini_db,
        mini_settings,
        "getafe",
        "barcelona",
        None,
        None,
        (0.3, 0.3, 0.4),
        status="scheduled",
    )
    perf = evaluate_season(mini_db, mini_settings)
    assert perf.n_resolved == 2
    assert perf.n_pending == 1
    assert 0.0 < perf.metrics["log_loss"] < 5.0
    assert perf.metrics["accuracy"] == 0.5  # acertó el 2-0 (H), falló el empate

    resolved = resolved_predictions(mini_db, mini_settings.current_season)
    assert set(resolved["result"]) == {"H", "D"}


def test_evaluate_season_sin_predicciones(mini_db, mini_settings):
    perf = evaluate_season(mini_db, mini_settings)
    assert perf.n_resolved == 0 and perf.metrics == {}


def test_assign_scheduled_matchdays_agrupa_por_fechas(mini_db, mini_settings):
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    season = mini_settings.current_season
    now = datetime.now(UTC).isoformat()
    # dos "jornadas": 16-17 ago (juntas) y 23 ago (>3 días después)
    partidos = [
        ("alaves", "getafe", "2026-08-16"),
        ("barcelona", "real-sociedad", "2026-08-17"),
        ("getafe", "barcelona", "2026-08-23"),
    ]
    for home, away, date in partidos:
        db.upsert(
            mini_db,
            "matches",
            {
                "match_id": make_match_id(season, home, away),
                "season": season,
                "matchday": None,
                "date": date,
                "home_id": home,
                "away_id": away,
                "home_goals": None,
                "away_goals": None,
                "status": "scheduled",
                "source": "test",
                "fetched_at": now,
            },
            key_cols=["match_id"],
        )
    mini_db.commit()
    assign_scheduled_matchdays(mini_db, season)
    md = {
        r["match_id"]: r["matchday"]
        for r in mini_db.execute("SELECT match_id, matchday FROM matches").fetchall()
    }
    assert md[make_match_id(season, "alaves", "getafe")] == 1
    assert md[make_match_id(season, "barcelona", "real-sociedad")] == 1  # misma jornada
    assert md[make_match_id(season, "getafe", "barcelona")] == 2  # salto de fechas
