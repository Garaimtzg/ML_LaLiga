"""Capa de BD: esquema, upserts idempotentes y fusión de fuentes en match_stats."""

from alaves_predictor.etl import db
from alaves_predictor.etl.teams import TeamRegistry

from .conftest import MINI_TEAMS


def _seed(conn) -> None:
    TeamRegistry(MINI_TEAMS).seed_db(conn)


def test_esquema_crea_todas_las_tablas(mini_db) -> None:
    counts = db.table_counts(mini_db)
    expected = {
        "teams",
        "matches",
        "match_stats",
        "odds",
        "elo",
        "squad_values",
        "injuries",
        "features",
        "predictions",
        "model_registry",
    }
    assert expected <= set(counts)


def test_upsert_de_partido_es_idempotente(mini_db) -> None:
    _seed(mini_db)
    row = {
        "match_id": "2018-19_alaves_barcelona",
        "season": "2018-19",
        "matchday": 1,
        "date": "2018-08-18",
        "home_id": "alaves",
        "away_id": "barcelona",
        "home_goals": 1,
        "away_goals": 2,
        "status": "finished",
        "source": "football-data",
        "fetched_at": "2026-01-01T00:00:00",
    }
    db.upsert(mini_db, "matches", row, key_cols=["match_id"])
    db.upsert(mini_db, "matches", row, key_cols=["match_id"])
    assert mini_db.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"] == 1


def test_upsert_no_pisa_con_nulos(mini_db) -> None:
    """Dos fuentes completan columnas distintas de la misma fila sin pisarse."""
    _seed(mini_db)
    db.upsert(
        mini_db,
        "matches",
        {
            "match_id": "m1",
            "season": "2018-19",
            "matchday": None,
            "date": "2018-08-18",
            "home_id": "alaves",
            "away_id": "barcelona",
            "home_goals": 1,
            "away_goals": 2,
            "status": "finished",
            "source": "football-data",
            "fetched_at": "t",
        },
        key_cols=["match_id"],
    )
    stats = {
        "match_id": "m1",
        "team_id": "alaves",
        "is_home": 1,
        "shots": 10,
        "xg": None,
        "source": "football-data",
        "fetched_at": "t",
    }
    db.upsert(mini_db, "match_stats", stats, key_cols=["match_id", "team_id"])
    # Segunda fuente: solo xg; shots va a None en el dict pero no debe borrarse.
    db.upsert(
        mini_db,
        "match_stats",
        {
            "match_id": "m1",
            "team_id": "alaves",
            "is_home": 1,
            "shots": None,
            "xg": 1.23,
            "source": "understat",
            "fetched_at": "t2",
        },
        key_cols=["match_id", "team_id"],
    )
    row = mini_db.execute("SELECT shots, xg FROM match_stats").fetchone()
    assert row["shots"] == 10  # preservado
    assert row["xg"] == 1.23  # añadido


def test_merge_source_tag() -> None:
    assert db.merge_source_tag(None, "football-data") == "football-data"
    assert db.merge_source_tag("football-data", "understat") == "football-data+understat"
    assert db.merge_source_tag("football-data+understat", "understat") == (
        "football-data+understat"
    )
