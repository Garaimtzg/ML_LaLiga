"""Próxima jornada y persistencia de predicciones (ADR-031).

Mini-liga de 8 equipos (4 partidos por jornada) construida a mano: basta con
las columnas que mira `next_round` para cubrir los casos reales que rompían la
regla anterior (aplazados, partidos con fecha pasada, jornadas a medias).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from alaves_predictor.dashboard.data import default_matchday
from alaves_predictor.etl import db
from alaves_predictor.etl.teams import TeamRegistry
from alaves_predictor.features.build import feature_payload
from alaves_predictor.models.predict import features_hash, next_round, persist_predictions

SEASON = "2026-27"
PER_ROUND = 4


def _frame(rows: list[tuple[int, str, bool]]) -> pd.DataFrame:
    """(jornada, fecha, jugado) -> frame con las columnas que usa next_round."""
    return pd.DataFrame(
        {
            "match_id": [f"{SEASON}_m{i}" for i in range(len(rows))],
            "season": SEASON,
            "matchday": [md for md, _, _ in rows],
            "date": [d for _, d, _ in rows],
            "result": ["H" if played else None for _, _, played in rows],
        }
    )


def _round(md: int, day: str, played: bool, n: int = PER_ROUND) -> list[tuple[int, str, bool]]:
    return [(md, day, played)] * n


def test_jornada_siguiente_normal() -> None:
    df = _frame(_round(1, "2026-08-15", True) + _round(2, "2026-08-22", False))
    md, rows = next_round(df, SEASON, PER_ROUND, date(2026, 8, 18))
    assert md == 2 and len(rows) == PER_ROUND


def test_un_aplazado_no_secuestra_la_proxima_jornada() -> None:
    """El caso real: un partido de la J3 aplazado a un mes vista."""
    played = _round(1, "2026-08-15", True) + _round(2, "2026-08-22", True)
    j3 = _round(3, "2026-08-29", True, n=3) + [(3, "2026-10-01", False)]  # aplazado
    df = _frame(played + j3 + _round(4, "2026-09-05", False) + _round(5, "2026-09-12", False))
    md, rows = next_round(df, SEASON, PER_ROUND, date(2026, 9, 1))
    assert md == 4  # antes: 3, y se predecía solo el aplazado
    assert list(rows["matchday"]) == [4] * PER_ROUND  # el aplazado, cuando le toque


def test_el_aplazado_se_predice_con_la_jornada_en_la_que_se_juega() -> None:
    j3 = _round(3, "2026-08-29", True, n=3) + [(3, "2026-09-09", False)]  # entre semana
    df = _frame(j3 + _round(4, "2026-09-12", False))
    md, rows = next_round(df, SEASON, PER_ROUND, date(2026, 9, 1))
    assert md == 4
    assert sorted(rows["matchday"]) == [3, 4, 4, 4, 4]


def test_programado_con_fecha_pasada_no_se_predice() -> None:
    """Jugado pero sin resultado publicado aún: predecirlo sería a posteriori."""
    j1 = _round(1, "2026-08-15", True, n=3) + [(1, "2026-08-17", False)]
    df = _frame(j1 + _round(2, "2026-08-22", False))
    md, rows = next_round(df, SEASON, PER_ROUND, date(2026, 8, 19))
    assert md == 2 and len(rows) == PER_ROUND
    assert (rows["date"] >= "2026-08-19").all()


def test_jornada_empezada_sigue_siendo_la_proxima() -> None:
    j2 = [(2, "2026-08-21", True)] + _round(2, "2026-08-23", False, n=3)
    df = _frame(_round(1, "2026-08-15", True) + j2 + _round(3, "2026-08-29", False))
    md, rows = next_round(df, SEASON, PER_ROUND, date(2026, 8, 22))
    assert md == 2 and len(rows) == 3


def test_sin_nada_por_jugar() -> None:
    df = _frame(_round(1, "2026-08-15", True))
    md, rows = next_round(df, SEASON, PER_ROUND, date(2026, 8, 20))
    assert md is None and rows.empty


def test_dashboard_abre_en_la_proxima_jornada(mini_settings) -> None:
    mini_settings.league.teams_per_season = 2 * PER_ROUND
    df = _frame(_round(1, "2026-08-15", True) + _round(2, "2026-08-22", False))
    assert default_matchday(df, mini_settings, SEASON, date(2026, 8, 18)) == 2
    cerrada = _frame(_round(1, "2025-08-15", True)).assign(season="2025-26")
    assert default_matchday(cerrada, mini_settings, "2025-26", date(2026, 8, 18)) is None


def test_persistencia_guarda_el_hash_y_no_pisa_el_parquet(mini_db, mini_settings) -> None:
    TeamRegistry(mini_settings.teams).seed_db(mini_db)
    season = mini_settings.current_season
    match_id = f"{season}_alaves_getafe"
    db.upsert(
        mini_db,
        "matches",
        {
            "match_id": match_id,
            "season": season,
            "matchday": 7,
            "date": "2026-11-14",
            "home_id": "alaves",
            "away_id": "getafe",
            "home_goals": None,
            "away_goals": None,
            "status": "scheduled",
            "source": "test",
            "fetched_at": "2026-11-05T00:00:00+00:00",
        },
        key_cols=["match_id"],
    )
    rows = pd.DataFrame(
        {
            "match_id": [match_id],
            "season": [season],
            "date": ["2026-11-14"],
            "as_of_date": ["2026-11-13"],
            "home_id": ["alaves"],
            "away_id": ["getafe"],
            "matchday": [7],
            "elo_internal_diff": [12.5],
            "home_points_ma5": [float("nan")],
        }
    )
    preds = pd.DataFrame(
        {
            "match_id": [match_id],
            "p_home": [0.45],
            "p_draw": [0.3],
            "p_away": [0.25],
            "pred_result": ["H"],
            "pred_score": ["1-0"],
            "expected_goals_h": [1.4],
            "expected_goals_a": [1.0],
        }
    )
    assert persist_predictions(mini_db, rows, preds, "v-test", mini_settings) == 1
    stored = mini_db.execute("SELECT features_hash FROM predictions").fetchone()["features_hash"]
    expected = features_hash(feature_payload(rows.to_dict(orient="records")[0]))
    assert stored == expected and len(stored) == 16
    assert mini_db.execute("SELECT COUNT(*) AS n FROM features").fetchone()["n"] == 1
    # el Parquet es el feature store completo de `alaves features`: no se toca
    assert not (mini_settings.data.features_dir / "features_v1.parquet").exists()


def test_hash_cambia_si_cambian_las_features() -> None:
    base = {"elo_internal_diff": 12.5, "home_points_ma5": None}
    assert features_hash(base) == features_hash(dict(reversed(base.items())))  # orden da igual
    assert features_hash(base) != features_hash({**base, "elo_internal_diff": 12.6})


@pytest.mark.parametrize("veces", [1, 2])
def test_migracion_anade_features_hash_a_bds_antiguas(tmp_path, veces) -> None:
    conn = db.connect(tmp_path / "vieja.db")
    conn.execute(
        "CREATE TABLE predictions (prediction_id INTEGER PRIMARY KEY, match_id TEXT NOT NULL, "
        "model_version TEXT NOT NULL, created_at TEXT NOT NULL, p_home REAL NOT NULL, "
        "p_draw REAL NOT NULL, p_away REAL NOT NULL, pred_result TEXT NOT NULL)"
    )
    for _ in range(veces):  # idempotente
        db.init_schema(conn)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    assert "features_hash" in columns
    conn.close()
