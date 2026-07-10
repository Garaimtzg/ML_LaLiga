"""Feature store v1 sobre la mini-liga: corte temporal, forma y anti-leakage.

El test anti-leakage es OBLIGATORIO (CLAUDE.md §5.1): se verifica
empíricamente que alterar un partido futuro no cambia las features de
partidos anteriores.
"""

import pandas as pd
import pytest

from alaves_predictor.etl import db
from alaves_predictor.etl.ingest import ingest_historical, make_match_id
from alaves_predictor.features.build import build_features, feature_columns, persist_features


@pytest.fixture()
def features_df(mini_settings, fake_fetch):
    conn = db.connect(mini_settings.data.db_path)
    try:
        ingest_historical(conn, mini_settings)
        yield build_features(conn, mini_settings), conn, mini_settings
    finally:
        conn.close()


def test_estructura_y_corte_temporal(features_df) -> None:
    df, _conn, _settings = features_df
    assert len(df) == 12  # una fila por partido
    # as_of = día anterior al partido (SPEC §4)
    first = df.sort_values("date").iloc[0]
    assert first["date"] == "2018-08-18"
    assert first["as_of_date"] == "2018-08-17"
    # target coherente con los goles
    assert first["result"] == "A"  # alaves 1-2 barcelona


def test_forma_sin_historia_es_nan_y_luego_correcta(features_df) -> None:
    df, _conn, _settings = features_df
    df = df.sort_values("date").reset_index(drop=True)
    # Jornada 1: nadie tiene historia -> forma NaN (no se inventa)
    j1 = df.iloc[0]
    assert pd.isna(j1["home_points_ma5"])
    # Partido de la J3 alaves-getafe: alaves lleva D(0)+E(1) = 0.5 puntos/partido
    row = df[df["match_id"] == make_match_id("2018-19", "alaves", "getafe")].iloc[0]
    assert row["home_points_ma5"] == pytest.approx(0.5)
    # y el getafe (visitante) lleva D(0)+D(0) = 0.0
    assert row["away_points_ma5"] == pytest.approx(0.0)


def test_elo_clubelo_asof_usa_el_rating_vigente(features_df) -> None:
    df, _conn, _settings = features_df
    # J1 (as_of 2018-08-17): rating vigente del fixture de ClubElo = base (p.ej. 1550 alaves)
    j1 = df[df["match_id"] == make_match_id("2018-19", "alaves", "barcelona")].iloc[0]
    assert j1["elo_clubelo_home"] == pytest.approx(1550.0)
    assert j1["elo_clubelo_away"] == pytest.approx(1950.0)
    # J3 (as_of 2018-08-31): ya rige el tramo 2018-08-20 -> +8
    j3 = df[df["match_id"] == make_match_id("2018-19", "alaves", "getafe")].iloc[0]
    assert j3["elo_clubelo_home"] == pytest.approx(1558.0)


def test_cuotas_implicitas_normalizadas(features_df) -> None:
    df, _conn, _settings = features_df
    sums = df[["imp_home", "imp_draw", "imp_away"]].sum(axis=1)
    assert sums.dropna().apply(lambda s: abs(s - 1.0) < 1e-9).all()
    # el margen se ha eliminado: cada prob < su inversa de cuota original
    j1 = df.sort_values("date").iloc[0]
    assert j1["imp_home"] < 1 / 1.5


def test_derbi_y_ascendidos(features_df) -> None:
    df, _conn, settings = features_df
    # mini config: alaves vs real-sociedad es derbi
    derby_row = df[df["match_id"] == make_match_id("2018-19", "alaves", "real-sociedad")]
    assert derby_row.iloc[0]["derby"] == 1
    other = df[df["match_id"] == make_match_id("2018-19", "alaves", "barcelona")]
    assert other.iloc[0]["derby"] == 0
    # primera temporada de la BD: sin referencia anterior -> promoted 0
    assert (df["promoted_home"] == 0).all()


def test_anti_leakage_alterar_el_futuro_no_cambia_el_pasado(features_df) -> None:
    """OBLIGATORIO (CLAUDE.md §5.1): las features de un partido no cambian si
    se altera cualquier partido POSTERIOR."""
    df_before, conn, settings = features_df
    df_before = df_before.sort_values("date").reset_index(drop=True)

    # Se corrompe el ÚLTIMO partido (2018-09-22): marcador y xG absurdos.
    last_id = make_match_id("2018-19", "real-sociedad", "barcelona")
    conn.execute("UPDATE matches SET home_goals = 9, away_goals = 9 WHERE match_id = ?", (last_id,))
    conn.execute("UPDATE match_stats SET xg = 9.9 WHERE match_id = ?", (last_id,))
    conn.commit()

    df_after = build_features(conn, settings).sort_values("date").reset_index(drop=True)

    cols = feature_columns(df_before)
    earlier = df_before["date"] < "2018-09-22"
    pd.testing.assert_frame_equal(
        df_before.loc[earlier, cols].reset_index(drop=True),
        df_after.loc[earlier, cols].reset_index(drop=True),
        check_dtype=False,
    )
    # y las features del propio partido corrupto tampoco cambian
    # (describen lo PREVIO al partido, no su resultado)
    own_before = df_before[df_before["match_id"] == last_id][cols].reset_index(drop=True)
    own_after = df_after[df_after["match_id"] == last_id][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(own_before, own_after, check_dtype=False)


def test_persistencia_reproducible(features_df) -> None:
    """SPEC §12.4: match_id + feature_set_version recuperan el mismo payload."""
    import json

    df, conn, settings = features_df
    parquet_path = persist_features(conn, df, settings)
    assert parquet_path.exists()
    row = conn.execute(
        "SELECT payload_json, as_of_date FROM features WHERE match_id = ? "
        "AND feature_set_version = ?",
        (make_match_id("2018-19", "alaves", "getafe"), "v1"),
    ).fetchone()
    payload = json.loads(row["payload_json"])
    assert payload["home_points_ma5"] == pytest.approx(0.5)
    assert row["as_of_date"] == "2018-08-31"
    # re-persistir es idempotente
    persist_features(conn, df, settings)
    n = conn.execute("SELECT COUNT(*) AS n FROM features").fetchone()["n"]
    assert n == 12
