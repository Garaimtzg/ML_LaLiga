"""Tests de la capa de datos del dashboard (lógica pura, sin Streamlit)."""

from __future__ import annotations

import pandas as pd
import pytest

from alaves_predictor.dashboard import data as dd
from alaves_predictor.models import train as train_mod
from alaves_predictor.simulation.project import project_standings, split_season


@pytest.fixture()
def bundle(synthetic_features, model_settings):
    return train_mod.train_models(synthetic_features, model_settings)


def test_split_season_modo_demo_y_en_curso(synthetic_features):
    season_df = synthetic_features[synthetic_features["season"] == "2021-22"]
    played, remaining = split_season(season_df, from_matchday=6)
    assert (played["matchday"] < 6).all()
    assert (remaining["matchday"] >= 6).all()
    # modo "en curso": sin partidos programados en el sintético, no queda nada por jugar
    played2, remaining2 = split_season(season_df, from_matchday=None)
    assert remaining2.empty and len(played2) == len(season_df)


def test_project_standings_devuelve_proyeccion(bundle, synthetic_features, model_settings):
    proj = project_standings(
        bundle, synthetic_features, model_settings, "2021-22", from_matchday=6, n=500
    )
    assert proj is not None
    assert proj.n_remaining > 0
    assert set(proj.teams) == set(synthetic_features["home_id"]) | set(
        synthetic_features["away_id"]
    )
    # sin nada por simular => None
    assert (
        project_standings(bundle, synthetic_features, model_settings, "2021-22", from_matchday=None)
        is None
    )


def test_standings_table_ordenada(synthetic_features, model_settings):
    season_df = synthetic_features[synthetic_features["season"] == "2021-22"]
    table = dd.standings_table(season_df, model_settings)
    assert list(table["Pos"]) == list(range(1, len(table) + 1))
    # ordenada por puntos descendente
    assert table["Pts"].is_monotonic_decreasing


def test_projection_table_y_heatmap(bundle, synthetic_features, model_settings):
    proj = project_standings(
        bundle, synthetic_features, model_settings, "2021-22", from_matchday=6, n=500
    )
    table = dd.projection_table(proj, model_settings)
    assert {"Equipo", "Pts esperados", "P(descenso)"} <= set(table.columns)
    # ordenada por posición esperada
    assert table["Pos esperada"].is_monotonic_increasing
    heat = dd.position_heatmap(proj, model_settings)
    # cada fila (equipo) es una distribución que suma 1
    assert heat.sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_matchday_predictions_con_nombres(bundle, synthetic_features, model_settings):
    def predict(rows):
        return bundle.predict_matches(rows, "sin_cuotas")

    preds = dd.matchday_predictions(predict, synthetic_features, model_settings, "2021-22", 1)
    assert not preds.empty
    assert {"Local", "Visitante", "Predicho", "Real"} <= set(preds.columns)
    assert preds["p_home"].between(0, 1).all()


def test_focus_timeline(model_settings):
    # frame mínimo con las columnas reales del feature store que lee focus_timeline
    model_settings.focus_team = "alaves"
    df = pd.DataFrame(
        [
            {  # alaves de local en la J1
                "season": "2025-26",
                "matchday": 1,
                "date": "2025-08-15",
                "home_id": "alaves",
                "away_id": "getafe",
                "elo_clubelo_home": 1650.0,
                "elo_clubelo_away": 1600.0,
                "home_xg": 1.8,
                "away_xg": 0.9,
                "home_points_ma5": 1.4,
                "away_points_ma5": 1.1,
            },
            {  # alaves de visitante en la J2
                "season": "2025-26",
                "matchday": 2,
                "date": "2025-08-22",
                "home_id": "barcelona",
                "away_id": "alaves",
                "elo_clubelo_home": 1720.0,
                "elo_clubelo_away": 1655.0,
                "home_xg": 1.2,
                "away_xg": 1.5,
                "home_points_ma5": 1.6,
                "away_points_ma5": 1.5,
            },
        ]
    )
    tl = dd.focus_timeline(df, model_settings, "2025-26")
    assert len(tl) == 2
    assert {"elo", "xg_favor", "xg_contra", "forma_pts_ma5", "rival"} <= set(tl.columns)
    # J1 (local): su Elo es el de casa, xG a favor = home_xg
    assert tl.iloc[0]["elo"] == 1650.0 and tl.iloc[0]["xg_favor"] == 1.8
    # J2 (visitante): su Elo es el de fuera, xG a favor = away_xg, xG en contra = home_xg
    assert tl.iloc[1]["elo"] == 1655.0
    assert tl.iloc[1]["xg_favor"] == 1.5 and tl.iloc[1]["xg_contra"] == 1.2
    assert tl.iloc[1]["rival"] == "FC Barcelona"


def test_registry_y_prediction_log(bundle, model_settings, mini_db):
    train_mod.register_model(mini_db, model_settings, bundle)
    reg = dd.model_registry_table(mini_db)
    assert len(reg) == 1
    assert reg.iloc[0]["Promocionado"] == "sí"
    # sin predicciones persistidas, el log está vacío
    assert dd.prediction_log(mini_db, model_settings).empty


def test_adr_list(tmp_path):
    (tmp_path / "001-primero.md").write_text("# ADR-001 — Primero\n\ncuerpo", encoding="utf-8")
    (tmp_path / "010-decimo.md").write_text("# ADR-010 — Décimo\n", encoding="utf-8")
    (tmp_path / "notas.md").write_text("no es un ADR", encoding="utf-8")
    adrs = dd.adr_list(tmp_path)
    assert list(adrs["num"]) == [1, 10]  # ordenado, ignora notas.md
    assert "Primero" in adrs.iloc[0]["titulo"]
