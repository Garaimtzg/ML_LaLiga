"""Tests de explicabilidad: SHAP (TreeSHAP nativo), dependencia parcial y ablation."""

from __future__ import annotations

import numpy as np
import pytest

from alaves_predictor.explain import ablation, importance
from alaves_predictor.models import gbm_classifier as gbm


@pytest.fixture()
def trained_no_odds(synthetic_features, model_settings):
    """GBM sin cuotas entrenado sobre el sintético, con una feature muy informativa."""
    train = synthetic_features[synthetic_features["season"] < "2021-22"]
    cols = ["strength_diff", "noise", "elo_clubelo_diff"]
    model = gbm.fit(train, cols, model_settings.models.lightgbm, gbm.VARIANT_NO_ODDS)
    return model, train


def test_shap_tiene_forma_muestras_variables_clases(trained_no_odds):
    model, train = trained_no_odds
    contribs = importance.shap_contributions(model, train.head(20))
    assert contribs.shape == (20, len(model.feature_names), 3)


def test_shap_reconstruye_la_prediccion(trained_no_odds):
    """SHAP + valor base = log-odds de la predicción (propiedad de TreeSHAP)."""
    model, train = trained_no_odds
    rows = train.head(10)
    matrix = gbm.to_matrix(rows, model.feature_names)
    raw = np.asarray(model.classifier.booster_.predict(matrix, pred_contrib=True))
    n_feat, n_cls = len(model.feature_names), len(model.classifier.classes_)
    reshaped = raw.reshape(raw.shape[0], n_cls, n_feat + 1)
    # suma de contribuciones + base = raw_score del booster para cada clase
    raw_scores = model.classifier.booster_.predict(matrix, raw_score=True)
    assert reshaped.sum(axis=2) == pytest.approx(raw_scores, abs=1e-5)


def test_importancia_global_ordenada_y_completa(trained_no_odds):
    model, train = trained_no_odds
    imp = importance.global_importance(model, train)
    assert list(imp["feature"]) and set(imp["feature"]) == set(model.feature_names)
    # ordenada de mayor a menor
    assert imp["mean_abs_shap"].is_monotonic_decreasing
    # la señal informativa (strength/elo, colineales) pesa más que el ruido
    informative = imp[imp["feature"].isin(["strength_diff", "elo_clubelo_diff"])][
        "mean_abs_shap"
    ].sum()
    noise = imp.loc[imp["feature"] == "noise", "mean_abs_shap"].iloc[0]
    assert informative > noise


def test_dependencia_parcial_es_distribucion_valida(trained_no_odds):
    model, train = trained_no_odds
    pdp = importance.partial_dependence(model, train, "strength_diff", grid_size=8)
    assert len(pdp) == 8
    totals = pdp[["H", "D", "A"]].sum(axis=1)
    assert totals.to_numpy() == pytest.approx(np.ones(8))


def test_plots_generan_png(trained_no_odds, tmp_path):
    model, train = trained_no_odds
    imp = importance.global_importance(model, train)
    bar = importance.plot_bar(imp, tmp_path / "bar.png")
    bee = importance.plot_beeswarm(model, train, imp, tmp_path / "bee.png")
    pdp = importance.plot_partial_dependence(model, train, ["strength_diff"], tmp_path / "pdp.png")
    for p in (bar, bee, pdp):
        assert p.exists() and p.stat().st_size > 0


def test_block_columns_agrupa_por_patron():
    cols = ["elo_clubelo_diff", "home_xg_for_ma5", "rest_days", "derby", "noise"]
    assert ablation.block_columns("elo", cols) == ["elo_clubelo_diff"]
    assert ablation.block_columns("xg", cols) == ["home_xg_for_ma5"]
    assert ablation.block_columns("descanso", cols) == ["rest_days"]
    assert ablation.block_columns("contexto", cols) == ["derby"]


def test_ablation_devuelve_completo_y_bloques(synthetic_features, model_settings):
    rows = ablation.run_ablation(synthetic_features, model_settings, n_test_seasons=2)
    assert rows[0].block == "(modelo completo)"
    assert rows[0].delta_vs_full == 0.0
    names = {r.block for r in rows}
    # los bloques presentes en el sintético: elo (elo_clubelo_diff) y contexto (matchday)
    assert {"elo", "contexto"} <= names
    for r in rows[1:]:
        assert r.n_features_removed > 0
        assert np.isfinite(r.log_loss)
