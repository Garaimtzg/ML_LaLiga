"""Tests del clasificador LightGBM (variantes con/sin cuotas, orden H/D/A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alaves_predictor.features.build import MARKET_COLS
from alaves_predictor.models import gbm_classifier as gbm


def _train_test(synthetic_features):
    train = synthetic_features[synthetic_features["season"] < "2021-22"]
    test = synthetic_features[synthetic_features["season"] == "2021-22"]
    return train, test


def test_variant_features_excluye_cuotas():
    cols = ["strength_diff", "noise", "imp_home", "imp_draw", "imp_away"]
    assert gbm.variant_features(cols, gbm.VARIANT_WITH_ODDS) == cols
    sin = gbm.variant_features(cols, gbm.VARIANT_NO_ODDS)
    assert sin == ["strength_diff", "noise"]
    assert not set(sin) & set(MARKET_COLS)


def test_to_matrix_convierte_na_en_nan():
    df = pd.DataFrame({"a": pd.array([1.5, pd.NA], dtype="Float64"), "b": [1, 2]})
    matrix = gbm.to_matrix(df, ["a", "b"]).to_numpy()
    assert matrix.dtype == np.float64
    assert matrix[0, 0] == 1.5
    assert np.isnan(matrix[1, 0])


def test_predicciones_validas_y_orden_hda(synthetic_features, model_settings):
    train, test = _train_test(synthetic_features)
    cols = ["strength_diff", "noise", "imp_home", "imp_draw", "imp_away"]
    model = gbm.fit(train, cols, model_settings.models.lightgbm, gbm.VARIANT_WITH_ODDS)
    probs = gbm.predict_proba(model, test)
    assert probs.shape == (len(test), 3)
    assert probs.sum(axis=1) == pytest.approx(np.ones(len(test)))
    # el mejor equipo en casa contra el peor debe ser favorito (orden [H, D, A])
    row = test[(test["home_id"] == "t1") & (test["away_id"] == "t6")]
    if not row.empty:
        p = gbm.predict_proba(model, row)[0]
        assert p[0] > p[2]


def test_la_variante_informada_bate_al_ruido(synthetic_features, model_settings):
    """Con features informativas el log-loss debe ser mejor que adivinar frecuencias."""
    from alaves_predictor.evaluation.metrics import log_loss

    train, test = _train_test(synthetic_features)
    cols = ["strength_diff", "imp_home", "imp_draw", "imp_away"]
    model = gbm.fit(train, cols, model_settings.models.lightgbm, gbm.VARIANT_WITH_ODDS)
    ours = log_loss(list(test["result"]), gbm.predict_proba(model, test))
    freqs = train["result"].value_counts(normalize=True)
    base = np.tile([freqs.get(o, 0.0) for o in ("H", "D", "A")], (len(test), 1))
    assert ours < log_loss(list(test["result"]), base)
