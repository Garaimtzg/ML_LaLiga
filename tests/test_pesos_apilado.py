"""Selección de pesos del apilado con calibración leave-one-season-out (ADR-022).

El sesgo que corrige: si un componente se calibra sobre el mismo pool con el
que se eligen los pesos (in-sample), parece mejor de lo que generaliza y roba
peso a un componente honesto. La calibración LOSO evalúa los tres fuera de
muestra, de modo que gana el que de verdad predice mejor.
"""

from __future__ import annotations

import numpy as np
import pytest

from alaves_predictor.models import train as train_mod
from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS

XI, C = 0.0019, 1.0


def _season(name, y, dc, gbm, lin):
    return train_mod.SeasonPredictions(
        season=name,
        y_true=list(y),
        dc_by_xi={XI: dc},
        gbm={VARIANT_NO_ODDS: gbm},
        market=np.tile([0.4, 0.3, 0.3], (len(y), 1)),
        linear_by_c={C: lin},
    )


def _sample(p_true, rng):
    return ["H" if r < p_true[0] else ("D" if r < p_true[0] + p_true[1] else "A") for r in rng]


def test_el_componente_honesto_mejor_se_lleva_el_peso():
    """El tercer componente (lineal) es el mejor y honesto: debe dominar el apilado."""
    rng = np.random.default_rng(0)
    preds = []
    for s in range(4):  # 4 temporadas × 400 = pool > umbral de calibración
        n = 400
        p_true = rng.dirichlet([6, 3, 4], size=n)
        y = [
            "H" if u < p_true[i, 0] else ("D" if u < p_true[i, :2].sum() else "A")
            for i, u in enumerate(rng.random(n))
        ]
        # lineal: casi la verdad (mejor y bien calibrado)
        lin = np.clip(p_true + rng.normal(0, 0.02, p_true.shape), 0.02, None)
        lin /= lin.sum(1, keepdims=True)
        # dc y gbm: ruido en torno a las frecuencias base (peores)
        dc = np.clip(np.tile([0.42, 0.28, 0.30], (n, 1)) + rng.normal(0, 0.03, (n, 3)), 0.02, None)
        dc /= dc.sum(1, keepdims=True)
        gbm = np.clip(np.tile([0.40, 0.30, 0.30], (n, 1)) + rng.normal(0, 0.03, (n, 3)), 0.02, None)
        gbm /= gbm.sum(1, keepdims=True)
        preds.append(_season(f"20{18 + s}-{19 + s}", y, dc, gbm, lin))

    _calibrators, weights = train_mod._calibrate_and_weigh(preds, VARIANT_NO_ODDS, 0.05, XI, C)
    # el componente lineal (índice 2) es el mejor => se lleva la mayor parte del peso
    assert weights[2] > 0.5
    assert weights.sum() == pytest.approx(1.0)


def test_loso_calibra_cada_temporada_con_las_demas():
    """La calibración de una temporada NO usa sus propios datos (out-of-fold)."""
    rng = np.random.default_rng(1)
    preds = []
    for s in range(3):
        n = 400
        y = _sample([0.5, 0.25, 0.25], rng.random(n))
        # probabilidades idénticas y sobreconfiadas en las tres temporadas
        raw = np.tile([0.9, 0.05, 0.05], (n, 1))
        preds.append(_season(f"S{s}", y, raw.copy(), raw.copy(), raw.copy()))

    y_oof, comps = train_mod._loso_calibrated(preds, VARIANT_NO_ODDS, XI, C)
    # la isotónica LOSO corrige el 0.9 sobreconfiado hacia la frecuencia real (~0.5)
    assert comps[0][:, 0].mean() == pytest.approx(0.5, abs=0.1)
    assert len(y_oof) == 1200
