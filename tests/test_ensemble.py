"""Tests del ensemble ponderado Dixon-Coles + LightGBM."""

from __future__ import annotations

import numpy as np
import pytest

from alaves_predictor.models import ensemble


def test_blend_es_media_ponderada():
    p_dc = np.array([[0.6, 0.2, 0.2]])
    p_gbm = np.array([[0.2, 0.2, 0.6]])
    assert ensemble.blend(p_dc, p_gbm, 0.5)[0] == pytest.approx([0.4, 0.2, 0.4])
    assert ensemble.blend(p_dc, p_gbm, 1.0)[0] == pytest.approx(p_dc[0])
    assert ensemble.blend(p_dc, p_gbm, 0.0)[0] == pytest.approx(p_gbm[0])


def test_optimal_weight_elige_al_modelo_bueno():
    y_true = ["H", "A", "D", "H"]
    perfect = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    awful = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    # si el DC es perfecto, todo el peso al DC; si es horrible, nada
    assert ensemble.optimal_weight(perfect, awful, y_true) == pytest.approx(1.0)
    assert ensemble.optimal_weight(awful, perfect, y_true) == pytest.approx(0.0)


def test_optimal_weights_apilado_de_tres():
    y_true = ["H", "A", "D", "H"]
    perfect = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    awful = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    uniform = np.full((4, 3), 1 / 3)
    weights = ensemble.optimal_weights([awful, perfect, uniform], y_true)
    # todo el peso al componente perfecto; el símplex suma 1
    assert weights == pytest.approx([0.0, 1.0, 0.0])
    assert weights.sum() == pytest.approx(1.0)


def test_blend_many_es_media_ponderada():
    a = np.array([[0.6, 0.2, 0.2]])
    b = np.array([[0.2, 0.2, 0.6]])
    c = np.array([[0.2, 0.6, 0.2]])
    out = ensemble.blend_many([a, b, c], np.array([0.5, 0.25, 0.25]))
    assert out[0] == pytest.approx([0.4, 0.3, 0.3])


def test_el_ensemble_nunca_es_peor_que_sus_partes_en_validacion():
    from alaves_predictor.evaluation.metrics import log_loss

    rng = np.random.default_rng(42)
    y_true = ["H" if r < 0.5 else "A" for r in rng.random(200)]
    p_a = rng.dirichlet(np.ones(3), 200)
    p_b = rng.dirichlet(np.ones(3), 200)
    w = ensemble.optimal_weight(p_a, p_b, y_true)
    blended = log_loss(y_true, ensemble.blend(p_a, p_b, w))
    assert blended <= log_loss(y_true, p_a) + 1e-9
    assert blended <= log_loss(y_true, p_b) + 1e-9
