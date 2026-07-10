"""Métricas probabilísticas con valores conocidos (SPEC §11: tests obligatorios)."""

import math

import numpy as np
import pytest

from alaves_predictor.evaluation.metrics import (
    accuracy,
    brier_score,
    evaluate,
    log_loss,
    rps,
)


def test_log_loss_prediccion_perfecta_y_uniforme() -> None:
    perfect = np.array([[1.0, 0.0, 0.0]])
    assert log_loss(["H"], perfect) == pytest.approx(0.0, abs=1e-9)
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert log_loss(["D"], uniform) == pytest.approx(math.log(3))


def test_brier_resuelto_a_mano() -> None:
    # p=(0.5,0.3,0.2), sale H: (0.5−1)² + 0.3² + 0.2² = 0.25+0.09+0.04 = 0.38
    probs = np.array([[0.5, 0.3, 0.2]])
    assert brier_score(["H"], probs) == pytest.approx(0.38)


def test_rps_resuelto_a_mano() -> None:
    # p=(0.5,0.3,0.2), sale H: acumuladas p=(0.5,0.8), o=(1,1)
    # RPS = ((0.5−1)² + (0.8−1)²) / 2 = (0.25 + 0.04)/2 = 0.145
    probs = np.array([[0.5, 0.3, 0.2]])
    assert rps(["H"], probs) == pytest.approx(0.145)


def test_rps_penaliza_mas_equivocarse_de_lado() -> None:
    """Predecir victoria local cuando gana el visitante debe costar más (RPS)
    que la misma masa puesta en el empate — el Brier no distingue, el RPS sí."""
    favors_home = np.array([[0.6, 0.2, 0.2]])
    favors_draw = np.array([[0.2, 0.6, 0.2]])
    assert rps(["A"], favors_home) > rps(["A"], favors_draw)
    assert brier_score(["A"], favors_home) == pytest.approx(brier_score(["A"], favors_draw))


def test_accuracy_y_evaluate() -> None:
    probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    assert accuracy(["H", "H"], probs) == pytest.approx(0.5)
    result = evaluate(["H", "A"], probs)
    assert set(result) == {"log_loss", "brier", "rps", "accuracy"}
    assert result["accuracy"] == pytest.approx(1.0)
