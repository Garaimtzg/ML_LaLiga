"""Tests de la calibración isotónica y la tabla de fiabilidad."""

from __future__ import annotations

import numpy as np
import pytest

from alaves_predictor.evaluation.metrics import log_loss
from alaves_predictor.models import calibration


def _overconfident_dataset(n: int = 600, seed: int = 42):
    """Predicciones sobreconfiadas a propósito: dicen 0.8 cuando la real es 0.55."""
    rng = np.random.default_rng(seed)
    y_true = ["H" if rng.random() < 0.55 else "A" for _ in range(n)]
    probs = np.tile([0.8, 0.05, 0.15], (n, 1))
    return probs, y_true


def test_la_calibracion_corrige_la_sobreconfianza():
    probs, y_true = _overconfident_dataset()
    calibrators = calibration.fit_isotonic(probs, y_true)
    calibrated = calibration.apply_isotonic(calibrators, probs)
    # el 0.8 sobreconfiado baja hacia la frecuencia real (~0.55)
    assert calibrated[0, 0] == pytest.approx(0.55, abs=0.05)
    assert log_loss(y_true, calibrated) < log_loss(y_true, probs)


def test_apply_devuelve_distribuciones_validas():
    probs, y_true = _overconfident_dataset()
    calibrators = calibration.fit_isotonic(probs, y_true)
    rng = np.random.default_rng(0)
    fresh = rng.dirichlet(np.ones(3), size=50)
    calibrated = calibration.apply_isotonic(calibrators, fresh)
    assert calibrated.shape == (50, 3)
    assert calibrated.sum(axis=1) == pytest.approx(np.ones(50))
    assert (calibrated > 0).all()  # nunca ceros exactos (protección de logs)


def test_la_isotonica_preserva_el_orden():
    probs, y_true = _overconfident_dataset()
    # añadimos variedad de probabilidades para que la isotónica tenga curva
    rng = np.random.default_rng(1)
    varied = rng.dirichlet(np.ones(3), size=len(y_true))
    calibrators = calibration.fit_isotonic(varied, y_true)
    grid = np.linspace(0, 1, 21)
    for iso in calibrators:
        mapped = iso.predict(grid)
        assert (np.diff(mapped) >= -1e-12).all()  # monótona no decreciente


def test_reliability_table_agrupa_bien():
    probs = np.array([[0.85, 0.10, 0.05], [0.85, 0.10, 0.05], [0.15, 0.10, 0.75]])
    y_true = ["H", "A", "A"]
    table = calibration.reliability_table(y_true, probs, n_bins=10)
    # clase H, bin 0.8-0.9: dos predicciones de 0.85, una acertada
    row = table[(table["clase"] == "H") & (table["bin"] == "0.8-0.9")].iloc[0]
    assert row["n"] == 2
    assert row["prob_media_predicha"] == pytest.approx(0.85)
    assert row["frecuencia_observada"] == pytest.approx(0.5)
