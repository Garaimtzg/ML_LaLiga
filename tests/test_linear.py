"""Tests del componente lineal Elo+forma del ensemble sin cuotas (ADR-020)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alaves_predictor.evaluation.metrics import log_loss
from alaves_predictor.models import linear


def _dataset(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Partidos con las columnas reales que consume el lineal y señal conocida."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1.2, n)  # diferencia de nivel local−visitante
    p_home = 1 / (1 + np.exp(-(0.4 + 0.9 * strength)))
    results = ["H" if rng.random() < ph else ("A" if rng.random() < 0.55 else "D") for ph in p_home]
    return pd.DataFrame(
        {
            "result": results,
            "elo_clubelo_home": 1500 + 100 * strength,
            "elo_clubelo_away": 1500 - 100 * strength,
            "elo_internal_home_pre": 1500 + 80 * strength,
            "elo_internal_away_pre": 1500 - 80 * strength,
            "home_points_ma10": 1.5 + 0.3 * strength,
            "away_points_ma10": 1.5 - 0.3 * strength,
            "home_xg_for_ma10": 1.4 + 0.2 * strength,
            "away_xg_for_ma10": 1.4 - 0.2 * strength,
            "home_xg_against_ma10": 1.4 - 0.1 * strength,
            "away_xg_against_ma10": 1.4 + 0.1 * strength,
            "home_rest_days": rng.integers(3, 8, n).astype(float),
            "away_rest_days": rng.integers(3, 8, n).astype(float),
        }
    )


def test_predicciones_validas_y_orden_hda():
    train, test = _dataset(seed=1), _dataset(seed=2)
    model = linear.fit_linear(train)
    probs = linear.predict_linear(model, test)
    assert probs.shape == (len(test), 3)
    assert probs.sum(axis=1) == pytest.approx(np.ones(len(test)))
    # partido con local muy superior => P(H) alta
    strong = test.iloc[[int(np.argmax(test["elo_clubelo_home"] - test["elo_clubelo_away"]))]]
    p = linear.predict_linear(model, strong)[0]
    assert p[0] > p[2]


def test_bate_a_las_frecuencias_del_entrenamiento():
    train, test = _dataset(seed=1), _dataset(seed=2)
    model = linear.fit_linear(train)
    ours = log_loss(list(test["result"]), linear.predict_linear(model, test))
    freqs = train["result"].value_counts(normalize=True)
    base = np.tile([freqs.get(o, 0.0) for o in ("H", "D", "A")], (len(test), 1))
    assert ours < log_loss(list(test["result"]), base)


def test_columnas_ausentes_no_rompen():
    """Si faltan columnas (p. ej. sin Elo interno), esa feature se anula, no falla."""
    train = _dataset(seed=1).drop(columns=["elo_internal_home_pre", "elo_internal_away_pre"])
    model = linear.fit_linear(train)
    probs = linear.predict_linear(model, train.head(5))
    assert probs.shape == (5, 3)
    assert probs.sum(axis=1) == pytest.approx(np.ones(5))
