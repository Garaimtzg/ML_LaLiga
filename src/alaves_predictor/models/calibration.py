"""Calibración isotónica por clase (SPEC §6.3, ADR-017).

Un modelo puede ordenar bien los partidos y aun así dar probabilidades
desajustadas (decir "60 %" cuando la frecuencia real es 52 %). La regresión
isotónica aprende, sobre predicciones de validación TEMPORAL (nunca del propio
entrenamiento), la corrección monótona probabilidad_predicha → frecuencia_real
de cada clase, y luego se renormaliza a suma 1.

La verificación es la tabla de fiabilidad (`reliability_table`): por bin de
probabilidad predicha, comparar la media predicha con la frecuencia observada.
El informe de backtest la incluye en forma numérica; el diagrama gráfico llega
con el dashboard (F6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from alaves_predictor.evaluation.metrics import OUTCOME_ORDER

# Suelo de probabilidad tras calibrar: la isotónica es una función escalonada
# y puede devolver 0 exacto en los extremos, lo que revienta el log-loss con
# un solo acierto del resultado "imposible". En fútbol ningún 1X2 baja en la
# práctica del 1 %, así que el suelo no distorsiona y protege la métrica.
_FLOOR = 0.01


def _one_hot(y_true: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(OUTCOME_ORDER)}
    out = np.zeros((len(y_true), len(OUTCOME_ORDER)))
    for i, label in enumerate(y_true):
        out[i, index[label]] = 1.0
    return out


# Mínimo de muestras para fiarse de la isotónica: por debajo sobreajusta (una
# función escalonada con pocos puntos memoriza el ruido) y, si sus pesos se
# eligen sobre ese mismo pool, generaliza mal. Con menos, se pasa el crudo.
_MIN_CALIBRATION_SAMPLES = 300


def _identity_calibrators() -> list[IsotonicRegression]:
    """Calibradores que devuelven la probabilidad tal cual (recta y=x)."""
    anchor = np.array([0.0, 1.0])
    out = []
    for _ in OUTCOME_ORDER:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(anchor, anchor)
        out.append(iso)
    return out


def fit_isotonic(probs: np.ndarray, y_true: list[str]) -> list[IsotonicRegression]:
    """Ajusta un calibrador por clase (orden H/D/A) sobre predicciones de validación.

    Con menos de `_MIN_CALIBRATION_SAMPLES` puntos devuelve calibradores
    identidad: no hay muestra para calibrar con seguridad (ADR-020).
    """
    if len(y_true) < _MIN_CALIBRATION_SAMPLES:
        return _identity_calibrators()
    targets = _one_hot(y_true)
    calibrators = []
    for k in range(len(OUTCOME_ORDER)):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(np.asarray(probs, dtype=float)[:, k], targets[:, k])
        calibrators.append(iso)
    return calibrators


def apply_isotonic(calibrators: list[IsotonicRegression], probs: np.ndarray) -> np.ndarray:
    """Aplica los calibradores y renormaliza cada fila a suma 1."""
    probs = np.asarray(probs, dtype=float)
    out = np.column_stack([calibrators[k].predict(probs[:, k]) for k in range(len(calibrators))])
    out = np.clip(out, _FLOOR, None)
    return out / out.sum(axis=1, keepdims=True)


def reliability_table(y_true: list[str], probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Tabla de fiabilidad por clase: media predicha vs frecuencia observada por bin.

    En un modelo bien calibrado ambas columnas coinciden (dentro del ruido
    muestral de `n` por bin).
    """
    probs = np.asarray(probs, dtype=float)
    targets = _one_hot(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for k, outcome in enumerate(OUTCOME_ORDER):
        bins = np.clip(np.digitize(probs[:, k], edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            mask = bins == b
            if not mask.any():
                continue
            rows.append(
                {
                    "clase": outcome,
                    "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                    "n": int(mask.sum()),
                    "prob_media_predicha": float(probs[mask, k].mean()),
                    "frecuencia_observada": float(targets[mask, k].mean()),
                }
            )
    return pd.DataFrame(rows)
