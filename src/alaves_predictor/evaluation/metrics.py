"""Métricas de calidad probabilística para 1X2 (SPEC §6.5).

Convención: las probabilidades van en columnas ordenadas [H, D, A] (victoria
local, empate, victoria visitante). Ese ORDEN importa para el RPS, que
penaliza más equivocarse "de lado" (predecir H cuando sale A) que fallar
entre resultados adyacentes — por eso es el estándar en fútbol.
"""

from __future__ import annotations

import numpy as np

OUTCOME_ORDER = ["H", "D", "A"]

_EPS = 1e-15


def _one_hot(y_true: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(OUTCOME_ORDER)}
    out = np.zeros((len(y_true), len(OUTCOME_ORDER)))
    for i, label in enumerate(y_true):
        out[i, index[label]] = 1.0
    return out


def log_loss(y_true: list[str], probs: np.ndarray) -> float:
    """Log-loss multiclase (métrica principal). Menor es mejor."""
    clipped = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0)
    return float(-np.mean(np.sum(_one_hot(y_true) * np.log(clipped), axis=1)))


def brier_score(y_true: list[str], probs: np.ndarray) -> float:
    """Brier multiclase: media de la distancia cuadrática al resultado real."""
    diff = np.asarray(probs, dtype=float) - _one_hot(y_true)
    return float(np.mean(np.sum(diff**2, axis=1)))


def rps(y_true: list[str], probs: np.ndarray) -> float:
    """Ranked Probability Score: como el Brier, pero sobre las ACUMULADAS.

    Al acumular en el orden H>D>A, colocar masa en el extremo equivocado
    cuesta más que colocarla en el empate — respeta el orden natural del 1X2.
    """
    cum_probs = np.cumsum(np.asarray(probs, dtype=float), axis=1)
    cum_true = np.cumsum(_one_hot(y_true), axis=1)
    # la última columna acumulada siempre es 1 en ambos: se excluye (K-1 términos)
    diff = cum_probs[:, :-1] - cum_true[:, :-1]
    return float(np.mean(np.sum(diff**2, axis=1) / (len(OUTCOME_ORDER) - 1)))


def accuracy(y_true: list[str], probs: np.ndarray) -> float:
    """Acierto del resultado más probable (métrica secundaria; poco informativa sola)."""
    predicted = np.asarray(probs, dtype=float).argmax(axis=1)
    actual = _one_hot(y_true).argmax(axis=1)
    return float(np.mean(predicted == actual))


def evaluate(y_true: list[str], probs: np.ndarray) -> dict[str, float]:
    """Las cuatro métricas de SPEC §6.5 de una vez."""
    return {
        "log_loss": log_loss(y_true, probs),
        "brier": brier_score(y_true, probs),
        "rps": rps(y_true, probs),
        "accuracy": accuracy(y_true, probs),
    }
