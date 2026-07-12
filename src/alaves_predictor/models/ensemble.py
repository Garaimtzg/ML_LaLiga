"""Ensemble Dixon-Coles + LightGBM: media ponderada de probabilidades (SPEC §6.3).

El peso del Dixon-Coles se elige por búsqueda en rejilla [0, 1] minimizando el
log-loss sobre la validación temporal (nunca sobre el entrenamiento). Decisión
del SPEC: el ensemble suele batir a cada modelo por separado y da robustez si
uno degenera — con peso 0 o 1 la rejilla recupera el mejor modelo individual,
así que el ensemble nunca es peor en validación que sus componentes.
"""

from __future__ import annotations

import numpy as np

from alaves_predictor.evaluation.metrics import log_loss


def blend(p_dc: np.ndarray, p_gbm: np.ndarray, dc_weight: float) -> np.ndarray:
    """Media ponderada: dc_weight·Dixon-Coles + (1−dc_weight)·LightGBM."""
    return dc_weight * np.asarray(p_dc, dtype=float) + (1.0 - dc_weight) * np.asarray(
        p_gbm, dtype=float
    )


def optimal_weight(
    p_dc: np.ndarray, p_gbm: np.ndarray, y_true: list[str], step: float = 0.05
) -> float:
    """Peso del Dixon-Coles que minimiza el log-loss en validación (rejilla)."""
    weights = np.arange(0.0, 1.0 + step / 2, step)
    losses = [log_loss(y_true, blend(p_dc, p_gbm, w)) for w in weights]
    return float(weights[int(np.argmin(losses))])


def blend_many(components: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Media ponderada de N conjuntos de probabilidades (los pesos suman 1)."""
    out = np.zeros_like(np.asarray(components[0], dtype=float))
    for probs, w in zip(components, weights, strict=True):
        out += float(w) * np.asarray(probs, dtype=float)
    return out


def _simplex_grid(n: int, step: float) -> list[tuple[float, ...]]:
    """Todas las combinaciones de n pesos ≥ 0 en rejilla de paso `step` que suman 1."""
    ticks = int(round(1.0 / step))

    def rec(remaining: int, left: int) -> list[tuple[int, ...]]:
        if remaining == 1:
            return [(left,)]
        return [(i, *rest) for i in range(left + 1) for rest in rec(remaining - 1, left - i)]

    return [tuple(i * step for i in combo) for combo in rec(n, ticks)]


def optimal_weights(
    components: list[np.ndarray], y_true: list[str], step: float = 0.05
) -> np.ndarray:
    """Pesos del ensemble apilado (ADR-019): rejilla sobre el símplex por log-loss.

    Con el peso 1 en un solo componente la rejilla recupera ese modelo, así
    que el apilado nunca es peor que su mejor componente en validación.
    """
    best_weights, best_loss = None, np.inf
    for candidate in _simplex_grid(len(components), step):
        loss = log_loss(y_true, blend_many(components, np.array(candidate)))
        if loss < best_loss:
            best_weights, best_loss = candidate, loss
    return np.array(best_weights)
