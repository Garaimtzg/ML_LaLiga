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
