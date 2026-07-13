"""Ablation study: aportación real de cada bloque de variables (SPEC §7.4).

Se mide el log-loss walk-forward de la variante sin cuotas quitando bloques
enteros de features y comparando con el modelo completo. Un bloque "aporta" si
quitarlo empeora el log-loss (delta positivo). Convierte "¿sirve el xG?" en una
pregunta empírica que responde el propio pipeline (SPEC §4.1).

Se usa el LightGBM sin cuotas (no el ensemble completo) por coste: aísla la
contribución de las features al clasificador, que es lo que interesa medir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alaves_predictor.config import Settings
from alaves_predictor.evaluation import metrics
from alaves_predictor.features.build import feature_columns
from alaves_predictor.models import gbm_classifier
from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS

# Bloques de features por patrón de nombre (sobre el feature set v1, ADR-012).
# Cada bloque agrupa columnas cuyo nombre contiene alguno de sus patrones.
FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "elo": ("elo_",),
    "xg": ("xg", "g_minus_xg"),  # rendimiento subyacente
    "forma": ("points_ma", "goals_for_ma", "goals_against_ma", "win_streak", "loss_streak"),
    "descanso": ("rest_days",),
    "contexto": ("matchday", "month", "no_crowd", "derby", "promoted", "h2h"),
}


@dataclass
class AblationRow:
    block: str
    n_features_removed: int
    log_loss: float
    delta_vs_full: float  # log_loss(sin bloque) − log_loss(completo); >0 = el bloque aporta


def block_columns(block: str, all_features: list[str]) -> list[str]:
    """Columnas del feature set que pertenecen a un bloque (por patrón de nombre)."""
    patterns = FEATURE_BLOCKS[block]
    return [c for c in all_features if any(p in c for p in patterns)]


def _walkforward_logloss(
    features: pd.DataFrame, cols: list[str], settings: Settings, n_test_seasons: int
) -> float:
    """Log-loss agrupado del GBM sin cuotas, walk-forward sobre las últimas temporadas."""
    seasons = sorted(set(features["season"]))
    test_seasons = seasons[-n_test_seasons:]
    y_all: list[str] = []
    probs_all = []
    for season in test_seasons:
        train = features[features["season"] < season]
        test = features[features["season"] == season]
        if train.empty or test.empty:
            continue
        model = gbm_classifier.fit(train, cols, settings.models.lightgbm, VARIANT_NO_ODDS)
        probs_all.append(gbm_classifier.predict_proba(model, test))
        y_all.extend(test["result"])
    return metrics.log_loss(y_all, np.vstack(probs_all))


def run_ablation(
    features: pd.DataFrame, settings: Settings, n_test_seasons: int = 3
) -> list[AblationRow]:
    """Log-loss del modelo completo y de cada variante sin un bloque (SPEC §7.4)."""
    finished = features[features["result"].notna()].copy()
    all_features = gbm_classifier.variant_features(feature_columns(finished), VARIANT_NO_ODDS)
    full_loss = _walkforward_logloss(finished, all_features, settings, n_test_seasons)

    rows = [AblationRow("(modelo completo)", 0, full_loss, 0.0)]
    for block in FEATURE_BLOCKS:
        removed = block_columns(block, all_features)
        kept = [c for c in all_features if c not in removed]
        if not removed or not kept:
            continue
        loss = _walkforward_logloss(finished, kept, settings, n_test_seasons)
        rows.append(AblationRow(block, len(removed), loss, loss - full_loss))
    # los bloques que más aportan primero (mayor empeoramiento al quitarlos)
    head, tail = rows[0], sorted(rows[1:], key=lambda r: r.delta_vs_full, reverse=True)
    return [head, *tail]
