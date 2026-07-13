"""Clasificador LightGBM multiclase 1X2 (SPEC §6.3, ADR-016).

Dos variantes sobre el mismo feature set v1 (SPEC §4.1):
- "con_cuotas": incluye las probabilidades implícitas de apertura — el techo
  de rendimiento (las cuotas contienen casi toda la información pública).
- "sin_cuotas": las excluye — es la variante que se interpreta con SHAP (F5),
  porque revela qué variables futbolísticas importan.

Los hiperparámetros v1 viven en config/settings.toml ([models.lightgbm]):
valores conservadores documentados, sin búsqueda automática todavía (optuna
es opcional en SPEC §6.3 y llegará, si hace falta, tras el primer backtest).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from alaves_predictor.config import LightGBMConfig
from alaves_predictor.evaluation.metrics import OUTCOME_ORDER
from alaves_predictor.features.build import MARKET_COLS

# Variantes de entrenamiento (SPEC §4.1): las cuotas de apertura son las
# únicas features de mercado; quitarlas define la variante interpretable.
VARIANT_WITH_ODDS = "con_cuotas"
VARIANT_NO_ODDS = "sin_cuotas"
VARIANTS = (VARIANT_WITH_ODDS, VARIANT_NO_ODDS)


@dataclass
class GBMModel:
    classifier: LGBMClassifier
    feature_names: list[str]
    variant: str


def variant_features(all_features: list[str], variant: str) -> list[str]:
    """Columnas de entrada de cada variante: sin_cuotas excluye las de mercado."""
    if variant == VARIANT_WITH_ODDS:
        return list(all_features)
    return [c for c in all_features if c not in MARKET_COLS]


def to_matrix(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Entrada del modelo: float64 con NaN donde falte el dato (LightGBM los maneja).

    Se devuelve como DataFrame para que LightGBM asocie nombres de columna
    idénticos en fit y en predict.
    """
    values = df[feature_names].astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
    return pd.DataFrame(values, columns=feature_names)


def fit(
    train: pd.DataFrame, feature_names: list[str], cfg: LightGBMConfig, variant: str
) -> GBMModel:
    """Entrena la variante sobre partidos con resultado conocido."""
    classifier = LGBMClassifier(
        objective="multiclass",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        colsample_bytree=cfg.feature_fraction,
        subsample=cfg.bagging_fraction,
        subsample_freq=cfg.bagging_freq,
        reg_lambda=cfg.lambda_l2,
        random_state=42,  # semilla fija del proyecto (CLAUDE.md §2)
        verbosity=-1,
    )
    classifier.fit(to_matrix(train, feature_names), train["result"])
    return GBMModel(classifier=classifier, feature_names=feature_names, variant=variant)


def predict_proba(model: GBMModel, df: pd.DataFrame) -> np.ndarray:
    """Probabilidades en el orden canónico [H, D, A] del proyecto."""
    raw = model.classifier.predict_proba(to_matrix(df, model.feature_names))
    class_index = {c: i for i, c in enumerate(model.classifier.classes_)}
    return raw[:, [class_index[o] for o in OUTCOME_ORDER]]
