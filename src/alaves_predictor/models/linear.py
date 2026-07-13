"""Componente lineal del ensemble sin cuotas: logística Elo + forma (ADR-020).

Multinomial logística regularizada sobre un puñado de diferencias locales −
visitantes con mucha señal y poco riesgo de sobreajuste. Amplía el baseline
Elo logístico (que solo usaba `elo_clubelo_diff`): al contener esa misma
feature con regularización L2 estándar, en la práctica nunca es peor que él,
y la forma reciente y el xG le añaden lo que al Elo le falta.

La ventaja de campo la capturan los interceptos por clase (constante para
todos los partidos), igual que en el baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from alaves_predictor.evaluation.metrics import OUTCOME_ORDER

# Cada feature derivada = resta de dos columnas del feature set v1.
# Pocas y fuertes a propósito: un lineal con 5 señales es muy difícil de romper.
DERIVED_FEATURES: dict[str, tuple[str, str]] = {
    "elo_clubelo_diff": ("elo_clubelo_home", "elo_clubelo_away"),
    "elo_internal_diff": ("elo_internal_home_pre", "elo_internal_away_pre"),
    "ppg10_diff": ("home_points_ma10", "away_points_ma10"),
    "xg_for10_diff": ("home_xg_for_ma10", "away_xg_for_ma10"),
    "xg_against10_diff": ("home_xg_against_ma10", "away_xg_against_ma10"),
    "rest_diff": ("home_rest_days", "away_rest_days"),
}


@dataclass
class LinearModel:
    pipeline: Pipeline  # imputación (mediana) → estandarización → logística


def _design(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de diseño con las diferencias; columna ausente => NaN (se imputa)."""
    out = {}
    for name, (home_col, away_col) in DERIVED_FEATURES.items():
        if home_col in df.columns and away_col in df.columns:
            out[name] = pd.to_numeric(df[home_col], errors="coerce") - pd.to_numeric(
                df[away_col], errors="coerce"
            )
        else:
            out[name] = pd.Series(np.nan, index=df.index)
    design = pd.DataFrame(out, index=df.index)
    # una columna completamente vacía rompería la imputación por mediana
    return design.fillna({c: 0.0 for c in design.columns[design.isna().all()]})


def fit_linear(train: pd.DataFrame, c: float = 1.0) -> LinearModel:
    """Ajusta la logística sobre partidos con resultado conocido.

    `c` es el inverso de la fuerza de regularización L2. Con las features
    estandarizadas, un C bajo encoge en exceso la señal Elo (ADR-021); se
    elige por validación walk-forward.
    """
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=c),
    )
    pipeline.fit(_design(train), train["result"])
    return LinearModel(pipeline=pipeline)


def predict_linear(model: LinearModel, df: pd.DataFrame) -> np.ndarray:
    """P(1X2) en el orden canónico H/D/A."""
    raw = model.pipeline.predict_proba(_design(df))
    classes = model.pipeline.classes_
    class_index = {c: i for i, c in enumerate(classes)}
    return raw[:, [class_index[o] for o in OUTCOME_ORDER]]
