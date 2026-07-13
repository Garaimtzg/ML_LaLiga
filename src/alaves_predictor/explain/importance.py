"""Importancia de variables por SHAP (SPEC §7, ADR-024).

Se usa el **TreeSHAP nativo de LightGBM** (`Booster.predict(pred_contrib=True)`),
que da valores SHAP exactos sin depender de la librería `shap` (su cadena
`numba` choca con numpy 2.x y con Python 3.11+; ADR-024). Todo el análisis se
hace sobre la variante **sin cuotas** del modelo, la interpretable (SPEC §4.1):
las cuotas contienen casi toda la información pública y enmascararían qué
variables futbolísticas importan de verdad.

Salidas: importancia global (media de |SHAP| por variable), beeswarm de la
clase "victoria local", dependencia parcial de las variables top y las figuras
PNG que acompañan al informe.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: solo escribe PNG
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alaves_predictor.evaluation.metrics import OUTCOME_ORDER
from alaves_predictor.models.gbm_classifier import GBMModel, to_matrix


def shap_contributions(model: GBMModel, df: pd.DataFrame) -> np.ndarray:
    """Valores SHAP por (muestra, variable, clase). Excluye el término base.

    LightGBM devuelve, para multiclase, bloques por clase de (n_features + 1)
    columnas: las n_features contribuciones + el valor base. Se reordena a
    (n_muestras, n_variables, n_clases) en el orden de clases del modelo.
    """
    matrix = to_matrix(df, model.feature_names)
    raw = np.asarray(model.classifier.booster_.predict(matrix, pred_contrib=True))
    n_features = len(model.feature_names)
    n_classes = len(model.classifier.classes_)
    # (n, n_classes, n_features + 1) -> quitar el bias -> (n, n_features, n_classes)
    reshaped = raw.reshape(raw.shape[0], n_classes, n_features + 1)
    return np.transpose(reshaped[:, :, :n_features], (0, 2, 1))


def _class_order(model: GBMModel) -> list[int]:
    """Índices para reordenar las clases del modelo al orden canónico H/D/A."""
    class_index = {c: i for i, c in enumerate(model.classifier.classes_)}
    return [class_index[o] for o in OUTCOME_ORDER]


def global_importance(model: GBMModel, df: pd.DataFrame) -> pd.DataFrame:
    """Media de |SHAP| por variable (agregada sobre clases y muestras), ordenada."""
    contribs = shap_contributions(model, df)  # (n, n_features, n_classes)
    mean_abs = np.abs(contribs).mean(axis=(0, 2))  # promedio sobre muestras y clases
    return (
        pd.DataFrame({"feature": model.feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def plot_bar(importance: pd.DataFrame, path: Path, top: int = 20) -> Path:
    """Gráfico de barras de la importancia global de las `top` variables."""
    data = importance.head(top).iloc[::-1]  # la más importante arriba
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(data))))
    ax.barh(data["feature"], data["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("media |SHAP| (impacto medio en la predicción)")
    ax.set_title("Importancia global de variables — variante sin cuotas")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_beeswarm(
    model: GBMModel, df: pd.DataFrame, importance: pd.DataFrame, path: Path, top: int = 15
) -> Path:
    """Beeswarm de la clase 'victoria local': cómo empuja cada variable el 1.

    Cada punto es un partido; su posición en x es el SHAP de esa variable para
    la clase H, y su color, el valor (estandarizado) de la variable — así se ve
    si valores altos empujan hacia victoria local o en contra.
    """
    contribs = shap_contributions(model, df)
    home_idx = _class_order(model)[0]  # clase H en el orden del modelo
    features = list(importance.head(top)["feature"])[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(features))))
    rng = np.random.default_rng(42)
    for row, feat in enumerate(features):
        col = model.feature_names.index(feat)
        shap_vals = contribs[:, col, home_idx]
        values = to_matrix(df, [feat]).iloc[:, 0].to_numpy()
        finite = np.isfinite(values)
        std = values[finite].std() or 1.0
        color = np.clip((values - np.nanmean(values)) / std, -2, 2)
        jitter = rng.uniform(-0.18, 0.18, size=len(shap_vals))
        sc = ax.scatter(
            shap_vals,
            np.full_like(shap_vals, row) + jitter,
            c=color,
            cmap="coolwarm",
            s=8,
            alpha=0.6,
        )
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("SHAP para victoria local (→ empuja hacia el 1)")
    ax.set_title("Beeswarm — clase victoria local (sin cuotas)")
    fig.colorbar(sc, ax=ax, label="valor de la variable (alto = rojo)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def partial_dependence(
    model: GBMModel, df: pd.DataFrame, feature: str, grid_size: int = 20
) -> pd.DataFrame:
    """Dependencia parcial: P(1X2) media al barrer una variable, fijando el resto.

    Para cada valor de la rejilla se sustituye la columna en TODAS las muestras
    y se promedia la predicción — el efecto marginal de la variable (SPEC §7.3).
    """
    values = to_matrix(df, [feature]).iloc[:, 0].to_numpy()
    finite = values[np.isfinite(values)]
    grid = np.linspace(np.quantile(finite, 0.05), np.quantile(finite, 0.95), grid_size)
    base = to_matrix(df, model.feature_names)
    col = model.feature_names.index(feature)
    rows = []
    for value in grid:
        perturbed = base.copy()
        perturbed.iloc[:, col] = value
        probs = model.classifier.predict_proba(perturbed).mean(axis=0)
        class_index = {c: i for i, c in enumerate(model.classifier.classes_)}
        rows.append(
            {
                "value": value,
                **{o: probs[class_index[o]] for o in OUTCOME_ORDER},
            }
        )
    return pd.DataFrame(rows)


def plot_partial_dependence(
    model: GBMModel, df: pd.DataFrame, features: list[str], path: Path
) -> Path:
    """Rejilla de gráficos de dependencia parcial de las variables `features`."""
    n = len(features)
    cols = 2
    fig_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(fig_rows, cols, figsize=(11, 3 * fig_rows), squeeze=False)
    colors = {"H": "#2ca02c", "D": "#7f7f7f", "A": "#d62728"}
    for i, feat in enumerate(features):
        ax = axes[i // cols][i % cols]
        pdp = partial_dependence(model, df, feat)
        for outcome in OUTCOME_ORDER:
            ax.plot(pdp["value"], pdp[outcome], label=outcome, color=colors[outcome])
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel("P")
        ax.legend(fontsize=7, loc="best")
    for j in range(n, fig_rows * cols):  # ejes sobrantes
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Dependencia parcial de las variables top (sin cuotas)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
