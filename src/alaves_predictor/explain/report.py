"""Informe de importancia de variables (SPEC §7.5): docs/reports/feature_importance.md.

Orquesta el análisis SHAP global, la dependencia parcial de las variables top y
el ablation study, guarda las figuras PNG y escribe un informe en lenguaje claro.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from alaves_predictor.config import Settings
from alaves_predictor.explain import ablation, importance
from alaves_predictor.features.dictionary import feature_dictionary
from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS
from alaves_predictor.models.train import ModelBundle

_TOP_PDP = 8  # variables para la dependencia parcial (SPEC §7.3)


def generate_report(
    bundle: ModelBundle,
    features: pd.DataFrame,
    settings: Settings,
    out_dir: Path,
    n_test_seasons: int = 3,
    sample: int = 1500,
) -> Path:
    """Genera figuras + informe Markdown. Devuelve la ruta del informe."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    model = bundle.variants[VARIANT_NO_ODDS].gbm
    finished = features[features["result"].notna()]
    shap_df = finished.sample(min(sample, len(finished)), random_state=42)

    imp = importance.global_importance(model, shap_df)
    bar_path = importance.plot_bar(imp, fig_dir / "importancia_global.png")
    bee_path = importance.plot_beeswarm(
        model, shap_df, imp, fig_dir / "beeswarm_victoria_local.png"
    )
    top_features = list(imp.head(_TOP_PDP)["feature"])
    pdp_path = importance.plot_partial_dependence(
        model, shap_df, top_features, fig_dir / "dependencia_parcial.png"
    )

    ablation_rows = ablation.run_ablation(features, settings, n_test_seasons)

    stamp = datetime.now(UTC)
    lines = [
        "# Importancia de variables (F5)",
        "",
        f"Generado: {stamp.isoformat(timespec='seconds')}",
        f"Modelo: `{bundle.model_version}` — variante **sin cuotas** (la interpretable).",
        "",
        "Todo el análisis se hace sobre la variante sin cuotas (SPEC §4.1): con",
        "cuotas el mercado domina y tapa qué variables *futbolísticas* importan.",
        "",
        "## 1. Importancia global (SHAP)",
        "",
        f"![Importancia global]({bar_path.relative_to(out_dir)})",
        "",
        "Media de |SHAP| por variable (impacto medio en la predicción, agregado",
        "sobre las tres clases). Las 15 primeras:",
        "",
        "| # | Variable | media \\|SHAP\\| |",
        "|---|----------|--------------|",
    ]
    for i, row in enumerate(imp.head(15).itertuples(index=False), start=1):
        lines.append(f"| {i} | `{row.feature}` | {row.mean_abs_shap:.4f} |")

    lines += [
        "",
        "## 2. Beeswarm — victoria local",
        "",
        f"![Beeswarm victoria local]({bee_path.relative_to(out_dir)})",
        "",
        "Cada punto es un partido; la x es cuánto empuja esa variable hacia la",
        "victoria local y el color, el valor de la variable (rojo = alto).",
        "",
        "## 3. Dependencia parcial de las variables top",
        "",
        f"![Dependencia parcial]({pdp_path.relative_to(out_dir)})",
        "",
        "Efecto marginal: cómo cambia P(1X2) al barrer cada variable dejando el",
        "resto fijo (H = victoria local, D = empate, A = victoria visitante).",
        "",
        "## 4. Ablation study — aportación de cada bloque",
        "",
        "Log-loss walk-forward quitando bloques enteros de features. Un bloque",
        "aporta si quitarlo **empeora** el log-loss (delta > 0).",
        "",
        "| Bloque | Variables quitadas | Log-loss | Δ vs completo |",
        "|--------|--------------------|----------|---------------|",
    ]
    for row in ablation_rows:
        delta = "—" if row.block.startswith("(") else f"{row.delta_vs_full:+.4f}"
        lines.append(f"| {row.block} | {row.n_features_removed} | {row.log_loss:.4f} | {delta} |")

    lines += [
        "",
        "> Nota: el ablation usa el LightGBM sin cuotas (no el ensemble completo)",
        "> para aislar la contribución de las features al clasificador. Deltas de",
        "> pocas milésimas están dentro del ruido; lo relevante es el signo y el",
        "> orden de magnitud relativo entre bloques.",
        "",
        "## Apéndice — diccionario de variables",
        "",
        "Qué significa cada columna del modelo (ordenadas por importancia SHAP;",
        "guía general en `docs/diccionario-features.md`):",
        "",
        "| Variable | Significado |",
        "|----------|-------------|",
    ]
    dictionary = feature_dictionary(list(imp["feature"]))
    for row in dictionary.itertuples(index=False):
        lines.append(f"| `{row.feature}` | {row.descripcion} |")
    lines.append("")
    path = out_dir / "feature_importance.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
