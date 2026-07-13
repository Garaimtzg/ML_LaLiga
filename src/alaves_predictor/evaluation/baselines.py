"""Baselines obligatorios (SPEC §6.1, ADR-014): la vara de medir de todo modelo.

1. Frecuencias históricas: P(H/D/A) del conjunto de entrenamiento, constantes.
2. Elo logístico: regresión logística multinomial sobre la diferencia de Elo
   (ClubElo) pre-partido; los interceptos capturan la ventaja de campo.
3. Cuotas de CIERRE normalizadas: la mejor estimación pública — el baseline
   a batir de verdad.

Evaluación walk-forward por temporada: para cada temporada de test se entrena
solo con temporadas ANTERIORES (nunca validación aleatoria, CLAUDE.md §5).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from alaves_predictor.config import Settings
from alaves_predictor.evaluation import metrics
from alaves_predictor.evaluation.metrics import OUTCOME_ORDER


@dataclass
class BaselineResult:
    baseline: str
    season: str
    n_matches: int
    metrics: dict[str, float]


def historical_frequencies(train: pd.DataFrame) -> np.ndarray:
    """Vector [P(H), P(D), P(A)] con las frecuencias del entrenamiento."""
    counts = train["result"].value_counts()
    total = counts.sum()
    return np.array([counts.get(o, 0) / total for o in OUTCOME_ORDER])


def fit_elo_logistic(train: pd.DataFrame) -> LogisticRegression:
    """Ajusta la logística multinomial sobre elo_clubelo_diff.

    La ventaja de campo no necesita feature: al ser constante, la capturan
    los interceptos por clase del modelo. Además de baseline, desde F3 es un
    componente del ensemble sin cuotas (ADR-019), por eso fit y predict van
    separados.
    """
    fit_rows = train.dropna(subset=["elo_clubelo_diff"])
    model = LogisticRegression(max_iter=1000)
    model.fit(fit_rows[["elo_clubelo_diff"]], fit_rows["result"])
    return model


def predict_elo_logistic(model: LogisticRegression, test: pd.DataFrame) -> np.ndarray:
    """P(1X2) del modelo Elo logístico, en el orden canónico H/D/A."""
    x_test = test[["elo_clubelo_diff"]].fillna(0.0)
    raw = model.predict_proba(x_test)
    class_index = {c: i for i, c in enumerate(model.classes_)}
    return raw[:, [class_index[o] for o in OUTCOME_ORDER]]


def elo_logistic_probs(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Ajusta y predice de una vez (uso como baseline walk-forward)."""
    return predict_elo_logistic(fit_elo_logistic(train), test)


def closing_odds_probs(conn: sqlite3.Connection, match_ids: pd.Series) -> pd.DataFrame:
    """Probabilidades implícitas de cuotas de CIERRE normalizadas por partido.

    Preferencia: media de mercado > bet365; si no hay cierre, cae a apertura
    (documentado: 2018-19 no publica cuotas de cierre).
    """
    odds = pd.read_sql_query(
        "SELECT match_id, bookmaker, open_h, open_d, open_a, close_h, close_d, close_a "
        "FROM odds WHERE bookmaker IN ('market_avg', 'bet365')",
        conn,
    )
    odds = odds[odds["match_id"].isin(set(match_ids))].copy()
    for side in ("h", "d", "a"):
        odds[f"best_{side}"] = odds[f"close_{side}"].fillna(odds[f"open_{side}"])
    odds["priority"] = (odds["bookmaker"] != "market_avg").astype(int)
    odds = (
        odds.dropna(subset=["best_h", "best_d", "best_a"])
        .sort_values(["match_id", "priority"])
        .drop_duplicates("match_id")
    )
    inv = 1.0 / odds[["best_h", "best_d", "best_a"]].to_numpy()
    probs = inv / inv.sum(axis=1, keepdims=True)
    odds[["p_home", "p_draw", "p_away"]] = probs
    return odds[["match_id", "p_home", "p_draw", "p_away"]]


def run_baselines(
    conn: sqlite3.Connection,
    features: pd.DataFrame,
    settings: Settings,
    n_test_seasons: int = 3,
) -> list[BaselineResult]:
    """Evalúa los 3 baselines walk-forward sobre las últimas `n_test_seasons` temporadas."""
    seasons = [s for s in settings.historical_seasons if s in set(features["season"])]
    test_seasons = seasons[-n_test_seasons:]
    results: list[BaselineResult] = []

    for season in test_seasons:
        train = features[features["season"] < season]
        test = features[features["season"] == season]
        if train.empty or test.empty:
            continue
        y_true = list(test["result"])

        freq = historical_frequencies(train)
        results.append(
            BaselineResult(
                "frecuencias",
                season,
                len(test),
                metrics.evaluate(y_true, np.tile(freq, (len(test), 1))),
            )
        )

        results.append(
            BaselineResult(
                "elo_logistico",
                season,
                len(test),
                metrics.evaluate(y_true, elo_logistic_probs(train, test)),
            )
        )

        odds_probs = closing_odds_probs(conn, test["match_id"])
        merged = test.merge(odds_probs, on="match_id", how="inner")
        results.append(
            BaselineResult(
                "cuotas_cierre",
                season,
                len(merged),
                metrics.evaluate(
                    list(merged["result"]),
                    merged[["p_home", "p_draw", "p_away"]].to_numpy(),
                ),
            )
        )
    return results


def write_report(results: list[BaselineResult], out_dir: Path) -> Path:
    """Informe Markdown en docs/reports/ con la tabla de métricas por baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    path = out_dir / f"baselines_{stamp}.md"
    lines = [
        "# Baselines 1X2 — evaluación walk-forward (F2)",
        "",
        f"Generado: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "Métricas: menor es mejor salvo accuracy. Las cuotas de cierre son el",
        "baseline exigente (SPEC §6.1): todo modelo de F3 se compara contra esto.",
        "",
        "| Baseline | Temporada | N | Log-loss | Brier | RPS | Accuracy |",
        "|----------|-----------|---|----------|-------|-----|----------|",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"| {r.baseline} | {r.season} | {r.n_matches} | {m['log_loss']:.4f} "
            f"| {m['brier']:.4f} | {m['rps']:.4f} | {m['accuracy']:.3f} |"
        )
    # medias por baseline
    lines += ["", "## Media por baseline", ""]
    lines += ["| Baseline | Log-loss | Brier | RPS | Accuracy |", "|---|---|---|---|---|"]
    by_name: dict[str, list[dict[str, float]]] = {}
    for r in results:
        by_name.setdefault(r.baseline, []).append(r.metrics)
    for name, items in by_name.items():
        avg = {k: sum(i[k] for i in items) / len(items) for k in items[0]}
        lines.append(
            f"| {name} | {avg['log_loss']:.4f} | {avg['brier']:.4f} "
            f"| {avg['rps']:.4f} | {avg['accuracy']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
