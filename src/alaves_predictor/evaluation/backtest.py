"""Backtesting walk-forward jornada a jornada (SPEC §6.5, ADR-018).

Para cada temporada de test T (las 3 últimas por defecto):

1. Los calibradores isotónicos y el peso del ensemble se ajustan con las
   predicciones walk-forward de las temporadas ANTERIORES a T (pool
   out-of-fold: cada una predicha con modelos entrenados solo con las previas).
2. La temporada T se predice jornada a jornada re-simulando el ciclo real de
   reentrenamiento (SPEC §6.4): antes de cada jornada, Dixon-Coles y LightGBM
   se reentrenan con TODO lo jugado hasta la víspera — incluidas las jornadas
   anteriores de la propia T. El Dixon-Coles arranca en caliente desde la
   jornada previa para que el bucle sea rápido.

Nada de la temporada T influye en su propia predicción salvo sus jornadas ya
jugadas, exactamente como ocurrirá en producción durante la 2026-27.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alaves_predictor.config import Settings
from alaves_predictor.evaluation import metrics
from alaves_predictor.evaluation.baselines import BaselineResult
from alaves_predictor.features.build import feature_columns
from alaves_predictor.models import calibration, dixon_coles, ensemble, gbm_classifier
from alaves_predictor.models.gbm_classifier import VARIANTS
from alaves_predictor.models.train import (
    _calibrate_and_weigh,
    dc_probs_for,
    fit_dc,
    season_walkforward,
)

MODEL_DC = "dixon_coles"


@dataclass
class BacktestRow:
    model: str
    season: str
    n_matches: int
    metrics: dict[str, float]
    alaves_accuracy: float | None = None  # solo partidos del equipo foco
    alaves_n: int = 0


@dataclass
class BacktestOutput:
    rows: list[BacktestRow]
    reliability: pd.DataFrame  # tabla de fiabilidad del ensemble sin cuotas


def _matchday_groups(season_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Partidos de la temporada agrupados por jornada, en orden cronológico."""
    if season_df["matchday"].isna().any():
        # sin jornada fiable: bloques por fecha (no debería ocurrir tras F1)
        return [g for _, g in season_df.groupby("date", sort=True)]
    return [g for _, g in season_df.groupby("matchday", sort=True)]


def run_backtest(
    features: pd.DataFrame,
    settings: Settings,
    n_test_seasons: int = 3,
    variants: tuple[str, ...] = VARIANTS,
    progress: Callable[[str], None] | None = None,
) -> BacktestOutput:
    """Evalúa Dixon-Coles, LightGBM calibrado y ensembles, walk-forward."""
    say = progress or (lambda _msg: None)
    finished = features[features["result"].notna()].copy()
    finished = finished.sort_values(["date", "match_id"])
    seasons = sorted(set(finished["season"]))
    test_seasons = seasons[-n_test_seasons:]
    all_cols = feature_columns(finished)
    step = settings.models.ensemble.weight_grid_step

    # Pool walk-forward por temporada, para calibrar sin fugas (se calcula una
    # vez; para la temporada de test T solo se usan temporadas < T).
    say("Preparando pool de calibración (walk-forward por temporada)...")
    oof = season_walkforward(finished, settings, variants)

    rows: list[BacktestRow] = []
    ens_no_odds_probs: list[np.ndarray] = []  # para la tabla de fiabilidad
    ens_no_odds_true: list[str] = []

    for season in test_seasons:
        prior = [p for p in oof if p.season < season]
        if not prior:
            continue  # sin temporadas previas no hay con qué calibrar ni entrenar
        calib: dict[str, tuple] = {v: _calibrate_and_weigh(prior, v, step) for v in variants}

        test = finished[finished["season"] == season]
        say(f"Temporada {season}: {len(test)} partidos, reentrenando jornada a jornada...")
        collected: dict[str, list[np.ndarray]] = {MODEL_DC: []}
        for v in variants:
            collected[f"lgbm_{v}"] = []
            collected[f"ensemble_{v}"] = []
        y_true: list[str] = []
        frames: list[pd.DataFrame] = []
        dc_model: dixon_coles.DixonColesModel | None = None

        for group in _matchday_groups(test):
            train = finished[finished["date"] < group["date"].min()]
            dc_model = fit_dc(train, settings, warm_start=dc_model)
            dc_probs = dc_probs_for(dc_model, group)
            collected[MODEL_DC].append(dc_probs)
            for v in variants:
                cols = gbm_classifier.variant_features(all_cols, v)
                gbm = gbm_classifier.fit(train, cols, settings.models.lightgbm, v)
                calibrators, dc_weight = calib[v]
                gbm_cal = calibration.apply_isotonic(
                    calibrators, gbm_classifier.predict_proba(gbm, group)
                )
                collected[f"lgbm_{v}"].append(gbm_cal)
                collected[f"ensemble_{v}"].append(ensemble.blend(dc_probs, gbm_cal, dc_weight))
            y_true.extend(group["result"])
            frames.append(group)

        season_frame = pd.concat(frames, ignore_index=True)
        is_alaves = (season_frame["home_id"] == settings.focus_team) | (
            season_frame["away_id"] == settings.focus_team
        )
        for model_name, chunks in collected.items():
            probs = np.vstack(chunks)
            alaves_acc = None
            if is_alaves.any():
                alaves_acc = metrics.accuracy(
                    [y for y, f in zip(y_true, is_alaves, strict=True) if f],
                    probs[is_alaves.to_numpy()],
                )
            rows.append(
                BacktestRow(
                    model=model_name,
                    season=season,
                    n_matches=len(y_true),
                    metrics=metrics.evaluate(y_true, probs),
                    alaves_accuracy=alaves_acc,
                    alaves_n=int(is_alaves.sum()),
                )
            )
        no_odds = f"ensemble_{gbm_classifier.VARIANT_NO_ODDS}"
        if no_odds in collected:
            ens_no_odds_probs.append(np.vstack(collected[no_odds]))
            ens_no_odds_true.extend(y_true)

    reliability = (
        calibration.reliability_table(ens_no_odds_true, np.vstack(ens_no_odds_probs))
        if ens_no_odds_probs
        else pd.DataFrame()
    )
    return BacktestOutput(rows=rows, reliability=reliability)


# --- Informe (SPEC §6.5: docs/reports/backtest_<fecha>.md) --------------------


def _mean_metrics(rows: list[BacktestRow]) -> dict[str, dict[str, float]]:
    by_model: dict[str, list[dict[str, float]]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r.metrics)
    return {
        m: {k: sum(i[k] for i in items) / len(items) for k in items[0]}
        for m, items in by_model.items()
    }


def _baseline_means(baselines: list[BaselineResult]) -> dict[str, float]:
    by_name: dict[str, list[float]] = {}
    for b in baselines:
        by_name.setdefault(b.baseline, []).append(b.metrics["log_loss"])
    return {name: sum(v) / len(v) for name, v in by_name.items()}


def acceptance_checks(
    rows: list[BacktestRow], baselines: list[BaselineResult]
) -> list[tuple[str, bool, str]]:
    """Criterios de aceptación de SPEC §12.1 sobre las medias de log-loss."""
    model_means = _mean_metrics(rows)
    base_means = _baseline_means(baselines)
    checks: list[tuple[str, bool, str]] = []
    no_odds = f"ensemble_{gbm_classifier.VARIANT_NO_ODDS}"
    with_odds = f"ensemble_{gbm_classifier.VARIANT_WITH_ODDS}"
    if no_odds in model_means and "elo_logistico" in base_means:
        ours, ref = model_means[no_odds]["log_loss"], base_means["elo_logistico"]
        checks.append(
            (
                "ensemble sin cuotas < baseline Elo",
                ours < ref,
                f"{ours:.4f} vs {ref:.4f}",
            )
        )
    if with_odds in model_means and "cuotas_cierre" in base_means:
        ours, ref = model_means[with_odds]["log_loss"], base_means["cuotas_cierre"]
        checks.append(
            (
                "ensemble con cuotas ≤ cuotas de cierre + 0.01",
                ours <= ref + 0.01,
                f"{ours:.4f} vs {ref:.4f} + 0.01",
            )
        )
    return checks


def write_report(
    output: BacktestOutput,
    baselines: list[BaselineResult],
    out_dir: Path,
) -> Path:
    """Informe Markdown con modelos vs baselines y criterios de SPEC §12.1."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    path = out_dir / f"backtest_{stamp}.md"

    lines = [
        "# Backtesting walk-forward (F3): modelos vs baselines",
        "",
        f"Generado: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "Protocolo (ADR-018): cada temporada de test se predice jornada a jornada,",
        "reentrenando antes de cada jornada con todo lo jugado hasta la víspera;",
        "calibración y peso del ensemble ajustados solo con temporadas anteriores.",
        "",
        "## Modelos (F3)",
        "",
        "| Modelo | Temporada | N | Log-loss | Brier | RPS | Accuracy | Acc. Alavés |",
        "|--------|-----------|---|----------|-------|-----|----------|-------------|",
    ]
    for r in output.rows:
        m = r.metrics
        alaves = "—"
        if r.alaves_accuracy is not None:
            alaves = f"{r.alaves_accuracy:.3f} (n={r.alaves_n})"
        lines.append(
            f"| {r.model} | {r.season} | {r.n_matches} | {m['log_loss']:.4f} "
            f"| {m['brier']:.4f} | {m['rps']:.4f} | {m['accuracy']:.3f} | {alaves} |"
        )

    lines += ["", "## Baselines (F2, mismas temporadas)", ""]
    lines += [
        "| Baseline | Temporada | N | Log-loss | Brier | RPS | Accuracy |",
        "|----------|-----------|---|----------|-------|-----|----------|",
    ]
    for b in baselines:
        m = b.metrics
        lines.append(
            f"| {b.baseline} | {b.season} | {b.n_matches} | {m['log_loss']:.4f} "
            f"| {m['brier']:.4f} | {m['rps']:.4f} | {m['accuracy']:.3f} |"
        )

    lines += ["", "## Medias por modelo", ""]
    lines += ["| Modelo | Log-loss | Brier | RPS | Accuracy |", "|---|---|---|---|---|"]
    for name, avg in _mean_metrics(output.rows).items():
        lines.append(
            f"| {name} | {avg['log_loss']:.4f} | {avg['brier']:.4f} "
            f"| {avg['rps']:.4f} | {avg['accuracy']:.3f} |"
        )

    lines += ["", "## Criterios de aceptación (SPEC §12.1)", ""]
    for label, passed, detail in acceptance_checks(output.rows, baselines):
        icon = "✅" if passed else "❌"
        lines.append(f"- {icon} {label}: {detail}")

    if not output.reliability.empty:
        lines += [
            "",
            "## Fiabilidad del ensemble sin cuotas (calibración)",
            "",
            "Por bin: la probabilidad media predicha debe parecerse a la frecuencia",
            "observada (el diagrama gráfico llega con el dashboard, F6).",
            "",
            "| Clase | Bin | N | Prob. media predicha | Frecuencia observada |",
            "|-------|-----|---|----------------------|----------------------|",
        ]
        for row in output.reliability.itertuples(index=False):
            lines.append(
                f"| {row.clase} | {row.bin} | {row.n} "
                f"| {row.prob_media_predicha:.3f} | {row.frecuencia_observada:.3f} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
