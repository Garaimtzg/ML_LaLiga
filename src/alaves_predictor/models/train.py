"""Entrenamiento, validación temporal, registro y carga de modelos (SPEC §6.3-§6.4).

Flujo de `train_models` (ADR-017 y ADR-019):

1. Predicciones walk-forward por temporada: cada temporada se predice con
   modelos entrenados SOLO con temporadas anteriores (folds temporales).
2. Con ese pool out-of-fold se eligen: el xi del Dixon-Coles (rejilla por
   log-loss), los calibradores isotónicos del LightGBM y los pesos del
   ensemble apilado — nunca con datos vistos en entrenamiento.
3. Las métricas de validación se calculan sobre la ÚLTIMA temporada, con
   todo lo anterior ajustado SOLO con las temporadas previas a ella.
4. Reentrenamiento final de todos los componentes con TODA la historia
   (calibradores y pesos del paso 2 se conservan).

El ensemble es un apilado de 3 componentes por variante (ADR-019):
    con_cuotas: Dixon-Coles + LightGBM calibrado + mercado (apertura)
    sin_cuotas: Dixon-Coles + LightGBM calibrado + Elo logístico

El registro (ADR-018) guarda el artefacto en models/registry/<versión>/ y una
fila en la tabla model_registry, y aplica la regla anti-sorpresa de SPEC §6.4:
si el log-loss de validación empeora >10 % respecto a la última versión
promocionada, la nueva se registra pero NO se promociona (predict la ignora).
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from alaves_predictor.config import Settings
from alaves_predictor.evaluation import baselines, metrics
from alaves_predictor.features.build import MARKET_COLS, feature_columns
from alaves_predictor.models import calibration, dixon_coles, ensemble, gbm_classifier
from alaves_predictor.models.gbm_classifier import (
    VARIANT_NO_ODDS,
    VARIANT_WITH_ODDS,
    VARIANTS,
    GBMModel,
)

# Componentes del ensemble apilado por variante (ADR-019), en orden fijo.
COMPONENTS = {
    VARIANT_WITH_ODDS: ["dixon_coles", "lightgbm", "mercado_apertura"],
    VARIANT_NO_ODDS: ["dixon_coles", "lightgbm", "elo_logistico"],
}


@dataclass
class SeasonPredictions:
    """Predicciones walk-forward de una temporada (sin fuga: train < temporada)."""

    season: str
    y_true: list[str]
    dc_by_xi: dict[float, np.ndarray]  # probabilidades del DC por candidato de xi
    gbm: dict[str, np.ndarray]  # por variante, sin calibrar
    market: np.ndarray  # probabilidades implícitas de apertura
    elo_lr: np.ndarray  # Elo logístico ajustado con las temporadas de train


@dataclass
class VariantModel:
    """Una variante completa: LightGBM + calibración + pesos del apilado."""

    gbm: GBMModel
    calibrators: list[IsotonicRegression]
    component_names: list[str]
    weights: np.ndarray  # mismo orden que component_names; suman 1

    def ensemble_probs(
        self, df: pd.DataFrame, dc_probs: np.ndarray, third: np.ndarray
    ) -> np.ndarray:
        gbm_cal = calibration.apply_isotonic(
            self.calibrators, gbm_classifier.predict_proba(self.gbm, df)
        )
        return ensemble.blend_many([dc_probs, gbm_cal, third], self.weights)


@dataclass
class ModelBundle:
    """Artefacto completo de un entrenamiento (lo que se serializa al registry)."""

    model_version: str
    feature_set_version: str
    trained_at: str
    train_window: str
    dixon_coles: dixon_coles.DixonColesModel
    elo_lr: LogisticRegression
    variants: dict[str, VariantModel]
    val_metrics: dict  # métricas walk-forward de la última temporada
    val_season: str
    xi: float  # xi elegido por validación (ADR-019)

    def _third_component(
        self, rows: pd.DataFrame, variant: str, dc_probs: np.ndarray
    ) -> np.ndarray:
        if variant == VARIANT_WITH_ODDS:
            return market_probs(rows, fallback=dc_probs)
        return baselines.predict_elo_logistic(self.elo_lr, rows)

    def predict_matches(self, rows: pd.DataFrame, variant: str) -> pd.DataFrame:
        """Predicción completa por partido: P(1X2) del apilado + marcador del DC."""
        dc_probs = dc_probs_for(self.dixon_coles, rows)
        third = self._third_component(rows, variant, dc_probs)
        probs = self.variants[variant].ensemble_probs(rows, dc_probs, third)
        records = []
        for i, m in enumerate(rows.itertuples(index=False)):
            lam, mu = self.dixon_coles.expected_goals(m.home_id, m.away_id)
            score_h, score_a, p_score = self.dixon_coles.most_likely_score(m.home_id, m.away_id)
            records.append(
                {
                    "match_id": m.match_id,
                    "home_id": m.home_id,
                    "away_id": m.away_id,
                    "date": m.date,
                    "matchday": m.matchday,
                    "p_home": float(probs[i, 0]),
                    "p_draw": float(probs[i, 1]),
                    "p_away": float(probs[i, 2]),
                    "pred_result": metrics.OUTCOME_ORDER[int(np.argmax(probs[i]))],
                    "pred_score": f"{score_h}-{score_a}",
                    "pred_score_prob": p_score,
                    "expected_goals_h": lam,
                    "expected_goals_a": mu,
                }
            )
        return pd.DataFrame(records)


def fit_dc(
    train: pd.DataFrame, settings: Settings, xi: float | None = None, **kwargs
) -> dixon_coles.DixonColesModel:
    """Ajusta el Dixon-Coles sobre las columnas de partido del frame de features."""
    cfg = settings.models.dixon_coles
    if xi is not None:
        cfg = cfg.model_copy(update={"xi": xi})
    cols = ["home_id", "away_id", "home_goals", "away_goals", "date"]
    return dixon_coles.fit(train[cols], cfg, **kwargs)


def dc_probs_for(model: dixon_coles.DixonColesModel, df: pd.DataFrame) -> np.ndarray:
    return np.vstack(
        [model.outcome_probs(h, a) for h, a in zip(df["home_id"], df["away_id"], strict=True)]
    )


def market_probs(df: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    """Probabilidades implícitas de apertura; sin cuotas, cae al fallback (DC)."""
    probs = df[MARKET_COLS].astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
    missing = np.isnan(probs).any(axis=1)
    probs[missing] = np.asarray(fallback, dtype=float)[missing]
    return probs


def season_walkforward(
    features: pd.DataFrame,
    settings: Settings,
    variants: tuple[str, ...] = VARIANTS,
) -> list[SeasonPredictions]:
    """Predice cada temporada (desde la 2ª) con modelos entrenados solo con las previas."""
    all_cols = feature_columns(features)
    xi_candidates = settings.models.dixon_coles.xi_candidates()
    out: list[SeasonPredictions] = []
    for season in sorted(set(features["season"]))[1:]:
        train = features[features["season"] < season]
        test = features[features["season"] == season]
        dc_by_xi: dict[float, np.ndarray] = {}
        previous: dixon_coles.DixonColesModel | None = None
        for xi in xi_candidates:
            previous = fit_dc(train, settings, xi=xi, warm_start=previous)
            dc_by_xi[xi] = dc_probs_for(previous, test)
        gbm_probs = {}
        for variant in variants:
            cols = gbm_classifier.variant_features(all_cols, variant)
            model = gbm_classifier.fit(train, cols, settings.models.lightgbm, variant)
            gbm_probs[variant] = gbm_classifier.predict_proba(model, test)
        out.append(
            SeasonPredictions(
                season=season,
                y_true=list(test["result"]),
                dc_by_xi=dc_by_xi,
                gbm=gbm_probs,
                market=market_probs(test, fallback=dc_by_xi[xi_candidates[0]]),
                elo_lr=baselines.elo_logistic_probs(train, test),
            )
        )
    return out


def choose_xi(preds: list[SeasonPredictions]) -> float:
    """xi con mejor log-loss medio sobre el pool walk-forward (ADR-019)."""
    candidates = list(preds[0].dc_by_xi)
    losses = {
        xi: float(np.mean([metrics.log_loss(p.y_true, p.dc_by_xi[xi]) for p in preds]))
        for xi in candidates
    }
    return min(losses, key=losses.get)  # type: ignore[arg-type]


def _pool(
    preds: list[SeasonPredictions], variant: str, xi: float
) -> tuple[list[str], list[np.ndarray]]:
    """Concatena el pool walk-forward: (y_true, [dc, gbm_sin_calibrar, tercero])."""
    y = [label for p in preds for label in p.y_true]
    dc = np.vstack([p.dc_by_xi[xi] for p in preds])
    gbm = np.vstack([p.gbm[variant] for p in preds])
    third_attr = "market" if variant == VARIANT_WITH_ODDS else "elo_lr"
    third = np.vstack([getattr(p, third_attr) for p in preds])
    return y, [dc, gbm, third]


def _calibrate_and_weigh(
    preds: list[SeasonPredictions], variant: str, step: float, xi: float
) -> tuple[list[IsotonicRegression], np.ndarray]:
    """Calibradores isotónicos + pesos del apilado a partir del pool out-of-fold."""
    y, (dc, gbm, third) = _pool(preds, variant, xi)
    calibrators = calibration.fit_isotonic(gbm, y)
    gbm_cal = calibration.apply_isotonic(calibrators, gbm)
    weights = ensemble.optimal_weights([dc, gbm_cal, third], y, step)
    return calibrators, weights


def train_models(
    features: pd.DataFrame,
    settings: Settings,
    variants: tuple[str, ...] = VARIANTS,
) -> ModelBundle:
    """Entrena el sistema completo sobre partidos jugados. Ver flujo en el docstring."""
    finished = features[features["result"].notna()].copy()
    seasons = sorted(set(finished["season"]))
    if len(seasons) < 2:
        raise ValueError(
            "Se necesitan al menos 2 temporadas para entrenar con validación temporal "
            f"(hay {len(seasons)}). Ejecuta `alaves ingest --historical` primero."
        )
    step = settings.models.ensemble.weight_grid_step
    default_xi = settings.models.dixon_coles.xi_candidates()[0]
    oof = season_walkforward(finished, settings, variants)

    # --- métricas de validación: última temporada, sin verse a sí misma ---
    val, prior = oof[-1], oof[:-1]
    xi_val = choose_xi(prior) if prior else default_xi
    # valores: dicts de métricas por modelo, salvo "xi" (el candidato elegido)
    val_metrics: dict = {
        "dixon_coles": metrics.evaluate(val.y_true, val.dc_by_xi[xi_val]),
        "xi": xi_val,
    }
    for variant in variants:
        third = val.market if variant == VARIANT_WITH_ODDS else val.elo_lr
        if prior:
            cal_v, weights_v = _calibrate_and_weigh(prior, variant, step, xi_val)
            gbm_val = calibration.apply_isotonic(cal_v, val.gbm[variant])
        else:
            # con solo 2 temporadas no hay pool previo: sin calibrar, pesos iguales
            gbm_val = val.gbm[variant]
            weights_v = np.full(3, 1.0 / 3.0)
        ens_val = ensemble.blend_many([val.dc_by_xi[xi_val], gbm_val, third], weights_v)
        val_metrics[variant] = {
            "lgbm": metrics.evaluate(val.y_true, gbm_val),
            "ensemble": metrics.evaluate(val.y_true, ens_val),
            "weights": dict(
                zip(COMPONENTS[variant], [round(float(w), 3) for w in weights_v], strict=True)
            ),
        }

    # --- selección/ajuste definitivos (todo el pool) + reentreno final ---
    xi_final = choose_xi(oof)
    all_cols = feature_columns(finished)
    dc_final = fit_dc(finished, settings, xi=xi_final)
    elo_lr_final = baselines.fit_elo_logistic(finished)
    bundle_variants: dict[str, VariantModel] = {}
    for variant in variants:
        calibrators, weights = _calibrate_and_weigh(oof, variant, step, xi_final)
        cols = gbm_classifier.variant_features(all_cols, variant)
        gbm_final = gbm_classifier.fit(finished, cols, settings.models.lightgbm, variant)
        bundle_variants[variant] = VariantModel(
            gbm=gbm_final,
            calibrators=calibrators,
            component_names=COMPONENTS[variant],
            weights=weights,
        )

    now = datetime.now(UTC)
    return ModelBundle(
        model_version=f"{settings.features.feature_set_version}-{now:%Y%m%d-%H%M%S}",
        feature_set_version=settings.features.feature_set_version,
        trained_at=now.isoformat(timespec="seconds"),
        train_window=f"{seasons[0]}..{seasons[-1]}",
        dixon_coles=dc_final,
        elo_lr=elo_lr_final,
        variants=bundle_variants,
        val_metrics=val_metrics,
        val_season=val.season,
        xi=xi_final,
    )


# --- Registro de modelos (SPEC §6.4 y §12.4) ---------------------------------

# Variante cuyo log-loss de ensemble sirve de referencia para la regla
# anti-sorpresa: sin_cuotas, porque se entrena SIEMPRE (con --no-odds y sin él)
# y así la comparación entre versiones es homogénea.
_REFERENCE_VARIANT = VARIANT_NO_ODDS


@dataclass
class RegistryDecision:
    model_version: str
    promoted: bool
    reason: str
    val_logloss: float
    previous_version: str | None = None
    previous_logloss: float | None = None


def _reference_logloss(val_metrics: dict) -> float:
    return float(val_metrics[_REFERENCE_VARIANT]["ensemble"]["log_loss"])


def register_model(
    conn: sqlite3.Connection, settings: Settings, bundle: ModelBundle
) -> RegistryDecision:
    """Serializa el artefacto, decide la promoción y escribe la fila del registry."""
    new_loss = _reference_logloss(bundle.val_metrics)
    previous = _latest_promoted_row(conn)
    promoted, reason = True, "primera versión registrada"
    prev_version = prev_loss = None
    if previous is not None:
        prev_version = previous["model_version"]
        prev_loss = float(json.loads(previous["metrics_json"])["reference_logloss"])
        limit = prev_loss * (1.0 + settings.models.max_logloss_regression)
        if new_loss > limit:
            promoted = False
            reason = (
                f"log-loss de validación {new_loss:.4f} empeora más del "
                f"{settings.models.max_logloss_regression:.0%} respecto a "
                f"{prev_version} ({prev_loss:.4f}); revisar antes de usar"
            )
        else:
            reason = f"mejora o mantiene a {prev_version} ({prev_loss:.4f} → {new_loss:.4f})"

    artifact_dir = settings.models.registry_dir / bundle.model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "model.pkl"
    with artifact_path.open("wb") as fh:
        pickle.dump(bundle, fh)

    metrics_payload = {
        "val_season": bundle.val_season,
        "val_metrics": bundle.val_metrics,
        "reference_logloss": new_loss,
        "promoted": promoted,
        "promotion_reason": reason,
    }
    config_payload = {
        "feature_set_version": bundle.feature_set_version,
        "variants": list(bundle.variants),
        "xi": bundle.xi,
        "models": settings.models.model_dump(mode="json"),
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (artifact_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    conn.execute(
        "INSERT INTO model_registry "
        "(model_version, trained_at, train_window, metrics_json, config_json, artifact_path) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            bundle.model_version,
            bundle.trained_at,
            bundle.train_window,
            json.dumps(metrics_payload, ensure_ascii=False),
            json.dumps(config_payload, ensure_ascii=False),
            str(artifact_path),
        ),
    )
    conn.commit()
    return RegistryDecision(
        model_version=bundle.model_version,
        promoted=promoted,
        reason=reason,
        val_logloss=new_loss,
        previous_version=prev_version,
        previous_logloss=prev_loss,
    )


def _latest_promoted_row(conn: sqlite3.Connection) -> sqlite3.Row | None:
    rows = conn.execute(
        "SELECT model_version, metrics_json, artifact_path FROM model_registry "
        "ORDER BY trained_at DESC, model_version DESC"
    ).fetchall()
    for row in rows:
        if json.loads(row["metrics_json"]).get("promoted", True):
            return row
    return None


def load_latest_model(conn: sqlite3.Connection) -> ModelBundle | None:
    """Carga la última versión PROMOCIONADA del registry (None si no hay ninguna)."""
    row = _latest_promoted_row(conn)
    if row is None:
        return None
    with open(row["artifact_path"], "rb") as fh:
        return pickle.load(fh)  # noqa: S301 — artefacto local generado por este código
