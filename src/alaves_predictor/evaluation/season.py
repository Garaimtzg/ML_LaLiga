"""Evaluación de las predicciones persistidas durante la temporada (SPEC §3.3 paso 5).

Cruza la tabla `predictions` con los resultados reales que ya se conocen y
mide cómo va el sistema en vivo: log-loss, Brier, RPS y acierto acumulados de
la temporada. Es la auditoría honesta del rendimiento real (CLAUDE.md §5.5),
distinta del backtest (que es sobre el pasado): aquí se juzga lo que el modelo
predijo ANTES de conocer el resultado.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alaves_predictor.config import Settings
from alaves_predictor.evaluation import metrics


@dataclass
class SeasonPerformance:
    season: str
    n_resolved: int  # predicciones con resultado ya conocido
    n_pending: int  # predicciones aún sin jugarse
    metrics: dict[str, float]  # log_loss, brier, rps, accuracy (vacío si n_resolved=0)


def resolved_predictions(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    """Predicciones de una temporada cuyo partido ya tiene resultado real.

    Si un partido tiene varias predicciones (reentrenos), se queda la más
    reciente (la vigente cuando se jugó).
    """
    df = pd.read_sql_query(
        "SELECT p.match_id, p.model_version, p.created_at, p.p_home, p.p_draw, p.p_away, "
        "m.home_goals, m.away_goals "
        "FROM predictions p JOIN matches m ON m.match_id = p.match_id "
        "WHERE m.season = ? AND m.status = 'finished' "
        "AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL "
        "ORDER BY p.created_at",
        conn,
        params=(season,),
    )
    if df.empty:
        return df
    df = df.drop_duplicates("match_id", keep="last").reset_index(drop=True)
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"],
        "H",
        np.where(df["home_goals"] == df["away_goals"], "D", "A"),
    )
    return df


def evaluate_season(conn: sqlite3.Connection, settings: Settings) -> SeasonPerformance:
    """Métricas acumuladas de la temporada actual sobre las predicciones resueltas."""
    season = settings.current_season
    resolved = resolved_predictions(conn, season)
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM predictions p JOIN matches m ON m.match_id = p.match_id "
        "WHERE m.season = ? AND m.status = 'scheduled'",
        (season,),
    ).fetchone()["n"]

    if resolved.empty:
        return SeasonPerformance(season=season, n_resolved=0, n_pending=pending, metrics={})
    probs = resolved[["p_home", "p_draw", "p_away"]].to_numpy()
    result = metrics.evaluate(list(resolved["result"]), probs)
    return SeasonPerformance(
        season=season,
        n_resolved=len(resolved),
        n_pending=pending,
        metrics=result,
    )
