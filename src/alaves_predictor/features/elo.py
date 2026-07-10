"""Elo interno recalculable (ADR-013).

Fórmula estándar de Elo aplicada a fútbol:

    esperado_local = 1 / (1 + 10^(-((R_local + ventaja_campo) - R_visitante) / 400))
    R' = R + K · (resultado - esperado)      con resultado ∈ {1, 0.5, 0}

Decisiones v1 (ADR-013): K fijo, sin multiplicador por margen de goles
(simplicidad y legibilidad primero; ambos ajustables por validación en F3),
rating inicial único para todos los equipos. Se calcula secuencialmente en
orden cronológico, de modo que el rating PRE-partido de cada fila usa solo
partidos anteriores — sin fugas temporales por construcción.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd

from alaves_predictor.config import EloInternalConfig
from alaves_predictor.etl import db


def expected_home_score(r_home: float, r_away: float, cfg: EloInternalConfig) -> float:
    """Probabilidad esperada de 'no perder ponderada' del local, escala Elo clásica."""
    diff = (r_home + cfg.home_advantage) - r_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def compute_internal_elo(matches: pd.DataFrame, cfg: EloInternalConfig) -> pd.DataFrame:
    """Calcula el Elo interno partido a partido.

    `matches` debe venir ordenado cronológicamente con columnas:
    match_id, date, home_id, away_id, home_goals, away_goals.

    Devuelve un DataFrame con match_id y los ratings PRE y POST de cada lado.
    """
    ratings: dict[str, float] = {}
    rows: list[dict] = []
    for m in matches.itertuples(index=False):
        r_home = ratings.get(m.home_id, cfg.initial_rating)
        r_away = ratings.get(m.away_id, cfg.initial_rating)
        expected = expected_home_score(r_home, r_away, cfg)
        if m.home_goals > m.away_goals:
            score = 1.0
        elif m.home_goals == m.away_goals:
            score = 0.5
        else:
            score = 0.0
        delta = cfg.k * (score - expected)
        rows.append(
            {
                "match_id": m.match_id,
                "date": m.date,
                "home_id": m.home_id,
                "away_id": m.away_id,
                "elo_internal_home_pre": r_home,
                "elo_internal_away_pre": r_away,
                "elo_internal_home_post": r_home + delta,
                "elo_internal_away_post": r_away - delta,
            }
        )
        ratings[m.home_id] = r_home + delta
        ratings[m.away_id] = r_away - delta
    return pd.DataFrame(rows)


def persist_internal_elo(conn: sqlite3.Connection, elo_history: pd.DataFrame) -> int:
    """Guarda el rating POST-partido de cada equipo en la tabla elo (fecha = día del partido).

    El upsert con COALESCE no pisa el elo_clubelo si coincidiera la fecha.
    Devuelve el nº de filas escritas.
    """
    now = datetime.now(UTC).isoformat()
    written = 0
    for row in elo_history.itertuples(index=False):
        for team_id, rating in (
            (row.home_id, row.elo_internal_home_post),
            (row.away_id, row.elo_internal_away_post),
        ):
            db.upsert(
                conn,
                "elo",
                {
                    "team_id": team_id,
                    "date": str(row.date),
                    "elo_clubelo": None,
                    "elo_internal": round(float(rating), 2),
                    "source": "internal",
                    "fetched_at": now,
                },
                key_cols=["team_id", "date"],
            )
            written += 1
    conn.commit()
    return written
