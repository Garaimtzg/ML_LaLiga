"""Qué partidos predecir y cómo guardarlos (SPEC §2, CLAUDE.md §5.5, ADR-031).

Lo usan `alaves predict`, el ciclo semanal (`alaves ingest --matchday`) y el
dashboard, para que los tres entiendan lo mismo por "la próxima jornada" y
guarden las predicciones del mismo modo.

**La próxima jornada** no es la de número más bajo con algo pendiente. Un
partido aplazado (en LaLiga los hay cada temporada: la Supercopa de enero
mueve partidos de cuatro equipos) conserva su jornada original, así que esa
regla elegía la jornada del aplazado y predecía solo ese partido, dejando sin
predecir la jornada que de verdad venía. La regla es:

1. Solo cuentan los programados con fecha de hoy en adelante. Uno con fecha
   pasada ya se ha jugado (y la fuente aún no ha publicado el resultado) o se
   aplazó sin fecha nueva: predecirlo ahora sería predecir a posteriori.
2. La próxima jornada es la de número más bajo que aún tiene por jugar al
   menos la mitad de sus partidos. Una jornada empezada (el viernes jugado y
   el resto por jugar) sigue siendo la próxima.
3. Los pendientes de jornadas anteriores (aplazados, o el resto de una
   jornada casi acabada) se predicen con ella si se juegan antes de que
   termine. Uno aplazado a dentro de un mes se predecirá cuando le toque.

**La persistencia** guarda, antes de conocerse el resultado, una fila por
partido con la versión del modelo, el instante y el hash de las features con
que se predijo (CLAUDE.md §5.5): así se puede auditar después que una
predicción se hizo con los datos que había entonces.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime

import pandas as pd

from alaves_predictor.config import Settings
from alaves_predictor.etl import db
from alaves_predictor.features.build import feature_payload, persist_features


def now_utc() -> datetime:
    """Instante actual en UTC: el sello `created_at` de cada predicción.

    Un único punto de lectura del reloj para todo el módulo; los tests que
    simulan una fecha lo sustituyen aquí (en producción es siempre el real: no
    hay forma de guardar una predicción con otra fecha).
    """
    return datetime.now(UTC)


def today_utc() -> date:
    """Fecha de hoy en UTC: la misma referencia que `created_at` y `alaves validate`."""
    return now_utc().date()


def next_round(
    features: pd.DataFrame, season: str, matches_per_round: int, today: date
) -> tuple[int | None, pd.DataFrame]:
    """(jornada, partidos a predecir) de la próxima jornada de `season`.

    `features` es el frame de `build_features(..., include_scheduled=True)`.
    Devuelve (None, vacío) si no queda nada por jugar con fecha.
    """
    pending = features[
        (features["season"] == season)
        & features["result"].isna()
        & features["matchday"].notna()
        & (features["date"].astype(str) >= today.isoformat())
    ]
    if pending.empty:
        return None, pending
    per_round = pending.groupby("matchday").size()
    open_rounds = per_round[per_round * 2 >= matches_per_round]
    matchday = int(open_rounds.index.min() if not open_rounds.empty else per_round.index.min())
    round_rows = pending[pending["matchday"] == matchday]
    last_day = round_rows["date"].astype(str).max()
    stragglers = pending[
        (pending["matchday"] < matchday) & (pending["date"].astype(str) <= last_day)
    ]
    rows = pd.concat([stragglers, round_rows]).sort_values(["date", "match_id"])
    return matchday, rows


def features_hash(payload: dict) -> str:
    """Huella (sha256, 16 hex) del vector de features con que se predijo un partido."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=float)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def persist_predictions(
    conn: sqlite3.Connection,
    rows: pd.DataFrame,
    preds: pd.DataFrame,
    model_version: str,
    settings: Settings,
) -> int:
    """Guarda las predicciones (y el snapshot de sus features) antes del partido.

    El snapshot va a la tabla `features` pero NO al Parquet: ese archivo es el
    feature store completo que escribe `alaves features`, y sobrescribirlo con
    los diez partidos de una jornada lo dejaba inservible.
    """
    db.init_schema(conn)  # migra BDs antiguas (columna features_hash)
    persist_features(conn, rows, settings, write_parquet=False)
    hashes = {
        row["match_id"]: features_hash(feature_payload(row))
        for row in rows.to_dict(orient="records")
    }
    now = now_utc().isoformat(timespec="seconds")
    for p in preds.itertuples(index=False):
        conn.execute(
            "INSERT INTO predictions (match_id, model_version, created_at, p_home, p_draw, "
            "p_away, pred_result, pred_score, expected_goals_h, expected_goals_a, "
            "features_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                p.match_id,
                model_version,
                now,
                p.p_home,
                p.p_draw,
                p.p_away,
                p.pred_result,
                p.pred_score,
                p.expected_goals_h,
                p.expected_goals_a,
                hashes.get(p.match_id),
            ),
        )
    conn.commit()
    return len(preds)
