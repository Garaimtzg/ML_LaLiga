"""Constructor del feature set v1 (SPEC §4, ADR-012).

Cada fila = un partido, con corte temporal estricto: `as_of_date` = día
anterior al partido, y toda feature usa solo información previa a esa fecha.
Salidas: tabla `features` (payload JSON versionado, reproducibilidad de SPEC
§12.4) y snapshot Parquet en data/features/.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from alaves_predictor.config import Settings
from alaves_predictor.features import form
from alaves_predictor.features.elo import compute_internal_elo, persist_internal_elo

# Columnas del feature set v1 (el payload JSON las guarda todas).
CONTEXT_COLS = ["matchday", "month", "no_crowd", "derby", "promoted_home", "promoted_away"]
MARKET_COLS = ["imp_home", "imp_draw", "imp_away"]


def load_matches_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Partidos con el xG de cada lado, ordenados cronológicamente."""
    matches = pd.read_sql_query(
        "SELECT match_id, season, matchday, date, home_id, away_id, home_goals, away_goals "
        "FROM matches WHERE status = 'finished' ORDER BY date, match_id",
        conn,
    )
    stats = pd.read_sql_query("SELECT match_id, team_id, is_home, xg FROM match_stats", conn)
    for is_home, col in ((1, "home_xg"), (0, "away_xg")):
        side = stats[stats["is_home"] == is_home][["match_id", "xg"]].rename(columns={"xg": col})
        matches = matches.merge(side, on="match_id", how="left")
    return matches


def _clubelo_asof(conn: sqlite3.Connection, matches: pd.DataFrame) -> pd.DataFrame:
    """Último Elo de ClubElo vigente en la fecha as_of (merge_asof hacia atrás)."""
    elo = pd.read_sql_query(
        "SELECT team_id, date, elo_clubelo FROM elo WHERE elo_clubelo IS NOT NULL ORDER BY date",
        conn,
    )
    elo["date"] = pd.to_datetime(elo["date"])
    out = matches.copy()
    out["as_of_ts"] = pd.to_datetime(out["date"]) - timedelta(days=1)
    for side in ("home", "away"):
        merged = pd.merge_asof(
            out[["match_id", "as_of_ts", f"{side}_id"]].sort_values("as_of_ts"),
            elo.rename(columns={"team_id": f"{side}_id", "elo_clubelo": f"elo_clubelo_{side}"}),
            left_on="as_of_ts",
            right_on="date",
            by=f"{side}_id",
            direction="backward",
        )[["match_id", f"elo_clubelo_{side}"]]
        out = out.merge(merged, on="match_id", how="left")
    return out.drop(columns=["as_of_ts"])


def _implied_odds(conn: sqlite3.Connection) -> pd.DataFrame:
    """Probabilidades implícitas de las cuotas de APERTURA, sin margen del bookmaker.

    Preferencia media de mercado > bet365 (SPEC §4.1: las de apertura como
    feature; las de CIERRE se reservan como baseline, no como feature).
    """
    odds = pd.read_sql_query(
        "SELECT match_id, bookmaker, open_h, open_d, open_a FROM odds "
        "WHERE bookmaker IN ('market_avg', 'bet365')",
        conn,
    )
    odds["priority"] = (odds["bookmaker"] != "market_avg").astype(int)  # avg primero
    odds = (
        odds.dropna(subset=["open_h", "open_d", "open_a"])
        .sort_values(["match_id", "priority"])
        .drop_duplicates("match_id")
    )
    inv = 1.0 / odds[["open_h", "open_d", "open_a"]].to_numpy()
    total = inv.sum(axis=1, keepdims=True)  # quita el margen: normaliza a suma 1
    probs = inv / total
    odds[["imp_home", "imp_draw", "imp_away"]] = probs
    return odds[["match_id", "imp_home", "imp_draw", "imp_away"]]


def _h2h_points(matches: pd.DataFrame, window: int = 5) -> pd.Series:
    """Puntos/partido del equipo local en los últimos `window` cruces directos previos.

    Muestra pequeña a propósito (SPEC: peso bajo; que SHAP lo juzgue en F5).
    """
    history: dict[tuple[str, str], list[tuple[str, int]]] = {}
    values: list[float | None] = []
    for m in matches.itertuples(index=False):
        key = tuple(sorted((m.home_id, m.away_id)))
        previous = history.setdefault(key, [])
        relevant = previous[-window:]
        if relevant:
            points = 0
            for winner, draw in relevant:
                if draw:
                    points += 1
                elif winner == m.home_id:
                    points += 3
            values.append(points / len(relevant))
        else:
            values.append(None)
        if m.home_goals > m.away_goals:
            previous.append((m.home_id, 0))
        elif m.home_goals < m.away_goals:
            previous.append((m.away_id, 0))
        else:
            previous.append(("", 1))
    return pd.Series(values, index=matches.index, dtype="Float64")


def _promoted_flags(matches: pd.DataFrame) -> pd.DataFrame:
    """Equipo recién ascendido = juega la temporada S pero no jugó la S-1.

    Para la primera temporada de la BD no hay referencia anterior: 0.
    """
    seasons = sorted(matches["season"].unique())
    teams_by_season = {
        s: set(matches.loc[matches["season"] == s, "home_id"])
        | set(matches.loc[matches["season"] == s, "away_id"])
        for s in seasons
    }
    prev = dict(zip(seasons[1:], seasons[:-1], strict=False))

    def promoted(season: str, team: str) -> int:
        if season not in prev:
            return 0
        return int(team not in teams_by_season[prev[season]])

    out = matches.copy()
    out["promoted_home"] = [
        promoted(s, t) for s, t in zip(out["season"], out["home_id"], strict=True)
    ]
    out["promoted_away"] = [
        promoted(s, t) for s, t in zip(out["season"], out["away_id"], strict=True)
    ]
    return out


def build_features(conn: sqlite3.Connection, settings: Settings) -> pd.DataFrame:
    """Construye el feature set v1 completo (una fila por partido jugado)."""
    cfg = settings.features
    matches = load_matches_frame(conn)

    # --- Elo interno (secuencial => sin fugas) + persistencia ---
    elo_hist = compute_internal_elo(matches, cfg.elo_internal)
    persist_internal_elo(conn, elo_hist)
    features = matches.merge(
        elo_hist[["match_id", "elo_internal_home_pre", "elo_internal_away_pre"]],
        on="match_id",
    )
    features["elo_internal_diff"] = (
        features["elo_internal_home_pre"] - features["elo_internal_away_pre"]
    )

    # --- Elo de ClubElo as-of (día anterior al partido) ---
    features = _clubelo_asof(conn, features)
    features["elo_clubelo_diff"] = features["elo_clubelo_home"] - features["elo_clubelo_away"]

    # --- Forma (ventanas móviles, splits por condición, descanso, rachas) ---
    long_df = form.long_format(matches)
    long_df = form.add_rolling_form(long_df, cfg.form_windows)
    long_df = form.add_rest_days(long_df)
    long_df = form.add_streaks(long_df)
    form_cols = [
        c for c in long_df.columns if "_ma" in c or c in ("rest_days", "win_streak", "loss_streak")
    ]
    for side, is_home in (("home", True), ("away", False)):
        side_df = long_df[long_df["is_home"] == is_home][["match_id", *form_cols]]
        side_df = side_df.rename(columns={c: f"{side}_{c}" for c in form_cols})
        features = features.merge(side_df, on="match_id", how="left")

    # --- Contexto del partido ---
    dates = pd.to_datetime(features["date"])
    features["month"] = dates.dt.month
    features["no_crowd"] = features["season"].isin(cfg.no_crowd_seasons).astype(int)
    derby_pairs = {tuple(sorted(pair)) for pair in cfg.derbies}
    features["derby"] = [
        int(tuple(sorted((h, a))) in derby_pairs)
        for h, a in zip(features["home_id"], features["away_id"], strict=True)
    ]
    features = _promoted_flags(features)
    features["h2h_home_ppg"] = _h2h_points(features)

    # --- Mercado (cuotas de apertura, sin margen) ---
    features = features.merge(_implied_odds(conn), on="match_id", how="left")

    # --- Target y corte temporal ---
    features["result"] = [
        "H" if hg > ag else ("D" if hg == ag else "A")
        for hg, ag in zip(features["home_goals"], features["away_goals"], strict=True)
    ]
    features["as_of_date"] = (dates - timedelta(days=1)).dt.date.astype(str)
    return features


# Columnas que son metadatos/target, no features del modelo.
META_COLS = [
    "match_id",
    "season",
    "date",
    "as_of_date",
    "home_id",
    "away_id",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "result",
]


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Nombres de las columnas de features del modelo (excluye metadatos y target)."""
    return [c for c in features.columns if c not in META_COLS]


def persist_features(conn: sqlite3.Connection, features: pd.DataFrame, settings: Settings) -> Path:
    """Guarda el snapshot: tabla `features` (JSON por partido) + Parquet. Devuelve la ruta."""
    version = settings.features.feature_set_version
    cols = feature_columns(features)
    now = datetime.now(UTC).isoformat()
    for row in features.itertuples(index=False):
        row_dict = row._asdict()
        # pd.NA/NaN no son serializables a JSON: se guardan como null
        payload = {c: (None if pd.isna(row_dict[c]) else row_dict[c]) for c in cols}
        payload["_computed_at"] = now
        conn.execute(
            "INSERT INTO features (match_id, feature_set_version, as_of_date, payload_json) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (match_id, feature_set_version) "
            "DO UPDATE SET as_of_date = excluded.as_of_date, "
            "payload_json = excluded.payload_json",
            (
                row_dict["match_id"],
                version,
                row_dict["as_of_date"],
                json.dumps(payload, ensure_ascii=False, default=float),
            ),
        )
    conn.commit()

    settings.data.features_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = settings.data.features_dir / f"features_{version}.parquet"
    features.to_parquet(parquet_path, index=False)
    return parquet_path
