"""Datos para el dashboard (SPEC §9), como funciones puras y testeables.

Toda la lógica vive aquí (en `src/`, con tests); `app/dashboard.py` se limita a
llamar a estas funciones y pintarlas con Streamlit/Plotly (CLAUDE.md §2: nada
de lógica solo en la capa de presentación).
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from alaves_predictor.config import Settings
from alaves_predictor.simulation import monte_carlo as mc
from alaves_predictor.simulation.project import Projection

_OUTCOME_LABEL = {"H": "1 (local)", "D": "X (empate)", "A": "2 (visitante)"}


def team_name(settings: Settings, team_id: str) -> str:
    return settings.teams[team_id].name if team_id in settings.teams else team_id


# --- Clasificación real y proyectada -----------------------------------------


def standings_table(played: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Clasificación real a partir de los partidos jugados, ordenada por pts y DG."""
    table = mc.current_standings(played)
    rows = [
        {
            "team_id": t,
            "Equipo": team_name(settings, t),
            "PJ": s.played,
            "Pts": s.points,
            "DG": s.goal_diff,
        }
        for t, s in table.items()
    ]
    df = pd.DataFrame(rows).sort_values(["Pts", "DG"], ascending=False).reset_index(drop=True)
    df.insert(0, "Pos", range(1, len(df) + 1))
    return df


def projection_table(projection: Projection, settings: Settings) -> pd.DataFrame:
    """Tabla de proyección por equipo, ordenada por posición esperada."""
    result = projection.result
    ranked = sorted(projection.teams, key=result.expected_position)
    rows = []
    for team in ranked:
        rows.append(
            {
                "team_id": team,
                "Equipo": team_name(settings, team),
                "Pts esperados": round(result.points_for(team), 1),
                "Pos esperada": round(result.expected_position(team), 1),
                "P(título)": result.prob_zone(team, "titulo"),
                "P(Champions)": result.prob_zone(team, "champions"),
                "P(Europa)": result.prob_zone(team, "europa"),
                "P(descenso)": result.prob_zone(team, "descenso"),
            }
        )
    return pd.DataFrame(rows)


def position_heatmap(projection: Projection, settings: Settings) -> pd.DataFrame:
    """Matriz equipo × posición con la probabilidad de acabar en cada puesto.

    Filas ordenadas por posición esperada; columnas = posiciones 1..N.
    """
    result = projection.result
    ranked = sorted(projection.teams, key=result.expected_position)
    data = {team_name(settings, t): result.position_distribution(t) for t in ranked}
    df = pd.DataFrame(data, index=range(1, len(projection.teams) + 1)).T
    df.columns.name = "posicion"
    return df


# --- Predicciones de una jornada ---------------------------------------------


def available_matchdays(features: pd.DataFrame, season: str) -> list[int]:
    md = features.loc[features["season"] == season, "matchday"].dropna().unique()
    return sorted(int(x) for x in md)


def matchday_predictions(
    predict_matches, features: pd.DataFrame, settings: Settings, season: str, matchday: int
) -> pd.DataFrame:
    """Predicciones formateadas de una jornada (SPEC §2), con nombres de equipo.

    `predict_matches(rows)` es un callable que devuelve el DataFrame de
    predicciones (normalmente `lambda r: bundle.predict_matches(r, variant)`).
    """
    rows = features[(features["season"] == season) & (features["matchday"] == matchday)]
    if rows.empty:
        return pd.DataFrame()
    preds = predict_matches(rows)
    preds = preds.merge(
        rows[["match_id", "result"]], on="match_id", how="left", suffixes=("", "_real")
    )
    preds["Local"] = preds["home_id"].map(lambda t: team_name(settings, t))
    preds["Visitante"] = preds["away_id"].map(lambda t: team_name(settings, t))
    preds["Predicho"] = preds["pred_result"].map(_OUTCOME_LABEL)
    preds["Real"] = preds["result"].map(
        lambda r: _OUTCOME_LABEL.get(r, "—") if pd.notna(r) else "—"
    )
    return preds


# --- Detalle del equipo foco -------------------------------------------------


def focus_timeline(features: pd.DataFrame, settings: Settings, season: str) -> pd.DataFrame:
    """Serie por jornada del equipo foco: Elo, xG a favor/en contra y forma.

    Extrae, de cada partido del equipo, los valores que le corresponden según
    juegue de local o visitante.
    """
    focus = settings.focus_team
    df = features[
        (features["season"] == season)
        & ((features["home_id"] == focus) | (features["away_id"] == focus))
    ].copy()
    if df.empty:
        return pd.DataFrame()
    is_home = df["home_id"] == focus

    def pick(home_col: str, away_col: str) -> np.ndarray:
        return np.where(is_home, df[home_col], df[away_col])

    out = pd.DataFrame(
        {
            "matchday": df["matchday"].to_numpy(),
            "date": df["date"].to_numpy(),
            "rival": np.where(is_home, df["away_id"], df["home_id"]),
            "local": is_home.to_numpy(),
            "elo": pick("elo_clubelo_home", "elo_clubelo_away"),
            "xg_favor": pick("home_xg", "away_xg"),
            "xg_contra": pick("away_xg", "home_xg"),
            "forma_pts_ma5": pick("home_points_ma5", "away_points_ma5"),
        }
    )
    out["rival"] = out["rival"].map(lambda t: team_name(settings, t))
    return out.sort_values("matchday").reset_index(drop=True)


# --- Rendimiento del modelo y registro ---------------------------------------


def model_registry_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Versiones registradas con sus métricas de validación (SPEC §9.5)."""
    rows = conn.execute(
        "SELECT model_version, trained_at, train_window, metrics_json "
        "FROM model_registry ORDER BY trained_at DESC"
    ).fetchall()
    out = []
    for r in rows:
        metrics = json.loads(r["metrics_json"])
        out.append(
            {
                "Versión": r["model_version"],
                "Entrenado": r["trained_at"],
                "Ventana": r["train_window"],
                "Temporada val": metrics.get("val_season", "—"),
                "Log-loss ref": round(metrics.get("reference_logloss", float("nan")), 4),
                "Promocionado": "sí" if metrics.get("promoted", True) else "no",
            }
        )
    return pd.DataFrame(out)


def prediction_log(conn: sqlite3.Connection, settings: Settings) -> pd.DataFrame:
    """Predicciones persistidas cruzadas con el resultado real (si ya se conoce)."""
    preds = pd.read_sql_query(
        "SELECT p.match_id, p.model_version, p.created_at, p.p_home, p.p_draw, p.p_away, "
        "p.pred_result, m.home_id, m.away_id, m.home_goals, m.away_goals, m.matchday "
        "FROM predictions p JOIN matches m ON m.match_id = p.match_id ORDER BY p.created_at",
        conn,
    )
    if preds.empty:
        return preds
    played = preds["home_goals"].notna() & preds["away_goals"].notna()
    real = np.where(
        preds["home_goals"] > preds["away_goals"],
        "H",
        np.where(preds["home_goals"] == preds["away_goals"], "D", "A"),
    )
    preds["result"] = pd.Series(real, index=preds.index).where(played, None)
    preds["acierto"] = (preds["pred_result"] == preds["result"]).where(played, np.nan)
    preds["Local"] = preds["home_id"].map(lambda t: team_name(settings, t))
    preds["Visitante"] = preds["away_id"].map(lambda t: team_name(settings, t))
    return preds


# --- Registro de decisiones (ADRs) -------------------------------------------

_ADR_RE = re.compile(r"^(\d+)-(.+)\.md$")


def adr_list(decisions_dir: Path) -> pd.DataFrame:
    """Índice de ADRs (número, título del H1, ruta) ordenado por número."""
    rows = []
    for path in sorted(decisions_dir.glob("*.md")):
        match = _ADR_RE.match(path.name)
        if not match:
            continue
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        title = first_line.lstrip("# ").strip()
        rows.append({"num": int(match.group(1)), "titulo": title, "path": str(path)})
    return pd.DataFrame(rows).sort_values("num").reset_index(drop=True)
