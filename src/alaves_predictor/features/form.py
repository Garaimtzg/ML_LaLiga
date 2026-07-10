"""Features de forma reciente (SPEC §4.1): ventanas móviles sin fuga temporal.

Regla anti-leakage central: toda media móvil se calcula sobre valores
DESPLAZADOS una posición (`shift(1)`), de modo que la fila de un partido solo
ve partidos ANTERIORES de ese equipo — nunca el propio partido ni futuros.
El test anti-leakage de tests/test_features.py lo verifica empíricamente.
"""

from __future__ import annotations

import pandas as pd

# Estadísticas por equipo-partido sobre las que se calculan las medias móviles.
FORM_STATS = ["points", "goals_for", "goals_against", "xg_for", "xg_against", "g_minus_xg"]


def long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Convierte partidos (una fila por partido) a formato largo (una por equipo-partido)."""

    def side(is_home: bool) -> pd.DataFrame:
        prefix, other = ("home", "away") if is_home else ("away", "home")
        df = pd.DataFrame(
            {
                "match_id": matches["match_id"],
                "season": matches["season"],
                "date": matches["date"],
                "team": matches[f"{prefix}_id"],
                "is_home": is_home,
                "goals_for": matches[f"{prefix}_goals"],
                "goals_against": matches[f"{other}_goals"],
                "xg_for": matches[f"{prefix}_xg"],
                "xg_against": matches[f"{other}_xg"],
            }
        )
        df["points"] = df.apply(
            lambda r: (
                3
                if r["goals_for"] > r["goals_against"]
                else (1 if r["goals_for"] == r["goals_against"] else 0)
            ),
            axis=1,
        )
        df["won"] = (df["goals_for"] > df["goals_against"]).astype(int)
        df["lost"] = (df["goals_for"] < df["goals_against"]).astype(int)
        return df

    long_df = pd.concat([side(True), side(False)], ignore_index=True)
    long_df["g_minus_xg"] = long_df["goals_for"] - long_df["xg_for"]
    return long_df.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _shifted_rolling_mean(group: pd.Series, window: int) -> pd.Series:
    # shift(1): el partido actual no se ve a sí mismo (anti-leakage)
    return group.shift(1).rolling(window, min_periods=1).mean()


def add_rolling_form(long_df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Añade medias móviles generales y separadas por condición local/visitante."""
    out = long_df.copy()
    for window in windows:
        for stat in FORM_STATS:
            out[f"{stat}_ma{window}"] = out.groupby("team")[stat].transform(
                _shifted_rolling_mean, window=window
            )
            # Forma específica en la MISMA condición (local en casa, visitante fuera):
            # ventana sobre los partidos previos del equipo en esa condición.
            out[f"{stat}_venue_ma{window}"] = out.groupby(["team", "is_home"])[stat].transform(
                _shifted_rolling_mean, window=window
            )
    return out


def add_rest_days(long_df: pd.DataFrame) -> pd.DataFrame:
    """Días desde el partido anterior del equipo (solo liga; ver limitación en ADR-012)."""
    out = long_df.copy()
    dates = pd.to_datetime(out["date"])
    out["rest_days"] = (dates - dates.groupby(out["team"]).shift(1)).dt.days.astype("Float64")
    return out


def _streak_before(series: pd.Series) -> pd.Series:
    """Racha de la estadística binaria ANTES de cada partido (0 si se rompió)."""
    streaks: list[float] = []
    current = 0
    for value in series:
        streaks.append(current)
        current = current + 1 if value == 1 else 0
    return pd.Series(streaks, index=series.index, dtype="Float64")


def add_streaks(long_df: pd.DataFrame) -> pd.DataFrame:
    """Rachas de victorias/derrotas consecutivas previas al partido."""
    out = long_df.copy()
    out["win_streak"] = out.groupby("team")["won"].transform(_streak_before)
    out["loss_streak"] = out.groupby("team")["lost"].transform(_streak_before)
    return out
