"""Features de forma reciente (SPEC §4.1): ventanas móviles sin fuga temporal.

Regla anti-leakage central: toda media móvil se calcula sobre valores
DESPLAZADOS una posición (`shift(1)`), de modo que la fila de un partido solo
ve partidos ANTERIORES de ese equipo — nunca el propio partido ni futuros.
El test anti-leakage de tests/test_features.py lo verifica empíricamente.
"""

from __future__ import annotations

import numpy as np
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
        # partidos sin jugar (goles NaN): puntos/won/lost quedan NaN para no
        # contaminar ventanas ni rachas de partidos posteriores
        played = df["goals_for"].notna() & df["goals_against"].notna()
        points = np.where(
            df["goals_for"] > df["goals_against"],
            3.0,
            np.where(df["goals_for"] == df["goals_against"], 1.0, 0.0),
        )
        df["points"] = np.where(played, points, np.nan)
        df["won"] = np.where(played, (df["goals_for"] > df["goals_against"]).astype(float), np.nan)
        df["lost"] = np.where(played, (df["goals_for"] < df["goals_against"]).astype(float), np.nan)
        return df

    long_df = pd.concat([side(True), side(False)], ignore_index=True)
    long_df["g_minus_xg"] = long_df["goals_for"] - long_df["xg_for"]
    return long_df.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _shifted_rolling_mean(group: pd.Series, window: int, played: pd.Series) -> pd.Series:
    """Media de la estadística en los `window` últimos partidos JUGADOS antes de cada fila.

    - Partido jugado: la ventana son sus `window` partidos jugados anteriores
      (`shift(1)`: nunca el propio, anti-leakage). Si a alguno le falta el dato
      (xG ausente) se promedia el resto; si no lo tiene ninguno, queda vacía.
    - Partido por jugar: hereda la media tal como quedó tras el último partido
      jugado, igual que las rachas.

    Los partidos por jugar no ocupan sitio en la ventana. Antes lo ocupaban como
    huecos: proyectando la temporada, a 6 jornadas vista la media de 5 ya no
    tenía datos y casi todas las features de forma iban vacías (ADR-031).

    Solo heredan los partidos por jugar, no los jugados sin dato: con una
    temporada entera sin xG (Understat caído, ADR-033), heredar congelaba la
    media de mayo de 2025 en todos los partidos de año y medio, un valor que
    parece válido y no lo es. Un dato que no existe queda vacío.
    """
    is_played = played.loc[group.index].to_numpy()
    done = group[is_played]
    before_each = done.shift(1).rolling(window, min_periods=1).mean()
    after_each = done.rolling(window, min_periods=1).mean().to_numpy()
    out = before_each.reindex(group.index)
    # Por posición, no con ffill: ffill saltaría también los jugados cuya media
    # es nula y rescataría un valor viejo. n_before = jugados anteriores a la fila.
    n_before = np.cumsum(is_played) - is_played
    for pos in np.flatnonzero(~is_played):
        k = n_before[pos]
        out.iloc[pos] = after_each[k - 1] if k > 0 else np.nan
    return out


def add_rolling_form(long_df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Añade medias móviles generales y separadas por condición local/visitante."""
    out = long_df.copy()
    played = out["points"].notna()  # points solo es nulo en un partido por jugar
    for window in windows:
        for stat in FORM_STATS:
            out[f"{stat}_ma{window}"] = out.groupby("team")[stat].transform(
                _shifted_rolling_mean, window=window, played=played
            )
            # Forma específica en la MISMA condición (local en casa, visitante fuera):
            # ventana sobre los partidos previos del equipo en esa condición.
            out[f"{stat}_venue_ma{window}"] = out.groupby(["team", "is_home"])[stat].transform(
                _shifted_rolling_mean, window=window, played=played
            )
    return out


def add_rest_days(long_df: pd.DataFrame) -> pd.DataFrame:
    """Días desde el partido anterior del equipo (solo liga; ver limitación en ADR-012)."""
    out = long_df.copy()
    dates = pd.to_datetime(out["date"])
    out["rest_days"] = (dates - dates.groupby(out["team"]).shift(1)).dt.days.astype("Float64")
    return out


def _streak_before(series: pd.Series) -> pd.Series:
    """Racha de la estadística binaria ANTES de cada partido (0 si se rompió).

    Un partido sin jugar (NaN) no rompe ni alarga la racha: la mejor estimación
    disponible para partidos posteriores es la racha actual.
    """
    streaks: list[float] = []
    current = 0
    for value in series:
        streaks.append(current)
        if not pd.isna(value):
            current = current + 1 if value == 1 else 0
    return pd.Series(streaks, index=series.index, dtype="Float64")


def add_streaks(long_df: pd.DataFrame) -> pd.DataFrame:
    """Rachas de victorias/derrotas consecutivas previas al partido."""
    out = long_df.copy()
    out["win_streak"] = out.groupby("team")["won"].transform(_streak_before)
    out["loss_streak"] = out.groupby("team")["lost"].transform(_streak_before)
    return out
