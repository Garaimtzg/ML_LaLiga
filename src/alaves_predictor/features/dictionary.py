"""Diccionario de variables del feature set v1: qué significa cada columna.

Muchas columnas se generan por patrón (`{lado}_{stat}[_venue]_ma{N}`), así que
en vez de un dict gigante que se desincronizaría, `describe()` RESUELVE el
significado de cualquier nombre: primero mira las columnas fijas y, si no,
parsea el patrón. Un test obligatorio verifica que ninguna feature del
pipeline se queda sin descripción.

Usos: apéndice del informe de importancia (F5), tooltips del dashboard (F6) y
consulta rápida (`docs/diccionario-features.md` explica lo mismo en prosa).
"""

from __future__ import annotations

import re

import pandas as pd

# --- Columnas fijas (no generadas por patrón) --------------------------------

_FIXED: dict[str, str] = {
    # Fuerza estructural (Elo)
    "elo_clubelo_home": "Elo de ClubElo del equipo LOCAL, el último publicado antes del partido",
    "elo_clubelo_away": "Elo de ClubElo del VISITANTE, el último publicado antes del partido",
    "elo_clubelo_diff": "Diferencia de Elo de ClubElo (local − visitante): ventaja de nivel",
    "elo_internal_home_pre": "Elo interno (ADR-013) del LOCAL justo antes del partido",
    "elo_internal_away_pre": "Elo interno (ADR-013) del VISITANTE justo antes del partido",
    "elo_internal_diff": "Diferencia de Elo interno (local − visitante)",
    "promoted_home": "1 si el LOCAL es recién ascendido (no jugó la temporada anterior)",
    "promoted_away": "1 si el VISITANTE es recién ascendido",
    # Contexto del partido
    "matchday": "Jornada oficial (fase de la temporada)",
    "month": "Mes del partido (estacionalidad)",
    "no_crowd": "1 si la temporada se jugó sin público (COVID, config)",
    "derby": "1 si el cruce es un derbi (pares definidos en config)",
    "h2h_home_ppg": "Puntos/partido del LOCAL en los últimos 5 enfrentamientos directos",
    # Descanso
    "home_rest_days": "Días de descanso del LOCAL desde su partido anterior de liga",
    "away_rest_days": "Días de descanso del VISITANTE desde su partido anterior de liga",
    # Rachas
    "home_win_streak": "Victorias consecutivas del LOCAL antes del partido",
    "home_loss_streak": "Derrotas consecutivas del LOCAL antes del partido",
    "away_win_streak": "Victorias consecutivas del VISITANTE antes del partido",
    "away_loss_streak": "Derrotas consecutivas del VISITANTE antes del partido",
    # Mercado (solo variante con cuotas)
    "imp_home": "Probabilidad implícita de victoria local en las cuotas de APERTURA (sin margen)",
    "imp_draw": "Probabilidad implícita de empate en las cuotas de APERTURA (sin margen)",
    "imp_away": "Prob. implícita de victoria visitante en las cuotas de APERTURA (sin margen)",
}

# --- Patrón de forma: {lado}_{stat}[_venue]_ma{N} ----------------------------

_SIDE = {"home": "LOCAL", "away": "VISITANTE"}

_STAT = {
    "points": "puntos por partido",
    "goals_for": "goles a favor",
    "goals_against": "goles en contra",
    "xg_for": "xG generado (calidad de ocasiones creadas)",
    "xg_against": "xG concedido (calidad de ocasiones del rival)",
    "g_minus_xg": "goles − xG (sobre/infrarrendimiento; tiende a revertir)",
}

_FORM_RE = re.compile(r"^(home|away)_([a-z_]+?)(_venue)?_ma(\d+)$")


def describe(feature: str) -> str | None:
    """Descripción en claro de una columna del feature set (None si no se reconoce)."""
    if feature in _FIXED:
        return _FIXED[feature]
    match = _FORM_RE.match(feature)
    if match:
        side, stat, venue, window = match.groups()
        if stat not in _STAT:
            return None
        scope = f" solo en su condición ({'casa' if side == 'home' else 'fuera'})" if venue else ""
        return (
            f"Media de {_STAT[stat]} del {_SIDE[side]} en sus últimos "
            f"{window} partidos{scope} (siempre anteriores al actual)"
        )
    return None


def feature_dictionary(columns: list[str]) -> pd.DataFrame:
    """Tabla (feature, descripción) para todas las columnas dadas, en su orden."""
    rows = []
    for col in columns:
        rows.append({"feature": col, "descripcion": describe(col) or "(sin descripción)"})
    return pd.DataFrame(rows)


def undocumented(columns: list[str]) -> list[str]:
    """Columnas sin descripción — el test del diccionario exige que sea vacío."""
    return [c for c in columns if describe(c) is None]
