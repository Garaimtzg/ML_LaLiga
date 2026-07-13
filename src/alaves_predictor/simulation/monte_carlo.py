"""Simulación Monte Carlo de la clasificación (SPEC §8, ADR-023).

Idea (SPEC §8): en vez de sumar los resultados más probables — que sesga al
ignorar la varianza y los empates — se muestrea la temporada completa muchas
veces. En cada simulación:

1. Cada partido pendiente se resuelve muestreando su resultado 1X2 de la
   distribución del ensemble (no del más probable).
2. Los puntos se reparten según ese resultado; el desempate fino usa la
   diferencia de goles muestreada del Dixon-Coles CONDICIONADA al resultado
   (el head-to-head reglamentario de LaLiga se aproxima por diferencia de
   goles — limitación documentada en SPEC §8.2).
3. Se ordena la tabla y se anota la posición final de cada equipo.

Agregando las N simulaciones se obtiene, por equipo, la distribución de
posiciones y de ahí P(título), P(Champions), P(Europa), P(descenso), la
posición esperada y los puntos esperados.

El cálculo está vectorizado con numpy: por cada partido se muestrean las N
simulaciones de golpe, de modo que N=10.000 sobre una liga entera corre en
segundos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alaves_predictor.models import dixon_coles

# Puntos por resultado, en el orden canónico [H, D, A] del proyecto.
_HOME_POINTS = np.array([3, 1, 0])
_AWAY_POINTS = np.array([0, 1, 3])


@dataclass
class Standing:
    """Situación de un equipo antes de simular (partidos ya jugados)."""

    points: int = 0
    goal_diff: int = 0
    played: int = 0


@dataclass
class RemainingMatch:
    """Un partido por jugar: probabilidades 1X2 y distribución de DG del DC."""

    home_id: str
    away_id: str
    probs: np.ndarray  # [P(H), P(D), P(A)] del ensemble
    # diferencia de goles (local−visitante) muestreable, condicionada al signo
    # del resultado: valores y probabilidades para victoria local y visitante
    home_gd_values: np.ndarray
    home_gd_probs: np.ndarray
    away_gd_values: np.ndarray
    away_gd_probs: np.ndarray


@dataclass
class SimulationResult:
    """Distribución de posiciones y agregados por equipo tras N simulaciones."""

    teams: list[str]
    n_sims: int
    position_counts: np.ndarray  # (n_teams, n_teams): veces en cada posición
    expected_points: np.ndarray  # (n_teams,)
    zones: dict[str, list[int]] = field(default_factory=dict)

    def _team_index(self, team: str) -> int:
        return self.teams.index(team)

    def position_distribution(self, team: str) -> np.ndarray:
        """Probabilidad de terminar en cada posición (1..n), suma 1."""
        counts = self.position_counts[self._team_index(team)]
        return counts / counts.sum()

    def expected_position(self, team: str) -> float:
        positions = np.arange(1, len(self.teams) + 1)
        return float((self.position_distribution(team) * positions).sum())

    def prob_between(self, team: str, low: int, high: int) -> float:
        """P(posición final en [low, high]) (posiciones 1-indexadas, inclusive)."""
        dist = self.position_distribution(team)
        return float(dist[low - 1 : high].sum())

    def prob_zone(self, team: str, zone: str) -> float:
        low, high = self.zones[zone]
        return self.prob_between(team, low, high)

    def points_for(self, team: str) -> float:
        return float(self.expected_points[self._team_index(team)])


def current_standings(played: pd.DataFrame) -> dict[str, Standing]:
    """Clasificación acumulada a partir de los partidos ya jugados.

    `played` necesita columnas home_id, away_id, home_goals, away_goals.
    """
    table: dict[str, Standing] = {}
    for m in played.itertuples(index=False):
        home = table.setdefault(m.home_id, Standing())
        away = table.setdefault(m.away_id, Standing())
        home.played += 1
        away.played += 1
        gd = int(m.home_goals) - int(m.away_goals)
        home.goal_diff += gd
        away.goal_diff -= gd
        if gd > 0:
            home.points += 3
        elif gd < 0:
            away.points += 3
        else:
            home.points += 1
            away.points += 1
    return table


def _conditional_gd(matrix: np.ndarray, sign: int) -> tuple[np.ndarray, np.ndarray]:
    """Distribución de la diferencia de goles (local−visitante) de un signo dado.

    sign > 0: victoria local (dg > 0); sign < 0: visitante (dg < 0). Devuelve
    (valores de dg, probabilidades normalizadas). Si el signo es imposible en
    la matriz (probabilidad ~0), devuelve un único valor ±1 de reserva.
    """
    size = matrix.shape[0]
    idx = np.arange(size)
    gd_grid = idx[:, None] - idx[None, :]  # dg de cada celda (i local, j visitante)
    mask = gd_grid > 0 if sign > 0 else gd_grid < 0
    gd_values = gd_grid[mask]
    weights = matrix[mask]
    total = weights.sum()
    if total <= 0:
        return np.array([sign]), np.array([1.0])
    # agrupar por valor de dg
    order = np.argsort(gd_values)
    gd_sorted, w_sorted = gd_values[order], weights[order]
    uniq, inv = np.unique(gd_sorted, return_inverse=True)
    probs = np.zeros(len(uniq))
    np.add.at(probs, inv, w_sorted)
    return uniq, probs / probs.sum()


def build_remaining(rows: pd.DataFrame, model: dixon_coles.DixonColesModel) -> list[RemainingMatch]:
    """Prepara los partidos pendientes: probs 1X2 + distribución de DG del DC.

    `rows` necesita home_id, away_id, p_home, p_draw, p_away (las probabilidades
    del ensemble). La diferencia de goles se toma del Dixon-Coles, condicionada
    al signo del resultado muestreado.
    """
    remaining: list[RemainingMatch] = []
    for m in rows.itertuples(index=False):
        matrix = model.score_matrix(m.home_id, m.away_id)
        home_vals, home_probs = _conditional_gd(matrix, sign=1)
        away_vals, away_probs = _conditional_gd(matrix, sign=-1)
        remaining.append(
            RemainingMatch(
                home_id=m.home_id,
                away_id=m.away_id,
                probs=np.array([m.p_home, m.p_draw, m.p_away], dtype=float),
                home_gd_values=home_vals,
                home_gd_probs=home_probs,
                away_gd_values=away_vals,
                away_gd_probs=away_probs,
            )
        )
    return remaining


def simulate(
    standings: dict[str, Standing],
    remaining: list[RemainingMatch],
    teams: list[str],
    n: int = 10000,
    seed: int = 42,
    zones: dict[str, list[int]] | None = None,
) -> SimulationResult:
    """Simula la temporada N veces y agrega la distribución de posiciones.

    `teams` es la lista completa de equipos de la liga (incluye los que ya no
    tienen partidos pendientes). Semilla fija por defecto (reproducibilidad,
    CLAUDE.md §2); parametrizable para las bandas de incertidumbre.
    """
    rng = np.random.default_rng(seed)
    index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    points = np.zeros((n, n_teams), dtype=float)
    goal_diff = np.zeros((n, n_teams), dtype=float)
    for team, standing in standings.items():
        points[:, index[team]] = standing.points
        goal_diff[:, index[team]] = standing.goal_diff

    for match in remaining:
        h, a = index[match.home_id], index[match.away_id]
        outcomes = rng.choice(3, size=n, p=match.probs)  # 0=H, 1=D, 2=A
        points[:, h] += _HOME_POINTS[outcomes]
        points[:, a] += _AWAY_POINTS[outcomes]

        gd = np.zeros(n)
        home_win = outcomes == 0
        away_win = outcomes == 2
        gd[home_win] = rng.choice(
            match.home_gd_values, size=int(home_win.sum()), p=match.home_gd_probs
        )
        gd[away_win] = rng.choice(
            match.away_gd_values, size=int(away_win.sum()), p=match.away_gd_probs
        )
        goal_diff[:, h] += gd
        goal_diff[:, a] -= gd

    position_counts = _rank_positions(points, goal_diff, rng)
    return SimulationResult(
        teams=teams,
        n_sims=n,
        position_counts=position_counts,
        expected_points=points.mean(axis=0),
        zones=zones or {},
    )


def _rank_positions(
    points: np.ndarray, goal_diff: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Cuenta cuántas veces cada equipo acaba en cada posición.

    Orden: puntos, luego diferencia de goles; los empates exactos se rompen al
    azar (ruido pequeño) para no favorecer sistemáticamente a un índice.
    """
    n, n_teams = points.shape
    # score con puntos dominantes sobre la DG, más un desempate aleatorio ínfimo
    tiebreak = rng.random((n, n_teams))
    score = points * 1000.0 + goal_diff + tiebreak * 1e-6
    # posición 1 = mayor score: doble argsort sobre el score negado
    order = np.argsort(-score, axis=1)
    positions = np.empty_like(order)
    rows = np.arange(n)[:, None]
    positions[rows, order] = np.arange(1, n_teams + 1)[None, :]

    counts = np.zeros((n_teams, n_teams), dtype=np.int64)
    for pos in range(1, n_teams + 1):
        counts[:, pos - 1] = (positions == pos).sum(axis=0)
    return counts
