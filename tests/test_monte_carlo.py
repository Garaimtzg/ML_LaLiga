"""Tests del simulador Monte Carlo de la clasificación (SPEC §8/§11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alaves_predictor.config import DixonColesConfig
from alaves_predictor.models import dixon_coles as dc
from alaves_predictor.simulation import monte_carlo as mc


def _match(home, away, probs, gd_home=1, gd_away=1):
    """RemainingMatch simple: DG determinista (±gd) para poder razonar a mano."""
    return mc.RemainingMatch(
        home_id=home,
        away_id=away,
        probs=np.array(probs, dtype=float),
        home_gd_values=np.array([gd_home]),
        home_gd_probs=np.array([1.0]),
        away_gd_values=np.array([-gd_away]),
        away_gd_probs=np.array([1.0]),
    )


def test_current_standings_cuenta_puntos_y_diferencia():
    played = pd.DataFrame(
        [
            {"home_id": "a", "away_id": "b", "home_goals": 2, "away_goals": 0},  # a gana
            {"home_id": "b", "away_id": "c", "home_goals": 1, "away_goals": 1},  # empate
            {"home_id": "c", "away_id": "a", "home_goals": 3, "away_goals": 1},  # c gana
        ]
    )
    table = mc.current_standings(played)
    assert table["a"].points == 3 and table["a"].goal_diff == 0  # +2, -2
    assert table["a"].played == 2
    assert table["b"].points == 1 and table["b"].goal_diff == -2
    assert table["c"].points == 4 and table["c"].goal_diff == 2


def test_conditional_gd_es_distribucion_valida_del_signo():
    matrix = np.array([[0.20, 0.10, 0.05], [0.15, 0.20, 0.05], [0.10, 0.10, 0.05]])
    vals_h, probs_h = mc._conditional_gd(matrix, sign=1)
    assert (vals_h > 0).all()
    assert probs_h.sum() == pytest.approx(1.0)
    vals_a, probs_a = mc._conditional_gd(matrix, sign=-1)
    assert (vals_a < 0).all()
    assert probs_a.sum() == pytest.approx(1.0)


def test_posiciones_suman_uno_por_equipo():
    """SPEC §11: la distribución de posiciones de cada equipo debe sumar 1."""
    teams = ["a", "b", "c", "d"]
    remaining = [
        _match("a", "b", [0.5, 0.3, 0.2]),
        _match("c", "d", [0.4, 0.3, 0.3]),
        _match("a", "c", [0.4, 0.3, 0.3]),
        _match("b", "d", [0.4, 0.3, 0.3]),
    ]
    result = mc.simulate({}, remaining, teams, n=2000, seed=1)
    for team in teams:
        assert result.position_distribution(team).sum() == pytest.approx(1.0)
    # cada posición se reparte entre los 4 equipos: columnas suman n_sims
    assert result.position_counts.sum(axis=0).tolist() == [2000] * 4


def test_reproducible_con_semilla():
    teams = ["a", "b", "c"]
    remaining = [_match("a", "b", [0.4, 0.3, 0.3]), _match("b", "c", [0.4, 0.3, 0.3])]
    r1 = mc.simulate({}, remaining, teams, n=500, seed=7)
    r2 = mc.simulate({}, remaining, teams, n=500, seed=7)
    assert np.array_equal(r1.position_counts, r2.position_counts)
    r3 = mc.simulate({}, remaining, teams, n=500, seed=8)
    assert not np.array_equal(r1.position_counts, r3.position_counts)


def test_lider_destacado_gana_casi_siempre():
    teams = ["fuerte", "x", "y", "z"]
    standings = {"fuerte": mc.Standing(points=30, goal_diff=20, played=10)}
    # el líder gana sus dos partidos con casi total seguridad
    remaining = [
        _match("fuerte", "x", [0.95, 0.03, 0.02]),
        _match("y", "fuerte", [0.02, 0.03, 0.95]),
    ]
    result = mc.simulate(standings, remaining, teams, n=3000, seed=3)
    assert result.prob_between("fuerte", 1, 1) > 0.98
    assert result.expected_position("fuerte") == pytest.approx(1.0, abs=0.05)


def test_puntos_esperados_sin_partidos_pendientes():
    teams = ["a", "b"]
    standings = {"a": mc.Standing(points=10, goal_diff=5), "b": mc.Standing(points=7)}
    result = mc.simulate(standings, [], teams, n=100, seed=1)
    assert result.points_for("a") == pytest.approx(10.0)
    assert result.points_for("b") == pytest.approx(7.0)
    assert result.prob_between("a", 1, 1) == pytest.approx(1.0)  # a siempre 1º


def test_zonas_configurables():
    teams = ["a", "b", "c", "d"]
    result = mc.simulate(
        {"a": mc.Standing(points=9)},
        [_match("b", "c", [0.4, 0.3, 0.3])],
        teams,
        n=500,
        seed=1,
        zones={"champions": [1, 2], "descenso": [4, 4]},
    )
    assert 0.0 <= result.prob_zone("a", "champions") <= 1.0
    assert result.prob_zone("a", "champions") == pytest.approx(1.0)  # a domina


def test_build_remaining_desde_un_dixon_coles_real():
    rng = np.random.default_rng(0)
    teams = ["t1", "t2", "t3", "t4"]
    rows = []
    date = pd.Timestamp("2020-01-01")
    for _ in range(6):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append(
                        {
                            "home_id": h,
                            "away_id": a,
                            "home_goals": rng.poisson(1.3),
                            "away_goals": rng.poisson(1.1),
                            "date": str(date.date()),
                        }
                    )
                    date += pd.Timedelta(days=1)
    model = dc.fit(pd.DataFrame(rows), DixonColesConfig(xi=0.0))
    future = pd.DataFrame(
        [{"home_id": "t1", "away_id": "t2", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}]
    )
    remaining = mc.build_remaining(future, model)
    assert len(remaining) == 1
    r = remaining[0]
    assert r.home_gd_probs.sum() == pytest.approx(1.0)
    assert (r.home_gd_values > 0).all() and (r.away_gd_values < 0).all()
