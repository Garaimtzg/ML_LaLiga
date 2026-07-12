"""Tests del Dixon-Coles (SPEC §11: verificado contra un caso resuelto a mano).

Caso a mano: lambda=1.2, mu=0.8, rho=-0.1. Con e^-2 = 0.1353352832:

    P(0,0) = e^-2 · tau(0,0) = e^-2 · (1 - 1.2·0.8·(-0.1)) = e^-2 · 1.096 = 0.14832747
    P(0,1) = e^-2 · 0.8 · (1 + 1.2·(-0.1))                 = e^-2 · 0.704 = 0.09527604
    P(1,0) = e^-2 · 1.2 · (1 + 0.8·(-0.1))                 = e^-2 · 1.104 = 0.14941015
    P(1,1) = e^-2 · 0.96 · (1 - (-0.1))                    = e^-2 · 1.056 = 0.14291406
    P(2,1) = e^-2 · (1.2²/2) · 0.8   (sin corrección)      = e^-2 · 0.576 = 0.07795312
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alaves_predictor.config import DixonColesConfig
from alaves_predictor.models import dixon_coles as dc

LAM, MU, RHO = 1.2, 0.8, -0.1

# (goles_local, goles_visitante) -> probabilidad calculada a mano (docstring)
HAND_VALUES = {
    (0, 0): 0.14832747,
    (0, 1): 0.09527604,
    (1, 0): 0.14941015,
    (1, 1): 0.14291406,
    (2, 1): 0.07795312,
}


def test_score_matrix_contra_caso_a_mano():
    matrix = dc.score_matrix(LAM, MU, RHO, max_goals=10)
    for (h, a), expected in HAND_VALUES.items():
        assert matrix[h, a] == pytest.approx(expected, abs=1e-6)


def test_tau_solo_corrige_marcadores_bajos():
    xs = np.array([0, 0, 1, 1, 2, 0])
    ys = np.array([0, 1, 0, 1, 2, 3])
    lam = np.full(6, LAM)
    mu = np.full(6, MU)
    out = dc.tau(xs, ys, lam, mu, RHO)
    assert out[:4] == pytest.approx([1.096, 0.88, 0.92, 1.1])
    assert out[4] == out[5] == 1.0  # el resto de marcadores no se toca


def test_la_correccion_conserva_la_probabilidad_total():
    # tau redistribuye masa entre los 4 marcadores bajos pero la suma sigue en 1
    for rho in (-0.15, 0.0, 0.1):
        matrix_sum = dc.score_matrix(LAM, MU, rho, max_goals=10).sum()
        assert matrix_sum == pytest.approx(1.0)


def test_outcome_probs_de_una_matriz_a_mano():
    # H = triángulo inferior (local marca más), D = diagonal, A = superior
    matrix = np.array([[0.2, 0.1], [0.3, 0.4]])
    probs = dc.outcome_probs(matrix)
    assert probs == pytest.approx([0.3, 0.6, 0.1])


def test_ponderacion_temporal_exponencial():
    # xi = ln(2)/365 => un partido de hace un año pesa exactamente 0.5
    xi = float(np.log(2) / 365)
    dates = pd.Series(["2025-06-01", "2024-06-01"])  # hoy y hace 365 días
    weights = dc.time_weights(dates, pd.Timestamp("2025-06-01"), xi)
    assert weights == pytest.approx([1.0, 0.5])


def _synthetic_league(seed: int = 42) -> tuple[pd.DataFrame, dict[str, float], float]:
    """Mini-liga de 6 equipos con parámetros conocidos: 8 vueltas completas."""
    rng = np.random.default_rng(seed)
    teams = ["t1", "t2", "t3", "t4", "t5", "t6"]
    attack = dict(zip(teams, [0.4, 0.2, 0.0, 0.0, -0.2, -0.4], strict=True))
    defense = dict(zip(teams, [0.3, 0.0, 0.1, -0.1, 0.0, -0.3], strict=True))
    gamma = 0.25
    rows = []
    date = pd.Timestamp("2020-01-01")
    for _ in range(8):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam = np.exp(attack[home] - defense[away] + gamma)
                mu = np.exp(attack[away] - defense[home])
                rows.append(
                    {
                        "home_id": home,
                        "away_id": away,
                        "home_goals": rng.poisson(lam),
                        "away_goals": rng.poisson(mu),
                        "date": str(date.date()),
                    }
                )
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows), attack, gamma


def test_fit_recupera_los_parametros_generadores():
    matches, true_attack, true_gamma = _synthetic_league()
    model = dc.fit(matches, DixonColesConfig(xi=0.0))  # xi=0: todos los partidos pesan igual

    # la ventaja de campo se recupera con margen razonable (240 partidos)
    assert model.home_advantage == pytest.approx(true_gamma, abs=0.15)
    # el ranking de ataques recuperado correlaciona fuerte con el real
    teams = list(true_attack)
    recovered = np.array([model.attack[t] for t in teams])
    real = np.array([true_attack[t] for t in teams])
    corr = np.corrcoef(recovered, real)[0, 1]
    assert corr > 0.8
    # identificabilidad: los ataques quedan centrados en 0
    assert np.mean(list(model.attack.values())) == pytest.approx(0.0, abs=1e-6)


def test_prediccion_y_marcador_mas_probable():
    matches, _, _ = _synthetic_league()
    model = dc.fit(matches, DixonColesConfig(xi=0.0))
    probs = model.outcome_probs("t1", "t6")
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] > probs[2]  # el mejor equipo, en casa, es favorito
    h, a, p = model.most_likely_score("t1", "t6")
    matrix = model.score_matrix("t1", "t6")
    assert matrix[h, a] == pytest.approx(p)
    assert p == pytest.approx(matrix.max())


def test_equipo_no_visto_hereda_el_proxy_de_colista():
    matches, _, _ = _synthetic_league()
    model = dc.fit(matches, DixonColesConfig(xi=0.0))
    atk, dfn = model._params_for("recien-ascendido")
    expected_atk = np.mean([model.attack[t] for t in model.proxy_teams])
    expected_dfn = np.mean([model.defense[t] for t in model.proxy_teams])
    assert atk == pytest.approx(expected_atk)
    assert dfn == pytest.approx(expected_dfn)
    # y sus probabilidades siguen siendo una distribución válida
    assert model.outcome_probs("recien-ascendido", "t1").sum() == pytest.approx(1.0)


def test_warm_start_produce_el_mismo_modelo():
    matches, _, _ = _synthetic_league()
    cfg = DixonColesConfig(xi=0.0)
    cold = dc.fit(matches, cfg)
    warm = dc.fit(matches, cfg, warm_start=cold)
    for team in cold.attack:
        assert warm.attack[team] == pytest.approx(cold.attack[team], abs=0.02)
    assert warm.home_advantage == pytest.approx(cold.home_advantage, abs=0.02)
