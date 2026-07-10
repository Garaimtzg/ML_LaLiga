"""Elo interno verificado contra un caso resuelto a mano (SPEC §11)."""

import pandas as pd
import pytest

from alaves_predictor.config import EloInternalConfig
from alaves_predictor.features.elo import compute_internal_elo, expected_home_score

CFG = EloInternalConfig(k=20.0, home_advantage=60.0, initial_rating=1500.0)


def test_esperado_con_ratings_iguales_y_ventaja_de_campo() -> None:
    # A mano: diff = 60 -> E = 1/(1+10^(-60/400)) = 1/(1+0.70795) = 0.58550
    assert expected_home_score(1500, 1500, CFG) == pytest.approx(0.58550, abs=1e-4)


def test_actualizacion_tras_victoria_local_resuelta_a_mano() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2018-08-18",
                "home_id": "a",
                "away_id": "b",
                "home_goals": 2,
                "away_goals": 0,
            },
            {
                "match_id": "m2",
                "date": "2018-08-25",
                "home_id": "b",
                "away_id": "a",
                "home_goals": 1,
                "away_goals": 1,
            },
        ]
    )
    hist = compute_internal_elo(matches, CFG)

    # m1: pre 1500/1500; delta = 20·(1 − 0.58566) = 8.287
    assert hist.loc[0, "elo_internal_home_pre"] == 1500.0
    assert hist.loc[0, "elo_internal_home_post"] == pytest.approx(1508.287, abs=1e-2)
    assert hist.loc[0, "elo_internal_away_post"] == pytest.approx(1491.713, abs=1e-2)

    # m2: los PRE son los POST del partido anterior (secuencial, sin fugas)
    assert hist.loc[1, "elo_internal_home_pre"] == pytest.approx(1491.713, abs=1e-2)
    assert hist.loc[1, "elo_internal_away_pre"] == pytest.approx(1508.287, abs=1e-2)
    # empate: b (local, inferior) gana puntos; E_b = 1/(1+10^(-(1491.71+60−1508.29)/400))
    expected_b = expected_home_score(1491.713, 1508.287, CFG)
    delta = 20.0 * (0.5 - expected_b)
    assert hist.loc[1, "elo_internal_home_post"] == pytest.approx(1491.713 + delta, abs=1e-2)


def test_elo_se_conserva_en_suma() -> None:
    """El Elo es de suma cero: lo que gana uno lo pierde el otro."""
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2018-08-18",
                "home_id": "a",
                "away_id": "b",
                "home_goals": 0,
                "away_goals": 3,
            }
        ]
    )
    hist = compute_internal_elo(matches, CFG)
    total_pre = hist.loc[0, "elo_internal_home_pre"] + hist.loc[0, "elo_internal_away_pre"]
    total_post = hist.loc[0, "elo_internal_home_post"] + hist.loc[0, "elo_internal_away_post"]
    assert total_pre == pytest.approx(total_post)
