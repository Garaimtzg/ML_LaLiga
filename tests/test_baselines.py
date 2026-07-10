"""Baselines de SPEC §6.1: frecuencias, Elo logístico y cuotas de cierre."""

import numpy as np
import pandas as pd
import pytest

from alaves_predictor.etl import db
from alaves_predictor.etl.ingest import ingest_historical
from alaves_predictor.evaluation.baselines import (
    BaselineResult,
    closing_odds_probs,
    elo_logistic_probs,
    historical_frequencies,
    run_baselines,
    write_report,
)
from alaves_predictor.features.build import build_features


def test_frecuencias_historicas() -> None:
    train = pd.DataFrame({"result": ["H", "H", "D", "A"]})
    probs = historical_frequencies(train)
    assert probs == pytest.approx([0.5, 0.25, 0.25])
    assert probs.sum() == pytest.approx(1.0)


def test_elo_logistico_aprende_la_direccion() -> None:
    """Con datos sintéticos separables, más elo_diff debe dar más P(victoria local)."""
    rng = np.random.default_rng(42)
    diffs = rng.uniform(-400, 400, 600)
    results = ["H" if d > 50 else ("A" if d < -50 else "D") for d in diffs]
    train = pd.DataFrame({"elo_clubelo_diff": diffs, "result": results})
    test = pd.DataFrame({"elo_clubelo_diff": [300.0, -300.0]})
    probs = elo_logistic_probs(train, test)
    assert probs.shape == (2, 3)
    assert probs[0].sum() == pytest.approx(1.0, abs=1e-9)
    assert probs[0, 0] > probs[1, 0]  # más diff -> más P(H)
    assert probs[1, 2] > probs[0, 2]  # menos diff -> más P(A)


def test_cuotas_cierre_normalizadas_en_mini_liga(mini_settings, fake_fetch) -> None:
    conn = db.connect(mini_settings.data.db_path)
    try:
        ingest_historical(conn, mini_settings)
        match_ids = pd.read_sql_query("SELECT match_id FROM matches", conn)["match_id"]
        probs = closing_odds_probs(conn, match_ids)
        assert len(probs) == 12
        sums = probs[["p_home", "p_draw", "p_away"]].sum(axis=1)
        assert sums.apply(lambda s: abs(s - 1.0) < 1e-9).all()
    finally:
        conn.close()


def test_walk_forward_sin_temporadas_de_entrenamiento_no_evalua(mini_settings, fake_fetch) -> None:
    """Con una sola temporada no hay train previo: cero resultados, nunca
    entrenar y evaluar sobre lo mismo (CLAUDE.md §5.2)."""
    conn = db.connect(mini_settings.data.db_path)
    try:
        ingest_historical(conn, mini_settings)
        features = build_features(conn, mini_settings)
        results = run_baselines(conn, features, mini_settings, n_test_seasons=3)
        assert results == []
    finally:
        conn.close()


def test_write_report_genera_markdown(tmp_path) -> None:
    results = [
        BaselineResult(
            "cuotas_cierre",
            "2024-25",
            380,
            {"log_loss": 0.98, "brier": 0.58, "rps": 0.19, "accuracy": 0.55},
        )
    ]
    path = write_report(results, tmp_path)
    text = path.read_text()
    assert "cuotas_cierre" in text and "0.9800" in text
    assert "Media por baseline" in text
