"""Tests del backtesting walk-forward jornada a jornada (F3)."""

from __future__ import annotations

import numpy as np

from alaves_predictor.evaluation import backtest as bt
from alaves_predictor.evaluation.baselines import BaselineResult


def test_run_backtest_evalua_todos_los_modelos(synthetic_features, model_settings):
    output = bt.run_backtest(synthetic_features, model_settings, n_test_seasons=2)
    models = {r.model for r in output.rows}
    assert models == {
        "dixon_coles",
        "lgbm_con_cuotas",
        "lgbm_sin_cuotas",
        "ensemble_con_cuotas",
        "ensemble_sin_cuotas",
    }
    seasons = {r.season for r in output.rows}
    assert seasons == {"2020-21", "2021-22"}  # las 2 últimas del sintético
    for row in output.rows:
        assert row.n_matches == 60  # liga sintética completa por temporada
        assert np.isfinite(row.metrics["log_loss"])
        # sin partidos del equipo foco (alaves) en la liga sintética
        assert row.alaves_accuracy is None
    assert not output.reliability.empty


def test_los_modelos_baten_a_las_frecuencias(synthetic_features, model_settings):
    """Sanidad: con features informativas, el ensemble bate al azar frecuentista."""
    output = bt.run_backtest(synthetic_features, model_settings, n_test_seasons=1)
    test = synthetic_features[synthetic_features["season"] == "2021-22"]
    train = synthetic_features[synthetic_features["season"] < "2021-22"]
    freqs = train["result"].value_counts(normalize=True)
    base = np.tile([freqs.get(o, 0.0) for o in ("H", "D", "A")], (len(test), 1))
    from alaves_predictor.evaluation.metrics import log_loss

    base_loss = log_loss(list(test["result"]), base)
    ens = next(r for r in output.rows if r.model == "ensemble_con_cuotas")
    assert ens.metrics["log_loss"] < base_loss


def _fake_rows_and_baselines(model_loss: float, elo_loss: float, odds_loss: float):
    metric = {"log_loss": model_loss, "brier": 0.6, "rps": 0.2, "accuracy": 0.5}
    rows = [
        bt.BacktestRow("ensemble_sin_cuotas", "2023-24", 380, dict(metric)),
        bt.BacktestRow("ensemble_con_cuotas", "2023-24", 380, dict(metric)),
    ]
    baselines = [
        BaselineResult("elo_logistico", "2023-24", 380, {**metric, "log_loss": elo_loss}),
        BaselineResult("cuotas_cierre", "2023-24", 380, {**metric, "log_loss": odds_loss}),
    ]
    return rows, baselines


def test_criterios_de_aceptacion_spec_12_1():
    # modelo mejor que Elo y dentro del margen de las cuotas: ambos pasan
    rows, baselines = _fake_rows_and_baselines(0.95, 0.97, 0.955)
    checks = dict((label, passed) for label, passed, _ in bt.acceptance_checks(rows, baselines))
    assert checks["ensemble sin cuotas < baseline Elo"] is True
    assert checks["ensemble con cuotas ≤ cuotas de cierre + 0.01"] is True
    # modelo peor que Elo y lejos de las cuotas: ambos fallan
    rows, baselines = _fake_rows_and_baselines(1.00, 0.97, 0.95)
    checks = dict((label, passed) for label, passed, _ in bt.acceptance_checks(rows, baselines))
    assert checks["ensemble sin cuotas < baseline Elo"] is False
    assert checks["ensemble con cuotas ≤ cuotas de cierre + 0.01"] is False


def test_write_report_genera_el_informe(synthetic_features, model_settings, tmp_path):
    output = bt.run_backtest(synthetic_features, model_settings, n_test_seasons=1)
    path = bt.write_report(output, baselines=[], out_dir=tmp_path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "## Modelos (F3)" in content
    assert "ensemble_sin_cuotas" in content
    assert "## Fiabilidad del ensemble sin cuotas" in content


def test_progress_callback_recibe_mensajes(synthetic_features, model_settings):
    messages: list[str] = []
    bt.run_backtest(synthetic_features, model_settings, n_test_seasons=1, progress=messages.append)
    assert any("2021-22" in m for m in messages)


def test_sin_temporadas_previas_no_hay_evaluacion(synthetic_features, model_settings):
    # la primera temporada no puede evaluarse: no hay con qué entrenar ni calibrar
    first = synthetic_features[synthetic_features["season"] == "2018-19"]
    output = bt.run_backtest(first, model_settings, n_test_seasons=1)
    assert output.rows == []
    assert output.reliability.empty
