"""Tests del entrenamiento completo, el registro de modelos y la regla anti-sorpresa."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alaves_predictor.models import train as train_mod
from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS, VARIANT_WITH_ODDS


def test_train_models_produce_un_bundle_completo(synthetic_features, model_settings):
    bundle = train_mod.train_models(synthetic_features, model_settings)
    assert set(bundle.variants) == {VARIANT_WITH_ODDS, VARIANT_NO_ODDS}
    assert bundle.val_season == "2021-22"  # la última temporada del sintético
    for variant, vm in bundle.variants.items():
        # apilado de 3 componentes (ADR-019): pesos en el símplex
        assert len(vm.weights) == len(vm.component_names) == 3
        assert np.all(vm.weights >= 0)
        assert vm.weights.sum() == pytest.approx(1.0)
        # un calibrador isotónico por componente: [dc, lightgbm, tercero]
        assert len(vm.component_calibrators) == 3
        assert "ensemble" in bundle.val_metrics[variant]
        assert set(bundle.val_metrics[variant]["weights"]) == set(vm.component_names)
    # el xi definitivo sale de la rejilla de candidatos
    assert bundle.xi in model_settings.models.dixon_coles.xi_candidates()
    # el train final ve todas las temporadas
    assert bundle.train_window == "2018-19..2021-22"


def test_train_exige_dos_temporadas(synthetic_features, model_settings):
    single = synthetic_features[synthetic_features["season"] == "2018-19"]
    with pytest.raises(ValueError, match="al menos 2 temporadas"):
        train_mod.train_models(single, model_settings)


def test_predict_matches_da_salida_completa(synthetic_features, model_settings):
    bundle = train_mod.train_models(synthetic_features, model_settings)
    rows = synthetic_features[synthetic_features["season"] == "2021-22"].head(3)
    preds = bundle.predict_matches(rows, VARIANT_NO_ODDS)
    assert len(preds) == 3
    totals = preds[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert totals.to_numpy() == pytest.approx(np.ones(3))
    assert preds["pred_result"].isin(["H", "D", "A"]).all()
    assert preds["pred_score"].str.match(r"^\d+-\d+$").all()


def test_registro_y_carga_del_modelo(synthetic_features, model_settings, mini_db):
    bundle = train_mod.train_models(synthetic_features, model_settings)
    decision = train_mod.register_model(mini_db, model_settings, bundle)
    assert decision.promoted  # la primera versión siempre se promociona

    artifact_dir = model_settings.models.registry_dir / bundle.model_version
    assert (artifact_dir / "model.pkl").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "config.json").exists()

    loaded = train_mod.load_latest_model(mini_db)
    assert loaded is not None
    assert loaded.model_version == bundle.model_version
    # el artefacto cargado predice igual que el original (reproducibilidad, SPEC §12.4)
    rows = synthetic_features.head(2)
    original = bundle.predict_matches(rows, VARIANT_NO_ODDS)
    reloaded = loaded.predict_matches(rows, VARIANT_NO_ODDS)
    assert reloaded["p_home"].to_numpy() == pytest.approx(original["p_home"].to_numpy())


def test_regla_anti_sorpresa_no_promociona_un_modelo_peor(
    synthetic_features, model_settings, mini_db
):
    good = train_mod.train_models(synthetic_features, model_settings)
    train_mod.register_model(mini_db, model_settings, good)

    worse = train_mod.train_models(synthetic_features, model_settings)
    worse.model_version = good.model_version + "-b"
    # sabotaje controlado: log-loss de validación un 50 % peor
    ref = worse.val_metrics[VARIANT_NO_ODDS]["ensemble"]["log_loss"]
    worse.val_metrics[VARIANT_NO_ODDS]["ensemble"]["log_loss"] = ref * 1.5
    decision = train_mod.register_model(mini_db, model_settings, worse)

    assert not decision.promoted
    assert "empeora" in decision.reason
    # queda registrado (auditable) pero predict sigue usando la versión buena
    n_rows = mini_db.execute("SELECT COUNT(*) AS n FROM model_registry").fetchone()["n"]
    assert n_rows == 2
    assert train_mod.load_latest_model(mini_db).model_version == good.model_version


def test_registry_row_es_auditable(synthetic_features, model_settings, mini_db):
    bundle = train_mod.train_models(synthetic_features, model_settings)
    train_mod.register_model(mini_db, model_settings, bundle)
    row = mini_db.execute("SELECT * FROM model_registry").fetchone()
    metrics_json = json.loads(row["metrics_json"])
    config_json = json.loads(row["config_json"])
    assert metrics_json["promoted"] is True
    assert metrics_json["val_season"] == bundle.val_season
    assert config_json["feature_set_version"] == bundle.feature_set_version
