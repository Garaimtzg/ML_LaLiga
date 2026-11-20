"""Test punta a punta: el sistema entero sobre una liga sintética, sin red.

Los tests unitarios cubren cada pieza con fixtures de 12 partidos; este
recorre el camino real de principio a fin con ~2.000 partidos en el formato
exacto de cada fuente (ver `liga_sintetica.py`): ingesta histórica →
validación → ciclo semanal de la temporada en curso (con un aplazado) →
entrenamiento y registro → predicción persistida de la próxima jornada →
proyección Monte Carlo → la semana siguiente, que evalúa lo predicho.

Solo se sustituye el transporte HTTP (`http_cache._download`), la espera del
rate limit y la fecha de "hoy": cache, parsers, BD, features y modelos son el
código de producción. Comprueba que todo encaja, no la calidad del modelo
(los datos son sintéticos).

Tarda ~30 s. Para saltarlo: `uv run pytest -m "not e2e"`.
"""

from __future__ import annotations

import types
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from alaves_predictor.config import DataConfig, Settings, load_settings
from alaves_predictor.etl import db, http_cache
from alaves_predictor.etl.ingest import ingest_historical, ingest_matchday
from alaves_predictor.etl.validate import validate_db
from alaves_predictor.evaluation.season import evaluate_season
from alaves_predictor.features.build import build_features, feature_columns
from alaves_predictor.models import predict
from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS
from alaves_predictor.models.predict import next_round, persist_predictions
from alaves_predictor.models.train import load_latest_model, register_model, train_models
from alaves_predictor.simulation.project import project_standings
from tests.e2e import liga_sintetica

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """La configuración real del proyecto, con datos y modelos en tmp.

    Cuatro temporadas históricas en vez de ocho: bastan para la validación
    walk-forward y el test tarda la mitad.
    """
    s = load_settings(REPO / "config")
    s.historical_seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]
    s.data = DataConfig(
        db_path=tmp_path / "alaves.db", raw_dir=tmp_path / "raw", features_dir=tmp_path / "feat"
    )
    s.models.registry_dir = tmp_path / "registry"
    s.sources.football_data.local_fixtures_file = str(tmp_path / "sin_calendario_local.csv")
    return s


class _Internet:
    """La "internet" sintética en una fecha dada; se puede avanzar el calendario."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, today: date) -> None:
        self._monkeypatch = monkeypatch
        self.go_to(today)
        monkeypatch.setattr(http_cache, "_download", self._download)
        monkeypatch.setattr(
            http_cache, "time", types.SimpleNamespace(sleep=lambda _s: None, monotonic=lambda: 0.0)
        )

    def go_to(self, today: date) -> None:
        """Avanza el calendario: lo que sirve "internet" y la hora del reloj.

        El reloj importa: las predicciones se sellan con él y la evaluación
        descarta las posteriores al partido. Con el reloj real el test solo
        pasaba mientras la fecha de hoy fuera anterior a la J13 sintética.
        """
        self.today = today
        self._web = liga_sintetica.build_web(today)
        moment = datetime(today.year, today.month, today.day, 10, 0, tzinfo=UTC)
        self._monkeypatch.setattr(predict, "now_utc", lambda: moment)

    def _download(self, url: str, headers, impersonate: bool) -> tuple[int, bytes]:
        return self._web.get(url, (404, b""))


def _predict_next(conn, settings: Settings, today: date):
    features = build_features(conn, settings, include_scheduled=True)
    bundle = train_models(features, settings)
    decision = register_model(conn, settings, bundle)
    matchday, rows = next_round(
        features, settings.current_season, settings.league.matches_per_round, today
    )
    preds = bundle.predict_matches(rows, VARIANT_NO_ODDS)
    persist_predictions(conn, rows, preds, bundle.model_version, settings)
    return features, bundle, decision, matchday, rows


def test_pipeline_completo_sobre_liga_sintetica(settings, monkeypatch) -> None:
    web = _Internet(monkeypatch, date(2026, 11, 5))
    conn = db.connect(settings.data.db_path)
    try:
        # --- F1: ingesta histórica (football-data + FBref/Wayback + Understat + ClubElo)
        report = ingest_historical(conn, settings)
        assert report.matches_by_season == {s: 380 for s in settings.historical_seasons}
        assert report.xg_matched_by_season == report.matches_by_season  # Understat rellena

        # --- F7: ciclo semanal a 5-nov (jugadas J1-J12, con un aplazado de la J9)
        md_report = ingest_matchday(conn, settings)
        assert (md_report.finished, md_report.scheduled) == (119, 261)
        assert md_report.warnings == []
        failed = [r for r in validate_db(conn, settings) if not r.passed]
        assert failed == [], failed

        # --- F3: entrenar, registrar y predecir la próxima jornada
        features, bundle, decision, matchday, rows = _predict_next(conn, settings, web.today)
        assert decision.promoted
        assert bundle.val_season == "2025-26"  # la temporada a medias nunca valida
        assert matchday == 13, "el aplazado de la J9 no debe ser 'la próxima jornada'"
        assert len(rows) == 10
        stored = conn.execute(
            "SELECT COUNT(*) AS n, COUNT(features_hash) AS h FROM predictions"
        ).fetchone()
        assert (stored["n"], stored["h"]) == (10, 10)

        # --- F4: la proyección no predice a ciegas las jornadas lejanas
        remaining = features[(features["season"] == settings.current_season)]
        remaining = remaining[remaining["result"].isna()]
        form_cols = [c for c in feature_columns(features) if "_ma" in c]
        assert remaining[form_cols].isna().mean().mean() < 0.05
        projection = project_standings(bundle, features, settings, settings.current_season, n=500)
        assert projection is not None and len(projection.teams) == 20
        assert (projection.n_played, projection.n_remaining) == (119, 261)
        p_title = sum(projection.result.prob_zone(t, "titulo") for t in projection.teams)
        assert p_title == pytest.approx(1.0)

        # --- La semana siguiente: la J13 se ha jugado y se evalúa lo predicho
        web.go_to(date(2026, 11, 17))
        assert ingest_matchday(conn, settings).finished == 129
        performance = evaluate_season(conn, settings)
        assert performance.n_resolved == 10
        assert 0.0 < performance.metrics["log_loss"] < 2.0
        assert load_latest_model(conn) is not None

        # --- La semana del aplazado: se predice junto a la jornada en la que se juega
        web.go_to(date(2026, 12, 1))
        ingest_matchday(conn, settings)
        _, _, _, matchday, rows = _predict_next(conn, settings, web.today)
        assert matchday == 16
        assert sorted(rows["matchday"].astype(int).tolist()) == [9] + [16] * 10
        failed = [r for r in validate_db(conn, settings) if not r.passed]
        assert failed == [], failed
    finally:
        conn.close()
