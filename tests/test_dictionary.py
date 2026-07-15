"""Diccionario de variables: toda feature del pipeline debe tener descripción."""

from __future__ import annotations

from alaves_predictor.etl import db
from alaves_predictor.etl.ingest import ingest_historical
from alaves_predictor.features import dictionary
from alaves_predictor.features.build import build_features, feature_columns


def test_ninguna_feature_del_pipeline_queda_sin_documentar(mini_settings, fake_fetch):
    """Si se añade una feature nueva sin entrada en el diccionario, este test falla."""
    conn = db.connect(mini_settings.data.db_path)
    try:
        ingest_historical(conn, mini_settings)
        df = build_features(conn, mini_settings)
    finally:
        conn.close()
    missing = dictionary.undocumented(feature_columns(df))
    assert missing == [], f"Features sin descripción en features/dictionary.py: {missing}"


def test_describe_columnas_fijas():
    assert "LOCAL" in dictionary.describe("elo_clubelo_home")
    assert "ClubElo" in dictionary.describe("elo_clubelo_home")
    assert "apertura" in dictionary.describe("imp_draw").lower()
    assert "derbi" in dictionary.describe("derby")


def test_describe_resuelve_el_patron_de_forma():
    text = dictionary.describe("away_xg_against_venue_ma10")
    assert "VISITANTE" in text
    assert "10" in text
    assert "fuera" in text  # condición venue del visitante
    assert "xG concedido" in text
    # sin venue y con otra ventana
    text2 = dictionary.describe("home_points_ma5")
    assert "LOCAL" in text2 and "5" in text2 and "puntos" in text2
    assert "condición" not in text2


def test_nombres_desconocidos_devuelven_none():
    assert dictionary.describe("invento_total") is None
    assert dictionary.describe("home_invento_ma5") is None


def test_feature_dictionary_mantiene_el_orden():
    cols = ["derby", "elo_clubelo_diff", "home_points_ma5"]
    table = dictionary.feature_dictionary(cols)
    assert list(table["feature"]) == cols
    assert (table["descripcion"] != "(sin descripción)").all()
