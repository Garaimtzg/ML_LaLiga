"""Resolución de nombres entre fuentes: alias correctos y fallo ruidoso (ADR-005)."""

import json

import pytest

from alaves_predictor.etl.errors import UnknownTeamError
from alaves_predictor.etl.teams import TeamRegistry

from .conftest import MINI_TEAMS


def test_resuelve_alias_por_fuente() -> None:
    registry = TeamRegistry(MINI_TEAMS)
    assert registry.resolve("football_data", "Sociedad") == "real-sociedad"
    assert registry.resolve("understat", "Real Sociedad") == "real-sociedad"
    assert registry.resolve("clubelo", "Sociedad") == "real-sociedad"
    assert registry.resolve("football_data", "Alaves") == "alaves"


def test_nombre_desconocido_falla_con_mensaje_util() -> None:
    registry = TeamRegistry(MINI_TEAMS)
    with pytest.raises(UnknownTeamError) as exc_info:
        registry.resolve("football_data", "Sporting Gijon")
    message = str(exc_info.value)
    assert "Sporting Gijon" in message
    assert "teams.toml" in message


def test_seed_db_inserta_equipos_y_alias(mini_db) -> None:
    registry = TeamRegistry(MINI_TEAMS)
    registry.seed_db(mini_db)
    rows = mini_db.execute("SELECT * FROM teams ORDER BY team_id").fetchall()
    assert [r["team_id"] for r in rows] == ["alaves", "barcelona", "getafe", "real-sociedad"]
    aliases = json.loads(rows[0]["aliases_json"])
    assert aliases["football_data"] == "Alaves"
    # re-seed idempotente
    registry.seed_db(mini_db)
    assert mini_db.execute("SELECT COUNT(*) AS n FROM teams").fetchone()["n"] == 4
