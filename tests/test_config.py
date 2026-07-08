"""La configuración real del repo debe cargar y ser coherente con SPEC §5."""

from pathlib import Path

from alaves_predictor.config import load_settings

CONFIG_DIR = Path(__file__).parent.parent / "config"


def test_settings_reales_cargan() -> None:
    settings = load_settings(CONFIG_DIR)
    assert settings.focus_team == "alaves"
    # SPEC §5: temporadas 2018-19 -> 2025-26
    assert settings.historical_seasons[0] == "2018-19"
    assert settings.historical_seasons[-1] == "2025-26"
    assert len(settings.historical_seasons) == 8
    # LaLiga: 20 equipos, 38 jornadas, 380 partidos
    assert settings.league.teams_per_season == 20
    assert settings.league.rounds == 38
    assert settings.league.matches_per_season == 380


def test_equipos_cubren_todas_las_fuentes() -> None:
    settings = load_settings(CONFIG_DIR)
    # 28 clubes del histórico 2018-26 + ascendidos a la 2026-27
    assert len(settings.teams) >= 30
    assert settings.focus_team in settings.teams
    assert "racing-santander" in settings.teams
    assert "deportivo-la-coruna" in settings.teams
    for team_id, cfg in settings.teams.items():
        assert cfg.name, team_id
        for source in ("football_data", "fbref", "understat", "clubelo"):
            assert cfg.aliases_for(source), f"{team_id} sin alias {source}"
        # el alias de clubelo es un componente de URL: sin espacios
        assert all(" " not in a for a in cfg.aliases_for("clubelo")), team_id


def test_alias_unicos_por_fuente() -> None:
    settings = load_settings(CONFIG_DIR)
    for source in ("football_data", "fbref", "understat", "clubelo"):
        aliases = [a for cfg in settings.teams.values() for a in cfg.aliases_for(source)]
        assert len(aliases) == len(set(aliases)), f"alias duplicado en {source}"
