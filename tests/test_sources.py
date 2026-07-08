"""Tests de los parsers de cada fuente sobre fixtures congelados (CLAUDE.md §6)."""

from datetime import date
from pathlib import Path

import pytest

from alaves_predictor.etl.errors import SourceFormatError
from alaves_predictor.etl.sources import clubelo, fbref, football_data, understat

FIXTURES = Path(__file__).parent / "fixtures"


# --- football-data -----------------------------------------------------------


def test_football_data_parsea_fixture() -> None:
    matches = football_data.parse_csv((FIXTURES / "football_data_mini.csv").read_text())
    assert len(matches) == 12  # la fila vacía de cola se ignora
    first = matches[0]
    assert first.match_date == date(2018, 8, 18)
    assert (first.home_team, first.away_team) == ("Alaves", "Barcelona")
    assert (first.home_goals, first.away_goals) == (1, 2)
    assert first.full_time_result == "A"
    assert first.home_shots == 10 and first.away_shots == 8
    # Cuotas de apertura y cierre de las 4 casas configuradas
    assert set(first.odds_open) == {"bet365", "pinnacle", "market_max", "market_avg"}
    assert first.odds_open["bet365"] == (1.5, 3.3, 4.6)
    assert first.odds_close["bet365"] == (1.45, 3.25, 4.55)


def test_football_data_fechas_de_dos_digitos() -> None:
    csv_text = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nSP1,18/08/18,Alaves,Barcelona,0,0,D\n"
    matches = football_data.parse_csv(csv_text)
    assert matches[0].match_date == date(2018, 8, 18)


def test_football_data_columnas_faltantes_falla_ruidosamente() -> None:
    with pytest.raises(SourceFormatError, match="FTR"):
        football_data.parse_csv("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nSP1,x,x,x,0,0\n")


def test_football_data_csv_vacio_falla() -> None:
    with pytest.raises(SourceFormatError):
        football_data.parse_csv("")


def test_football_data_season_code() -> None:
    assert football_data.season_code("2018-19") == "1819"
    assert football_data.season_code("2025-26") == "2526"


# --- FBref -------------------------------------------------------------------


def test_fbref_parsea_fixture() -> None:
    matches = fbref.parse_schedule((FIXTURES / "fbref_schedule_mini.html").read_text())
    # 12 jugados; se ignoran la fila espaciadora y el partido futuro sin marcador
    assert len(matches) == 12
    first = matches[0]
    assert first.matchday == 1
    assert first.match_date == date(2018, 8, 18)
    assert (first.home_team, first.away_team) == ("Alavés", "Barcelona")
    assert (first.home_goals, first.away_goals) == (1, 2)
    assert first.home_xg == pytest.approx(1.2)
    assert first.away_xg == pytest.approx(2.0)


def test_fbref_sin_tabla_falla_ruidosamente() -> None:
    with pytest.raises(SourceFormatError, match="sched"):
        fbref.parse_schedule("<html><body>Attention Required! | Cloudflare</body></html>")


def test_fbref_marcador_raro_falla() -> None:
    html = (FIXTURES / "fbref_schedule_mini.html").read_text().replace(">1–2<", ">1:2<", 1)
    with pytest.raises(SourceFormatError, match="formato inesperado"):
        fbref.parse_schedule(html)


def test_fbref_season_slug_y_url() -> None:
    from alaves_predictor.config import FBrefConfig

    assert fbref.season_slug("2018-19") == "2018-2019"
    assert fbref.season_slug("2025-26") == "2025-2026"
    cfg = FBrefConfig(base_url="https://fbref.com/en/comps")
    assert fbref.schedule_url("2018-19", cfg) == (
        "https://fbref.com/en/comps/12/2018-2019/schedule/2018-2019-La-Liga-Scores-and-Fixtures"
    )
    # Snapshot de Wayback: agosto posterior al fin de temporada, HTML original (id_)
    assert fbref.wayback_url("2018-19", cfg) == (
        "https://web.archive.org/web/20190801000000id_/"
        "https://fbref.com/en/comps/12/2018-2019/schedule/"
        "2018-2019-La-Liga-Scores-and-Fixtures"
    )


# --- Understat (API interna getLeagueData, ADR-011) ---------------------------


def test_understat_parsea_fixture_json() -> None:
    matches = understat.parse_league_data((FIXTURES / "understat_league_mini.json").read_text())
    assert len(matches) == 12  # el partido con isResult=false se ignora
    first = matches[0]
    assert first.home_team == "Alaves"
    assert first.away_team == "Barcelona"
    assert (first.home_goals, first.away_goals) == (1, 2)
    assert first.home_xg == pytest.approx(1.2)
    assert first.match_date == date(2018, 8, 18)


def test_understat_acepta_lista_directa_y_otros_sobres() -> None:
    """El sobre exterior del endpoint interno puede variar: lista o dict."""
    import json

    payload = json.loads((FIXTURES / "understat_league_mini.json").read_text())
    entries = payload["datesData"]
    assert len(understat.parse_league_data(json.dumps(entries))) == 12
    assert len(understat.parse_league_data(json.dumps({"matchesData": entries}))) == 12


def test_understat_sobre_desconocido_lista_claves() -> None:
    with pytest.raises(SourceFormatError, match="claves presentes"):
        understat.parse_league_data('{"otraCosa": []}')


def test_understat_no_json_falla_ruidosamente() -> None:
    with pytest.raises(SourceFormatError, match="JSON"):
        understat.parse_league_data("<html>bloqueo</html>")


def test_understat_url_y_season_year() -> None:
    from alaves_predictor.config import UnderstatConfig

    assert understat.season_year("2018-19") == 2018
    assert understat.season_year("2025-26") == 2025
    cfg = UnderstatConfig(base_url="https://understat.com", league="La liga")
    assert understat.league_data_url("2025-26", cfg) == (
        "https://understat.com/getLeagueData/La%20liga/2025"
    )


# --- ClubElo -----------------------------------------------------------------


def test_clubelo_parsea_fixture() -> None:
    ratings = clubelo.parse_csv((FIXTURES / "clubelo_Alaves.csv").read_text(), "Alaves")
    assert len(ratings) == 4
    assert ratings[1].elo == 1550.0
    assert ratings[1].valid_from == date(2018, 7, 1)


def test_clubelo_vacio_apunta_a_config() -> None:
    with pytest.raises(SourceFormatError, match="teams.toml"):
        clubelo.parse_csv("Rank,Club,Country,Level,Elo,From,To\n", "NombreMalo")


def test_clubelo_formato_cambiado_falla() -> None:
    with pytest.raises(SourceFormatError, match="Elo"):
        clubelo.parse_csv("otra,cosa\n1,2\n", "Alaves")
