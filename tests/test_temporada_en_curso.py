"""Tests del comportamiento con la temporada YA EMPEZADA pero sin terminar (ADR-027).

Una temporada a medias (3 jornadas de 38) entra al entrenamiento como cualquier
otra, pero no puede usarse para juzgar el modelo ni para validar la BD con las
reglas de una temporada completa. Aquí se fija ese contrato.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from alaves_predictor.etl import db
from alaves_predictor.etl.ingest import ingest_fixtures
from alaves_predictor.etl.teams import TeamRegistry
from alaves_predictor.etl.validate import validate_db
from alaves_predictor.evaluation import backtest as bt
from alaves_predictor.features.build import in_progress_seasons
from alaves_predictor.models import train as train_mod

CURRENT = "2021-22"  # última temporada del frame sintético
FIXTURES = Path(__file__).parent / "fixtures"


def partial_features(synthetic_features, played_matchdays: int = 3):
    """Frame sintético con la última temporada a medias (el resto, sin jugar)."""
    df = synthetic_features.copy()
    unplayed = (df["season"] == CURRENT) & (df["matchday"] > played_matchdays)
    for col in ("result", "home_goals", "away_goals", "home_xg", "away_xg"):
        df.loc[unplayed, col] = None
    return df


def test_in_progress_por_partidos_sin_jugar(synthetic_features, model_settings):
    """Señal 1: el frame trae calendario, así que la temporada a medias se delata."""
    df = partial_features(synthetic_features)
    assert in_progress_seasons(df) == {CURRENT}
    # el histórico puro no tiene ninguna en curso
    assert in_progress_seasons(synthetic_features) == set()


def test_in_progress_por_conteo_cuando_no_hay_calendario(synthetic_features, model_settings):
    """Señal 2: sin calendario en el frame, delata el conteo de la temporada actual."""
    model_settings.current_season = CURRENT
    df = partial_features(synthetic_features)
    solo_jugados = df[df["result"].notna()]
    assert in_progress_seasons(solo_jugados) == set()  # sin la señal 1 no se ve
    assert in_progress_seasons(solo_jugados, model_settings) == {CURRENT}
    # una temporada actual COMPLETA no es "en curso"
    assert in_progress_seasons(synthetic_features, model_settings) == set()


def test_train_valida_sobre_la_ultima_temporada_completa(synthetic_features, model_settings):
    """La temporada a medias entrena, pero la validación se mide en la anterior."""
    model_settings.current_season = CURRENT
    df = partial_features(synthetic_features)
    bundle = train_mod.train_models(df, model_settings)

    assert bundle.val_season == "2020-21"  # no la 2021-22, que va por la jornada 3
    assert bundle.val_metrics["seasons_in_progress"] == [CURRENT]
    # pero lo jugado de la temporada en curso SÍ ha entrenado el modelo
    assert bundle.train_window.endswith(CURRENT)


def test_train_exige_dos_temporadas_completas(synthetic_features, model_settings):
    """Con una sola temporada cerrada no hay validación temporal posible."""
    model_settings.current_season = "2019-20"
    df = synthetic_features[synthetic_features["season"] <= "2019-20"].copy()
    df = df[~((df["season"] == "2019-20") & (df["matchday"] > 3))]
    with pytest.raises(ValueError, match="al menos 2 temporadas completas"):
        train_mod.train_models(df, model_settings)


def test_backtest_no_testea_la_temporada_a_medias(synthetic_features, model_settings):
    """Sus 30 partidos ensuciarían las medias de los criterios de SPEC §12.1."""
    model_settings.current_season = CURRENT
    df = partial_features(synthetic_features)
    output = bt.run_backtest(df, model_settings, n_test_seasons=2)
    seasons = {r.season for r in output.rows}
    # candidatas tras excluir la temporada a medias: 2019-20 y 2020-21; la
    # 2019-20 se salta porque no tiene ninguna temporada previa con la que
    # calibrar (comportamiento de siempre).
    assert seasons == {"2020-21"}
    assert CURRENT not in seasons
    assert all(r.n_matches == 60 for r in output.rows)  # solo temporadas enteras


def test_eleccion_de_xi_pondera_por_partidos(synthetic_features, model_settings):
    """El log-loss del pool se mide partido a partido, no como media de medias."""
    oof = train_mod.season_walkforward(synthetic_features, model_settings)
    # con temporadas del mismo tamaño ambas cuentas coinciden: el cambio solo
    # importa cuando una temporada tiene muchos menos partidos que las demás
    xi = train_mod.choose_xi(oof)
    assert xi in model_settings.models.dixon_coles.xi_candidates()


# --- Validación de la BD con la temporada en curso ---------------------------


def _seed_current_season(conn, settings, played_matchdays: int = 1) -> None:
    """Mini-liga de 4 equipos: `played_matchdays` jugadas y el resto programadas."""
    now = datetime.now(UTC).isoformat()
    teams = list(settings.teams)
    rounds = [
        [(teams[0], teams[1]), (teams[2], teams[3])],
        [(teams[1], teams[0]), (teams[3], teams[2])],
        [(teams[0], teams[2]), (teams[1], teams[3])],
    ]
    for md, pairs in enumerate(rounds, start=1):
        finished = md <= played_matchdays
        for home, away in pairs:
            match_id = f"{settings.current_season}_{home}_{away}"
            db.upsert(
                conn,
                "matches",
                {
                    "match_id": match_id,
                    "season": settings.current_season,
                    "matchday": md,
                    "date": f"2021-09-0{md}",
                    "home_id": home,
                    "away_id": away,
                    "home_goals": 1 if finished else None,
                    "away_goals": 0 if finished else None,
                    "status": "finished" if finished else "scheduled",
                    "source": "test",
                    "fetched_at": now,
                },
                key_cols=["match_id"],
            )
            if not finished:
                continue
            for team, is_home in ((home, 1), (away, 0)):
                db.upsert(
                    conn,
                    "match_stats",
                    {
                        "match_id": match_id,
                        "team_id": team,
                        "is_home": is_home,
                        "xg": 1.0,
                        "source": "test",
                        "fetched_at": now,
                    },
                    key_cols=["match_id", "team_id"],
                )
            db.upsert(
                conn,
                "odds",
                {
                    "match_id": match_id,
                    "bookmaker": "market_avg",
                    "open_h": 2.0,
                    "open_d": 3.3,
                    "open_a": 3.5,
                    "close_h": 2.0,
                    "close_d": 3.3,
                    "close_a": 3.5,
                    "source": "test",
                    "fetched_at": now,
                },
                key_cols=["match_id", "bookmaker"],
            )
    conn.commit()


def _current_checks(conn, settings) -> dict[str, bool]:
    prefix = f"[{settings.current_season}]"
    return {r.name: r.passed for r in validate_db(conn, settings) if r.name.startswith(prefix)}


def test_validate_acepta_una_temporada_a_medias(mini_db, mini_settings):
    """Con 1 de 3 jornadas jugadas, los chequeos de la temporada en curso pasan."""
    mini_settings.current_season = "2021-22"
    TeamRegistry(mini_settings.teams).seed_db(mini_db)
    _seed_current_season(mini_db, mini_settings, played_matchdays=1)
    checks = _current_checks(mini_db, mini_settings)
    assert checks, "la temporada en curso debe generar chequeos propios"
    assert all(checks.values()), [k for k, v in checks.items() if not v]


def test_validate_avisa_si_no_hay_datos_de_la_temporada(mini_db, mini_settings):
    mini_settings.current_season = "2026-27"
    checks = _current_checks(mini_db, mini_settings)
    assert list(checks) == ["[2026-27] temporada en curso"]
    assert all(checks.values())  # no haber empezado aún no es un fallo


def test_validate_detecta_prediccion_posterior_al_partido(mini_db, mini_settings):
    """CLAUDE.md §5.5: una predicción hecha después del partido no vale nada."""
    mini_settings.current_season = "2021-22"
    TeamRegistry(mini_settings.teams).seed_db(mini_db)
    _seed_current_season(mini_db, mini_settings, played_matchdays=1)
    match_id = mini_db.execute(
        "SELECT match_id FROM matches WHERE status = 'finished' LIMIT 1"
    ).fetchone()["match_id"]
    mini_db.execute(
        "INSERT INTO predictions (match_id, model_version, created_at, p_home, p_draw, p_away, "
        "pred_result) VALUES (?, 'v1', '2021-09-05T12:00:00+00:00', 0.5, 0.3, 0.2, 'H')",
        (match_id,),
    )
    mini_db.commit()
    checks = _current_checks(mini_db, mini_settings)
    assert checks["[2021-22] predicciones hechas antes del partido"] is False


def test_validate_detecta_jornadas_con_hueco(mini_db, mini_settings):
    """Saltar de la jornada 1 a la 3 delata una ingesta a medias."""
    mini_settings.current_season = "2021-22"
    TeamRegistry(mini_settings.teams).seed_db(mini_db)
    _seed_current_season(mini_db, mini_settings, played_matchdays=1)
    mini_db.execute(
        "UPDATE matches SET status = 'finished', home_goals = 2, away_goals = 1 "
        "WHERE season = ? AND matchday = 3",
        (mini_settings.current_season,),
    )
    mini_db.commit()
    checks = _current_checks(mini_db, mini_settings)
    assert checks["[2021-22] jornadas jugadas correlativas"] is False


# --- Calendario: el remoto manda sobre la siembra local -----------------------

_FIXTURE_CSV = "Div,Date,Time,HomeTeam,AwayTeam\nSP1,{date},21:00,Alaves,Getafe\n"


def test_el_calendario_remoto_manda_sobre_el_local(mini_db, mini_settings, tmp_path, monkeypatch):
    """La siembra local envejece: en cuanto football-data publica, gana el remoto."""
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    local = tmp_path / "fixtures.csv"
    local.write_text(_FIXTURE_CSV.format(date="01/02/2027"), encoding="utf-8")
    mini_settings.sources.football_data.local_fixtures_file = str(local)

    monkeypatch.setattr(
        "alaves_predictor.etl.ingest.fetch_text",
        lambda *a, **k: _FIXTURE_CSV.format(date="15/03/2027"),
    )
    fixtures = ingest_fixtures(mini_db, mini_settings, registry)

    assert fixtures.inserted == 1 and not fixtures.unknown_teams  # un partido, no dos
    row = mini_db.execute("SELECT date, status FROM matches").fetchone()
    assert row["date"] == "2027-03-15"  # la fecha del remoto, no la local
    assert row["status"] == "scheduled"


def test_el_local_rellena_lo_que_el_remoto_no_trae(mini_db, mini_settings, tmp_path, monkeypatch):
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    local = tmp_path / "fixtures.csv"
    local.write_text(_FIXTURE_CSV.format(date="01/02/2027"), encoding="utf-8")
    mini_settings.sources.football_data.local_fixtures_file = str(local)

    # remoto sin partidos de la división: solo queda la siembra local
    monkeypatch.setattr(
        "alaves_predictor.etl.ingest.fetch_text",
        lambda *a, **k: "Div,Date,Time,HomeTeam,AwayTeam\nE0,01/02/2027,21:00,Arsenal,Chelsea\n",
    )
    assert ingest_fixtures(mini_db, mini_settings, registry).inserted == 1
    assert mini_db.execute("SELECT date FROM matches").fetchone()["date"] == "2027-02-01"


def test_el_calendario_nunca_pisa_un_partido_jugado(mini_db, mini_settings, tmp_path, monkeypatch):
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    now = datetime.now(UTC).isoformat()
    db.upsert(
        mini_db,
        "matches",
        {
            "match_id": f"{mini_settings.current_season}_alaves_getafe",
            "season": mini_settings.current_season,
            "matchday": 1,
            "date": "2026-08-16",
            "home_id": "alaves",
            "away_id": "getafe",
            "home_goals": 2,
            "away_goals": 1,
            "status": "finished",
            "source": "test",
            "fetched_at": now,
        },
        key_cols=["match_id"],
    )
    mini_db.commit()
    local = tmp_path / "fixtures.csv"
    local.write_text(_FIXTURE_CSV.format(date="15/03/2027"), encoding="utf-8")
    mini_settings.sources.football_data.local_fixtures_file = str(local)
    monkeypatch.setattr("alaves_predictor.etl.ingest.fetch_text", lambda *a, **k: "")

    assert ingest_fixtures(mini_db, mini_settings, registry).inserted == 0
    row = mini_db.execute("SELECT date, status, home_goals FROM matches").fetchone()
    assert (row["date"], row["status"], row["home_goals"]) == ("2026-08-16", "finished", 2)


def test_focus_timeline_ignora_los_partidos_por_jugar(model_settings):
    """Dibujar el Elo de partidos no jugados sugiere un conocimiento que no existe."""
    from alaves_predictor.dashboard import data as dd

    model_settings.focus_team = "alaves"
    base = {
        "season": "2026-27",
        "elo_clubelo_home": 1650.0,
        "elo_clubelo_away": 1600.0,
        "home_points_ma5": 1.4,
        "away_points_ma5": 1.1,
    }
    df = pd.DataFrame(
        [
            {
                **base,
                "matchday": 1,
                "date": "2026-08-16",
                "home_id": "alaves",
                "away_id": "getafe",
                "home_goals": 2.0,
                "away_goals": 1.0,
                "home_xg": 1.8,
                "away_xg": 0.9,
                "result": "H",
            },
            {  # programado: sin goles, sin xG, sin resultado
                **base,
                "matchday": 2,
                "date": "2026-08-23",
                "home_id": "barcelona",
                "away_id": "alaves",
                "home_goals": None,
                "away_goals": None,
                "home_xg": None,
                "away_xg": None,
                "result": None,
            },
        ]
    )
    tl = dd.focus_timeline(df, model_settings, "2026-27")
    assert list(tl["matchday"]) == [1]
    assert list(tl["resultado"]) == ["V"] and list(tl["puntos_acumulados"]) == [3]


# --- Fallos que se dan con la temporada ya arrancada -------------------------


def test_todos_los_equipos_desconocidos_se_informan_de_una_vez(mini_db, mini_settings, monkeypatch):
    """Un ascenso de varios equipos no debe obligar a reingerir uno por uno.

    Antes se resolvía dentro del bucle y la ingesta abortaba en el PRIMER
    nombre desconocido: el siguiente solo aparecía tras arreglar el anterior.
    """
    from alaves_predictor.etl.errors import UnknownTeamError
    from alaves_predictor.etl.ingest import ingest_football_data_season

    csv = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "SP1,15/08/2026,Malaga,Alaves,1,2,A\n"
        "SP1,16/08/2026,Getafe,Cordoba,0,0,D\n"
    )
    monkeypatch.setattr("alaves_predictor.etl.ingest.fetch_text", lambda *a, **k: csv)
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)

    with pytest.raises(UnknownTeamError) as exc:
        ingest_football_data_season(mini_db, "2026-27", mini_settings, registry, force=True)
    assert exc.value.raw_names == ["Cordoba", "Malaga"]  # los dos, no solo el primero
    # y no ha entrado ningún partido a medias
    assert mini_db.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"] == 0


def test_un_calendario_vacio_explica_por_que(mini_db, mini_settings, tmp_path, monkeypatch):
    """Sin calendario no hay nada que predecir: nunca se despacha un 0 en silencio."""
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)
    # hay archivo local (así que no se lanza excepción) pero sin partidos de SP1
    local = tmp_path / "fixtures.csv"
    local.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam\nE0,01/02/2027,21:00,Arsenal,Chelsea\n", encoding="utf-8"
    )
    mini_settings.sources.football_data.local_fixtures_file = str(local)
    monkeypatch.setattr(
        "alaves_predictor.etl.ingest.fetch_text",
        lambda *a, **k: "Div,Date,Time,HomeTeam,AwayTeam\nE0,01/02/2027,21:00,Arsenal,Chelsea\n",
    )

    fixtures = ingest_fixtures(mini_db, mini_settings, registry)
    assert fixtures.inserted == 0
    assert fixtures.by_source == {}  # ningún origen aportó un solo encuentro
    assert "SP1" in fixtures.explain_empty()


def test_el_ciclo_semanal_siempre_redescarga(monkeypatch, tmp_path):
    """El CSV de la temporada en curso crece cada jornada: leer la cache lo congela.

    `ingest_matchday` documenta force=True, pero el CLI le pasaba el flag
    `--force` (False por defecto), así que el ciclo semanal habría releído la
    cache y no habría visto nunca la jornada siguiente.
    """
    from alaves_predictor import cli

    llamadas: list[bool] = []

    def _fake_ingest(conn, settings, *, force):
        llamadas.append(force)
        raise SystemExit(0)  # corta el ciclo: solo interesa con qué se llamó

    monkeypatch.setattr("alaves_predictor.etl.ingest.ingest_matchday", _fake_ingest)
    monkeypatch.setattr(cli, "_load_settings", lambda: _settings_en_tmp(tmp_path))
    with pytest.raises(SystemExit):
        cli._run_matchday_cycle()
    assert llamadas == [True]


def _settings_en_tmp(tmp_path):
    """Settings reales con la BD en tmp, para no tocar data/alaves.db."""
    from alaves_predictor.config import load_settings

    settings = load_settings(Path("config"))
    settings.data.db_path = tmp_path / "test.db"
    settings.data.raw_dir = tmp_path / "raw"
    return settings


# --- Calendario: de dónde sale y quién manda en cada campo (ADR-029) ---------

_FBREF_FIXTURES = (FIXTURES / "fbref_fixtures_mini.html").read_text(encoding="utf-8")
_UNDERSTAT_FIXTURES = (FIXTURES / "understat_league_pendientes.json").read_text(encoding="utf-8")
_FD_FIXTURES = (
    "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\n"
    "SP1,19/09/2026,18:30,Barcelona,Alaves,1.40,5.00,7.50\n"
)


def _calendar_fetch(monkeypatch, *, fbref_html="", understat_json="", fd_csv=""):
    """Sirve a cada origen su texto; cadena vacía = origen caído."""
    from alaves_predictor.etl.errors import SourceDownloadError

    def _fetch(url, cache_path, **kwargs):
        for marca, texto in (
            ("fbref.test", fbref_html),
            ("us.test", understat_json),
            ("fixtures.csv", fd_csv),
        ):
            if marca in url:
                if not texto:
                    raise SourceDownloadError(f"{marca} no disponible")
                return texto
        raise SourceDownloadError(f"origen no simulado: {url}")

    monkeypatch.setattr("alaves_predictor.etl.ingest.fetch_text", _fetch)


def _sin_calendario_local(settings, tmp_path):
    settings.sources.football_data.local_fixtures_file = str(tmp_path / "no_existe.csv")


def test_el_calendario_sale_de_fbref_con_jornada_oficial(
    mini_db, mini_settings, tmp_path, monkeypatch
):
    """FBref es el único origen con la Wk: si responde, no se deduce nada."""
    from alaves_predictor.etl.ingest import assign_scheduled_matchdays

    _sin_calendario_local(mini_settings, tmp_path)
    _calendar_fetch(monkeypatch, fbref_html=_FBREF_FIXTURES)
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)

    fixtures = ingest_fixtures(mini_db, mini_settings, registry)
    assert fixtures.inserted == 2 and fixtures.by_source == {"fbref": 2}

    # la jornada guardada es la OFICIAL (7 y 8), no un 1 y 2 deducidos
    rows = dict(
        mini_db.execute(
            "SELECT match_id, matchday FROM matches WHERE status = 'scheduled'"
        ).fetchall()
    )
    season = mini_settings.current_season
    assert rows[f"{season}_barcelona_alaves"] == 7
    assert rows[f"{season}_alaves_real-sociedad"] == 8

    # y la deducción por fechas no las pisa
    assign_scheduled_matchdays(mini_db, season)
    rows_despues = dict(
        mini_db.execute(
            "SELECT match_id, matchday FROM matches WHERE status = 'scheduled'"
        ).fetchall()
    )
    assert rows_despues == rows


def test_understat_da_el_calendario_cuando_fbref_no_responde(
    mini_db, mini_settings, tmp_path, monkeypatch
):
    """Understat publica la temporada entera: es la red de seguridad del calendario."""
    _sin_calendario_local(mini_settings, tmp_path)
    _calendar_fetch(monkeypatch, understat_json=_UNDERSTAT_FIXTURES)
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)

    fixtures = ingest_fixtures(mini_db, mini_settings, registry)
    assert fixtures.inserted == 2 and fixtures.by_source == {"understat": 2}
    # sin jornada oficial: queda NULL y la deduce assign_scheduled_matchdays
    nulls = mini_db.execute(
        "SELECT COUNT(*) AS n FROM matches WHERE status = 'scheduled' AND matchday IS NULL"
    ).fetchone()["n"]
    assert nulls == 2


def test_cada_origen_aporta_lo_suyo_al_mismo_partido(mini_db, mini_settings, tmp_path, monkeypatch):
    """football-data pone las cuotas y FBref la jornada: el partido se fusiona."""
    _sin_calendario_local(mini_settings, tmp_path)
    _calendar_fetch(
        monkeypatch,
        fd_csv=_FD_FIXTURES,
        fbref_html=_FBREF_FIXTURES,
        understat_json=_UNDERSTAT_FIXTURES,
    )
    registry = TeamRegistry(mini_settings.teams)
    registry.seed_db(mini_db)

    ingest_fixtures(mini_db, mini_settings, registry)
    mid = f"{mini_settings.current_season}_barcelona_alaves"
    row = mini_db.execute(
        "SELECT matchday, date FROM matches WHERE match_id = ?", (mid,)
    ).fetchone()
    assert row["matchday"] == 7  # de FBref
    assert row["date"] == "2026-09-19"
    odds = mini_db.execute("SELECT open_h FROM odds WHERE match_id = ?", (mid,)).fetchone()
    assert odds["open_h"] == 1.40  # de football-data
    # y el partido que solo tienen FBref/Understat también entra
    assert (
        mini_db.execute("SELECT COUNT(*) AS n FROM matches WHERE status = 'scheduled'").fetchone()[
            "n"
        ]
        == 2
    )


def test_fbref_usa_la_url_sin_ano_para_la_temporada_en_curso():
    """La URL versionada da 404 mientras la temporada está viva: era el 404 de la 2026-27."""
    from alaves_predictor.etl.ingest import _fbref_schedule_urls

    settings = _settings_en_tmp(Path("/tmp"))
    urls = _fbref_schedule_urls(settings.current_season, settings)
    assert len(urls) == 2
    assert "/comps/12/schedule/" in urls[0] and "2026-2027" not in urls[0]  # sin año, primero
    assert "2026-2027" in urls[1]  # la versionada, como respaldo

    # una temporada cerrada solo prueba la versionada
    pasadas = _fbref_schedule_urls("2024-25", settings)
    assert len(pasadas) == 1 and "2024-2025" in pasadas[0]
