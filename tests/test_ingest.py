"""Test de integración de la ingesta histórica sobre la mini-liga sintética
(SPEC §11: pipeline completo sobre mini-dataset) y de la validación de BD."""

import pytest

from alaves_predictor.etl import db
from alaves_predictor.etl.errors import SourceConsistencyError
from alaves_predictor.etl.ingest import assign_matchdays, ingest_historical, make_match_id
from alaves_predictor.etl.validate import validate_db


def test_ingesta_historica_completa(mini_settings, fake_fetch) -> None:
    conn = db.connect(mini_settings.data.db_path)
    try:
        report = ingest_historical(conn, mini_settings)

        # 12 partidos, todos con xG cruzado de Understat
        assert report.matches_by_season == {"2018-19": 12}
        assert report.xg_matched_by_season == {"2018-19": 12}
        assert report.warnings == []

        # Estadísticas: 2 filas por partido, con shots (football-data) y xg (fbref)
        n_stats = conn.execute("SELECT COUNT(*) AS n FROM match_stats").fetchone()["n"]
        assert n_stats == 24
        merged = conn.execute(
            "SELECT source, shots, xg FROM match_stats WHERE match_id = ? AND team_id = ?",
            (make_match_id("2018-19", "alaves", "barcelona"), "alaves"),
        ).fetchone()
        assert merged["source"] == "football-data+fbref"
        assert merged["shots"] == 10
        assert merged["xg"] == pytest.approx(1.2)

        # Cuotas: 4 casas por partido, con apertura y cierre
        n_odds = conn.execute("SELECT COUNT(*) AS n FROM odds").fetchone()["n"]
        assert n_odds == 12 * 4
        odds = conn.execute(
            "SELECT * FROM odds WHERE bookmaker = 'bet365' AND match_id = ?",
            (make_match_id("2018-19", "alaves", "barcelona"),),
        ).fetchone()
        assert odds["open_h"] == 1.5 and odds["close_h"] == 1.45

        # Jornadas: 6 jornadas de 2 partidos (la Wk oficial de FBref
        # sobreescribe la aproximación por conteo)
        rows = conn.execute(
            "SELECT matchday, COUNT(*) AS n FROM matches GROUP BY matchday ORDER BY matchday"
        ).fetchall()
        assert [(r["matchday"], r["n"]) for r in rows] == [(d, 2) for d in range(1, 7)]

        # Elo: 3 filas por equipo (la anterior a history_start se filtra)
        assert report.elo_rows_by_team == {
            "alaves": 3,
            "barcelona": 3,
            "real-sociedad": 3,
            "getafe": 3,
        }

        # La validación completa debe pasar
        results = validate_db(conn, mini_settings)
        failed = [r for r in results if not r.passed]
        assert failed == [], [f"{r.name}: {r.detail}" for r in failed]

        # Idempotencia: re-ingerir no duplica nada
        ingest_historical(conn, mini_settings)
        assert conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"] == 12
        assert conn.execute("SELECT COUNT(*) AS n FROM match_stats").fetchone()["n"] == 24
        assert conn.execute("SELECT COUNT(*) AS n FROM odds").fetchone()["n"] == 48
    finally:
        conn.close()


def test_marcador_discrepante_aborta(mini_settings, fake_fetch, monkeypatch) -> None:
    """CLAUDE.md §6: discrepancia entre fuentes -> error ruidoso, no inserción silenciosa."""
    import alaves_predictor.etl.ingest as ingest_mod

    original = fake_fetch

    def corrupted(url, cache_path, **kwargs):
        text = original(url, cache_path, **kwargs)
        if "fbref.test" in url:
            # Rompe el marcador 1–2 del primer partido (-> 9–2)
            return text.replace(">1–2<", ">9–2<", 1)
        return text

    monkeypatch.setattr(ingest_mod, "fetch_text", corrupted)
    conn = db.connect(mini_settings.data.db_path)
    try:
        with pytest.raises(SourceConsistencyError, match="Marcador discrepante"):
            ingest_historical(conn, mini_settings)
    finally:
        conn.close()


def test_cache_envenenada_de_fbref_se_redescarga_sola(
    mini_settings, fake_fetch, monkeypatch
) -> None:
    """Si la cache tiene una página de bloqueo (sin datos), se re-descarga una vez."""
    import alaves_predictor.etl.ingest as ingest_mod

    original = fake_fetch
    calls = {"fbref": 0}

    # Cache pre-existente con una página de bloqueo (de una ejecución anterior).
    poisoned = mini_settings.data.raw_dir / "fbref" / "schedule_2018-2019.html"
    poisoned.parent.mkdir(parents=True, exist_ok=True)
    poisoned.write_text("<html><body>Checking your browser...</body></html>")

    def contar_descargas(url, cache_path, **kwargs):
        if "fbref.test" in url:
            calls["fbref"] += 1
            # la cache envenenada nunca se sirve por red: el orquestador la
            # descarta al parsearla y pide la página real con force=True
            assert kwargs.get("force") is True
        return original(url, cache_path, **kwargs)

    monkeypatch.setattr(ingest_mod, "fetch_text", contar_descargas)
    conn = db.connect(mini_settings.data.db_path)
    try:
        report = ingest_historical(conn, mini_settings)
        assert report.xg_matched_by_season == {"2018-19": 12}
        assert calls["fbref"] == 1  # una sola re-descarga directa basta
    finally:
        conn.close()


def test_fbref_reporta_todos_los_nombres_desconocidos_de_una_vez(
    mini_settings, fake_fetch, monkeypatch
) -> None:
    """Un solo error con la lista completa de alias que faltan, no uno por ejecución."""
    import alaves_predictor.etl.ingest as ingest_mod
    from alaves_predictor.etl.errors import UnknownTeamError

    original = fake_fetch

    def nombres_antiguos(url, cache_path, **kwargs):
        text = original(url, cache_path, **kwargs)
        if "fbref.test" in url:
            # simula un snapshot de otra época con dos nomenclaturas desconocidas
            return text.replace(">Real Sociedad<", ">Erreala<").replace(">Getafe<", ">Getafe CF<")
        return text

    monkeypatch.setattr(ingest_mod, "fetch_text", nombres_antiguos)
    conn = db.connect(mini_settings.data.db_path)
    try:
        with pytest.raises(UnknownTeamError) as exc_info:
            ingest_historical(conn, mini_settings)
        message = str(exc_info.value)
        assert "'Erreala'" in message and "'Getafe CF'" in message
        assert "2018-19" in message
    finally:
        conn.close()


def test_fbref_bloqueado_cae_a_wayback(mini_settings, fake_fetch, monkeypatch) -> None:
    """Si FBref directo devuelve 403, el xG se obtiene del snapshot de la Wayback
    Machine (ADR-010), con aviso en el informe y procedencia 'fbref-wayback'."""
    from pathlib import Path

    import alaves_predictor.etl.ingest as ingest_mod
    from alaves_predictor.etl.errors import SourceDownloadError

    FIXTURES = Path(__file__).parent / "fixtures"
    original = fake_fetch

    def fbref_bloqueado(url, cache_path, **kwargs):
        if "fbref.test" in url and "web.archive.org" not in url:
            raise SourceDownloadError(f"HTTP 403 al descargar {url}: bloqueo anti-bot.")
        if "cdx/search" in url:
            return "20260701123456\n20260101123456\n"  # índice de snapshots
        if "web.archive.org" in url:
            assert "id_/" in url  # snapshot sin la barra de wayback
            assert "20260701123456" in url  # se prueba primero el más reciente
            return (FIXTURES / "fbref_schedule_mini.html").read_text()
        return original(url, cache_path, **kwargs)

    monkeypatch.setattr(ingest_mod, "fetch_text", fbref_bloqueado)
    conn = db.connect(mini_settings.data.db_path)
    try:
        report = ingest_historical(conn, mini_settings)
        assert report.xg_matched_by_season == {"2018-19": 12}
        assert any("Wayback" in w for w in report.warnings)
        source = conn.execute(
            "SELECT source FROM match_stats WHERE xg IS NOT NULL LIMIT 1"
        ).fetchone()["source"]
        assert "fbref-wayback" in source
    finally:
        conn.close()


def _cache_fbref_sin_xg(mini_settings) -> None:
    """Deja en cache un calendario de FBref con marcadores pero sin xG
    (la versión 2026 de FBref quitó el xG del calendario, ADR-011)."""
    import re
    from pathlib import Path

    good_html = (Path(__file__).parent / "fixtures" / "fbref_schedule_mini.html").read_text()
    no_xg_html = re.sub(r'(data-stat="(?:home|away)_xg">)[0-9.]+', r"\1", good_html)
    cache = mini_settings.data.raw_dir / "fbref" / "schedule_2018-2019.html"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(no_xg_html)


def test_fbref_sin_xg_se_rellena_con_understat(mini_settings, fake_fetch) -> None:
    """Si el calendario de FBref no trae xG, Understat lo rellena (ADR-011),
    manteniendo la jornada oficial de FBref y sin pisar nada."""
    _cache_fbref_sin_xg(mini_settings)
    conn = db.connect(mini_settings.data.db_path)
    try:
        report = ingest_historical(conn, mini_settings)
        assert report.xg_matched_by_season == {"2018-19": 12}
        assert any("Understat" in w for w in report.warnings)
        # el xG es el de Understat (fórmula sintética distinta a la de FBref:
        # para el Barça visitante con 2 goles, 1.95 vs 2.0)
        row = conn.execute(
            "SELECT xg, source FROM match_stats WHERE match_id = ? AND team_id = ?",
            (make_match_id("2018-19", "alaves", "barcelona"), "barcelona"),
        ).fetchone()
        assert row["xg"] == pytest.approx(1.95)
        assert "understat" in row["source"]
        # la validación completa pasa
        failed = [r for r in validate_db(conn, mini_settings) if not r.passed]
        assert failed == [], [f"{r.name}: {r.detail}" for r in failed]

        # Re-ejecución: el xG ya está en la BD, así que la cobertura se mide
        # contra la BD y no debe aparecer ningún aviso (ni falso "0/N").
        report2 = ingest_historical(conn, mini_settings)
        assert report2.xg_matched_by_season == {"2018-19": 12}
        assert report2.warnings == []
    finally:
        conn.close()


def test_relleno_understat_no_pisa_el_xg_de_fbref(mini_settings, fake_fetch) -> None:
    """Con FBref completo, Understat ni se consulta (sus xG difieren a propósito)."""
    conn = db.connect(mini_settings.data.db_path)
    try:
        report = ingest_historical(conn, mini_settings)
        assert report.warnings == []
        row = conn.execute(
            "SELECT xg, source FROM match_stats WHERE match_id = ? AND team_id = ?",
            (make_match_id("2018-19", "alaves", "barcelona"), "barcelona"),
        ).fetchone()
        assert row["xg"] == pytest.approx(2.0)  # 0.9·2 + 0.2: el de FBref
        assert "understat" not in row["source"]
    finally:
        conn.close()


def test_relleno_understat_marcador_discrepante_aborta(
    mini_settings, fake_fetch, monkeypatch
) -> None:
    import alaves_predictor.etl.ingest as ingest_mod

    _cache_fbref_sin_xg(mini_settings)
    original = fake_fetch

    def corrupto(url, cache_path, **kwargs):
        text = original(url, cache_path, **kwargs)
        if "us.test" in url:
            # rompe los goles 1-2 del primer partido (id 2000)
            return text.replace('"h": "1"', '"h": "9"', 1)
        return text

    monkeypatch.setattr(ingest_mod, "fetch_text", corrupto)
    conn = db.connect(mini_settings.data.db_path)
    try:
        with pytest.raises(SourceConsistencyError, match="Marcador discrepante"):
            ingest_historical(conn, mini_settings)
    finally:
        conn.close()


def test_fbref_y_wayback_bloqueados_explica_snapshot_manual(
    mini_settings, fake_fetch, monkeypatch
) -> None:
    import alaves_predictor.etl.ingest as ingest_mod
    from alaves_predictor.etl.errors import SourceDownloadError

    original = fake_fetch

    def todo_bloqueado(url, cache_path, **kwargs):
        if "fbref.test" in url or "web.archive.org" in url:
            raise SourceDownloadError(f"HTTP 403 al descargar {url}: bloqueo.")
        return original(url, cache_path, **kwargs)

    monkeypatch.setattr(ingest_mod, "fetch_text", todo_bloqueado)
    conn = db.connect(mini_settings.data.db_path)
    try:
        with pytest.raises(SourceDownloadError, match="navegador"):
            ingest_historical(conn, mini_settings)
    finally:
        conn.close()


def test_validacion_detecta_bd_incompleta(mini_settings, fake_fetch) -> None:
    conn = db.connect(mini_settings.data.db_path)
    try:
        ingest_historical(conn, mini_settings)
        match_id = make_match_id("2018-19", "alaves", "barcelona")
        for table in ("match_stats", "odds", "matches"):
            conn.execute(f"DELETE FROM {table} WHERE match_id = ?", (match_id,))
        conn.commit()
        results = validate_db(conn, mini_settings)
        failed = {r.name for r in results if not r.passed}
        assert "[2018-19] nº de partidos" in failed
        assert "[2018-19] partidos por equipo" in failed
    finally:
        conn.close()


def test_asignacion_de_jornadas_absorbe_aplazamientos(mini_db) -> None:
    """Un partido aplazado recibe la jornada correspondiente a su posición real
    en el calendario de ambos equipos (aproximación documentada en ADR-006)."""
    for team_id, name in (
        ("alaves", "A"),
        ("barcelona", "B"),
        ("real-sociedad", "S"),
        ("getafe", "G"),
    ):
        mini_db.execute(
            "INSERT INTO teams (team_id, name, aliases_json) VALUES (?, ?, '{}')",
            (team_id, name),
        )
    # J1 completa, y el partido alaves-getafe de la "J2" se aplaza a después de la J3
    matches = [
        ("m1", "2018-08-18", "alaves", "barcelona"),
        ("m2", "2018-08-18", "real-sociedad", "getafe"),
        ("m3", "2018-09-01", "barcelona", "real-sociedad"),  # J2 (para estos equipos)
        ("m4", "2018-09-08", "alaves", "real-sociedad"),  # J3 local, J3 visitante
        ("m5", "2018-09-08", "getafe", "barcelona"),
        ("m6", "2018-09-20", "alaves", "getafe"),  # aplazado de la J2
    ]
    for match_id, day, home, away in matches:
        mini_db.execute(
            "INSERT INTO matches (match_id, season, date, home_id, away_id, home_goals, "
            "away_goals, status, source, fetched_at) VALUES (?, '2018-19', ?, ?, ?, 0, 0, "
            "'finished', 'test', 't')",
            (match_id, day, home, away),
        )
    assign_matchdays(mini_db, "2018-19")
    matchdays = {
        r["match_id"]: r["matchday"]
        for r in mini_db.execute("SELECT match_id, matchday FROM matches")
    }
    # El aplazado es el 3er partido jugado tanto por Alavés como por Getafe:
    # la aproximación le asigna jornada 3 (la oficial habría sido la 2).
    assert matchdays["m6"] == 3
    assert matchdays["m1"] == 1 and matchdays["m2"] == 1
    assert matchdays["m3"] == 2
    # m4/m5: 2º partido de un equipo pero 3º del rival -> max() da jornada 3
    assert matchdays["m4"] == 3 and matchdays["m5"] == 3
