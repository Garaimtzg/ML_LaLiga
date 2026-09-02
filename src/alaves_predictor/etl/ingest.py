"""Orquestador de la ingesta histórica (Fase 1).

Flujo por temporada:
1. football-data.co.uk -> matches + match_stats (tiros, córners, faltas,
   tarjetas) + odds. Es la fuente autoritativa de resultados.
2. Asignación de jornada aproximada por conteo (ADR-006), como base.
3. FBref -> añade xG a match_stats y la jornada oficial (columna Wk),
   cruzando cada partido con el ya insertado y verificando que el marcador
   coincide (discrepancia -> error ruidoso, CLAUDE.md §6). Sustituye a
   Understat, cuyo rediseño de dic-2025 eliminó el JSON embebido (ADR-008).

Después, ClubElo -> tabla elo (histórico por club, una petición por equipo).

La ingesta es idempotente: los upserts se basan en claves naturales
(match_id determinista), así que re-ejecutar no duplica filas.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from alaves_predictor.config import Settings
from alaves_predictor.etl import db
from alaves_predictor.etl.errors import (
    ETLError,
    SourceConsistencyError,
    SourceDownloadError,
    SourceFormatError,
    UnknownTeamError,
)
from alaves_predictor.etl.http_cache import fetch_text
from alaves_predictor.etl.sources import clubelo, fbref, football_data, understat
from alaves_predictor.etl.teams import TeamRegistry


@dataclass
class IngestReport:
    """Resumen de lo ingerido, para mostrar al usuario al final."""

    matches_by_season: dict[str, int] = field(default_factory=dict)
    xg_matched_by_season: dict[str, int] = field(default_factory=dict)
    elo_rows_by_team: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def make_match_id(season: str, home_id: str, away_id: str) -> str:
    """Id determinista y legible: cada emparejamiento ocurre una vez por temporada."""
    return f"{season}_{home_id}_{away_id}"


def assign_matchdays(conn: sqlite3.Connection, season: str) -> None:
    """Asigna jornada aproximada a los partidos de una temporada (ADR-006).

    football-data no publica la jornada oficial. Aproximación: ordenados por
    fecha, el partido N de un equipo pertenece a su jornada N; se toma
    max(nº de partido del local, nº del visitante) para absorber aplazamientos.
    """
    rows = conn.execute(
        "SELECT match_id, home_id, away_id FROM matches WHERE season = ? ORDER BY date, match_id",
        (season,),
    ).fetchall()
    played: dict[str, int] = {}
    for row in rows:
        home_n = played.get(row["home_id"], 0) + 1
        away_n = played.get(row["away_id"], 0) + 1
        matchday = max(home_n, away_n)
        played[row["home_id"]] = home_n
        played[row["away_id"]] = away_n
        conn.execute(
            "UPDATE matches SET matchday = ? WHERE match_id = ?", (matchday, row["match_id"])
        )


def assign_scheduled_matchdays(conn: sqlite3.Connection, season: str, gap_days: int = 3) -> None:
    """Asigna jornada a los partidos PROGRAMADOS agrupando por fechas cercanas (F7).

    Es una DEDUCCIÓN, y solo se usa cuando no hay más remedio: si el calendario
    vino de FBref, cada partido ya trae su jornada oficial (Wk) y esta función
    no toca nada (ADR-029). La deducción falla justo donde más duele —jornadas
    entre semana, partidos aplazados—, así que el dato oficial siempre gana.

    Cuando hay que deducir: se ordenan los programados por fecha y un salto de
    más de `gap_days` días abre jornada nueva (una jornada de LaLiga ocupa
    ~viernes-lunes), continuando desde la última jugada. No toca los partidos
    jugados (esos conservan la jornada oficial de FBref).
    """
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM matches WHERE season = ? AND status = 'scheduled' "
        "AND matchday IS NULL",
        (season,),
    ).fetchone()["n"]
    if not pending:
        return  # todos traen jornada oficial: nada que deducir

    max_finished = (
        conn.execute(
            "SELECT MAX(matchday) AS m FROM matches WHERE season = ? AND status = 'finished'",
            (season,),
        ).fetchone()["m"]
        or 0
    )
    rows = conn.execute(
        "SELECT match_id, date FROM matches WHERE season = ? AND status = 'scheduled' "
        "ORDER BY date, match_id",
        (season,),
    ).fetchall()
    matchday = max_finished + 1
    prev: date | None = None
    for row in rows:
        current = date.fromisoformat(row["date"])
        if prev is not None and (current - prev).days > gap_days:
            matchday += 1
        conn.execute(
            "UPDATE matches SET matchday = ? WHERE match_id = ?", (matchday, row["match_id"])
        )
        prev = current
    conn.commit()


def ingest_football_data_season(
    conn: sqlite3.Connection,
    season: str,
    settings: Settings,
    registry: TeamRegistry,
    *,
    force: bool = False,
) -> int:
    """Descarga y carga una temporada desde football-data. Devuelve nº de partidos."""
    cfg = settings.sources.football_data
    url = football_data.csv_url(season, cfg)
    cache = settings.data.raw_dir / "football_data" / f"{football_data.season_code(season)}.csv"
    # latin-1: los CSV históricos de football-data no siempre son UTF-8 válido.
    text = fetch_text(
        url, cache, rate_limit_seconds=cfg.rate_limit_seconds, force=force, encoding="latin-1"
    )
    matches = football_data.parse_csv(text)
    now = datetime.now(UTC).isoformat()

    # Los nombres desconocidos se recogen TODOS antes de insertar nada, y se
    # informan de una vez. Resolver dentro del bucle abortaba en el primero, así
    # que un ascenso de varios equipos obligaba a repetir la ingesta una vez por
    # equipo, descubriendo el siguiente nombre solo tras arreglar el anterior.
    unknown = sorted(
        {
            name
            for m in matches
            for name in (m.home_team, m.away_team)
            if not registry.knows("football_data", name)
        }
    )
    if unknown:
        raise UnknownTeamError("football_data", unknown, context=f"temporada {season}")

    for m in matches:
        home_id = registry.resolve("football_data", m.home_team)
        away_id = registry.resolve("football_data", m.away_team)
        match_id = make_match_id(season, home_id, away_id)
        db.upsert(
            conn,
            "matches",
            {
                "match_id": match_id,
                "season": season,
                "matchday": None,  # se asigna después con assign_matchdays
                "date": m.match_date.isoformat(),
                "home_id": home_id,
                "away_id": away_id,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
                "status": "finished",
                "source": football_data.SOURCE_NAME,
                "fetched_at": now,
            },
            key_cols=["match_id"],
        )
        # Estadísticas básicas por equipo (perspectiva local y visitante).
        for team_id, is_home, shots, sot, fouls, corners, yellow, red in (
            (
                home_id,
                1,
                m.home_shots,
                m.home_shots_on_target,
                m.home_fouls,
                m.home_corners,
                m.home_yellow,
                m.home_red,
            ),
            (
                away_id,
                0,
                m.away_shots,
                m.away_shots_on_target,
                m.away_fouls,
                m.away_corners,
                m.away_yellow,
                m.away_red,
            ),
        ):
            _upsert_match_stats(
                conn,
                match_id=match_id,
                team_id=team_id,
                is_home=is_home,
                values={
                    "shots": shots,
                    "shots_on_target": sot,
                    "fouls": fouls,
                    "corners": corners,
                    "cards_yellow": yellow,
                    "cards_red": red,
                },
                source=football_data.SOURCE_NAME,
                fetched_at=now,
            )
        for bookmaker in set(m.odds_open) | set(m.odds_close):
            open_odds = m.odds_open.get(bookmaker)
            close_odds = m.odds_close.get(bookmaker)
            db.upsert(
                conn,
                "odds",
                {
                    "match_id": match_id,
                    "bookmaker": bookmaker,
                    "open_h": open_odds[0] if open_odds else None,
                    "open_d": open_odds[1] if open_odds else None,
                    "open_a": open_odds[2] if open_odds else None,
                    "close_h": close_odds[0] if close_odds else None,
                    "close_d": close_odds[1] if close_odds else None,
                    "close_a": close_odds[2] if close_odds else None,
                    "source": football_data.SOURCE_NAME,
                    "fetched_at": now,
                },
                key_cols=["match_id", "bookmaker"],
            )
    conn.commit()
    return len(matches)


def _upsert_match_stats(
    conn: sqlite3.Connection,
    *,
    match_id: str,
    team_id: str,
    is_home: int,
    values: dict[str, float | int | None],
    source: str,
    fetched_at: str,
) -> None:
    """Upsert de match_stats fusionando la etiqueta de procedencia por fila."""
    existing = conn.execute(
        "SELECT source FROM match_stats WHERE match_id = ? AND team_id = ?",
        (match_id, team_id),
    ).fetchone()
    merged_source = db.merge_source_tag(existing["source"] if existing else None, source)
    db.upsert(
        conn,
        "match_stats",
        {
            "match_id": match_id,
            "team_id": team_id,
            "is_home": is_home,
            **values,
            "source": merged_source,
            "fetched_at": fetched_at,
        },
        key_cols=["match_id", "team_id"],
    )


def _has_xg(matches: list[fbref.FBrefMatch]) -> bool:
    return any(m.home_xg is not None for m in matches)


def _xg_coverage(conn: sqlite3.Connection, season: str) -> int:
    """Nº de partidos de la temporada con xG almacenado para ambos equipos."""
    return conn.execute(
        "SELECT COUNT(*) AS n FROM ("
        "  SELECT ms.match_id FROM match_stats ms"
        "  JOIN matches m ON m.match_id = ms.match_id"
        "  WHERE m.season = ? AND ms.xg IS NOT NULL"
        "  GROUP BY ms.match_id HAVING COUNT(*) = 2"
        ")",
        (season,),
    ).fetchone()["n"]


def _fbref_schedule_cache(season: str, settings: Settings) -> Path:
    return settings.data.raw_dir / "fbref" / f"schedule_{fbref.season_slug(season)}.html"


def _fbref_schedule_urls(season: str, settings: Settings) -> list[str]:
    """URLs candidatas del calendario, en orden de preferencia (ADR-029).

    FBref versiona por temporada solo las ya cerradas: la vigente vive en la URL
    sin año, y la versionada responde 404 hasta que la temporada acaba. Se
    prueban las dos porque `current_season` es config, no una verdad de FBref.
    """
    cfg = settings.sources.fbref
    versioned = fbref.schedule_url(season, cfg)
    if season != settings.current_season:
        return [versioned]
    return [fbref.current_schedule_url(cfg), versioned]


def _fetch_fbref_schedule(
    season: str, settings: Settings, *, force: bool = False
) -> tuple[list[fbref.FBrefMatch], str]:
    """Resuelve el calendario de una temporada de FBref. Devuelve (partidos, fuente).

    Orden de intentos (ADR-009/010):
    1. Cache local (salvo force), si parsea. Puede no traer xG (la versión
       2026 de FBref lo quitó del calendario): el relleno llega de Understat
       (ADR-011), así que aquí basta con jornada oficial y marcadores.
    2. FBref directo con huella TLS de Chrome (curl_cffi).
    3. Snapshots de la Wayback Machine: se listan con la API CDX y se prueban
       del más reciente hacia atrás, prefiriendo el primero que tenga xG.
    Si todo falla, el error incluye cómo guardar el snapshot a mano en la cache.
    """
    cfg = settings.sources.fbref
    url = fbref.schedule_url(season, cfg)
    cache = _fbref_schedule_cache(season, settings)

    # 1. Cache local
    if cache.exists() and not force:
        try:
            return fbref.parse_schedule(cache.read_text(encoding="utf-8")), fbref.SOURCE_NAME
        except SourceFormatError:
            pass  # cache envenenada (página de bloqueo): re-resolver

    # 2. FBref directo (fuente primaria; imprescindible en F7 para la temporada
    #    en curso). Para la temporada vigente se prueba PRIMERO la URL sin año:
    #    FBref solo versiona las temporadas cerradas, y la versionada da 404
    #    mientras la temporada está en marcha (ADR-029).
    for candidate in _fbref_schedule_urls(season, settings):
        try:
            text = fetch_text(
                candidate,
                cache,
                rate_limit_seconds=cfg.rate_limit_seconds,
                force=True,
                impersonate=True,
            )
            return fbref.parse_schedule(text), fbref.SOURCE_NAME
        except (SourceDownloadError, SourceFormatError):
            continue  # bloqueado o página sin datos: siguiente candidato

    # 3. Wayback Machine: candidatos del índice CDX, más recientes primero
    try:
        cdx_cache = settings.data.raw_dir / "fbref" / f"cdx_{fbref.season_slug(season)}.txt"
        cdx_text = fetch_text(
            fbref.cdx_url(season, cfg),
            cdx_cache,
            rate_limit_seconds=cfg.rate_limit_seconds,
            force=force,
        )
        candidates = fbref.parse_cdx_timestamps(cdx_text)
    except (SourceDownloadError, SourceFormatError):
        candidates = []
    if not candidates:
        candidates = [fbref.default_wayback_timestamp(season)]

    last_error: Exception | None = None
    best_without_xg: list[fbref.FBrefMatch] | None = None
    for timestamp in candidates[:8]:  # acotado: 8 intentos como mucho
        try:
            text = fetch_text(
                fbref.snapshot_url(timestamp, season, cfg),
                cache,
                rate_limit_seconds=cfg.rate_limit_seconds,
                force=True,
            )
            matches = fbref.parse_schedule(text)
        except (SourceDownloadError, SourceFormatError) as exc:
            last_error = exc
            continue
        if _has_xg(matches):
            return matches, f"{fbref.SOURCE_NAME}-wayback"
        best_without_xg = best_without_xg or matches

    if best_without_xg is not None:
        # Ningún snapshot trae xG (FBref lo quitó del calendario en 2026):
        # se acepta el mejor por la jornada oficial; el xG lo pone Understat.
        return best_without_xg, f"{fbref.SOURCE_NAME}-wayback"
    raise SourceDownloadError(
        f"No se pudo obtener el calendario de FBref de {season} ni directo ni vía "
        f"Wayback Machine (último error: {last_error}). Alternativa manual: abre {url} "
        f"en tu navegador, guarda la página como HTML (Ctrl+S, 'solo HTML') en '{cache}' "
        "y relanza la ingesta: la leerá de la cache."
    )


def ingest_fbref_season(
    conn: sqlite3.Connection,
    season: str,
    settings: Settings,
    registry: TeamRegistry,
    *,
    force: bool = False,
) -> tuple[int, str]:
    """Añade xG y jornada oficial de FBref a los partidos ya cargados.

    Devuelve (nº de partidos cruzados con xG, fuente usada). La jornada oficial
    (columna Wk) sobreescribe la aproximación por conteo (ADR-006/008).
    """
    fb_matches, via = _fetch_fbref_schedule(season, settings, force=force)
    now = datetime.now(UTC).isoformat()

    # Los snapshots de FBref mezclan épocas con nomenclaturas distintas
    # ("Betis"/"Real Betis"): se recopilan TODOS los nombres desconocidos de la
    # página y se reportan de una vez, no uno por ejecución.
    unknown = sorted(
        {
            name
            for m in fb_matches
            for name in (m.home_team, m.away_team)
            if not registry.knows("fbref", name)
        }
    )
    if unknown:
        raise UnknownTeamError("fbref", unknown, context=f"temporada {season}")

    matched = 0
    for m in fb_matches:
        home_id = registry.resolve("fbref", m.home_team)
        away_id = registry.resolve("fbref", m.away_team)
        match_id = make_match_id(season, home_id, away_id)
        row = conn.execute(
            "SELECT date, home_goals, away_goals FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone()
        if row is None:
            raise SourceConsistencyError(
                f"FBref tiene el partido {home_id} vs {away_id} ({m.match_date}, temporada "
                f"{season}) pero football-data no. Revisa la ingesta antes de continuar."
            )
        # Verificación cruzada de marcadores (CLAUDE.md §6). La fecha puede
        # variar ±1 día entre fuentes por zonas horarias.
        if (row["home_goals"], row["away_goals"]) != (m.home_goals, m.away_goals):
            raise SourceConsistencyError(
                f"Marcador discrepante en {match_id}: football-data "
                f"{row['home_goals']}-{row['away_goals']} vs FBref "
                f"{m.home_goals}-{m.away_goals}. No se inserta nada."
            )
        stored_date = date.fromisoformat(row["date"])
        if abs((stored_date - m.match_date).days) > 1:
            raise SourceConsistencyError(
                f"Fecha discrepante en {match_id}: football-data {stored_date} vs "
                f"FBref {m.match_date} (>1 día de diferencia)."
            )
        if m.matchday is not None:
            # Jornada oficial de FBref > aproximación por conteo (ADR-006).
            conn.execute(
                "UPDATE matches SET matchday = ? WHERE match_id = ?", (m.matchday, match_id)
            )
        if m.home_xg is not None and m.away_xg is not None:
            for team_id, is_home, xg in ((home_id, 1, m.home_xg), (away_id, 0, m.away_xg)):
                _upsert_match_stats(
                    conn,
                    match_id=match_id,
                    team_id=team_id,
                    is_home=is_home,
                    values={"xg": xg},
                    source=via,  # "fbref" o "fbref-wayback" (procedencia real)
                    fetched_at=now,
                )
            matched += 1
    conn.commit()
    return matched, via


def ingest_understat_xg(
    conn: sqlite3.Connection,
    season: str,
    settings: Settings,
    registry: TeamRegistry,
    *,
    force: bool = False,
) -> int:
    """Rellena con Understat el xG de los partidos que aún no lo tienen (ADR-011).

    Solo toca partidos sin xG (nunca pisa el de FBref) y cruza el marcador con
    el almacenado antes de insertar. Devuelve el nº de partidos rellenados.
    """
    # Partidos de la temporada con algún equipo sin xG.
    pending = {
        row["match_id"]
        for row in conn.execute(
            "SELECT m.match_id FROM matches m JOIN match_stats ms "
            "ON ms.match_id = m.match_id WHERE m.season = ? AND ms.xg IS NULL",
            (season,),
        )
    }
    if not pending:
        return 0

    cfg = settings.sources.understat
    url = understat.league_data_url(season, cfg)
    cache = settings.data.raw_dir / "understat" / f"league_{understat.season_year(season)}.json"
    had_cache = cache.exists()
    text = fetch_text(
        url,
        cache,
        rate_limit_seconds=cfg.rate_limit_seconds,
        force=force,
        headers=understat.api_headers(season, cfg),
    )
    try:
        us_matches = understat.parse_league_data(text)
    except SourceFormatError:
        if not (had_cache and not force):
            raise
        # cache envenenada de una ejecución anterior: re-descarga una vez
        text = fetch_text(
            url,
            cache,
            rate_limit_seconds=cfg.rate_limit_seconds,
            force=True,
            headers=understat.api_headers(season, cfg),
        )
        us_matches = understat.parse_league_data(text)
    now = datetime.now(UTC).isoformat()

    filled = 0
    for m in us_matches:
        home_id = registry.resolve("understat", m.home_team)
        away_id = registry.resolve("understat", m.away_team)
        match_id = make_match_id(season, home_id, away_id)
        if match_id not in pending:
            continue  # ya tiene xG (FBref) o no existe: no tocar
        row = conn.execute(
            "SELECT date, home_goals, away_goals FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone()
        if row is None:
            continue
        # Verificación cruzada de marcadores (CLAUDE.md §6).
        if (row["home_goals"], row["away_goals"]) != (m.home_goals, m.away_goals):
            raise SourceConsistencyError(
                f"Marcador discrepante en {match_id}: almacenado "
                f"{row['home_goals']}-{row['away_goals']} vs Understat "
                f"{m.home_goals}-{m.away_goals}. No se inserta nada."
            )
        stored_date = date.fromisoformat(row["date"])
        if abs((stored_date - m.match_date).days) > 1:
            raise SourceConsistencyError(
                f"Fecha discrepante en {match_id}: almacenada {stored_date} vs "
                f"Understat {m.match_date} (>1 día de diferencia)."
            )
        for team_id, is_home, xg in ((home_id, 1, m.home_xg), (away_id, 0, m.away_xg)):
            _upsert_match_stats(
                conn,
                match_id=match_id,
                team_id=team_id,
                is_home=is_home,
                values={"xg": xg},
                source=understat.SOURCE_NAME,
                fetched_at=now,
            )
        filled += 1
    conn.commit()
    return filled


def ingest_clubelo(
    conn: sqlite3.Connection,
    settings: Settings,
    registry: TeamRegistry,
    *,
    force: bool = False,
) -> tuple[dict[str, int], list[str]]:
    """Carga el histórico Elo de cada equipo del registro.

    Resiliencia (la BD manda, como en el xG):
    - Si un equipo ya tiene Elo en la BD (y no hay --force), no se toca la red.
    - Si ClubElo no responde para un equipo, se anota y se continúa: la
      ingesta no muere por una fuente caída; `alaves validate` sigue siendo
      el juez de si falta algo que importe.

    Devuelve (filas en BD por equipo, equipos no descargables en esta pasada).
    """
    cfg = settings.sources.clubelo
    history_start = date.fromisoformat(cfg.history_start)
    now = datetime.now(UTC).isoformat()
    rows_by_team: dict[str, int] = {}
    unavailable: list[str] = []

    for team_id in registry.team_ids:
        existing = conn.execute(
            "SELECT COUNT(*) AS n FROM elo WHERE team_id = ?", (team_id,)
        ).fetchone()["n"]
        if existing and not force:
            rows_by_team[team_id] = existing  # ya en BD: cero peticiones
            continue

        alias = registry.alias(team_id, "clubelo")
        url = clubelo.club_url(alias, cfg)
        cache = settings.data.raw_dir / "clubelo" / f"{alias}.csv"
        try:
            text = fetch_text(url, cache, rate_limit_seconds=cfg.rate_limit_seconds, force=force)
        except SourceDownloadError:
            unavailable.append(team_id)
            rows_by_team[team_id] = existing
            continue
        ratings = clubelo.parse_csv(text, alias)
        inserted = 0
        for rating in ratings:
            if rating.valid_from < history_start:
                continue  # anterior a la ventana de entrenamiento
            db.upsert(
                conn,
                "elo",
                {
                    "team_id": team_id,
                    "date": rating.valid_from.isoformat(),
                    "elo_clubelo": rating.elo,
                    "elo_internal": None,  # se calculará en F2
                    "source": clubelo.SOURCE_NAME,
                    "fetched_at": now,
                },
                key_cols=["team_id", "date"],
            )
            inserted += 1
        rows_by_team[team_id] = inserted
    conn.commit()
    return rows_by_team, unavailable


@dataclass
class FixturesReport:
    """Resultado de leer el calendario: qué entró y qué no, y por qué (ADR-028)."""

    inserted: int = 0
    unknown_teams: list[str] = field(default_factory=list)
    # Encuentros del calendario que ya constan jugados en la BD (se saltan).
    skipped_finished: int = 0
    # Por origen ("remoto"/"local"): cuántos encuentros de la división aportó.
    by_source: dict[str, int] = field(default_factory=dict)
    # Motivos por los que un origen no aportó nada (descarga o formato).
    problems: list[str] = field(default_factory=list)

    def explain_empty(self) -> str:
        """Por qué no hay calendario, en una frase para el aviso del CLI."""
        if self.problems:
            return "; ".join(self.problems)
        if not self.by_source:
            return "ningún origen de calendario disponible"
        return (
            "los orígenes respondieron pero no traen partidos de la división "
            "(típico si football-data aún no publica los próximos encuentros)"
        )


@dataclass
class PendingFixture:
    """Un partido por jugar, ya resuelto a team_ids canónicos.

    Los orígenes aportan cosas distintas —football-data trae cuotas pero solo
    los encuentros inminentes; FBref trae la jornada oficial; Understat trae la
    temporada entera— así que se fusionan campo a campo en vez de quedarse con
    el primero entero (ADR-029).
    """

    home_id: str
    away_id: str
    match_date: date
    matchday: int | None = None
    odds_open: dict[str, tuple[float, float, float]] = field(default_factory=dict)


def _fbref_fixtures(settings: Settings, *, force: bool = False) -> list[fbref.FBrefFixture]:
    """Partidos por jugar según FBref, con su jornada oficial (ADR-029).

    Reutiliza la página que la ingesta de xG acaba de dejar en cache: FBref pide
    moderación a los bots (6 s entre peticiones) y esa página ya trae, en la
    misma tabla, lo jugado y lo que falta por jugar. Si no hay cache utilizable
    se descarga; si tampoco, se devuelve lista vacía (es una fuente opcional).
    """
    season = settings.current_season
    cfg = settings.sources.fbref
    cache = _fbref_schedule_cache(season, settings)
    if cache.exists() and not force:
        try:
            return fbref.parse_fixtures(cache.read_text(encoding="utf-8"))
        except SourceFormatError:
            pass  # cache envenenada: intentar descarga
    for candidate in _fbref_schedule_urls(season, settings):
        try:
            text = fetch_text(
                candidate,
                cache,
                rate_limit_seconds=cfg.rate_limit_seconds,
                force=True,
                impersonate=True,
            )
            return fbref.parse_fixtures(text)
        except (SourceDownloadError, SourceFormatError):
            continue
    return []


def _understat_fixtures(
    settings: Settings, *, force: bool = False
) -> list[understat.UnderstatFixture]:
    """Partidos por jugar según Understat: la temporada entera desde el día uno."""
    season = settings.current_season
    cfg = settings.sources.understat
    cache = settings.data.raw_dir / "understat" / f"league_{understat.season_year(season)}.json"
    text = fetch_text(
        understat.league_data_url(season, cfg),
        cache,
        rate_limit_seconds=cfg.rate_limit_seconds,
        force=force,
        headers=understat.api_headers(season, cfg),
    )
    return understat.parse_league_fixtures(text)


def ingest_fixtures(
    conn: sqlite3.Connection,
    settings: Settings,
    registry: TeamRegistry,
    *,
    force: bool = False,
) -> FixturesReport:
    """Inserta los partidos por jugar de la temporada actual como 'scheduled' (F7).

    Cuatro orígenes complementarios, en orden de preferencia (ADR-026/029):

    1. **football-data** (`fixtures.csv`): pocos encuentros —solo los
       inminentes— pero es el único con **cuotas de apertura**, que alimentan
       la variante `con_cuotas` del modelo.
    2. **FBref** (Scores & Fixtures): el único con la **jornada oficial** (Wk),
       así que no hay que deducirla agrupando por fechas.
    3. **Understat** (`getLeagueData`): la **temporada entera** desde el primer
       día; es lo que permite proyectar la clasificación final.
    4. Un archivo **local** opcional (`[sources.football_data].local_fixtures_file`),
       para sembrar el calendario a mano si todo lo demás falla.

    No es una cascada de "el primero que responda": cada partido se fusiona
    campo a campo, porque ningún origen lo tiene todo. La fecha la fija el
    primer origen que traiga el partido; la jornada y las cuotas, el primero
    que las tenga.

    Nunca pisa un partido ya 'finished' (el resultado manda sobre el
    calendario). Los equipos sin alias en config/teams.toml se saltan y se
    devuelven para avisar. Devuelve siempre un FixturesReport que explica el
    resultado: un calendario vacío nunca se despacha en silencio.
    """
    cfg = settings.sources.football_data
    season = settings.current_season
    problems: list[str] = []
    by_source: dict[str, int] = {}
    unknown: set[str] = set()
    merged: dict[tuple[str, str], PendingFixture] = {}

    def add(label: str, home: str, away: str, source_name: str, **fields) -> None:
        """Fusiona un encuentro en el acumulador, resolviendo antes los equipos."""
        if not (registry.knows(source_name, home) and registry.knows(source_name, away)):
            unknown.update(n for n in (home, away) if not registry.knows(source_name, n))
            return
        key = (registry.resolve(source_name, home), registry.resolve(source_name, away))
        existing = merged.get(key)
        if existing is None:
            merged[key] = PendingFixture(home_id=key[0], away_id=key[1], **fields)
        else:
            # Solo se rellenan huecos: el origen más prioritario ya decidió.
            if existing.matchday is None and fields.get("matchday") is not None:
                existing.matchday = fields["matchday"]
            if not existing.odds_open and fields.get("odds_open"):
                existing.odds_open = fields["odds_open"]
        by_source[label] = by_source.get(label, 0) + 1

    def add_football_data(label: str, text: str) -> None:
        try:
            parsed = football_data.parse_fixtures(text, cfg.division)
        except SourceFormatError as exc:
            problems.append(f"{label}: {exc}")
            return
        if not parsed:
            problems.append(f"{label}: sin partidos de la división {cfg.division}")
        for f in parsed:
            add(
                label,
                f.home_team,
                f.away_team,
                "football_data",
                match_date=f.match_date,
                odds_open=dict(f.odds_open),
            )

    # Los orígenes se procesan EN ORDEN DE PREFERENCIA y cada uno solo rellena
    # lo que el anterior dejó vacío; así la precedencia se lee de arriba abajo.

    # 1. football-data: pocos encuentros, pero es el único con cuotas.
    try:
        add_football_data(
            "football-data",
            fetch_text(
                football_data.fixtures_url(cfg),
                settings.data.raw_dir / "football_data" / "fixtures.csv",
                rate_limit_seconds=cfg.rate_limit_seconds,
                force=force,
                encoding="utf-8",
            ),
        )
    except ETLError as exc:
        problems.append(f"football-data: {exc}")

    # 2. FBref: el único con la jornada oficial (Wk). Sin `force` a propósito:
    #    el paso de xG del ciclo acaba de dejar esa misma página en cache, y
    #    FBref pide moderación a los bots (6 s entre peticiones).
    try:
        for f in _fbref_fixtures(settings):
            add(
                "fbref",
                f.home_team,
                f.away_team,
                "fbref",
                match_date=f.match_date,
                matchday=f.matchday,
            )
    except ETLError as exc:
        problems.append(f"fbref: {exc}")

    # 3. Understat: la temporada entera, que es lo que permite proyectar.
    try:
        for u in _understat_fixtures(settings, force=force):
            add("understat", u.home_team, u.away_team, "understat", match_date=u.match_date)
    except ETLError as exc:
        problems.append(f"understat: {exc}")

    # 4. Siembra local a mano (formato football-data), como último recurso.
    local = Path(cfg.local_fixtures_file)
    if local.exists():
        add_football_data("local", local.read_text(encoding="utf-8"))

    now = datetime.now(UTC).isoformat()
    inserted = 0
    skipped_finished = 0
    for fixture in merged.values():
        match_id = make_match_id(season, fixture.home_id, fixture.away_id)
        existing = conn.execute(
            "SELECT status FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone()
        if existing and existing["status"] == "finished":
            skipped_finished += 1  # ya jugado: el resultado manda sobre el calendario
            continue
        db.upsert(
            conn,
            "matches",
            {
                "match_id": match_id,
                "season": season,
                "matchday": fixture.matchday,
                "date": fixture.match_date.isoformat(),
                "home_id": fixture.home_id,
                "away_id": fixture.away_id,
                "home_goals": None,
                "away_goals": None,
                "status": "scheduled",
                "source": football_data.SOURCE_NAME,
                "fetched_at": now,
            },
            key_cols=["match_id"],
        )
        for bookmaker, triplet in fixture.odds_open.items():
            db.upsert(
                conn,
                "odds",
                {
                    "match_id": match_id,
                    "bookmaker": bookmaker,
                    "open_h": triplet[0],
                    "open_d": triplet[1],
                    "open_a": triplet[2],
                    "close_h": None,
                    "close_d": None,
                    "close_a": None,
                    "source": football_data.SOURCE_NAME,
                    "fetched_at": now,
                },
                key_cols=["match_id", "bookmaker"],
            )
        inserted += 1
    conn.commit()
    return FixturesReport(
        inserted=inserted,
        unknown_teams=sorted(unknown),
        skipped_finished=skipped_finished,
        by_source=by_source,
        problems=problems,
    )


@dataclass
class MatchdayReport:
    """Resumen de una ingesta post-jornada (F7)."""

    season: str
    finished: int = 0
    scheduled: int = 0
    xg_coverage: int = 0
    warnings: list[str] = field(default_factory=list)


def ingest_matchday(
    conn: sqlite3.Connection, settings: Settings, *, force: bool = True
) -> MatchdayReport:
    """Ingesta de la temporada en curso: resultados nuevos, xG, calendario y Elo (F7).

    Refresca todo lo temporal de la temporada actual. Cada fuente que falle
    degrada con aviso (la BD manda, la red es el medio); nunca aborta el ciclo
    entero por una fuente caída. `force=True` por defecto: los archivos de la
    temporada en curso se actualizan cada semana.
    """
    registry = TeamRegistry(settings.teams)
    db.init_schema(conn)
    registry.seed_db(conn)
    season = settings.current_season
    report = MatchdayReport(season=season)

    # 1. Resultados jugados de la temporada actual (el CSV crece cada jornada).
    try:
        report.finished = ingest_football_data_season(conn, season, settings, registry, force=force)
        assign_matchdays(conn, season)
    except UnknownTeamError as exc:
        # No es que falte el dato: está y no se puede leer. Sin esto, NADA de la
        # temporada entra (ni resultados ni xG ni cuotas), así que se dice claro.
        report.warnings.append(
            f"NO se ha ingerido NINGÚN resultado de {season}: {exc} "
            "Arréglalo en config/teams.toml y vuelve a lanzar `alaves ingest --matchday`."
        )
    except ETLError as exc:
        report.warnings.append(
            f"football-data aún no publica resultados de {season} ({exc}); "
            "se continúa con el calendario."
        )

    # 2. xG: FBref directo (o Wayback) y relleno con Understat (ADR-008/011).
    if report.finished:
        try:
            _, via = ingest_fbref_season(conn, season, settings, registry, force=force)
            if via != fbref.SOURCE_NAME:
                report.warnings.append(f"{season}: xG/calendario de FBref vía Wayback Machine.")
        except ETLError as exc:
            report.warnings.append(f"FBref no disponible para {season}: {exc}")
        if _xg_coverage(conn, season) < report.finished:
            try:
                filled = ingest_understat_xg(conn, season, settings, registry, force=force)
                if filled:
                    report.warnings.append(f"{season}: xG de {filled} partidos vía Understat.")
            except ETLError as exc:
                report.warnings.append(f"Understat no disponible para {season}: {exc}")
    report.xg_coverage = _xg_coverage(conn, season)

    # 3. Calendario de próximos partidos (fixtures) + jornada de los programados.
    try:
        fixtures = ingest_fixtures(conn, settings, registry, force=force)
        report.scheduled = fixtures.inserted
        assign_scheduled_matchdays(conn, season)
        if fixtures.unknown_teams:
            report.warnings.append(
                "Equipos del calendario sin alias en config/teams.toml: "
                f"{', '.join(fixtures.unknown_teams)}."
            )
        # Lo que importa no es cuántos entraron en ESTA pasada, sino si la BD
        # tiene partidos por jugar: sin ellos no hay próxima jornada que
        # predecir ni clasificación que proyectar. Es un chequeo de estado.
        pending = conn.execute(
            "SELECT COUNT(*) AS n FROM matches WHERE season = ? AND status = 'scheduled'",
            (season,),
        ).fetchone()["n"]
        if not pending:
            report.warnings.append(
                "La BD no tiene NINGÚN partido por jugar de la temporada, así que no "
                f"hay nada que predecir ni simular. Motivo: {fixtures.explain_empty()}. "
                f"El fixtures.csv de football-data solo lista los encuentros inminentes; "
                f"para tener la temporada entera, siembra el calendario oficial en "
                f"'{Path(settings.sources.football_data.local_fixtures_file)}' "
                "(formato football-data: Div,Date,Time,HomeTeam,AwayTeam)."
            )
    except ETLError as exc:
        report.warnings.append(f"No se pudo obtener el calendario de próximos partidos: {exc}")

    # 4. Elo reciente de ClubElo (force: queremos el rating más actual).
    elo_rows, elo_unavailable = ingest_clubelo(conn, settings, registry, force=force)
    if elo_unavailable:
        # Distinguir "no se ha podido refrescar" de "no tiene ningún Elo": lo
        # segundo sí afecta a las predicciones de ese equipo (un ascendido
        # recién dado de alta), y decir "conserva el Elo en BD" sería falso.
        sin_datos = [t for t in elo_unavailable if not elo_rows.get(t)]
        detail = f"ClubElo no responde para: {', '.join(elo_unavailable)}."
        if sin_datos:
            detail += (
                f" ATENCIÓN: {', '.join(sin_datos)} no tiene(n) NINGÚN Elo en la BD; sus "
                "features de Elo van vacías. Revisa su alias 'clubelo' en config/teams.toml "
                "(es el componente de la URL de api.clubelo.com)."
            )
        else:
            detail += " (Todos conservan su Elo ya almacenado en la BD.)"
        report.warnings.append(detail)
    return report


def ingest_historical(
    conn: sqlite3.Connection, settings: Settings, *, force: bool = False
) -> IngestReport:
    """Pipeline completo de la Fase 1: todas las temporadas históricas + Elo."""
    registry = TeamRegistry(settings.teams)
    db.init_schema(conn)
    registry.seed_db(conn)

    report = IngestReport()
    for season in settings.historical_seasons:
        n_matches = ingest_football_data_season(conn, season, settings, registry, force=force)
        report.matches_by_season[season] = n_matches
        # Base aproximada por conteo; FBref la sobreescribe con la Wk oficial.
        assign_matchdays(conn, season)
        _, via = ingest_fbref_season(conn, season, settings, registry, force=force)
        if via != fbref.SOURCE_NAME:
            report.warnings.append(
                f"{season}: FBref directo bloqueado; calendario obtenido de la "
                "Wayback Machine (verificado contra football-data)."
            )
        # Relleno de xG con Understat para lo que falte (ADR-011). La cobertura
        # se mide siempre contra la BD, no contra lo aportado en esta pasada:
        # en re-ejecuciones el xG ya está almacenado y no hay nada que rellenar.
        if _xg_coverage(conn, season) < n_matches:
            filled = ingest_understat_xg(conn, season, settings, registry, force=force)
            if filled:
                report.warnings.append(
                    f"{season}: xG de {filled} partidos rellenado con Understat."
                )
        n_xg = _xg_coverage(conn, season)
        report.xg_matched_by_season[season] = n_xg
        conn.commit()
        if n_xg < n_matches:
            report.warnings.append(
                f"{season}: solo {n_xg}/{n_matches} partidos con xG (FBref + Understat)."
            )

    report.elo_rows_by_team, elo_unavailable = ingest_clubelo(conn, settings, registry, force=force)
    if elo_unavailable:
        sin_datos = [t for t in elo_unavailable if not report.elo_rows_by_team.get(t)]
        detail = ", ".join(elo_unavailable)
        report.warnings.append(
            f"ClubElo no responde para: {detail}. Se reintentará en la próxima ingesta."
            + (
                f" ATENCIÓN: {', '.join(sin_datos)} no tiene(n) ningún Elo en la BD."
                if sin_datos
                else " (Todos conservan su Elo ya almacenado en la BD.)"
            )
        )
    return report
