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

from alaves_predictor.config import Settings
from alaves_predictor.etl import db
from alaves_predictor.etl.errors import (
    SourceConsistencyError,
    SourceDownloadError,
    SourceFormatError,
)
from alaves_predictor.etl.http_cache import fetch_text
from alaves_predictor.etl.sources import clubelo, fbref, football_data
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


def _fetch_fbref_schedule(
    season: str, settings: Settings, *, force: bool = False
) -> tuple[str, str]:
    """Descarga el calendario de una temporada de FBref. Devuelve (html, fuente).

    Orden de intentos (ADR-009/010):
    1. Cache local (salvo force).
    2. FBref directo con huella TLS de Chrome (curl_cffi).
    3. Snapshot de la Wayback Machine (temporadas pasadas, datos estáticos).
    Si todo falla, el error incluye cómo guardar el snapshot a mano en la cache.
    """
    cfg = settings.sources.fbref
    url = fbref.schedule_url(season, cfg)
    cache = settings.data.raw_dir / "fbref" / f"schedule_{fbref.season_slug(season)}.html"
    try:
        return (
            fetch_text(
                url,
                cache,
                rate_limit_seconds=cfg.rate_limit_seconds,
                force=force,
                impersonate=True,
            ),
            fbref.SOURCE_NAME,
        )
    except SourceDownloadError:
        pass  # FBref bloquea (desafío JS de Cloudflare): probar el archivo histórico
    wb_url = fbref.wayback_url(season, cfg)
    try:
        return (
            fetch_text(wb_url, cache, rate_limit_seconds=cfg.rate_limit_seconds, force=force),
            f"{fbref.SOURCE_NAME}-wayback",
        )
    except SourceDownloadError as exc:
        raise SourceDownloadError(
            f"No se pudo descargar el calendario de FBref de {season} ni directo ni vía "
            f"Wayback Machine ({exc}). Alternativa manual: abre {url} en tu navegador, "
            f"guarda la página como HTML (Ctrl+S, 'solo HTML') en '{cache}' y relanza "
            "la ingesta: la leerá de la cache."
        ) from exc


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
    cache = settings.data.raw_dir / "fbref" / f"schedule_{fbref.season_slug(season)}.html"
    had_cache = cache.exists()
    text, via = _fetch_fbref_schedule(season, settings, force=force)
    try:
        fb_matches = fbref.parse_schedule(text)
    except SourceFormatError:
        if not (had_cache and not force):
            raise
        # La cache puede contener una página de bloqueo/error de una descarga
        # antigua: se re-descarga UNA vez antes de rendirse.
        text, via = _fetch_fbref_schedule(season, settings, force=True)
        fb_matches = fbref.parse_schedule(text)
    now = datetime.now(UTC).isoformat()

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


def ingest_clubelo(
    conn: sqlite3.Connection,
    settings: Settings,
    registry: TeamRegistry,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Descarga el histórico Elo de cada equipo del registro. Devuelve filas por equipo."""
    cfg = settings.sources.clubelo
    history_start = date.fromisoformat(cfg.history_start)
    now = datetime.now(UTC).isoformat()
    rows_by_team: dict[str, int] = {}

    for team_id in registry.team_ids:
        alias = registry.alias(team_id, "clubelo")
        url = clubelo.club_url(alias, cfg)
        cache = settings.data.raw_dir / "clubelo" / f"{alias}.csv"
        text = fetch_text(url, cache, rate_limit_seconds=cfg.rate_limit_seconds, force=force)
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
    return rows_by_team


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
        n_xg, via = ingest_fbref_season(conn, season, settings, registry, force=force)
        report.xg_matched_by_season[season] = n_xg
        conn.commit()
        if via != fbref.SOURCE_NAME:
            report.warnings.append(
                f"{season}: FBref directo bloqueado; xG obtenido del snapshot de la "
                "Wayback Machine (verificado contra football-data)."
            )
        if n_xg < n_matches:
            report.warnings.append(
                f"{season}: FBref solo cubre {n_xg}/{n_matches} partidos con xG."
            )

    report.elo_rows_by_team = ingest_clubelo(conn, settings, registry, force=force)
    return report
