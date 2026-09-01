"""Validación de la base de datos tras la ingesta (entregable F1: "BD poblada y validada").

Cada chequeo devuelve un CheckResult; el CLI los imprime y devuelve código de
salida distinto de cero si alguno falla. Los umbrales dependen de la config
(teams_per_season), no de números mágicos.

Dos bloques distintos: las temporadas históricas se exigen COMPLETAS (380
partidos, 38 por equipo) y la temporada en curso se valida como lo que es, una
temporada a medias (ver `_current_season_checks`, ADR-027).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from alaves_predictor.config import Settings


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _check(name: str, passed: bool, detail: str) -> CheckResult:
    return CheckResult(name=name, passed=passed, detail=detail)


def _current_season_checks(conn: sqlite3.Connection, settings: Settings) -> list[CheckResult]:
    """Chequeos de la temporada EN CURSO, que por definición está a medias (ADR-027).

    Los de `validate_db` no sirven aquí: exigen 380 partidos y 38 por equipo.
    Lo que sí debe cumplirse con la temporada arrancada es que lo jugado esté
    completo (goles, xG, cuotas), que todo partido tenga jornada — también los
    programados, o `predict --next` no sabría cuál es la próxima — y que las
    predicciones guardadas se hicieran antes de jugarse el partido.
    """
    season = settings.current_season
    league = settings.league
    prefix = f"[{season}]"
    results: list[CheckResult] = []

    counts = conn.execute(
        "SELECT COALESCE(SUM(status = 'finished'), 0) AS played, "
        "COALESCE(SUM(status = 'scheduled'), 0) AS scheduled "
        "FROM matches WHERE season = ?",
        (season,),
    ).fetchone()
    played, scheduled = counts["played"], counts["scheduled"]
    if played == 0 and scheduled == 0:
        return [
            _check(
                f"{prefix} temporada en curso",
                True,
                "todavía sin datos (ejecuta `alaves ingest --matchday`)",
            )
        ]

    total = played + scheduled
    results.append(
        _check(
            f"{prefix} partidos jugados + programados",
            played > 0 and total <= league.matches_per_season,
            f"{played} jugados + {scheduled} programados = {total}/{league.matches_per_season}",
        )
    )

    n_teams = conn.execute(
        "SELECT COUNT(DISTINCT team) AS n FROM (SELECT home_id AS team FROM matches "
        "WHERE season = ? UNION SELECT away_id FROM matches WHERE season = ?)",
        (season, season),
    ).fetchone()["n"]
    results.append(
        _check(
            f"{prefix} nº de equipos",
            n_teams == league.teams_per_season,
            f"{n_teams}/{league.teams_per_season}",
        )
    )

    null_goals = conn.execute(
        "SELECT COUNT(*) AS n FROM matches WHERE season = ? AND status = 'finished' "
        "AND (home_goals IS NULL OR away_goals IS NULL)",
        (season,),
    ).fetchone()["n"]
    results.append(
        _check(f"{prefix} goles sin nulos", null_goals == 0, f"{null_goals} filas con NULL")
    )

    # Todo partido necesita jornada: los jugados para las features, los
    # programados para saber cuál es la próxima jornada a predecir.
    bad_matchday = conn.execute(
        "SELECT COUNT(*) AS n FROM matches WHERE season = ? AND "
        "(matchday IS NULL OR matchday < 1 OR matchday > ?)",
        (season, league.rounds),
    ).fetchone()["n"]
    results.append(
        _check(
            f"{prefix} jornadas asignadas (1-{league.rounds})",
            bad_matchday == 0,
            f"{bad_matchday} partidos sin jornada válida",
        )
    )

    # Las jornadas jugadas deben ir 1..N sin huecos; un hueco delata una
    # ingesta a medias o un aplazamiento mal absorbido.
    md = [
        r["matchday"]
        for r in conn.execute(
            "SELECT DISTINCT matchday FROM matches WHERE season = ? AND status = 'finished' "
            "AND matchday IS NOT NULL ORDER BY matchday",
            (season,),
        )
    ]
    results.append(
        _check(
            f"{prefix} jornadas jugadas correlativas",
            md == list(range(1, len(md) + 1)),
            f"jornadas 1-{len(md)}" if md else "ninguna jornada jugada aún",
        )
    )

    if played:
        xg_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM match_stats ms JOIN matches m ON m.match_id = ms.match_id "
            "WHERE m.season = ? AND m.status = 'finished' AND ms.xg IS NOT NULL",
            (season,),
        ).fetchone()["n"]
        results.append(
            _check(
                f"{prefix} cobertura de xG",
                xg_rows == 2 * played,
                f"{xg_rows}/{2 * played} filas equipo-partido con xG",
            )
        )

        odds_matches = conn.execute(
            "SELECT COUNT(DISTINCT o.match_id) AS n FROM odds o "
            "JOIN matches m ON m.match_id = o.match_id "
            "WHERE m.season = ? AND m.status = 'finished'",
            (season,),
        ).fetchone()["n"]
        results.append(
            _check(
                f"{prefix} cobertura de cuotas",
                odds_matches == played,
                f"{odds_matches}/{played} partidos jugados con cuotas",
            )
        )

    # Auditoría honesta (CLAUDE.md §5.5): una predicción hecha DESPUÉS del
    # partido no vale nada. `created_at` es un timestamp ISO en UTC y `date` la
    # fecha del partido: basta comparar la parte de fecha.
    late = conn.execute(
        "SELECT COUNT(*) AS n FROM predictions p JOIN matches m ON m.match_id = p.match_id "
        "WHERE m.season = ? AND substr(p.created_at, 1, 10) > m.date",
        (season,),
    ).fetchone()["n"]
    n_preds = conn.execute(
        "SELECT COUNT(*) AS n FROM predictions p JOIN matches m ON m.match_id = p.match_id "
        "WHERE m.season = ?",
        (season,),
    ).fetchone()["n"]
    results.append(
        _check(
            f"{prefix} predicciones hechas antes del partido",
            late == 0,
            f"{n_preds} predicciones, ninguna posterior al partido"
            if late == 0
            else f"{late}/{n_preds} predicciones con fecha posterior al partido",
        )
    )
    return results


def validate_db(conn: sqlite3.Connection, settings: Settings) -> list[CheckResult]:
    """Ejecuta todos los chequeos de integridad sobre la BD."""
    results: list[CheckResult] = []
    league = settings.league
    expected_matches = league.matches_per_season
    expected_rounds = league.rounds

    for season in settings.historical_seasons:
        prefix = f"[{season}]"

        n = conn.execute(
            "SELECT COUNT(*) AS n FROM matches WHERE season = ?", (season,)
        ).fetchone()["n"]
        results.append(
            _check(
                f"{prefix} nº de partidos",
                n == expected_matches,
                f"{n}/{expected_matches}",
            )
        )
        if n == 0:
            continue  # el resto de chequeos de la temporada no aportan nada

        n_teams = conn.execute(
            "SELECT COUNT(DISTINCT team) AS n FROM (SELECT home_id AS team FROM matches "
            "WHERE season = ? UNION SELECT away_id FROM matches WHERE season = ?)",
            (season, season),
        ).fetchone()["n"]
        results.append(
            _check(
                f"{prefix} nº de equipos",
                n_teams == league.teams_per_season,
                f"{n_teams}/{league.teams_per_season}",
            )
        )

        # Cada equipo debe jugar exactamente `rounds` partidos.
        bad_counts = conn.execute(
            "SELECT team, COUNT(*) AS n FROM (SELECT home_id AS team FROM matches "
            "WHERE season = ? UNION ALL SELECT away_id FROM matches WHERE season = ?) "
            "GROUP BY team HAVING n != ?",
            (season, season, expected_rounds),
        ).fetchall()
        results.append(
            _check(
                f"{prefix} partidos por equipo",
                len(bad_counts) == 0,
                f"todos juegan {expected_rounds}"
                if not bad_counts
                else ", ".join(f"{r['team']}={r['n']}" for r in bad_counts),
            )
        )

        null_goals = conn.execute(
            "SELECT COUNT(*) AS n FROM matches WHERE season = ? AND status = 'finished' "
            "AND (home_goals IS NULL OR away_goals IS NULL)",
            (season,),
        ).fetchone()["n"]
        results.append(
            _check(f"{prefix} goles sin nulos", null_goals == 0, f"{null_goals} filas con NULL")
        )

        bad_matchday = conn.execute(
            "SELECT COUNT(*) AS n FROM matches WHERE season = ? AND "
            "(matchday IS NULL OR matchday < 1 OR matchday > ?)",
            (season, expected_rounds),
        ).fetchone()["n"]
        results.append(
            _check(
                f"{prefix} jornadas en rango 1-{expected_rounds}",
                bad_matchday == 0,
                f"{bad_matchday} fuera de rango",
            )
        )

        # Cobertura de xG (Understat): debe existir para los dos equipos de cada partido.
        xg_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM match_stats ms JOIN matches m ON m.match_id = ms.match_id "
            "WHERE m.season = ? AND ms.xg IS NOT NULL",
            (season,),
        ).fetchone()["n"]
        results.append(
            _check(
                f"{prefix} cobertura de xG",
                xg_rows == 2 * n,
                f"{xg_rows}/{2 * n} filas equipo-partido con xG",
            )
        )

        # Cobertura de cuotas: al menos una casa por partido.
        odds_matches = conn.execute(
            "SELECT COUNT(DISTINCT o.match_id) AS n FROM odds o "
            "JOIN matches m ON m.match_id = o.match_id WHERE m.season = ?",
            (season,),
        ).fetchone()["n"]
        results.append(
            _check(
                f"{prefix} cobertura de cuotas",
                odds_matches == n,
                f"{odds_matches}/{n} partidos con cuotas",
            )
        )

    if settings.current_season not in settings.historical_seasons:
        results.extend(_current_season_checks(conn, settings))

    # Elo: cada equipo que aparece en matches debe tener historial en la ventana.
    teams_without_elo = conn.execute(
        "SELECT DISTINCT team FROM (SELECT home_id AS team FROM matches "
        "UNION SELECT away_id FROM matches) "
        "WHERE team NOT IN (SELECT DISTINCT team_id FROM elo)"
    ).fetchall()
    results.append(
        _check(
            "[global] Elo para todos los equipos",
            len(teams_without_elo) == 0,
            "ok"
            if not teams_without_elo
            else "sin Elo: " + ", ".join(r["team"] for r in teams_without_elo),
        )
    )

    # Consistencia interna: FTR implícito coincide con los goles almacenados.
    # (El resultado se recalcula, no se almacena, así que aquí solo se
    # comprueba que no haya goles negativos u otros valores absurdos.)
    weird = conn.execute(
        "SELECT COUNT(*) AS n FROM matches WHERE home_goals < 0 OR away_goals < 0 "
        "OR home_goals > 15 OR away_goals > 15"
    ).fetchone()["n"]
    results.append(_check("[global] marcadores plausibles", weird == 0, f"{weird} sospechosos"))

    return results
