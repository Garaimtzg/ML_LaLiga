"""Validación de la base de datos tras la ingesta (entregable F1: "BD poblada y validada").

Cada chequeo devuelve un CheckResult; el CLI los imprime y devuelve código de
salida distinto de cero si alguno falla. Los umbrales dependen de la config
(teams_per_season), no de números mágicos.
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
