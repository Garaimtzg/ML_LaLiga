"""Resolución de nombres de equipo entre fuentes (ADR-005).

Cada fuente escribe los equipos a su manera ("Alaves", "Ath Bilbao",
"Athletic Club"...). config/teams.toml define el id canónico y los alias;
este módulo resuelve nombre-de-fuente -> team_id y falla ruidosamente
(UnknownTeamError) ante nombres no registrados.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from alaves_predictor.config import TeamConfig
from alaves_predictor.etl.errors import UnknownTeamError


class TeamRegistry:
    """Índice alias -> team_id por fuente, construido desde config/teams.toml."""

    def __init__(self, teams: dict[str, TeamConfig]) -> None:
        self._teams = teams
        self._by_source: dict[str, dict[str, str]] = {}
        for source in ("football_data", "understat", "clubelo"):
            self._by_source[source] = {
                getattr(cfg, source): team_id for team_id, cfg in teams.items()
            }

    def resolve(self, source: str, raw_name: str) -> str:
        """Devuelve el team_id canónico para un nombre tal como lo escribe la fuente."""
        team_id = self._by_source[source].get(raw_name.strip())
        if team_id is None:
            raise UnknownTeamError(source, raw_name)
        return team_id

    def alias(self, team_id: str, source: str) -> str:
        """Alias de un equipo en una fuente (p. ej. para construir URLs de ClubElo)."""
        return getattr(self._teams[team_id], source)

    @property
    def team_ids(self) -> list[str]:
        return list(self._teams)

    def seed_db(self, conn: sqlite3.Connection) -> None:
        """Inserta/actualiza la tabla `teams` con los equipos y alias de la config."""
        now = datetime.now(UTC).isoformat()
        for team_id, cfg in self._teams.items():
            aliases = {
                "football_data": cfg.football_data,
                "understat": cfg.understat,
                "clubelo": cfg.clubelo,
                "_seeded_at": now,
            }
            conn.execute(
                "INSERT INTO teams (team_id, name, aliases_json) VALUES (?, ?, ?) "
                "ON CONFLICT (team_id) DO UPDATE SET name = excluded.name, "
                "aliases_json = excluded.aliases_json",
                (team_id, cfg.name, json.dumps(aliases, ensure_ascii=False)),
            )
        conn.commit()
