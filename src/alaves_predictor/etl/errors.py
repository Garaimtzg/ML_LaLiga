"""Errores del pipeline ETL.

Regla de datos (CLAUDE.md §6): si una fuente falla o cambia de formato, el
pipeline falla ruidosamente con un mensaje claro; nunca inserta datos corruptos
en silencio. Estas excepciones son ese mecanismo.
"""

from __future__ import annotations


class ETLError(Exception):
    """Base de todos los errores del pipeline ETL."""


class SourceFormatError(ETLError):
    """La fuente devolvió un contenido con formato inesperado (web cambiada, HTML de error...)."""


class UnknownTeamError(ETLError):
    """Una fuente usa un nombre de equipo no registrado en config/teams.toml."""

    def __init__(self, source: str, raw_name: str) -> None:
        super().__init__(
            f"La fuente '{source}' usa el nombre de equipo '{raw_name}', que no está "
            f"registrado en config/teams.toml. Añade el alias en la clave '{source}' "
            f"del equipo correspondiente y vuelve a lanzar la ingesta."
        )
        self.source = source
        self.raw_name = raw_name


class SourceConsistencyError(ETLError):
    """Dos fuentes discrepan sobre el mismo partido (p. ej. marcadores distintos)."""
