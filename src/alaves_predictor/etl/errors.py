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


class SourceDownloadError(ETLError):
    """La descarga falló (HTTP != 200, timeout, DNS...). El mensaje incluye la URL y pistas."""


class UnknownTeamError(ETLError):
    """Una fuente usa nombres de equipo no registrados en config/teams.toml."""

    def __init__(self, source: str, raw_names: str | list[str], context: str = "") -> None:
        names = [raw_names] if isinstance(raw_names, str) else list(raw_names)
        listado = ", ".join(f"'{n}'" for n in names)
        plural = "los nombres de equipo" if len(names) > 1 else "el nombre de equipo"
        suffix = f" ({context})" if context else ""
        super().__init__(
            f"La fuente '{source}'{suffix} usa {plural} {listado}, no registrado(s) en "
            f"config/teams.toml. Añade cada alias a la lista '{source}' del equipo "
            f"correspondiente y vuelve a lanzar la ingesta."
        )
        self.source = source
        self.raw_names = names


class SourceConsistencyError(ETLError):
    """Dos fuentes discrepan sobre el mismo partido (p. ej. marcadores distintos)."""
