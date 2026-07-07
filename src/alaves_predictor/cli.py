"""CLI del proyecto (`alaves ...`), según SPEC §10.

En F1 están implementados: ingest --historical, status y validate.
El resto de comandos existen como stubs que indican su fase, para que la
superficie del CLI coincida con la especificación sin prometer nada que no
funcione todavía.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import typer

from alaves_predictor.config import Settings, load_settings
from alaves_predictor.etl import db
from alaves_predictor.etl.errors import ETLError
from alaves_predictor.etl.ingest import ingest_historical
from alaves_predictor.etl.validate import validate_db

app = typer.Typer(
    name="alaves",
    help="Predictor probabilístico del Deportivo Alavés — LaLiga 2026-27.",
    no_args_is_help=True,
)


def _load_settings() -> Settings:
    try:
        return load_settings(Path("config"))
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def ingest(
    historical: bool = typer.Option(False, "--historical", help="ETL histórico completo (F1)."),
    matchday: int | None = typer.Option(None, "--matchday", help="Ingesta post-jornada (F7)."),
    force: bool = typer.Option(False, "--force", help="Re-descarga aunque exista cache local."),
) -> None:
    """Ingesta de datos: histórica (--historical) o post-jornada (--matchday, F7)."""
    if matchday is not None:
        typer.secho("La ingesta post-jornada llega en la Fase 7.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    if not historical:
        typer.secho("Indica --historical (o --matchday N cuando exista la F7).", err=True)
        raise typer.Exit(code=1)

    settings = _load_settings()
    conn = db.connect(settings.data.db_path)
    try:
        report = ingest_historical(conn, settings, force=force)
    except ETLError as exc:
        typer.secho(f"ERROR de ingesta: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except sqlite3.OperationalError as exc:
        typer.secho(
            f"ERROR de base de datos: {exc}. Si dice 'database is locked': cierra "
            "cualquier proceso que use data/alaves.db (visores SQLite, otra ingesta) "
            "y evita que OneDrive sincronice el repo mientras trabaja — lo más fiable "
            "es clonar el proyecto dentro del sistema de archivos de WSL (p. ej. "
            "~/proyectos/ML_LaLiga) en vez de /mnt/c/...OneDrive.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    typer.secho("Ingesta histórica completada.", fg=typer.colors.GREEN, bold=True)
    for season, n in report.matches_by_season.items():
        xg = report.xg_matched_by_season.get(season, 0)
        typer.echo(f"  {season}: {n} partidos, {xg} con xG de FBref")
    total_elo = sum(report.elo_rows_by_team.values())
    typer.echo(f"  Elo (ClubElo): {total_elo} filas para {len(report.elo_rows_by_team)} equipos")
    for warning in report.warnings:
        typer.secho(f"  AVISO: {warning}", fg=typer.colors.YELLOW)
    typer.echo("Ejecuta `alaves validate` para certificar la BD.")


@app.command()
def status() -> None:
    """Muestra el nº de filas por tabla y partidos por temporada."""
    settings = _load_settings()
    if not settings.data.db_path.exists():
        typer.secho(
            f"No existe {settings.data.db_path}. Ejecuta `alaves ingest --historical` primero.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)
    conn = db.connect(settings.data.db_path)
    try:
        typer.secho("Filas por tabla:", bold=True)
        for table, n in db.table_counts(conn).items():
            typer.echo(f"  {table:16s} {n:7d}")
        typer.secho("Partidos por temporada:", bold=True)
        for row in conn.execute(
            "SELECT season, COUNT(*) AS n, MIN(date) AS first, MAX(date) AS last "
            "FROM matches GROUP BY season ORDER BY season"
        ):
            typer.echo(f"  {row['season']}: {row['n']:4d}  ({row['first']} → {row['last']})")
    finally:
        conn.close()


@app.command()
def validate() -> None:
    """Valida la integridad de la BD (conteos, coberturas, consistencia)."""
    settings = _load_settings()
    if not settings.data.db_path.exists():
        typer.secho(
            f"No existe {settings.data.db_path}. Ejecuta `alaves ingest --historical` primero.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)
    conn = db.connect(settings.data.db_path)
    try:
        results = validate_db(conn, settings)
    finally:
        conn.close()

    failures = 0
    for result in results:
        if result.passed:
            typer.secho(f"  ✓ {result.name}: {result.detail}", fg=typer.colors.GREEN)
        else:
            failures += 1
            typer.secho(f"  ✗ {result.name}: {result.detail}", fg=typer.colors.RED)
    if failures:
        typer.secho(f"{failures} chequeos fallidos.", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)
    typer.secho(
        "Base de datos validada: todos los chequeos pasan.", fg=typer.colors.GREEN, bold=True
    )


def _stub(phase: str) -> None:
    typer.secho(f"Este comando se implementa en la {phase}.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=1)


@app.command()
def train() -> None:
    """Entrena Dixon-Coles + LightGBM + calibración (F3)."""
    _stub("Fase 3")


@app.command()
def predict() -> None:
    """Predicciones de la próxima jornada (F3+F7)."""
    _stub("Fase 3")


@app.command()
def simulate() -> None:
    """Simulación Monte Carlo de la clasificación (F4)."""
    _stub("Fase 4")


@app.command()
def backtest() -> None:
    """Backtesting walk-forward (F3)."""
    _stub("Fase 3")


@app.command()
def report() -> None:
    """Informes SHAP / importancia de variables (F5)."""
    _stub("Fase 5")


if __name__ == "__main__":
    app()
