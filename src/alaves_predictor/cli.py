"""CLI del proyecto (`alaves ...`), según SPEC §10.

Implementados: ingest --historical (F1), status, validate, features,
baselines (F2), train, predict, backtest (F3), simulate (F4) y
report --importance (F5). La superficie del CLI coincide con SPEC §10.
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
        typer.echo(f"  {season}: {n} partidos, {xg} con xG")
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


@app.command()
def features() -> None:
    """Construye el feature set v1 y lo persiste (tabla features + Parquet)."""
    settings = _load_settings()
    if not settings.data.db_path.exists():
        typer.secho("No existe la BD. Ejecuta `alaves ingest --historical` primero.", err=True)
        raise typer.Exit(code=1)
    from alaves_predictor.features.build import build_features, feature_columns, persist_features

    conn = db.connect(settings.data.db_path)
    try:
        df = build_features(conn, settings)
        parquet_path = persist_features(conn, df, settings)
    finally:
        conn.close()
    version = settings.features.feature_set_version
    typer.secho(
        f"Feature set {version}: {len(df)} partidos × {len(feature_columns(df))} features.",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.echo(f"  Persistido en tabla `features` y en {parquet_path}")


@app.command()
def baselines(
    seasons: int = typer.Option(3, "--seasons", help="Temporadas de test (walk-forward)."),
) -> None:
    """Evalúa los 3 baselines de SPEC §6.1 e imprime/guarda el informe."""
    settings = _load_settings()
    if not settings.data.db_path.exists():
        typer.secho("No existe la BD. Ejecuta `alaves ingest --historical` primero.", err=True)
        raise typer.Exit(code=1)
    from alaves_predictor.evaluation.baselines import run_baselines, write_report
    from alaves_predictor.features.build import build_features

    conn = db.connect(settings.data.db_path)
    try:
        df = build_features(conn, settings)
        results = run_baselines(conn, df, settings, n_test_seasons=seasons)
    finally:
        conn.close()

    typer.secho("Baselines (walk-forward):", bold=True)
    for r in results:
        m = r.metrics
        typer.echo(
            f"  {r.baseline:14s} {r.season}  n={r.n_matches:3d}  "
            f"log-loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
            f"rps={m['rps']:.4f}  acc={m['accuracy']:.3f}"
        )
    report_path = write_report(results, Path("docs/reports"))
    typer.echo(f"Informe guardado en {report_path}")


def _stub(phase: str) -> None:
    typer.secho(f"Este comando se implementa en la {phase}.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=1)


def _require_db(settings: Settings) -> None:
    if not settings.data.db_path.exists():
        typer.secho("No existe la BD. Ejecuta `alaves ingest --historical` primero.", err=True)
        raise typer.Exit(code=1)


def _echo_metrics(label: str, m: dict) -> None:
    typer.echo(
        f"  {label:22s} log-loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
        f"rps={m['rps']:.4f}  acc={m['accuracy']:.3f}"
    )


@app.command()
def train(
    no_odds: bool = typer.Option(
        False, "--no-odds", help="Entrena solo la variante sin cuotas (la interpretable)."
    ),
) -> None:
    """Entrena Dixon-Coles + LightGBM + calibración + ensemble y registra la versión (F3)."""
    settings = _load_settings()
    _require_db(settings)
    from alaves_predictor.features.build import build_features
    from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS, VARIANTS
    from alaves_predictor.models.train import register_model, train_models

    variants = (VARIANT_NO_ODDS,) if no_odds else VARIANTS
    conn = db.connect(settings.data.db_path)
    try:
        typer.echo("Construyendo features y entrenando (validación walk-forward por temporada)...")
        df = build_features(conn, settings)
        try:
            bundle = train_models(df, settings, variants)
        except ValueError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        decision = register_model(conn, settings, bundle)
    finally:
        conn.close()

    typer.secho(
        f"Modelo {bundle.model_version} entrenado "
        f"(métricas de validación sobre {bundle.val_season}):",
        bold=True,
    )
    _echo_metrics("dixon_coles", bundle.val_metrics["dixon_coles"])
    typer.echo(
        f"    (elegidos en validación: xi={bundle.xi} para el DC, C={bundle.c} para el lineal)"
    )
    for variant in variants:
        vm = bundle.val_metrics[variant]
        _echo_metrics(f"lgbm_{variant}", vm["lgbm"])
        _echo_metrics(f"ensemble_{variant}", vm["ensemble"])
        weights = ", ".join(f"{name}={w:.2f}" for name, w in vm["weights"].items())
        typer.echo(f"    (pesos del apilado: {weights})")
    if decision.promoted:
        typer.secho(f"Promocionado: {decision.reason}.", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho(f"NO promocionado: {decision.reason}.", fg=typer.colors.YELLOW, bold=True)
        typer.echo("`alaves predict` seguirá usando la última versión promocionada.")


@app.command()
def predict(
    next_matchday: bool = typer.Option(
        False, "--next", help="Predice la próxima jornada con partidos programados."
    ),
    matchday: int | None = typer.Option(
        None, "--matchday", help="Predice una jornada concreta de la temporada actual."
    ),
    no_odds: bool = typer.Option(
        False, "--no-odds", help="Fuerza la variante sin cuotas del modelo."
    ),
) -> None:
    """Predice partidos programados con el último modelo promocionado (salida SPEC §2)."""
    if not next_matchday and matchday is None:
        typer.secho("Indica --next o --matchday N.", err=True)
        raise typer.Exit(code=1)
    settings = _load_settings()
    _require_db(settings)
    from datetime import UTC, datetime

    import pandas as pd

    from alaves_predictor.features.build import build_features, persist_features
    from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS, VARIANT_WITH_ODDS
    from alaves_predictor.models.train import load_latest_model

    conn = db.connect(settings.data.db_path)
    try:
        bundle = load_latest_model(conn)
        if bundle is None:
            typer.secho(
                "No hay ningún modelo en el registry. Ejecuta `alaves train` primero.",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)

        df = build_features(conn, settings, include_scheduled=True)
        scheduled = df[df["result"].isna() & (df["season"] == settings.current_season)]
        if matchday is not None:
            rows = scheduled[scheduled["matchday"] == matchday]
        else:
            rows = (
                scheduled[scheduled["matchday"] == scheduled["matchday"].min()]
                if not scheduled.empty
                else scheduled
            )
        if rows.empty:
            typer.secho(
                "No hay partidos programados que predecir en la BD. El calendario "
                f"de la {settings.current_season} se ingiere en la F7 "
                "(`alaves ingest --matchday`).",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)

        # variante: con cuotas solo si el bundle la tiene y TODOS los partidos
        # tienen cuotas ingeridas (si no, degradaría en silencio)
        variant = VARIANT_NO_ODDS
        if not no_odds and VARIANT_WITH_ODDS in bundle.variants:
            if rows["imp_home"].notna().all():
                variant = VARIANT_WITH_ODDS
            else:
                typer.secho(
                    "Aviso: faltan cuotas de apertura de algún partido; "
                    "se usa la variante sin cuotas.",
                    fg=typer.colors.YELLOW,
                )
        preds = bundle.predict_matches(rows, variant)

        # SPEC §5.5: persistir SIEMPRE antes de conocer el resultado —
        # snapshot de features + fila por predicción, para auditoría real
        persist_features(conn, rows, settings)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        for p in preds.itertuples(index=False):
            conn.execute(
                "INSERT INTO predictions (match_id, model_version, created_at, p_home, "
                "p_draw, p_away, pred_result, pred_score, expected_goals_h, expected_goals_a) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    p.match_id,
                    bundle.model_version,
                    now,
                    p.p_home,
                    p.p_draw,
                    p.p_away,
                    p.pred_result,
                    p.pred_score,
                    p.expected_goals_h,
                    p.expected_goals_a,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    labels = {"H": "1 (victoria local)", "D": "X (empate)", "A": "2 (victoria visitante)"}
    for p in preds.itertuples(index=False):
        home = settings.teams[p.home_id].name if p.home_id in settings.teams else p.home_id
        away = settings.teams[p.away_id].name if p.away_id in settings.teams else p.away_id
        jornada = f"Jornada {int(p.matchday)}" if not pd.isna(p.matchday) else "Jornada ?"
        typer.secho(f"{home} vs {away} — {jornada} — {p.date}", bold=True)
        typer.echo(f"Resultado predicho: {labels[p.pred_result]}")
        typer.echo(f"P(victoria local):     {p.p_home * 100:5.1f} %")
        typer.echo(f"P(empate):             {p.p_draw * 100:5.1f} %")
        typer.echo(f"P(victoria visitante): {p.p_away * 100:5.1f} %")
        typer.echo(f"Marcador más probable: {p.pred_score} (p = {p.pred_score_prob * 100:.1f} %)")
        typer.echo("")
    typer.echo(
        f"{len(preds)} predicciones persistidas en BD "
        f"(modelo {bundle.model_version}, variante {variant})."
    )


@app.command()
def simulate(
    n: int = typer.Option(10000, "--n", help="Número de simulaciones Monte Carlo."),
    seed: int = typer.Option(42, "--seed", help="Semilla (reproducibilidad)."),
    season: str | None = typer.Option(
        None, "--season", help="Modo demo: proyecta una temporada histórica (p. ej. 2025-26)."
    ),
    from_matchday: int = typer.Option(
        1, "--from-matchday", help="En modo demo, jornada desde la que se proyecta."
    ),
    no_odds: bool = typer.Option(False, "--no-odds", help="Fuerza la variante sin cuotas."),
) -> None:
    """Proyecta la clasificación por Monte Carlo sobre las probabilidades del ensemble (F4).

    Sin `--season` simula la temporada actual (partidos jugados = clasificación,
    programados = por simular). Con `--season S --from-matchday N` proyecta una
    temporada histórica desde la jornada N tomando las anteriores como reales
    (modo demo/validación: permite contrastar la tabla proyectada con la real).
    """
    settings = _load_settings()
    _require_db(settings)
    from alaves_predictor.features.build import build_features
    from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS, VARIANT_WITH_ODDS
    from alaves_predictor.models.train import load_latest_model
    from alaves_predictor.simulation import monte_carlo as mc

    target_season = season or settings.current_season
    conn = db.connect(settings.data.db_path)
    try:
        bundle = load_latest_model(conn)
        if bundle is None:
            typer.secho("No hay modelo entrenado. Ejecuta `alaves train` primero.", err=True)
            raise typer.Exit(code=1)
        df = build_features(conn, settings, include_scheduled=True)
        df = df[df["season"] == target_season]
        if df.empty:
            typer.secho(f"No hay partidos de {target_season} en la BD.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

        if season is not None:  # modo demo sobre temporada histórica
            played = df[df["matchday"] < from_matchday]
            remaining_rows = df[df["matchday"] >= from_matchday]
        else:  # temporada en curso: jugados vs programados
            played = df[df["result"].notna()]
            remaining_rows = df[df["result"].isna()]

        if remaining_rows.empty:
            typer.secho(
                f"No hay partidos por simular en {target_season}. "
                "El calendario de la temporada en curso se ingiere en la F7.",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)

        has_odds = remaining_rows["imp_home"].notna().all()
        variant = VARIANT_NO_ODDS
        if not no_odds and VARIANT_WITH_ODDS in bundle.variants and has_odds:
            variant = VARIANT_WITH_ODDS
        preds = bundle.predict_matches(remaining_rows, variant)
        remaining = mc.build_remaining(preds, bundle.dixon_coles)
    finally:
        conn.close()

    teams = sorted(set(df["home_id"]) | set(df["away_id"]))
    standings = mc.current_standings(played)
    typer.echo(
        f"Simulando {target_season} desde {len(played)} partidos jugados y "
        f"{len(remaining)} por jugar ({n} simulaciones, variante {variant})..."
    )
    result = mc.simulate(standings, remaining, teams, n=n, seed=seed, zones=settings.league.zones)

    ranked = sorted(teams, key=result.expected_position)
    header = f"{'#':>2} {'Equipo':22} {'Pts':>5} {'Pos':>4}  {'Título':>6} {'Champ':>6} {'Eur':>5} {'Desc':>6}"  # noqa: E501
    typer.secho(f"Clasificación proyectada — {target_season}", bold=True)
    typer.secho(header, bold=True)
    for i, team in enumerate(ranked, start=1):
        name = settings.teams[team].name if team in settings.teams else team
        mark = "◀" if team == settings.focus_team else " "
        typer.echo(
            f"{i:>2} {name[:22]:22} {result.points_for(team):5.1f} "
            f"{result.expected_position(team):4.1f}  "
            f"{result.prob_zone(team, 'titulo') * 100:5.1f}% "
            f"{result.prob_zone(team, 'champions') * 100:5.1f}% "
            f"{result.prob_zone(team, 'europa') * 100:4.0f}% "
            f"{result.prob_zone(team, 'descenso') * 100:5.1f}% {mark}"
        )

    focus = settings.focus_team
    if focus in teams:
        name = settings.teams[focus].name if focus in settings.teams else focus
        typer.secho(f"\nFoco — {name}:", bold=True)
        typer.echo(
            f"  Posición esperada: {result.expected_position(focus):.1f}  "
            f"| Puntos esperados: {result.points_for(focus):.1f}"
        )
        typer.echo(
            f"  P(descenso): {result.prob_zone(focus, 'descenso') * 100:.1f}%  "
            f"| P(Europa o mejor, top-6): {result.prob_between(focus, 1, 6) * 100:.1f}%"
        )


@app.command()
def backtest(
    seasons: int = typer.Option(3, "--seasons", help="Temporadas de test (walk-forward)."),
) -> None:
    """Backtesting walk-forward jornada a jornada, comparado contra los baselines (F3)."""
    settings = _load_settings()
    _require_db(settings)
    from alaves_predictor.evaluation.backtest import acceptance_checks, run_backtest, write_report
    from alaves_predictor.evaluation.baselines import run_baselines
    from alaves_predictor.features.build import build_features

    conn = db.connect(settings.data.db_path)
    try:
        df = build_features(conn, settings)
        typer.echo("Backtest jornada a jornada (reentrena cada jornada: tarda unos minutos)...")
        output = run_backtest(
            df, settings, n_test_seasons=seasons, progress=lambda m: typer.echo(f"  {m}")
        )
        baselines = run_baselines(conn, df, settings, n_test_seasons=seasons)
    finally:
        conn.close()

    typer.secho("Resultados por modelo y temporada:", bold=True)
    for r in output.rows:
        _echo_metrics(f"{r.model} {r.season}", r.metrics)
    typer.secho("Criterios de aceptación (SPEC §12.1):", bold=True)
    for label, passed, detail in acceptance_checks(output.rows, baselines):
        color = typer.colors.GREEN if passed else typer.colors.RED
        icon = "✓" if passed else "✗"
        typer.secho(f"  {icon} {label}: {detail}", fg=color)
    report_path = write_report(output, baselines, Path("docs/reports"))
    typer.echo(f"Informe guardado en {report_path}")


@app.command()
def report(
    importance: bool = typer.Option(
        False, "--importance", help="Genera el informe SHAP + ablation (F5)."
    ),
    seasons: int = typer.Option(3, "--seasons", help="Temporadas del ablation walk-forward."),
) -> None:
    """Informe de importancia de variables: SHAP global, dependencia parcial y ablation (F5)."""
    if not importance:
        typer.secho("Indica --importance.", err=True)
        raise typer.Exit(code=1)
    settings = _load_settings()
    _require_db(settings)
    from alaves_predictor.explain.report import generate_report
    from alaves_predictor.features.build import build_features
    from alaves_predictor.models.train import load_latest_model

    conn = db.connect(settings.data.db_path)
    try:
        bundle = load_latest_model(conn)
        if bundle is None:
            typer.secho("No hay modelo entrenado. Ejecuta `alaves train` primero.", err=True)
            raise typer.Exit(code=1)
        typer.echo("Calculando SHAP, dependencia parcial y ablation (tarda un poco)...")
        df = build_features(conn, settings)
        report_path = generate_report(bundle, df, settings, Path("docs/reports"), seasons)
    finally:
        conn.close()
    typer.secho(
        f"Informe de importancia guardado en {report_path}", fg=typer.colors.GREEN, bold=True
    )
    typer.echo("  Figuras PNG en docs/reports/figures/")


if __name__ == "__main__":
    app()
