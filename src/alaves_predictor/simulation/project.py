"""Ensamblado de la proyección de clasificación (compartido por CLI y dashboard).

Junta las piezas del motor Monte Carlo (`monte_carlo.py`) con el modelo
entrenado: separa una temporada en jugado / por jugar, predice los pendientes
con el ensemble y simula. Se aísla aquí para que el comando `alaves simulate`
y la página del dashboard usen exactamente el mismo camino (sin duplicar
lógica en la capa de presentación, CLAUDE.md §2).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alaves_predictor.config import Settings
from alaves_predictor.models.gbm_classifier import VARIANT_NO_ODDS, VARIANT_WITH_ODDS
from alaves_predictor.models.train import ModelBundle
from alaves_predictor.simulation import monte_carlo as mc


@dataclass
class Projection:
    """Resultado de la proyección más el contexto que la generó."""

    season: str
    result: mc.SimulationResult
    teams: list[str]
    variant: str
    n_played: int
    n_remaining: int
    # Partidos de la temporada que no están en la BD ni jugados ni programados.
    # Con el calendario a medias (p. ej. Understat caído: football-data solo
    # trae los inminentes) la simulación cubre solo lo conocido y la tabla que
    # sale NO es la clasificación final: quien la muestre debe decirlo (ADR-033).
    n_missing: int = 0

    def warning(self) -> str | None:
        """Aviso para mostrar junto a la proyección, o None si está completa."""
        if not self.n_missing:
            return None
        known = self.n_played + self.n_remaining
        return (
            f"Calendario incompleto: faltan {self.n_missing} partidos de la temporada "
            f"(conocidos {known}). La simulación solo cubre esos, así que los puntos y "
            "puestos NO son la clasificación final. Falta el calendario completo, que "
            "aporta Understat (o FBref)."
        )


def split_season(
    season_df: pd.DataFrame, from_matchday: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa (jugados, por jugar).

    `from_matchday` None => modo temporada en curso: jugados = con resultado,
    por jugar = programados (resultado nulo). Con `from_matchday` => modo demo:
    las jornadas anteriores son el punto de partida y de la N en adelante se
    proyectan (ignorando su resultado real si lo hubiera).
    """
    if from_matchday is None:
        return season_df[season_df["result"].notna()], season_df[season_df["result"].isna()]
    return (
        season_df[season_df["matchday"] < from_matchday],
        season_df[season_df["matchday"] >= from_matchday],
    )


def choose_variant(bundle: ModelBundle, remaining: pd.DataFrame, no_odds: bool) -> str:
    """Variante con cuotas solo si el bundle la tiene y TODOS los pendientes las tienen."""
    if not no_odds and VARIANT_WITH_ODDS in bundle.variants and remaining["imp_home"].notna().all():
        return VARIANT_WITH_ODDS
    return VARIANT_NO_ODDS


def project_standings(
    bundle: ModelBundle,
    features: pd.DataFrame,
    settings: Settings,
    season: str,
    from_matchday: int | None = None,
    n: int = 10000,
    seed: int = 42,
    no_odds: bool = False,
) -> Projection | None:
    """Proyecta la clasificación de `season`. Devuelve None si no hay nada por simular.

    `features` debe venir de build_features(..., include_scheduled=True).
    """
    season_df = features[features["season"] == season]
    if season_df.empty:
        return None
    played, remaining_rows = split_season(season_df, from_matchday)
    if remaining_rows.empty:
        return None

    variant = choose_variant(bundle, remaining_rows, no_odds)
    preds = bundle.predict_matches(remaining_rows, variant)
    remaining = mc.build_remaining(preds, bundle.dixon_coles)

    teams = sorted(set(season_df["home_id"]) | set(season_df["away_id"]))
    standings = mc.current_standings(played)
    result = mc.simulate(standings, remaining, teams, n=n, seed=seed, zones=settings.league.zones)
    return Projection(
        season=season,
        result=result,
        teams=teams,
        variant=variant,
        n_played=len(played),
        n_remaining=len(remaining),
        n_missing=max(0, settings.league.matches_per_season - len(season_df)),
    )
