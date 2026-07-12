"""Modelo Dixon-Coles: Poisson bivariante con corrección de marcadores bajos.

Implementación propia por máxima verosimilitud con scipy.optimize (SPEC §6.2,
ADR-015; sin librerías de terceros abandonadas). El modelo:

    goles_local     ~ Poisson(lambda),  lambda = exp(atk_local − def_visit + gamma)
    goles_visitante ~ Poisson(mu),      mu     = exp(atk_visit − def_local)

donde `atk` es la fuerza de ataque de cada equipo, `def` su fuerza defensiva
(mayor = mejor defensa) y `gamma` la ventaja de campo. La independencia entre
ambos Poisson se corrige en los marcadores bajos (donde empíricamente falla)
con el factor tau de Dixon & Coles (1997), parametrizado por rho:

    tau(0,0) = 1 − lambda·mu·rho     tau(1,0) = 1 + mu·rho
    tau(0,1) = 1 + lambda·rho        tau(1,1) = 1 − rho        (resto: 1)

rho < 0 desplaza probabilidad hacia el 0-0 y el 1-1 (y la quita del 1-0/0-1),
que es el patrón observado en fútbol. La suma total sigue siendo 1: las cuatro
correcciones se cancelan exactamente entre sí.

Ponderación temporal (SPEC §6.2): cada partido pesa exp(−xi·días de
antigüedad) respecto a la fecha de referencia, de modo que lo reciente manda
sin descartar la historia.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from alaves_predictor.config import DixonColesConfig

# Nº de equipos "débiles" cuyos parámetros medios hereda un equipo no visto
# en entrenamiento (recién ascendido sin historia en la BD): tratarlo como
# equipo medio lo sobrevaloraría; como la media de los colistas, no (ADR-015).
_N_PROXY_WEAKEST = 3


@dataclass
class DixonColesModel:
    """Parámetros ajustados; los dicts ataque/defensa son el 'ranking' consultable."""

    attack: dict[str, float]
    defense: dict[str, float]
    home_advantage: float
    rho: float
    max_goals: int = 10
    reference_date: str = ""  # fecha respecto a la que se ponderó (auditoría)
    proxy_teams: list[str] = field(default_factory=list)  # colistas usados como proxy

    def _params_for(self, team_id: str) -> tuple[float, float]:
        """Ataque/defensa del equipo; si no se vio en entrenamiento, proxy de colista."""
        if team_id in self.attack:
            return self.attack[team_id], self.defense[team_id]
        proxy = self.proxy_teams or _weakest_teams(self.attack, self.defense)
        atk = float(np.mean([self.attack[t] for t in proxy]))
        dfn = float(np.mean([self.defense[t] for t in proxy]))
        return atk, dfn

    def expected_goals(self, home_id: str, away_id: str) -> tuple[float, float]:
        """(lambda, mu): goles esperados del local y del visitante."""
        atk_h, def_h = self._params_for(home_id)
        atk_a, def_a = self._params_for(away_id)
        lam = float(np.exp(atk_h - def_a + self.home_advantage))
        mu = float(np.exp(atk_a - def_h))
        return lam, mu

    def score_matrix(self, home_id: str, away_id: str) -> np.ndarray:
        """Matriz P[goles_local, goles_visitante], normalizada a suma 1."""
        lam, mu = self.expected_goals(home_id, away_id)
        return score_matrix(lam, mu, self.rho, self.max_goals)

    def outcome_probs(self, home_id: str, away_id: str) -> np.ndarray:
        """[P(H), P(D), P(A)] derivadas de la matriz de marcadores."""
        return outcome_probs(self.score_matrix(home_id, away_id))

    def most_likely_score(self, home_id: str, away_id: str) -> tuple[int, int, float]:
        """(goles_local, goles_visitante, probabilidad) del marcador más probable."""
        matrix = self.score_matrix(home_id, away_id)
        h, a = np.unravel_index(int(matrix.argmax()), matrix.shape)
        return int(h), int(a), float(matrix[h, a])


def _weakest_teams(attack: dict[str, float], defense: dict[str, float]) -> list[str]:
    """Los N equipos de menor fuerza total (ambos parámetros son 'mayor = mejor')."""
    strength = {t: attack[t] + defense[t] for t in attack}
    return sorted(strength, key=strength.get)[:_N_PROXY_WEAKEST]  # type: ignore[arg-type]


def tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Factor de corrección de Dixon-Coles para los cuatro marcadores bajos."""
    out = np.ones_like(lam, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


def time_weights(dates: pd.Series, reference: pd.Timestamp, xi: float) -> np.ndarray:
    """Peso exponencial exp(−xi·días) de cada partido respecto a `reference`."""
    days = (reference - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    return np.exp(-xi * days)


def score_matrix(lam: float, mu: float, rho: float, max_goals: int) -> np.ndarray:
    """Matriz conjunta de marcadores: Poisson independientes × tau, renormalizada.

    La renormalización solo compensa la masa perdida al truncar en max_goals
    (la corrección tau en sí conserva la suma total exactamente).
    """
    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals, lam)
    p_away = poisson.pmf(goals, mu)
    matrix = np.outer(p_home, p_away)
    xs, ys = np.meshgrid(goals, goals, indexing="ij")
    matrix *= tau(xs, ys, np.full_like(matrix, lam), np.full_like(matrix, mu), rho)
    matrix = np.clip(matrix, 0.0, None)  # tau extremo podría rozar negativo
    return matrix / matrix.sum()


def outcome_probs(matrix: np.ndarray) -> np.ndarray:
    """[P(H), P(D), P(A)]: triángulo inferior, diagonal y triángulo superior."""
    p_home = float(np.tril(matrix, k=-1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, k=1).sum())
    return np.array([p_home, p_draw, p_away])


def _negative_log_likelihood(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    """−log-verosimilitud ponderada (sin el término constante log x!·y!)."""
    attack, defense, gamma, rho = _unpack(params, n_teams)
    lam = np.exp(attack[home_idx] - defense[away_idx] + gamma)
    mu = np.exp(attack[away_idx] - defense[home_idx])
    # clip: con rho en su cota y lambda·mu grande, tau podría hacerse <= 0
    correction = np.clip(tau(x, y, lam, mu, rho), 1e-10, None)
    log_lik = np.log(correction) + x * np.log(lam) - lam + y * np.log(mu) - mu
    return float(-np.sum(weights * log_lik))


def _unpack(params: np.ndarray, n_teams: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Vector de optimización → (ataques, defensas, gamma, rho).

    Identificabilidad: sumar c a todos los ataques y a todas las defensas deja
    lambda y mu intactas, así que el ataque del primer equipo no es libre:
    se fija como −suma(resto) (equivale a imponer media de ataques = 0).
    """
    attack = np.concatenate(([-params[: n_teams - 1].sum()], params[: n_teams - 1]))
    defense = params[n_teams - 1 : 2 * n_teams - 1]
    gamma = params[2 * n_teams - 1]
    rho = params[2 * n_teams]
    return attack, defense, gamma, rho


def fit(
    matches: pd.DataFrame,
    cfg: DixonColesConfig,
    reference_date: pd.Timestamp | None = None,
    warm_start: DixonColesModel | None = None,
) -> DixonColesModel:
    """Ajusta el modelo por MLE sobre partidos jugados.

    `matches` requiere columnas: home_id, away_id, home_goals, away_goals, date.
    `reference_date`: fecha para la ponderación temporal (por defecto, el último
    partido del conjunto). `warm_start`: arranca desde un modelo previo (acelera
    los reajustes jornada a jornada del backtest).
    """
    teams = sorted(set(matches["home_id"]) | set(matches["away_id"]))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = matches["home_id"].map(index).to_numpy()
    away_idx = matches["away_id"].map(index).to_numpy()
    x = matches["home_goals"].to_numpy(dtype=float)
    y = matches["away_goals"].to_numpy(dtype=float)
    if reference_date is None:
        reference_date = pd.to_datetime(matches["date"]).max()
    weights = time_weights(matches["date"], reference_date, cfg.xi)

    initial = np.zeros(2 * n + 1)
    initial[2 * n - 1] = 0.25  # gamma: la ventaja de campo real ronda exp(0.25)≈1.3
    if warm_start is not None:
        for team, i in index.items():
            if team in warm_start.attack and i > 0:
                initial[i - 1] = warm_start.attack[team]
            if team in warm_start.defense:
                initial[n - 1 + i] = warm_start.defense[team]
        initial[2 * n - 1] = warm_start.home_advantage
        initial[2 * n] = warm_start.rho

    bounds = (
        [(-3.0, 3.0)] * (n - 1)  # ataques (menos el primero, fijado)
        + [(-3.0, 3.0)] * n  # defensas
        + [(-1.0, 1.0)]  # gamma
        + [(-cfg.rho_bound, cfg.rho_bound)]  # rho
    )
    result = minimize(
        _negative_log_likelihood,
        initial,
        args=(home_idx, away_idx, x, y, weights, n),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success and "ABNORMAL" in str(result.message).upper():
        raise RuntimeError(f"El ajuste Dixon-Coles no convergió: {result.message}")

    attack, defense, gamma, rho = _unpack(result.x, n)
    model = DixonColesModel(
        attack={t: float(attack[i]) for t, i in index.items()},
        defense={t: float(defense[i]) for t, i in index.items()},
        home_advantage=float(gamma),
        rho=float(rho),
        max_goals=cfg.max_goals,
        reference_date=str(pd.to_datetime(reference_date).date()),
    )
    model.proxy_teams = _weakest_teams(model.attack, model.defense)
    return model
