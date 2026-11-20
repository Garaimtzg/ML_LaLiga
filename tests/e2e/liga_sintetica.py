"""Liga sintética servida en el formato exacto de cada fuente (test punta a punta).

NO son datos reales: sirve para ejercitar todo el pipeline (descarga → parseo →
BD → features → modelos → simulación) sin red, que es lo que no cubren los
fixtures congelados de 12 partidos. Los equipos y la composición de cada
temporada son los reales; los resultados, xG, cuotas y Elo se generan con un
Poisson con fuerzas latentes por equipo (semilla fija). Imita también rarezas
reales de las fuentes: cabeceras repetidas en FBref, FBref sin xG desde 2024-25
(lo rellena Understat), BOM en el fixtures.csv, nombres "de época" en FBref y
un partido de la 2026-27 aplazado (J9 → 2 de diciembre).

`build_web(today)` devuelve {url: (status, bytes)}: lo que "internet" serviría
en la fecha `today`. Lo jugado antes de esa fecha tiene resultado; el resto es
calendario.
"""

from __future__ import annotations

import csv
import io
import json
import math
import tomllib
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TEAMS = tomllib.loads((REPO / "config/teams.toml").read_text(encoding="utf-8"))


def _teams(names: str) -> list[str]:
    return names.split()


SEASONS: dict[str, list[str]] = {
    "2018-19": _teams(
        "alaves athletic-club atletico-madrid barcelona betis celta-vigo eibar espanyol "
        "getafe girona huesca leganes levante rayo-vallecano real-madrid real-sociedad sevilla "
        "valencia valladolid villarreal"
    ),
    "2019-20": _teams(
        "alaves athletic-club atletico-madrid barcelona betis celta-vigo eibar espanyol "
        "getafe granada leganes levante mallorca osasuna real-madrid real-sociedad sevilla "
        "valencia valladolid villarreal"
    ),
    "2020-21": _teams(
        "alaves athletic-club atletico-madrid barcelona betis cadiz celta-vigo eibar "
        "elche getafe granada huesca levante osasuna real-madrid real-sociedad sevilla valencia "
        "valladolid villarreal"
    ),
    "2021-22": _teams(
        "alaves athletic-club atletico-madrid barcelona betis cadiz celta-vigo elche "
        "espanyol getafe granada levante mallorca osasuna rayo-vallecano real-madrid "
        "real-sociedad sevilla valencia villarreal"
    ),
    "2022-23": _teams(
        "almeria athletic-club atletico-madrid barcelona betis cadiz celta-vigo elche "
        "espanyol getafe girona mallorca osasuna rayo-vallecano real-madrid real-sociedad "
        "sevilla valencia valladolid villarreal"
    ),
    "2023-24": _teams(
        "alaves almeria athletic-club atletico-madrid barcelona betis cadiz celta-vigo "
        "getafe girona granada las-palmas mallorca osasuna rayo-vallecano real-madrid "
        "real-sociedad sevilla valencia villarreal"
    ),
    "2024-25": _teams(
        "alaves athletic-club atletico-madrid barcelona betis celta-vigo espanyol getafe "
        "girona las-palmas leganes mallorca osasuna rayo-vallecano real-madrid real-sociedad "
        "sevilla valencia valladolid villarreal"
    ),
    "2025-26": _teams(
        "alaves athletic-club atletico-madrid barcelona betis celta-vigo elche espanyol "
        "getafe girona levante mallorca osasuna oviedo rayo-vallecano real-madrid real-sociedad "
        "sevilla valencia villarreal"
    ),
    "2026-27": _teams(
        "alaves athletic-club atletico-madrid barcelona betis celta-vigo deportivo-la-coruna "
        "elche espanyol getafe malaga mallorca osasuna racing-santander rayo-vallecano "
        "real-madrid real-sociedad sevilla valencia villarreal"
    ),
}
CURRENT = "2026-27"
for s, t in SEASONS.items():
    assert len(t) == 20 and len(set(t)) == 20, s
    assert all(x in TEAMS for x in t), s

# Fuerza latente inicial (aprox. por niveles); se mueve cada temporada.
TIER = {
    "real-madrid": 0.75,
    "barcelona": 0.7,
    "atletico-madrid": 0.55,
    "sevilla": 0.3,
    "villarreal": 0.3,
    "real-sociedad": 0.3,
    "athletic-club": 0.25,
    "betis": 0.2,
    "valencia": 0.15,
    "girona": 0.05,
    "celta-vigo": 0.0,
    "osasuna": 0.0,
}


def alias(team: str, source: str, k: int = 0) -> str:
    value = TEAMS[team][source]
    if isinstance(value, list):
        return value[min(k, len(value) - 1)]
    return value


# ---------------------------------------------------------------------------
# Generación de la liga
# ---------------------------------------------------------------------------


def round_robin(teams: list[str], rng: np.random.Generator) -> list[list[tuple[str, str]]]:
    """Calendario a doble vuelta por el método del círculo (38 jornadas)."""
    t = list(teams)
    rng.shuffle(t)
    n = len(t)
    rounds = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = t[i], t[n - 1 - i]
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(pairs)
        t = [t[0], t[-1], *t[1:-1]]
    second = [[(b, a) for a, b in rnd] for rnd in rounds]
    return rounds + second


def round_dates(season: str) -> list[date]:
    """Fecha (viernes) de inicio de cada jornada, con parones y alguna entre semana."""
    y = int(season[:4])
    d = date(y, 8, 14)
    while d.weekday() != 4:
        d += timedelta(days=1)
    starts = []
    for r in range(1, 39):
        starts.append(d)
        if d.weekday() == 1:  # tras una jornada entre semana, vuelta al viernes
            d += timedelta(days=3)
            continue
        if r in (6, 15, 32):  # la siguiente es entre semana (martes)
            d += timedelta(days=4)
            continue
        step = 7
        if r in (3, 7, 12):
            step = 14  # parón de selecciones
        if r == 17:
            step = 21  # Navidad
        d += timedelta(days=step)
    return starts


def match_day_offsets(n: int, midweek: bool) -> list[int]:
    if midweek:
        return [0] * (n // 2) + [1] * (n - n // 2)
    return [0] + [1] * 4 + [2] * 4 + [3] * (n - 9)


def outcome_probs(lam: float, mu: float, maxg: int = 10) -> tuple[float, float, float]:
    ph = pd_ = pa = 0.0
    fh = [math.exp(-lam) * lam**i / math.factorial(i) for i in range(maxg + 1)]
    fa = [math.exp(-mu) * mu**j / math.factorial(j) for j in range(maxg + 1)]
    for i in range(maxg + 1):
        for j in range(maxg + 1):
            p = fh[i] * fa[j]
            if i > j:
                ph += p
            elif i == j:
                pd_ += p
            else:
                pa += p
    s = ph + pd_ + pa
    return ph / s, pd_ / s, pa / s


def odds_from(p: tuple[float, float, float], margin: float, noise: float, rng) -> list[float]:
    q = np.array(p) * np.exp(rng.normal(0, noise, 3))
    q = q / q.sum()
    return [round(1.0 / (x * (1 + margin)), 2) for x in q]


def generate_league(seed: int = 42) -> list[dict]:
    """Todos los partidos de todas las temporadas, con lo que cada fuente necesita."""
    rng = np.random.default_rng(seed)
    all_teams = list(TEAMS)
    strength = {t: TIER.get(t, -0.1) + rng.normal(0, 0.1) for t in all_teams}
    matches: list[dict] = []
    uid = 10000
    for season, teams in SEASONS.items():
        # deriva entre temporadas
        for t in all_teams:
            strength[t] = 0.85 * strength[t] + 0.15 * TIER.get(t, -0.1) + rng.normal(0, 0.08)
        att = {t: strength[t] * 0.55 + rng.normal(0, 0.05) for t in teams}
        dfn = {t: strength[t] * 0.45 + rng.normal(0, 0.05) for t in teams}
        home_adv = 0.08 if season == "2020-21" else 0.24
        schedule = round_robin(teams, rng)
        starts = round_dates(season)
        for r, (pairs, start) in enumerate(zip(schedule, starts, strict=True), start=1):
            offs = match_day_offsets(len(pairs), midweek=start.weekday() == 1)
            order = rng.permutation(len(pairs))
            for k, idx in enumerate(order):
                h, a = pairs[idx]
                lam = math.exp(0.05 + home_adv + att[h] - dfn[a])
                mu = math.exp(0.05 + att[a] - dfn[h])
                hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu))
                p_true = outcome_probs(lam, mu)
                hxg = max(0.05, 0.6 * lam + 0.4 * hg + rng.normal(0, 0.3))
                axg = max(0.05, 0.6 * mu + 0.4 * ag + rng.normal(0, 0.3))
                hs = int(rng.poisson(6 + 5 * lam))
                as_ = int(rng.poisson(6 + 5 * mu))
                uid += 1
                matches.append(
                    {
                        "id": uid,
                        "season": season,
                        "round": r,
                        "date": start + timedelta(days=offs[k]),
                        "time": ["18:30", "21:00", "16:15", "14:00"][k % 4],
                        "home": h,
                        "away": a,
                        "hg": hg,
                        "ag": ag,
                        "hxg": round(hxg, 2),
                        "axg": round(axg, 2),
                        "stats": [
                            hs,
                            as_,
                            int(rng.binomial(hs, 0.35)),
                            int(rng.binomial(as_, 0.35)),
                            int(rng.poisson(12)),
                            int(rng.poisson(12)),
                            int(rng.poisson(5)),
                            int(rng.poisson(4)),
                            int(rng.poisson(2.3)),
                            int(rng.poisson(2.5)),
                            int(rng.random() < 0.05),
                            int(rng.random() < 0.06),
                        ],
                        "odds_open": {
                            "B365": odds_from(p_true, 0.055, 0.12, rng),
                            "PS": odds_from(p_true, 0.025, 0.10, rng),
                            "Max": odds_from(p_true, -0.01, 0.10, rng),
                            "Avg": odds_from(p_true, 0.05, 0.10, rng),
                        },
                        "odds_close": {
                            "B365": odds_from(p_true, 0.05, 0.06, rng),
                            "PS": odds_from(p_true, 0.02, 0.05, rng),
                            "Max": odds_from(p_true, -0.01, 0.05, rng),
                            "Avg": odds_from(p_true, 0.045, 0.05, rng),
                        },
                    }
                )
    # Un aplazado en la temporada en curso: un partido de la J9 pasa a diciembre.
    postponed = next(
        m
        for m in matches
        if m["season"] == CURRENT and m["round"] == 9 and "alaves" not in (m["home"], m["away"])
    )
    postponed["original_date"] = postponed["date"]
    postponed["date"] = date(2026, 12, 2)
    postponed["time"] = "21:00"
    return matches


# ---------------------------------------------------------------------------
# Serialización en el formato de cada fuente
# ---------------------------------------------------------------------------

FD_COLS = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR"
FD_COLS_LIST = FD_COLS.split(",")


def _ftr(m: dict) -> str:
    return "H" if m["hg"] > m["ag"] else ("D" if m["hg"] == m["ag"] else "A")


def football_data_csv(season: str, played: list[dict]) -> bytes:
    with_close = season >= "2019-20"  # como la fuente real: cierre desde 2019-20
    cols = list(FD_COLS_LIST)
    for p in ("B365", "PS", "Max", "Avg"):
        cols += [f"{p}H", f"{p}D", f"{p}A"]
    if with_close:
        for p in ("B365", "PS", "Max", "Avg"):
            cols += [f"{p}CH", f"{p}CD", f"{p}CA"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(cols)
    for m in sorted(played, key=lambda x: (x["date"], x["time"])):
        row = [
            "SP1",
            m["date"].strftime("%d/%m/%Y"),
            m["time"],
            alias(m["home"], "football_data"),
            alias(m["away"], "football_data"),
            m["hg"],
            m["ag"],
            _ftr(m),
            *m["stats"],
        ]
        for p in ("B365", "PS", "Max", "Avg"):
            row += m["odds_open"][p]
        if with_close:
            for p in ("B365", "PS", "Max", "Avg"):
                row += m["odds_close"][p]
        w.writerow(row)
    w.writerow([""] * 5)  # fila vacía de cola, como en los CSV reales
    return buf.getvalue().encode("latin-1")


def football_data_fixtures(upcoming: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(
        [
            "Div",
            "Date",
            "Time",
            "HomeTeam",
            "AwayTeam",
            "B365H",
            "B365D",
            "B365A",
            "AvgH",
            "AvgD",
            "AvgA",
        ]
    )
    for m in upcoming:
        w.writerow(
            [
                "SP1",
                m["date"].strftime("%d/%m/%Y"),
                m["time"],
                alias(m["home"], "football_data"),
                alias(m["away"], "football_data"),
                *m["odds_open"]["B365"],
                *m["odds_open"]["Avg"],
            ]
        )
    w.writerow(
        [
            "E0",
            upcoming[0]["date"].strftime("%d/%m/%Y"),
            "15:00",
            "Arsenal",
            "Chelsea",
            2.1,
            3.4,
            3.3,
            2.1,
            3.4,
            3.3,
        ]
    )
    return ("﻿" + buf.getvalue()).encode("utf-8")  # con BOM, como a veces la real


def fbref_html(season: str, ms: list[dict], today: date, with_xg: bool) -> bytes:
    y = int(season[:4])
    slug = f"{y}-{y + 1}"
    head = (
        "<tr><th data-stat='gameweek'>Wk</th><th data-stat='dayofweek'>Day</th>"
        "<th data-stat='date'>Date</th><th data-stat='home_team'>Home</th>"
        "<th data-stat='home_xg'>xG</th><th data-stat='score'>Score</th>"
        "<th data-stat='away_xg'>xG</th><th data-stat='away_team'>Away</th>"
        "<th data-stat='notes'>Notes</th></tr>"
    )
    rows = []
    last_round = None
    k_alias = 0 if season >= "2022-23" else 1  # nombres "de época" en temporadas viejas
    for m in sorted(ms, key=lambda x: (x["round"], x["date"], x["time"])):
        if last_round is not None and m["round"] != last_round:
            rows.append("<tr class='spacer partial_table'><td colspan='9'></td></tr>")
            if m["round"] % 10 == 0:
                rows.append(
                    "<tr class='thead'>" + head.replace("<tr>", "").replace("</tr>", "") + "</tr>"
                )
        last_round = m["round"]
        played = m["date"] < today
        score = f"{m['hg']}&ndash;{m['ag']}" if played else ""
        hxg = f"{m['hxg']:.1f}" if played and with_xg else ""
        axg = f"{m['axg']:.1f}" if played and with_xg else ""
        note = "Match Postponed" if "original_date" in m and not played else ""
        rows.append(
            f"<tr><th scope='row' data-stat='gameweek'>{m['round']}</th>"
            f"<td data-stat='dayofweek'>{m['date']:%a}</td>"
            f"<td data-stat='date'><a href='/en/matches/{m['date']}'>{m['date']}</a></td>"
            f"<td data-stat='home_team'><a href='/x'>{alias(m['home'], 'fbref', k_alias)}</a></td>"
            f"<td data-stat='home_xg'>{hxg}</td>"
            f"<td data-stat='score'><a href='/y'>{score}</a></td>"
            f"<td data-stat='away_xg'>{axg}</td>"
            f"<td data-stat='away_team'><a href='/z'>{alias(m['away'], 'fbref', k_alias)}</a></td>"
            f"<td data-stat='notes'>{note}</td></tr>"
        )
    html = (
        f"<!DOCTYPE html><html><head><title>{slug} La Liga Scores &amp; Fixtures | FBref.com"
        f"</title></head><body><div class='table_container'>"
        f"<table class='stats_table' id='sched_{slug}_12_1'><thead>{head}</thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></body></html>"
    )
    return html.encode("utf-8")


def understat_json(ms: list[dict], today: date) -> bytes:
    entries = []
    for m in sorted(ms, key=lambda x: (x["date"], x["time"])):
        played = m["date"] < today
        entries.append(
            {
                "id": str(m["id"]),
                "isResult": played,
                "h": {"id": "1", "title": alias(m["home"], "understat"), "short_title": "H"},
                "a": {"id": "2", "title": alias(m["away"], "understat"), "short_title": "A"},
                "goals": {"h": str(m["hg"]), "a": str(m["ag"])}
                if played
                else {"h": None, "a": None},
                "xG": {"h": f"{m['hxg']:.6f}", "a": f"{m['axg']:.6f}"}
                if played
                else {"h": None, "a": None},
                "datetime": f"{m['date']} {m['time']}:00",
                "forecast": {"w": "0.4", "d": "0.3", "l": "0.3"} if played else None,
            }
        )
    return json.dumps({"teams": {}, "players": [], "dates": entries}).encode("utf-8")


def clubelo_csv(team: str, matches: list[dict], today: date, rng) -> bytes:
    """Histórico tipo ClubElo: un tramo de vigencia por partido (o semana en Segunda)."""
    rating = 1500 + 350 * (TIER.get(team, -0.1)) + rng.normal(0, 20)
    mine = sorted(
        (m for m in matches if team in (m["home"], m["away"]) and m["date"] < today),
        key=lambda x: x["date"],
    )
    # puntos de cambio: cada partido de liga + semanas sueltas (otras competiciones)
    d = date(2016, 7, 1)
    changes: list[tuple[date, float]] = [(d, rating)]
    for m in mine:
        exp = 1 / (1 + 10 ** (-(rating + (60 if m["home"] == team else -60) - 1600) / 400))
        gf, ga = (m["hg"], m["ag"]) if m["home"] == team else (m["ag"], m["hg"])
        res = 1.0 if gf > ga else (0.5 if gf == ga else 0.0)
        rating += 20 * (res - exp) + rng.normal(0, 3)
        changes.append((m["date"] + timedelta(days=1), rating))
    if len(mine) < 50:  # equipo sobre todo de Segunda: deriva semanal
        d = date(2016, 7, 8)
        while d < today:
            rating += rng.normal(0, 4)
            changes.append((d, rating))
            d += timedelta(days=7)
    changes.sort(key=lambda x: x[0])
    dedup: dict[date, float] = {}
    for when, r in changes:
        dedup[when] = r
    items = sorted(dedup.items())
    buf = io.StringIO()
    buf.write("Rank,Club,Country,Level,Elo,From,To\n")
    for i, (when, r) in enumerate(items):
        to = (
            items[i + 1][0] - timedelta(days=1)
            if i + 1 < len(items)
            else today + timedelta(days=30)
        )
        buf.write(f"None,{alias(team, 'clubelo')},ESP,1,{r:.5f},{when},{to}\n")
    return buf.getvalue().encode("utf-8")


def build_web(today: date) -> dict[str, tuple[int, bytes]]:
    """URL -> (status, cuerpo) de la "internet" sintética en la fecha `today`."""
    matches = generate_league()
    web: dict[str, tuple[int, bytes]] = {}
    by_season: dict[str, list[dict]] = {}
    for m in matches:
        by_season.setdefault(m["season"], []).append(m)
    for season, ms in by_season.items():
        y = int(season[:4])
        code = f"{str(y)[-2:]}{str(y + 1)[-2:]}"
        slug = f"{y}-{y + 1}"
        played = [m for m in ms if m["date"] < today]
        if played:
            web[f"https://www.football-data.co.uk/mmz4281/{code}/SP1.csv"] = (
                200,
                football_data_csv(season, played),
            )
        # FBref quitó el xG del calendario en 2026 (ADR-011): las temporadas
        # recientes llegan sin xG y lo rellena Understat.
        html = fbref_html(season, ms, today, with_xg=season <= "2023-24")
        if season == CURRENT:
            web["https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures"] = (200, html)
        else:
            web[
                f"https://fbref.com/en/comps/12/{slug}/schedule/{slug}-La-Liga-Scores-and-Fixtures"
            ] = (200, html)
        web[f"https://understat.com/getLeagueData/La%20liga/{y}"] = (200, understat_json(ms, today))
    cur = [m for m in by_season[CURRENT] if m["date"] >= today]
    upcoming = [m for m in cur if m["date"] < today + timedelta(days=10)]
    if upcoming:
        web["https://www.football-data.co.uk/fixtures.csv"] = (
            200,
            football_data_fixtures(upcoming),
        )
    rng = np.random.default_rng(7)
    for team in TEAMS:
        web[f"http://api.clubelo.com/{alias(team, 'clubelo')}"] = (
            200,
            clubelo_csv(team, matches, today, rng),
        )
    return web
