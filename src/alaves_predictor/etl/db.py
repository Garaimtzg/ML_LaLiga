"""Capa de acceso a la base de datos SQLite (esquema de SPEC §3.2).

Decisiones (ADR-002):
- SQLite como almacén único: proyecto monousuario local, cero infraestructura.
- Identificadores de texto legibles: team_id es un slug ("alaves") y match_id
  es determinista ("2018-19_alaves_barcelona"), lo que hace la BD auditable a
  ojo y las ingestas idempotentes (re-ejecutar no duplica filas).
- Se crea el esquema completo desde F1 (incluidas tablas que se poblarán en
  fases posteriores: features, predictions, model_registry...) para que el
  contrato de datos quede fijado desde el principio.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Columnas de estadísticas por equipo y partido (SPEC §3.2, tabla match_stats).
# Se almacenan TODAS aunque F1 solo puebla un subconjunto (football-data:
# tiros/córners/faltas/tarjetas; understat: xg). El resto lo llenará FBref en F2.
MATCH_STATS_COLUMNS: list[str] = [
    # Tiro
    "xg",
    "npxg",
    "shots",
    "shots_on_target",
    "shot_distance_avg",
    "goals_per_shot",
    # Pase
    "passes_completed",
    "passes_attempted",
    "pass_accuracy_pct",
    "progressive_passes",
    "passes_final_third",
    "passes_penalty_area",
    "key_passes",
    "crosses",
    "xa",
    # Posesión y conducción
    "possession_pct",
    "touches",
    "touches_att_third",
    "progressive_carries",
    "dribbles_completed",
    "dispossessed",
    # Defensa
    "tackles",
    "tackles_won",
    "interceptions",
    "blocks",
    "clearances",
    "errors_leading_to_shot",
    "ppda",
    # Portería
    "psxg",
    "saves",
    "save_pct",
    # Balón parado
    "corners",
    "set_piece_shots",
    "set_piece_goals",
    # Disciplina
    "fouls",
    "cards_yellow",
    "cards_red",
    "penalties_conceded",
    # Otros
    "aerials_won_pct",
    "offsides",
]

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS teams (
    team_id      TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    aliases_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS matches (
    match_id   TEXT PRIMARY KEY,
    season     TEXT NOT NULL,
    matchday   INTEGER,
    date       TEXT NOT NULL,          -- ISO YYYY-MM-DD
    home_id    TEXT NOT NULL REFERENCES teams(team_id),
    away_id    TEXT NOT NULL REFERENCES teams(team_id),
    home_goals INTEGER,
    away_goals INTEGER,
    status     TEXT NOT NULL,          -- 'finished' | 'scheduled'
    source     TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    UNIQUE (season, home_id, away_id)
);

CREATE TABLE IF NOT EXISTS match_stats (
    match_id   TEXT NOT NULL REFERENCES matches(match_id),
    team_id    TEXT NOT NULL REFERENCES teams(team_id),
    is_home    INTEGER NOT NULL,
    {", ".join(f"{col} REAL" for col in MATCH_STATS_COLUMNS)},
    source     TEXT NOT NULL,          -- fuentes que aportaron columnas ("fd+understat")
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (match_id, team_id)
);

CREATE TABLE IF NOT EXISTS odds (
    match_id   TEXT NOT NULL REFERENCES matches(match_id),
    bookmaker  TEXT NOT NULL,          -- bet365 | pinnacle | market_max | market_avg
    open_h     REAL, open_d REAL, open_a REAL,
    close_h    REAL, close_d REAL, close_a REAL,
    source     TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (match_id, bookmaker)
);

CREATE TABLE IF NOT EXISTS elo (
    team_id      TEXT NOT NULL REFERENCES teams(team_id),
    date         TEXT NOT NULL,        -- inicio de vigencia del rating (ISO)
    elo_clubelo  REAL,
    elo_internal REAL,                 -- se calcula en F2
    source       TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (team_id, date)
);

CREATE TABLE IF NOT EXISTS squad_values (
    team_id          TEXT NOT NULL REFERENCES teams(team_id),
    date             TEXT NOT NULL,
    market_value_eur REAL,
    mean_age         REAL,
    n_players        INTEGER,
    source           TEXT NOT NULL,
    fetched_at       TEXT NOT NULL,
    PRIMARY KEY (team_id, date)
);

CREATE TABLE IF NOT EXISTS injuries (
    team_id         TEXT NOT NULL REFERENCES teams(team_id),
    date            TEXT NOT NULL,
    player          TEXT NOT NULL,
    expected_return TEXT,
    source          TEXT NOT NULL,
    fetched_at      TEXT NOT NULL,
    PRIMARY KEY (team_id, date, player)
);

CREATE TABLE IF NOT EXISTS features (
    match_id            TEXT NOT NULL REFERENCES matches(match_id),
    feature_set_version TEXT NOT NULL,
    as_of_date          TEXT NOT NULL,
    payload_json        TEXT NOT NULL,
    PRIMARY KEY (match_id, feature_set_version)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id         TEXT NOT NULL REFERENCES matches(match_id),
    model_version    TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    p_home           REAL NOT NULL,
    p_draw           REAL NOT NULL,
    p_away           REAL NOT NULL,
    pred_result      TEXT NOT NULL,
    pred_score       TEXT,
    expected_goals_h REAL,
    expected_goals_a REAL
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version TEXT PRIMARY KEY,
    trained_at    TEXT NOT NULL,
    train_window  TEXT NOT NULL,
    metrics_json  TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    artifact_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_elo_date ON elo(date);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Abre la BD (creando el directorio si falta) con claves foráneas activas.

    timeout=30: si otro proceso (u OneDrive sincronizando el archivo) tiene el
    bloqueo, SQLite reintenta durante 30 s antes de rendirse con
    'database is locked', en vez de fallar al instante.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def upsert(conn: sqlite3.Connection, table: str, row: dict, key_cols: list[str]) -> None:
    """INSERT ... ON CONFLICT DO UPDATE genérico.

    En caso de conflicto solo sobreescribe con valores no nulos (COALESCE),
    de modo que dos fuentes pueden completar columnas distintas de la misma
    fila sin pisarse (p. ej. football-data pone shots y understat pone xg).
    """
    cols = list(row.keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(
        f"{c} = COALESCE(excluded.{c}, {table}.{c})" for c in cols if c not in key_cols
    )
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({', '.join(key_cols)}) DO UPDATE SET {updates}"
    )
    conn.execute(sql, row)


def merge_source_tag(existing: str | None, new: str) -> str:
    """Combina etiquetas de procedencia: "football-data" + "understat" se unen con '+'."""
    if not existing:
        return new
    parts = existing.split("+")
    if new in parts:
        return existing
    return "+".join([*parts, new])


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Nº de filas por tabla, para el comando `alaves status`."""
    tables = [
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        if not r["name"].startswith("sqlite_")
    ]
    return {t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"] for t in tables}
