# Predictor del Deportivo Alavés — LaLiga 2026-27

Sistema de machine learning en Python que predice los resultados (1X2 con
probabilidades calibradas y marcador más probable) de los partidos del
Deportivo Alavés — y de toda LaLiga — durante la temporada 2026-27, y proyecta
la clasificación final mediante simulación Monte Carlo.

> Documentos de referencia: [BRIEF.md](BRIEF.md) (visión general),
> [SPEC.md](SPEC.md) (especificación técnica normativa),
> [CLAUDE.md](CLAUDE.md) (forma de trabajo) y
> [docs/decisions/](docs/decisions/) (registro de decisiones, ADRs).

## Estado del proyecto

| Fase | Contenido | Estado |
|------|-----------|--------|
| **F1** | Setup del repo, entorno, ETL de datos históricos (2018-19 → 2025-26) | ✅ **Completada** (pipeline listo; ejecutar la ingesta en local, ver abajo) |
| F2 | Feature engineering + baselines (Elo simple, cuotas implícitas) | Pendiente |
| F3 | Modelos (Dixon-Coles + LightGBM 1X2), calibración, backtesting | Pendiente |
| F4 | Simulador Monte Carlo de la clasificación | Pendiente |
| F5 | Explicabilidad (SHAP) y análisis de variables | Pendiente |
| F6 | Dashboard Streamlit | Pendiente |
| F7 | Modo temporada: ingesta post-jornada + reentrenamiento semanal | Pendiente |

## Requisitos

- **WSL (Ubuntu) o Linux** con Python 3.11+.
- [`uv`](https://docs.astral.sh/uv/) como gestor de entorno. Instalación en WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# reinicia la shell o: source ~/.local/bin/env
```

## Puesta en marcha (en tu WSL)

> **Importante — dónde clonar**: clona dentro del sistema de archivos de WSL
> (p. ej. `~/proyectos/`), **no** en `/mnt/c/...` ni en una carpeta de OneDrive.
> SQLite sobre el puente Windows↔WSL sufre bloqueos espurios ("database is
> locked") y OneDrive puede bloquear/corromper `data/alaves.db` al
> sincronizarla a mitad de escritura. El código reintenta 30 s, pero la
> solución de verdad es el sistema de archivos nativo de WSL (además, todo el
> I/O va mucho más rápido).

```bash
cd ~ && mkdir -p proyectos && cd proyectos
git clone https://github.com/Garaimtzg/ML_LaLiga.git
cd ML_LaLiga
uv sync                          # crea .venv e instala dependencias (usa uv.lock)
uv run pytest -q                 # verifica que todo pasa (48 tests, sin red)

# Población de la base de datos histórica (necesita internet; ~5 min la 1ª vez)
uv run alaves ingest --historical
uv run alaves validate           # certifica la BD: conteos, coberturas, consistencia
uv run alaves status             # resumen de filas por tabla y temporada
```

> **Por qué la ingesta se ejecuta en tu máquina**: el entorno remoto de
> desarrollo (Claude Code web) no tiene acceso de red a las fuentes de datos.
> El pipeline está completo y testeado con fixtures congelados; la descarga
> real se hace en local, donde sí hay red (ADR-007). Las descargas quedan
> cacheadas en `data/raw/`, así que re-ejecutar es gratis. Si una fuente
> difiere de lo esperado (p. ej. un alias de ClubElo), el comando falla con un
> mensaje que indica el arreglo exacto — normalmente una línea en
> `config/teams.toml`.

## Comandos del CLI

| Comando | Función | Fase |
|---------|---------|------|
| `uv run alaves ingest --historical` | ETL histórico completo (con cache; `--force` re-descarga) | ✅ F1 |
| `uv run alaves validate` | Chequeos de integridad de la BD (falla con exit code ≠ 0) | ✅ F1 |
| `uv run alaves status` | Filas por tabla y partidos por temporada | ✅ F1 |
| `uv run alaves ingest --matchday N` | Ingesta post-jornada | F7 |
| `uv run alaves train` | Entrenar Dixon-Coles + LightGBM + calibración | F3 |
| `uv run alaves predict --next` | Predicciones de la próxima jornada | F3/F7 |
| `uv run alaves simulate --n 10000` | Monte Carlo de la clasificación | F4 |
| `uv run alaves backtest` | Backtesting walk-forward | F3 |
| `uv run alaves report --importance` | Informes SHAP / importancia de variables | F5 |

Los comandos de fases futuras existen como stubs que lo indican honestamente.

Desarrollo:

```bash
uv run pytest -q                       # tests
uv run ruff check src tests --fix      # lint
uv run ruff format src tests           # formato
uv run mypy src                        # tipos (modo básico)
```

## Datos: fuentes y qué aporta cada una (F1)

| Fuente | Aporta | Tablas |
|--------|--------|--------|
| [football-data.co.uk](https://www.football-data.co.uk/spainm.php) | Resultados, tiros/córners/faltas/tarjetas y **cuotas** (bet365, Pinnacle, máx./media de mercado; apertura y cierre) | `matches`, `match_stats`, `odds` |
| [FBref](https://fbref.com/en/comps/12/) | **xG** histórico + **jornada oficial** (Wk) | `match_stats.xg`, `matches.matchday` |
| [Understat](https://understat.com) | **xG de relleno** vía su API interna `getLeagueData` (donde FBref no lo aporta; fuente en vivo prevista para F7) | `match_stats.xg` |
| [ClubElo](http://clubelo.com) | Rating **Elo** histórico por club | `elo` |

Cobertura: temporadas **2018-19 → 2025-26** (≈ 3.040 partidos). Las
estadísticas técnico-tácticas detalladas de FBref, Transfermarkt (valor de
plantillas) y API-Football (calendario 2026-27, lesiones) se incorporan en
F2/F7 (ADR-003). Historia accidentada de las fuentes de xG — Understat
rediseñó su web (ADR-008), FBref bloquea bots y quitó el xG del calendario en
2026 (ADR-009/010/011) — resuelta con una cascada: FBref directo → snapshots
de la Wayback Machine elegidos vía API CDX → relleno con Understat.

Garantías del pipeline (CLAUDE.md §6):

- **Validación de esquema** con pydantic en cada parser; formato inesperado →
  error ruidoso, nunca inserción silenciosa.
- **Consistencia entre fuentes**: el marcador de FBref se cruza con el de
  football-data antes de insertar el xG; discrepancia → aborta.
- **Procedencia**: cada fila lleva `source` y `fetched_at`.
- **Cache y rate limit**: nada se descarga dos veces; peticiones espaciadas
  por fuente (ADR-004).
- **Idempotencia**: re-ejecutar la ingesta no duplica filas (upserts sobre
  claves naturales deterministas, ADR-002).

## Base de datos

SQLite en `data/alaves.db` (gitignored). Tablas principales (esquema completo
en [`src/alaves_predictor/etl/db.py`](src/alaves_predictor/etl/db.py)):

- `teams` — id canónico, nombre y alias por fuente (`config/teams.toml`, ADR-005)
- `matches` — partidos con temporada, jornada oficial de FBref (ADR-006/008), goles y estado
- `match_stats` — estadísticas por (partido, equipo); F1 puebla básicas + xG,
  el resto de columnas (pases, presión, portería...) esperan a FBref en F2
- `odds` — cuotas 1X2 de apertura y cierre por casa
- `elo` — Elo de ClubElo por club y fecha (el Elo interno se calcula en F2)
- `features`, `predictions`, `model_registry` — creadas ya, se pueblan en F2-F3

Identificadores legibles: `team_id = "alaves"`,
`match_id = "2018-19_alaves_barcelona"`.

## Estructura del repositorio

```
├── CLAUDE.md / SPEC.md / BRIEF.md    # forma de trabajo, especificación, visión
├── pyproject.toml                    # paquete + dependencias (gestión con uv)
├── config/
│   ├── settings.toml                 # temporadas, fuentes, rutas, liga
│   └── teams.toml                    # alias de equipos por fuente
├── data/                             # BD y descargas crudas (gitignored)
├── docs/
│   ├── decisions/                    # ADRs (una decisión por archivo)
│   └── reports/                      # informes de backtesting/importancia (F3+)
├── src/alaves_predictor/
│   ├── config.py                     # carga tipada de la configuración
│   ├── cli.py                        # CLI typer (`alaves ...`)
│   └── etl/
│       ├── db.py                     # esquema SQLite + upserts
│       ├── http_cache.py             # descargas con cache y rate limit
│       ├── teams.py                  # resolución de nombres entre fuentes
│       ├── ingest.py                 # orquestador de la ingesta histórica
│       ├── validate.py               # chequeos de integridad de la BD
│       └── sources/                  # un adaptador por fuente
│           ├── football_data.py
│           ├── fbref.py
│           ├── understat.py          # xG de relleno vía API interna (ADR-011)
│           └── clubelo.py
└── tests/                            # 48 tests; fixtures congelados en tests/fixtures/
```

## Decisiones tomadas (ADRs)

| ADR | Decisión |
|-----|----------|
| [001](docs/decisions/001-stack-y-layout-del-proyecto.md) | Stack, layout `src/` y dependencias por fase |
| [002](docs/decisions/002-esquema-sqlite-e-identificadores.md) | SQLite, esquema completo desde F1, ids legibles, fusión de fuentes |
| [003](docs/decisions/003-fuentes-de-datos-fase-1.md) | Fuentes de F1 (football-data + Understat + ClubElo); FBref/Transfermarkt/API-Football diferidas |
| [004](docs/decisions/004-cache-local-y-rate-limiting.md) | Cache manual en `data/raw/` + rate limit por host |
| [005](docs/decisions/005-mapeo-de-nombres-de-equipos.md) | Alias explícitos en `config/teams.toml` con fallo ruidoso |
| [006](docs/decisions/006-jornada-aproximada.md) | Jornada aproximada por conteo de partidos jugados |
| [007](docs/decisions/007-ingesta-en-local-por-restriccion-de-red.md) | La ingesta real se ejecuta en local (WSL); tests con fixtures sin red |
| [008](docs/decisions/008-xg-de-fbref-en-vez-de-understat.md) | xG desde FBref (+ jornada oficial); Understat en pausa tras su rediseño de dic-2025 |
| [009](docs/decisions/009-transporte-tls-curl-cffi-para-fbref.md) | curl_cffi (huella TLS de Chrome) solo para FBref, cuyo Cloudflare rechaza clientes Python |
| [010](docs/decisions/010-fallback-wayback-machine-para-fbref.md) | Cascada de descarga de FBref: cache → directo → Wayback Machine → snapshot manual |
| [011](docs/decisions/011-understat-via-api-interna-como-relleno-de-xg.md) | Understat vuelve vía su endpoint interno getLeagueData como relleno de xG (y fuente en vivo para F7) |

## Principios de ML del proyecto (resumen de CLAUDE.md §5)

- **Sin data leakage**: toda feature se calcula con corte temporal `as_of_date`.
- **Validación temporal** (walk-forward), nunca splits aleatorios.
- **Baselines primero**: frecuencias históricas, Elo simple y cuotas de cierre.
- **Probabilidades calibradas** y verificadas con reliability diagrams.
- **Predicciones persistidas** antes de conocer el resultado (auditables).
- **Honestidad estadística**: el objetivo es batir baselines y acercarse a las
  cuotas de mercado, no "acertar todo".

## Próximos pasos (F2)

1. Elo interno recalculable (factor K ajustable) para comparar con ClubElo.
2. Features de forma (ventanas móviles 5/10), separadas local/visitante.
3. Adaptador FBref → completar `match_stats`; Transfermarkt → `squad_values`.
4. Baselines: frecuencias 1X2, Elo logístico, probabilidades implícitas de
   cuotas de cierre; métricas log-loss / Brier / RPS.
