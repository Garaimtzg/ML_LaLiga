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
| **F1** | Setup del repo, entorno, ETL de datos históricos (2018-19 → 2025-26) | ✅ **Completada** — BD poblada y validada: 3.040 partidos, xG completo, 11.209 líneas de cuotas, 25.306 registros Elo de 30 clubes |
| **F2** | Feature engineering + baselines (Elo simple, cuotas implícitas) | ✅ **Completada** — feature set v1 (~50 features, corte temporal estricto + test anti-leakage) y 3 baselines walk-forward |
| **F3** | Modelos (Dixon-Coles + LightGBM 1X2), calibración, backtesting | ✅ **Completada** — Dixon-Coles propio (MLE + ponderación temporal), LightGBM con/sin cuotas, calibración isotónica, ensemble ponderado, registro de modelos y backtest jornada a jornada |
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

- **libgomp** (runtime de OpenMP, lo usa LightGBM desde F3). En un Ubuntu de
  WSL recién instalado no viene de serie; si `alaves train` falla con
  `OSError: libgomp.so.1: cannot open shared object file`:

```bash
sudo apt-get update && sudo apt-get install -y libgomp1
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
uv run pytest -q                 # verifica que todo pasa (103 tests, sin red)

# Población de la base de datos histórica (necesita internet; ~5 min la 1ª vez)
uv run alaves ingest --historical
uv run alaves validate           # certifica la BD: conteos, coberturas, consistencia
uv run alaves status             # resumen de filas por tabla y temporada

# Modelado (F3): entrenar y evaluar contra los baselines
uv run alaves train              # entrena y registra la versión del modelo
uv run alaves backtest --seasons 3   # backtest jornada a jornada (~unos minutos)
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
| `uv run alaves features` | Construye el feature set v1 (tabla `features` + Parquet) | ✅ F2 |
| `uv run alaves baselines` | Evalúa los 3 baselines walk-forward e informa en `docs/reports/` | ✅ F2 |
| `uv run alaves train [--no-odds]` | Entrena DC + LightGBM + calibración + ensemble y registra la versión | ✅ F3 |
| `uv run alaves backtest --seasons 3` | Backtest jornada a jornada vs baselines + informe en `docs/reports/` | ✅ F3 |
| `uv run alaves predict --next` / `--matchday N` | Predice partidos programados y persiste las predicciones | ✅ F3* |
| `uv run alaves ingest --matchday N` | Ingesta post-jornada | F7 |
| `uv run alaves simulate --n 10000` | Monte Carlo de la clasificación | F4 |
| `uv run alaves report --importance` | Informes SHAP / importancia de variables | F5 |

\* `predict` está completo, pero necesita partidos con estado `scheduled` en la
BD; el calendario de la 2026-27 se ingiere en la F7 (API-Football). Hasta
entonces avisa honestamente de que no hay nada que predecir.

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

## Modelos (F3)

Tres piezas que se combinan (SPEC §6, ADRs 015-018):

1. **Dixon-Coles** (`models/dixon_coles.py`): Poisson bivariante con parámetros
   de ataque/defensa por equipo, ventaja de campo y corrección ρ de marcadores
   bajos. Implementación propia por máxima verosimilitud (`scipy.optimize`),
   con ponderación temporal exponencial (un partido de hace un año pesa ~0.5).
   Aporta goles esperados, matriz de marcadores y el "marcador más probable".
2. **LightGBM multiclase** (`models/gbm_classifier.py`) sobre el feature set
   v1, en dos variantes: **con cuotas** (techo de rendimiento) y **sin cuotas**
   (la que se interpretará con SHAP en F5).
3. **Calibración isotónica + ensemble apilado** (`models/calibration.py`,
   `models/ensemble.py`, `models/linear.py`, ADR-019/020): las probabilidades
   del LightGBM y del Dixon-Coles se calibran por clase sobre predicciones
   walk-forward (nunca sobre el entrenamiento) y se apilan con un tercer
   componente — el mercado de apertura (con cuotas) o una logística Elo+forma
   (sin cuotas) — con pesos elegidos por log-loss en validación. El ξ de la
   ponderación temporal del Dixon-Coles también se elige por validación en una
   rejilla. La calibración se desactiva sola con pocos datos (evita
   sobreajuste).

Cada `alaves train` guarda el artefacto en `models/registry/<versión>/`
(gitignored) y una fila auditable en la tabla `model_registry`. **Regla
anti-sorpresa** (SPEC §6.4): si el log-loss de validación empeora >10 %
respecto a la última versión promocionada, la nueva se registra pero no se
promociona y `predict` la ignora.

El **backtest** (`evaluation/backtest.py`) re-simula el ciclo real: para cada
temporada de test reentrena los modelos antes de cada jornada con todo lo
jugado hasta la víspera, y compara contra los tres baselines de F2. El informe
queda en `docs/reports/backtest_<fecha>.md` con el veredicto de los criterios
de aceptación (SPEC §12.1): ensemble sin cuotas < baseline Elo (~0.971 de
log-loss) y ensemble con cuotas ≤ cuotas de cierre + 0.01 (~0.964).

## Estructura del repositorio

```
├── CLAUDE.md / SPEC.md / BRIEF.md    # forma de trabajo, especificación, visión
├── pyproject.toml                    # paquete + dependencias (gestión con uv)
├── config/
│   ├── settings.toml                 # temporadas, fuentes, rutas, liga
│   └── teams.toml                    # alias de equipos por fuente
├── data/                             # BD y descargas crudas (gitignored)
├── models/registry/                  # artefactos entrenados (gitignored)
├── docs/
│   ├── decisions/                    # ADRs (una decisión por archivo)
│   └── reports/                      # informes de baselines/backtesting
├── src/alaves_predictor/
│   ├── features/                     # elo.py (Elo interno), form.py, build.py
│   ├── models/                       # dixon_coles.py, gbm_classifier.py,
│   │                                 # calibration.py, ensemble.py, linear.py, train.py
│   ├── evaluation/                   # metrics.py, baselines.py, backtest.py
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
└── tests/                            # 103 tests; fixtures congelados en tests/fixtures/
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
| [012](docs/decisions/012-feature-set-v1-y-dependencias-f2.md) | Feature set v1 (~50 features, as_of estricto); bloque técnico-táctico aplazado; deps de F2 |
| [013](docs/decisions/013-elo-interno.md) | Elo interno clásico: K=20, ventaja 60, inicio 1500 (parámetros en config) |
| [014](docs/decisions/014-baselines-y-evaluacion-walk-forward.md) | Baselines (frecuencias, Elo logístico, cuotas de cierre) y protocolo walk-forward |
| [015](docs/decisions/015-dixon-coles.md) | Dixon-Coles propio: parametrización, ξ=0.0019/día, ρ acotado, proxy de colista para ascendidos |
| [016](docs/decisions/016-lightgbm-variantes.md) | LightGBM: hiperparámetros v1 documentados, variantes con/sin cuotas, NaN nativos |
| [017](docs/decisions/017-calibracion-y-ensemble.md) | Calibración isotónica sobre folds temporales (suelo 1 %) + ensemble ponderado por log-loss |
| [018](docs/decisions/018-backtest-y-registro.md) | Backtest jornada a jornada, registro de modelos y regla anti-sorpresa del 10 % |
| [019](docs/decisions/019-ensemble-apilado-y-xi-por-validacion.md) | Ensemble apilado de 3 componentes (DC + GBM + mercado/Elo) y ξ elegido por validación |
| [020](docs/decisions/020-componente-lineal-y-calibracion-dc.md) | Componente lineal Elo+forma, calibración del Dixon-Coles y guarda de calibración con pocos datos |

## Principios de ML del proyecto (resumen de CLAUDE.md §5)

- **Sin data leakage**: toda feature se calcula con corte temporal `as_of_date`.
- **Validación temporal** (walk-forward), nunca splits aleatorios.
- **Baselines primero**: frecuencias históricas, Elo simple y cuotas de cierre.
- **Probabilidades calibradas** y verificadas con reliability diagrams.
- **Predicciones persistidas** antes de conocer el resultado (auditables).
- **Honestidad estadística**: el objetivo es batir baselines y acercarse a las
  cuotas de mercado, no "acertar todo".

## Próximos pasos

1. **En tu WSL**: `git pull`, `uv sync`, `uv run alaves train` y
   `uv run alaves backtest --seasons 3` sobre la BD real — el informe dirá si
   los criterios de SPEC §12.1 se cumplen con datos de verdad.
2. **F4**: simulador Monte Carlo de la clasificación (`alaves simulate`).
3. **F5**: explicabilidad — SHAP sobre la variante sin cuotas, ablation study.
