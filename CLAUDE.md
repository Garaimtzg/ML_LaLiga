# CLAUDE.md — Instrucciones para Claude Code

Este archivo define cómo debe trabajar Claude Code en este proyecto. Léelo íntegro antes de tocar código. La especificación técnica completa está en `SPEC.md` y la visión general en `brief.md`.

## 1. Principio rector

**El usuario debe entender y controlar todo el proyecto en todo momento.** Esto implica:

1. **Cada decisión no trivial se documenta** como ADR en `docs/decisions/NNN-titulo.md` (formato: contexto → opciones consideradas → decisión → consecuencias). Ejemplos de decisión no trivial: elección de modelo, fuente de datos, esquema de BD, estrategia de validación, librería nueva.
2. **Antes de implementar un módulo nuevo**, explica en el chat qué vas a hacer, por qué, y qué alternativas descartas. Si hay varias opciones razonables, pregunta.
3. **Nada de magia**: código legible por encima de código ingenioso. Comentarios donde la lógica no sea obvia (fórmulas de Elo, Dixon-Coles, ajustes de calibración...).
4. **Commits pequeños y descriptivos** (Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`).

## 2. Stack y convenciones

- **Python 3.11+**, gestión de entorno con `uv` (o `venv` + `pip-tools` si `uv` no está disponible).
- **Estructura de paquete bajo `src/`** (layout de producción, no notebooks sueltos). Los notebooks solo para exploración en `notebooks/` y nunca contienen lógica que no exista también en `src/`.
- Librerías principales: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `shap`, `streamlit`, `plotly`, `requests`/`httpx`, `beautifulsoup4`, `pydantic` (validación de esquemas), `typer` (CLI), `pytest`.
- **Formato y lint**: `ruff` (format + lint). Type hints en todas las funciones públicas; `mypy` en modo básico.
- **Datos**: SQLite (`data/alaves.db`) como almacén principal + Parquet para snapshots de features. Nunca commitear datos crudos pesados; `data/` está en `.gitignore` salvo pequeños fixtures de test.
- **Configuración** en `config/settings.toml` + variables de entorno para API keys (`.env`, nunca commiteado).
- **Semillas fijas** (`random_state=42`) en todo lo estocástico para reproducibilidad; las simulaciones Monte Carlo aceptan semilla por parámetro.

## 3. Estructura del repositorio

```
alaves-predictor/
├── CLAUDE.md
├── SPEC.md
├── brief.md
├── README.md
├── pyproject.toml
├── config/settings.toml
├── data/                     # gitignored (salvo fixtures)
├── docs/
│   ├── decisions/            # ADRs numerados
│   └── reports/              # backtesting, importancia de variables
├── notebooks/                # solo exploración
├── src/alaves_predictor/
│   ├── etl/                  # adaptadores por fuente + carga a BD
│   ├── features/             # cálculo de features (Elo, forma, xG...)
│   ├── models/               # dixon_coles.py, gbm_classifier.py, calibration.py
│   ├── simulation/           # monte_carlo.py (clasificación)
│   ├── evaluation/           # métricas, backtesting walk-forward
│   ├── explain/              # SHAP y análisis de variables
│   └── cli.py                # comandos typer
├── app/dashboard.py          # Streamlit
└── tests/
```

## 4. Comandos habituales

```bash
uv sync                                  # instalar dependencias
uv run pytest -q                         # tests
uv run ruff check src tests --fix        # lint
uv run alaves ingest --historical        # ETL histórico (F1)
uv run alaves ingest --matchday          # ingesta post-jornada (F7)
uv run alaves train                      # entrenar/reentrenar modelos
uv run alaves predict --next             # predecir próxima jornada
uv run alaves simulate --n 10000         # simular clasificación
uv run alaves report --importance        # informe SHAP
uv run streamlit run app/dashboard.py    # dashboard
```

(Implementa estos comandos en `cli.py`; si alguno aún no existe, créalo antes de usarlo en documentación.)

## 5. Reglas de ML innegociables

1. **Prohibido el data leakage.** Toda feature de un partido se calcula solo con información anterior a su fecha (`as_of_date`). Los tests deben verificar esto explícitamente.
2. **Validación temporal siempre** (walk-forward por jornadas/temporadas). Nunca `train_test_split` aleatorio.
3. **Baselines primero.** Ningún modelo se acepta sin comparación contra: (a) frecuencias históricas 1X2, (b) Elo simple, (c) probabilidades implícitas de las cuotas de cierre. Métricas: log-loss, Brier score, RPS, accuracy.
4. **Probabilidades calibradas.** Tras entrenar, aplicar calibración (isotónica o Platt) y verificar con reliability diagrams.
5. **Las predicciones se persisten** en BD antes de conocerse el resultado real (tabla `predictions` con timestamp, versión del modelo y hash de features), para poder auditar el rendimiento real del sistema durante la temporada.
6. **Versionado de modelos**: cada entrenamiento guarda artefacto + métricas + config en `models/registry/` con identificador de fecha.

## 6. Reglas de datos

- Cada fuente externa tiene su adaptador en `src/alaves_predictor/etl/sources/` con: función de descarga, parser, validación de esquema con `pydantic`, y test con fixture HTML/CSV congelado.
- Respetar `robots.txt` y rate limits; añadir `time.sleep` y cache local de respuestas (no re-descargar lo ya guardado).
- Si una fuente falla o cambia de formato, el pipeline debe fallar ruidosamente con un mensaje claro, nunca insertar datos corruptos en silencio.
- Registrar en la BD la procedencia (`source`, `fetched_at`) de cada fila.

## 7. Flujo de trabajo por sesión

1. Lee `SPEC.md` y el estado actual (`docs/decisions/`, últimos commits) antes de empezar.
2. Propón un plan corto para la tarea de la sesión y espera confirmación si hay decisiones abiertas.
3. Implementa con tests (los módulos de features y de métricas requieren tests obligatoriamente).
4. Ejecuta `pytest` y `ruff` antes de dar la tarea por cerrada.
5. Actualiza documentación afectada (README, ADRs, SPEC si algo cambia de la especificación — cualquier desviación de SPEC.md requiere ADR).
6. Resume al final: qué se hizo, qué queda pendiente, qué decisiones se tomaron.

## 8. Qué NO hacer

- No introducir dependencias nuevas sin justificarlas (y sin ADR si son estructurales).
- No entrenar solo con partidos del Alavés (ver SPEC §5: se entrena con toda LaLiga).
- No prometer precisión irreal en documentación ni en el dashboard: mostrar siempre probabilidades e intervalos, no certezas.
- No mezclar exploración de notebooks con código de producción.
- No hardcodear temporadas, equipos ni fechas: todo parametrizado en `config/`.
