# SPEC.md — Especificación técnica: Predictor Deportivo Alavés, LaLiga 2026-27

> Documento normativo. Cualquier desviación durante la implementación requiere un ADR en `docs/decisions/`.

---

## 1. Resumen del sistema

Sistema de predicción probabilística de partidos de LaLiga con foco en el Deportivo Alavés. Componentes:

1. **ETL**: ingesta de datos históricos (temporadas 2018-19 → 2025-26) y de la temporada 2026-27 jornada a jornada.
2. **Feature store**: cálculo versionado de features por partido, con corte temporal estricto (`as_of_date`).
3. **Modelos**: (a) Dixon-Coles/Poisson bivariante para goles esperados y marcador, (b) LightGBM multiclase para probabilidades 1X2, (c) capa de calibración.
4. **Simulador**: Monte Carlo de la temporada restante → clasificación proyectada.
5. **Explicabilidad**: SHAP + análisis de importancia de variables.
6. **Dashboard**: Streamlit.
7. **CLI**: orquestación de todo lo anterior.

**Decisión clave (explicada)**: se predicen *todos* los partidos de LaLiga, no solo los del Alavés. Motivos: (1) la clasificación proyectada exige predecir la liga completa; (2) entrenar solo con ~38 partidos/temporada del Alavés es estadísticamente inviable; entrenar con ~380 partidos/temporada × 8 temporadas ≈ 3.000 partidos da un modelo robusto que luego se aplica al Alavés.

---

## 2. Objetivos de predicción (targets)

| Target | Tipo | Modelo responsable |
|--------|------|--------------------|
| Resultado 1X2 | Clasificación multiclase (H/D/A) | LightGBM + calibración |
| Goles local / goles visitante | Conteo | Dixon-Coles (Poisson bivariante con corrección de marcadores bajos) |
| Marcador más probable | Derivado | Matriz de probabilidades del Dixon-Coles |
| Clasificación final | Distribución de posiciones por equipo | Monte Carlo sobre las probabilidades 1X2 |

**Salida obligatoria por partido** (CLI y dashboard):

```
Alavés vs Real Sociedad — Jornada 12 — 2026-11-08
Resultado predicho: 1 (victoria local)
P(victoria local):     46.3 %
P(empate):             27.9 %
P(victoria visitante): 25.8 %
Marcador más probable: 1-0 (p = 11.2 %)
```

**Decisión (explicada)**: aunque el usuario pidió P(local) y P(visitante), se incluye siempre P(empate). En fútbol el empate es ~25 % de los resultados; omitirlo haría que las probabilidades no sumaran 1 y sesgaría la simulación de la clasificación.

---

## 3. Datos: fuentes, contenido y método de obtención

### 3.1 Fuentes

| Fuente | Datos | Método | Cadencia | Coste |
|--------|-------|--------|----------|-------|
| **football-data.co.uk** | Resultados históricos, estadísticas básicas (tiros, córners, tarjetas), **cuotas de apuestas** de LaLiga (SP1.csv) | Descarga CSV directa | Histórico + actualización semanal | Gratis |
| **FBref** | Estadísticas avanzadas por equipo y partido: xG, xGA, posesión, pases progresivos, presión, etc. | Scraping HTML (respetando rate limit ~1 req/6s) | Post-jornada | Gratis |
| **Understat** | xG por partido y por tiro (alternativa/contraste de FBref) | Scraping del JSON embebido | Post-jornada | Gratis |
| **ClubElo (clubelo.com)** | Rating Elo histórico y actual de cada club | API CSV pública | Post-jornada | Gratis |
| **API-Football (free tier)** | Calendario oficial 2026-27, alineaciones, lesiones, árbitro | REST API (100 req/día en free tier) | Pre y post-jornada | Gratis (limitado) |
| **Transfermarkt** | Valor de mercado de plantillas, fichajes, edad media | Scraping (moderado, con cache) | Mensual / ventanas de mercado | Gratis |

**Decisiones (explicadas)**:
- *football-data.co.uk como columna vertebral*: es la fuente más estable (CSV con formato invariable desde hace años) e incluye cuotas de cierre, que son la mejor estimación pública de probabilidad de un partido y sirven como **baseline exigente** y como feature.
- *FBref + Understat en paralelo*: el xG es la métrica moderna con más señal predictiva; tener dos fuentes permite validación cruzada y resiliencia si un scraper se rompe.
- *ClubElo en lugar de calcular Elo propio desde cero*: da series históricas largas ya calibradas; además el sistema mantiene un **Elo propio interno** (recalculable) para no depender de la fuente y poder ajustar el factor K. Ambos se comparan en el análisis de variables.
- *Transfermarkt para "estatus" de equipos*: el valor de mercado de la plantilla es el mejor proxy público de la diferencia estructural de calidad entre equipos (esto captura el "ranking y estatus de cada equipo" pedido, junto con el Elo).

### 3.2 Esquema de base de datos (SQLite)

Tablas mínimas:

- `teams(team_id, name, aliases_json)` — mapeo de nombres entre fuentes (problema real: "Alavés" vs "Alaves" vs "Deportivo Alavés").
- `matches(match_id, season, matchday, date, home_id, away_id, home_goals, away_goals, status, source, fetched_at)`
- `match_stats(match_id, team_id, ...)` — estadísticas completas por equipo y partido, organizadas en bloques (todos disponibles en FBref):
  - *Tiro*: `xg`, `npxg` (xG sin penaltis), `shots`, `shots_on_target`, `shot_distance_avg`, `goals_per_shot`
  - *Pase*: `passes_completed`, `passes_attempted`, `pass_accuracy_pct`, `progressive_passes`, `passes_final_third`, `passes_penalty_area`, `key_passes`, `crosses`, `xa` (expected assists)
  - *Posesión y conducción*: `possession_pct`, `touches`, `touches_att_third`, `progressive_carries`, `dribbles_completed`, `dispossessed`
  - *Defensa*: `tackles`, `tackles_won`, `interceptions`, `blocks`, `clearances`, `errors_leading_to_shot`, `ppda` (pases del rival por acción defensiva — proxy de intensidad de presión, de Understat)
  - *Portería*: `psxg` (post-shot xG en contra — mide la calidad de las paradas), `saves`, `save_pct`
  - *Balón parado*: `corners`, `set_piece_shots`, `set_piece_goals`
  - *Disciplina*: `fouls`, `cards_yellow`, `cards_red`, `penalties_conceded`
  - *Otros*: `aerials_won_pct`, `offsides`
- `odds(match_id, bookmaker, open_h, open_d, open_a, close_h, close_d, close_a)`
- `elo(team_id, date, elo_clubelo, elo_internal)`
- `squad_values(team_id, date, market_value_eur, mean_age, n_players)`
- `injuries(team_id, date, player, expected_return)` — opcional según límites de API-Football.
- `features(match_id, feature_set_version, as_of_date, payload_json)` — snapshot exacto de las features usadas en cada predicción.
- `predictions(prediction_id, match_id, model_version, created_at, p_home, p_draw, p_away, pred_result, pred_score, expected_goals_h, expected_goals_a)`
- `model_registry(model_version, trained_at, train_window, metrics_json, config_json, artifact_path)`

**Decisión (explicada)**: SQLite y no PostgreSQL porque es un proyecto monousuario local; cero infraestructura, archivo único versionable por backup, y `pandas.read_sql` directo. Si algún día se despliega multiusuario, la capa de acceso (repositorio en `etl/db.py`) permite migrar.

### 3.3 Ciclo de ingesta durante la temporada (F7)

Tras cada jornada (`alaves ingest --matchday`):

1. Descargar resultados y estadísticas de la jornada de todas las fuentes.
2. Validar esquemas (pydantic) y consistencia entre fuentes (¿coinciden los marcadores?). Discrepancia → error ruidoso, no inserción silenciosa.
3. Insertar en BD con `source` y `fetched_at`.
4. Recalcular features dependientes del tiempo (Elo interno, formas, medias móviles).
5. Evaluar las predicciones que ya tienen resultado real → actualizar métricas de temporada en curso.
6. Reentrenar (ver §6.4) y generar predicciones de la siguiente jornada.
7. Ejecutar simulación de clasificación actualizada.

---

## 4. Feature engineering

Todas las features se calculan con corte `as_of_date` = día anterior al partido. Versión del conjunto de features registrada en `features.feature_set_version`.

### 4.1 Catálogo de features (v1)

**Fuerza estructural del equipo ("estatus/ranking")**
- `elo_home`, `elo_away`, `elo_diff` (ClubElo e interno)
- `market_value_home`, `market_value_away`, `log_value_ratio`
- `position_table_home/away`, `points_per_game_season`
- `promoted_flag` (equipo recién ascendido)

**Forma reciente (ventanas móviles de 5 y 10 partidos)**
- Puntos por partido, goles a favor/en contra
- `xg_for_ma5`, `xg_against_ma5` (xG medio reciente — más estable que los goles reales)
- Diferencia goles reales − xG (proxy de sobre/infra-rendimiento, tiende a revertir)
- Rachas (victorias/derrotas consecutivas)

**Local/visitante**
- Todas las anteriores separadas por condición de local/visitante
- Rendimiento histórico en ese estadio (para el Alavés: Mendizorroza)

**Contexto del partido**
- Días de descanso de cada equipo; partido europeo/copa entre medias
- Jornada (fase de temporada), mes
- Head-to-head reciente (últimos 5 enfrentamientos directos) — peso bajo, muestra pequeña
- Derbi (flag; p. ej. vs Athletic, Real Sociedad, Osasuna)
- Importancia del partido (distancia a descenso/Europa en puntos, jornadas restantes)

**Rendimiento técnico-táctico (medias móviles de 5 y 10 partidos, separadas local/visitante)**
Derivadas de la tabla `match_stats` ampliada. Se agregan como medias móviles, nunca como valores del propio partido (eso sería leakage: no conoces la precisión de pase de un partido antes de jugarlo):
- *Estilo con balón*: `pass_accuracy_ma5`, `progressive_passes_ma5`, `passes_final_third_ma5`, `possession_ma5`, `dribbles_completed_ma5`
- *Creación*: `key_passes_ma5`, `xa_ma5`, `npxg_per_shot_ma5` (calidad media de tiro)
- *Presión e intensidad*: `ppda_ma5` (presión propia), `ppda_allowed_ma5` (presión sufrida), `tackles_won_ma5`, `interceptions_ma5`
- *Solidez*: `psxg_minus_goals_conceded_ma5` (rendimiento del portero por encima/debajo de lo esperado), `errors_leading_to_shot_ma10`, `set_piece_goals_conceded_ma10`
- *Enfrentamiento de estilos* (interacciones): `possession_home_ma5 − possession_away_ma5`, `ppda_home vs pass_accuracy_away` (¿equipo presionante contra equipo que sufre bajo presión?)

**Mercado**
- Probabilidades implícitas de cuotas de apertura (normalizadas quitando el margen del bookmaker)

**Plantilla (si API-Football lo permite)**
- Nº de titulares habituales lesionados/sancionados
- Cambio de entrenador reciente (flag + partidos desde el cambio)

**Decisiones (explicadas)**:
- *Se incluyen las cuotas como feature y también se entrena una variante sin ellas.* Las cuotas contienen casi toda la información pública; incluirlas maximiza la precisión, pero enmascara qué variables "futbolísticas" importan. Por eso el análisis de importancia (§7) se hace sobre la **variante sin cuotas**, y la variante con cuotas se usa como techo de rendimiento.
- *xG por encima de goles reales en las features de forma*: los goles tienen enorme varianza; el xG es mejor predictor del rendimiento futuro (consenso en la literatura de football analytics).
- *Head-to-head con peso bajo*: intuitivamente atractivo pero con poca señal real (plantillas cambian); se incluye para que el análisis SHAP lo demuestre empíricamente.
- *Control de dimensionalidad (importante)*: con el bloque técnico-táctico el catálogo supera fácilmente las 150 features para ~3.000 filas de entrenamiento — riesgo real de overfitting. Estrategia obligatoria: (1) ingerir y almacenar **todo** en `match_stats` (los datos crudos siempre se guardan completos, cuesta lo mismo y permiten iterar); (2) pero el modelo v1 entrena con un subconjunto curado (~40-60 features); (3) el resto se incorpora por rondas de *forward selection* guiadas por log-loss en walk-forward: una feature entra solo si mejora la validación. Esto convierte "¿aporta la precisión de pase?" en una pregunta empírica que responde el propio pipeline, y el resultado se documenta en el informe de importancia de variables (§7).
- *Agregación por medias móviles y no valores del partido*: todas las estadísticas del partido (pases, posesión, tackles...) describen lo que *pasó*, no lo que *pasará*; como features solo son válidas agregadas sobre partidos anteriores.

---

## 5. Conjunto de entrenamiento

- **Partidos**: todas las temporadas de LaLiga desde 2018-19 hasta la última completada (2025-26), ≈ 3.000 partidos. Se descartan temporadas anteriores a 2018-19 porque no hay xG fiable y el fútbol cambia de dinámica (decisión revisable por ADR).
- La temporada COVID (2020-21 sin público) se marca con flag `no_crowd` para que el modelo pueda descontar el efecto de jugar sin afición.
- El Alavés estuvo en Segunda en 2022-23: esos partidos **no** se incluyen (liga distinta, nivel distinto), pero el ascenso queda capturado vía `promoted_flag` y Elo.

---

## 6. Modelado

### 6.1 Baselines (obligatorios antes de cualquier modelo)

1. **Frecuencias históricas**: P(H)=0.45, P(D)=0.25, P(A)=0.30 aprox. (calcular con datos reales).
2. **Elo simple**: probabilidad logística sobre `elo_diff` + ventaja de campo.
3. **Cuotas de cierre** (probabilidades implícitas normalizadas): el baseline a batir de verdad; si el modelo sin cuotas se acerca al log-loss de las cuotas, es un buen modelo.

### 6.2 Modelo A — Dixon-Coles (goles y marcador)

- Poisson bivariante con parámetros de ataque/defensa por equipo, ventaja de campo y corrección ρ para marcadores bajos (0-0, 1-0, 0-1, 1-1).
- **Ponderación temporal exponencial** (ξ ≈ 0.0019/día, ajustar por validación): los partidos recientes pesan más — esto implementa de forma natural el requisito de "tener en cuenta los partidos anteriores" con más peso a lo reciente.
- Salidas: λ_home, λ_away (goles esperados), matriz de probabilidad de marcadores → marcador más probable y P(1X2) derivadas.
- Implementación: optimización por máxima verosimilitud con `scipy.optimize` (código propio, documentado; no depender de librerías abandonadas).

**Por qué Dixon-Coles**: es el estándar académico para fútbol, interpretable (cada equipo tiene fuerza de ataque y defensa explícitas), da marcadores además de 1X2, y sus parámetros son en sí mismos un "ranking" de equipos consultable en el dashboard.

### 6.3 Modelo B — LightGBM multiclase (1X2)

- `LGBMClassifier(objective="multiclass", num_class=3)` sobre el catálogo de features §4.
- Búsqueda de hiperparámetros con `optuna` (opcional, v1 puede usar valores razonables documentados) validando **solo con esquema walk-forward**.
- **Calibración** posterior con regresión isotónica por clase (sobre folds de validación temporal), verificada con reliability diagrams.
- **Ensemble final**: media ponderada de probabilidades Dixon-Coles y LightGBM (peso ajustado por log-loss en validación). Decisión: el ensemble suele batir a cada modelo por separado y da robustez si uno degenera.

**Por qué LightGBM y no una red neuronal**: ~3.000 filas es territorio de gradient boosting, no de deep learning; LightGBM maneja nativamente no linealidades e interacciones, entrena en segundos (clave para reentrenar cada jornada) y se integra perfectamente con SHAP.

### 6.4 Reentrenamiento durante la temporada

- **Cadencia**: tras cada jornada.
- **Estrategia**: reentrenamiento completo (no incremental interno) con toda la historia + partidos nuevos, manteniendo la ponderación temporal. Con estos volúmenes tarda segundos, y evita la complejidad/fragilidad del online learning.
- Cada reentrenamiento genera nueva `model_version` en el registry con sus métricas; el dashboard muestra la evolución del rendimiento del modelo durante la temporada.
- **Regla anti-sorpresa**: si las métricas de la nueva versión empeoran >10 % en log-loss respecto a la anterior sobre el conjunto de validación, se alerta y no se promociona automáticamente.

### 6.5 Validación y backtesting

- **Walk-forward por temporada**: entrenar hasta la temporada T-1, predecir la T completa jornada a jornada (re-simulando el ciclo real de reentrenamiento). Repetir para las 3 últimas temporadas disponibles.
- Métricas: **log-loss** (principal), Brier score, **RPS** (Ranked Probability Score, estándar en fútbol por respetar el orden H>D>A), accuracy, y accuracy específica en partidos del Alavés.
- Informe en `docs/reports/backtest_<fecha>.md` con comparación contra los 3 baselines.

---

## 7. Análisis de importancia de variables

Requisito explícito del proyecto. Entregables:

1. **SHAP values globales** (beeswarm + bar plot) sobre la variante del modelo **sin cuotas**: ranking de qué variables más afectan a las predicciones.
2. **SHAP por partido** en el dashboard: para cada predicción del Alavés, desglose de qué factores empujan hacia victoria/empate/derrota (waterfall plot).
3. **Análisis de dependencia parcial** de las 8 variables top.
4. **Ablation study**: log-loss del modelo quitando bloques de features (sin xG, sin Elo, sin valores de mercado...) para cuantificar la aportación real de cada bloque.
5. Informe interpretativo en `docs/reports/feature_importance.md`, en lenguaje claro.

---

## 8. Simulación de la clasificación (Monte Carlo)

1. Para cada partido restante de la temporada, obtener P(H/D/A) del ensemble.
2. Simular la temporada completa N=10.000 veces muestreando cada resultado de su distribución; en cada simulación acumular puntos (para el desempate fino se usa diferencia de goles muestreada del Dixon-Coles; el head-to-head reglamentario de LaLiga se aproxima por diferencia de goles — limitación documentada).
3. Salidas por equipo: posición esperada, distribución de posiciones, P(título), P(Champions, top-4), P(Europa), P(descenso), puntos esperados.
4. Para el Alavés: gráfico específico de distribución de posición final y evolución de P(descenso)/P(Europa) jornada a jornada.

**Decisión (explicada)**: Monte Carlo en lugar de "sumar resultados más probables" porque este último sesga sistemáticamente (ignora la varianza y los empates); la simulación da distribuciones completas y probabilidades honestas.

---

## 9. Dashboard (Streamlit)

Páginas:

1. **Próxima jornada**: predicción del partido del Alavés (resultado, P(1)/P(X)/P(2), marcador más probable, goles esperados) + resto de la jornada en tabla.
2. **Clasificación proyectada**: tabla actual + posición esperada + P(descenso/Europa/título) por equipo, con heatmap de distribución de posiciones.
3. **Alavés en detalle**: evolución de Elo, xG a favor/en contra por jornada, forma, calendario restante con dificultad estimada.
4. **Explicabilidad**: SHAP global + waterfall del próximo partido + ablation.
5. **Rendimiento del modelo**: log-loss/Brier acumulado de la temporada vs baselines, reliability diagram, historial de predicciones vs resultados reales, registro de versiones de modelo.
6. **Registro de decisiones**: render de los ADRs (transparencia total del proyecto).

**Por qué Streamlit**: puro Python (sin frontend aparte), integración directa con pandas/plotly/shap, y despliegue trivial en local o Streamlit Community Cloud. Alternativas descartadas: Dash (más verboso), Grafana (orientado a series temporales de infra, ya lo domina el usuario pero encaja peor con SHAP y tablas interactivas de ML).

---

## 10. CLI (typer)

| Comando | Función |
|---------|---------|
| `alaves ingest --historical` | ETL completo de temporadas históricas |
| `alaves ingest --matchday [N]` | Ingesta post-jornada + validación + evaluación de predicciones pasadas |
| `alaves train [--no-odds]` | Entrena Dixon-Coles + LightGBM + calibración + ensemble; registra versión |
| `alaves predict --next` / `--matchday N` | Predicciones con salida formateada (§2) y persistencia en BD |
| `alaves simulate --n 10000 [--seed 42]` | Simulación Monte Carlo y actualización de la clasificación proyectada |
| `alaves backtest --seasons 3` | Backtesting walk-forward |
| `alaves report --importance` | Genera informes SHAP/ablation |

---

## 11. Testing

- `pytest` con cobertura mínima en: features (corte temporal correcto — test anti-leakage obligatorio), métricas, Elo interno, Dixon-Coles (verificar contra un caso resuelto a mano), simulador (las probabilidades de posición deben sumar 1 por equipo), y parsers de cada fuente (fixtures congelados).
- Test de integración: pipeline completo sobre un mini-dataset sintético de 2 jornadas.

## 12. Criterios de aceptación del sistema

1. Backtesting sobre 3 temporadas: log-loss del ensemble sin cuotas < log-loss del baseline Elo; con cuotas ≤ log-loss de cuotas + 0.01.
2. Reliability diagram sin desviaciones groseras (calibración visualmente razonable en los bins centrales).
3. Ciclo completo post-jornada (`ingest → train → predict → simulate`) ejecuta en < 10 min en un portátil.
4. Toda predicción mostrada es reproducible: `match_id + model_version + feature_set_version` recuperan exactamente las mismas probabilidades.
5. Existe al menos un ADR por cada decisión de §3, §6, §8 y §9 (pueden partir de las justificaciones de este documento).
