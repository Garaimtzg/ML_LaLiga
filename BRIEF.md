# Brief — Predictor de Resultados del Deportivo Alavés (LaLiga 2026-27)

## 1. Qué es este proyecto

Sistema de machine learning en Python que predice los resultados de los partidos del Deportivo Alavés durante la temporada 2026-27 de LaLiga, y que a partir de esas predicciones (y las del resto de partidos de la jornada) proyecta la clasificación final de la liga.

El sistema es **incremental**: tras cada jornada se ingieren los datos reales de los partidos jugados, se actualizan las features (forma, Elo, xG acumulado, etc.) y se reentrenan/recalibran los modelos para predecir las jornadas siguientes con la información más reciente.

## 2. Objetivos

| # | Objetivo | Criterio de éxito |
|---|----------|-------------------|
| O1 | Predecir el resultado de cada partido del Alavés (1X2) | Log-loss mejor que el baseline de cuotas implícitas; accuracy > baseline "siempre gana el favorito" |
| O2 | Mostrar por cada predicción: resultado predicho, P(victoria local), P(empate), P(victoria visitante) y marcador más probable | Salida clara en CLI y dashboard |
| O3 | Proyectar la clasificación de LaLiga vía simulación Monte Carlo | Tabla con posiciones esperadas y probabilidades (título, Europa, descenso) |
| O4 | Análisis de importancia de variables | Ranking SHAP + informe interpretativo de qué variables pesan más |
| O5 | Dashboard interactivo | Streamlit con predicciones, clasificación proyectada, evolución del modelo y explicabilidad |
| O6 | Trazabilidad total | Cada decisión documentada en `docs/decisions/` (ADRs); el usuario nunca pierde el control del proyecto |

## 3. Alcance

**Dentro del alcance**
- Predicción 1X2 con probabilidades calibradas para todos los partidos de LaLiga (necesario para simular la clasificación), con foco especial en el Alavés.
- Predicción de goles esperados por equipo (modelo Poisson/Dixon-Coles) para derivar el marcador más probable.
- Pipeline ETL reproducible: descarga, limpieza, validación y almacenamiento de datos históricos y de la temporada en curso.
- Reentrenamiento/actualización automática tras cada jornada.
- Simulación Monte Carlo de la temporada restante (≥10.000 simulaciones) para la clasificación.
- Análisis de explicabilidad (SHAP, feature importance, análisis por variable).
- Dashboard en Streamlit.
- Backtesting sobre temporadas anteriores para validar el modelo antes de la 2026-27.

**Fuera del alcance (v1)**
- Apuestas o estrategias de bankroll.
- Predicción de estadísticas de jugadores individuales (goleadores, tarjetas...).
- Datos de tracking/eventos en vivo (posicionales por segundo).
- Otras ligas o competiciones (Copa del Rey solo como feature de fatiga/calendario, no como objetivo de predicción).

## 4. Usuarios y contexto

- **Usuario único**: Garai, desarrollando con Claude Code. Perfil técnico (Industria 4.0, Python, data science), quiere entender cada pieza del sistema, no una caja negra.
- El proyecto también sirve como pieza de **portfolio** con arquitectura de producción (paquete `src/`, tests, documentación), coherente con proyectos anteriores.

## 5. Entregables

1. Paquete Python `alaves_predictor/` bajo `src/` con pipeline ETL, feature engineering, modelos, simulador y CLI.
2. Dashboard Streamlit (`app/dashboard.py`).
3. Base de datos local (SQLite o Parquet) con datos históricos y de temporada en curso.
4. Informe de backtesting y de importancia de variables.
5. Documentación: `README.md`, `SPEC.md`, `CLAUDE.md`, ADRs en `docs/decisions/`.

## 6. Restricciones y supuestos

- **Lenguaje**: Python 3.11+.
- **Datos**: solo fuentes gratuitas o con free tier (football-data.co.uk, FBref, Understat, API-Football free tier, ClubElo, Transfermarkt para valores de mercado). Ver SPEC §3.
- **Cadencia**: LaLiga tiene jornadas aproximadamente semanales (agosto 2026 – mayo 2027); el pipeline de actualización se ejecuta manualmente o por cron tras cada jornada.
- **Honestidad estadística**: el fútbol tiene mucha varianza. El objetivo realista es batir baselines simples y acercarse a las cuotas de mercado, no "acertar todo". Las métricas de evaluación (log-loss, Brier, RPS) reflejan calidad probabilística, no solo aciertos.
- **Sin data leakage**: cada predicción usa exclusivamente información disponible antes del partido.

## 7. Fases del proyecto

| Fase | Contenido | Resultado |
|------|-----------|-----------|
| F1 | Setup del repo, entorno, ETL de datos históricos (≥ temporadas 2018-19 → 2025-26) | Base de datos poblada y validada |
| F2 | Feature engineering + baselines (Elo simple, cuotas implícitas) | Features versionadas + métricas baseline |
| F3 | Modelos (Poisson/Dixon-Coles + Gradient Boosting 1X2), calibración, backtesting | Modelo validado sobre temporadas pasadas |
| F4 | Simulador Monte Carlo de la clasificación | Tabla proyectada con probabilidades |
| F5 | Explicabilidad (SHAP) y análisis de variables | Informe de importancia de variables |
| F6 | Dashboard Streamlit | App funcional |
| F7 | Modo temporada: ingesta post-jornada, reentrenamiento incremental, predicción de la siguiente jornada | Ciclo semanal operativo desde agosto 2026 |

## 8. Riesgos principales

- **Disponibilidad de datos de la 2026-27**: los scrapers pueden romperse si cambian las webs → mitigación: capa de adaptadores por fuente + tests de esquema.
- **Pocos datos del Alavés en escenarios nuevos** (fichajes, cambio de entrenador) → el modelo se entrena con toda LaLiga, no solo con el Alavés, y usa features que capturan estado actual (Elo, forma, valor de plantilla).
- **Overfitting con features abundantes** → validación temporal estricta (walk-forward), nunca validación aleatoria.
