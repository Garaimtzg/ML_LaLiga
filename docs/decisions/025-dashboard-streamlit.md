# ADR-025 — Dashboard Streamlit y separación lógica/presentación

- **Fecha**: 2026-07-13 (Fase 6)
- **Estado**: aceptada

## Contexto

SPEC §9 pide un dashboard con seis páginas (próxima jornada, clasificación
proyectada, el Alavés en detalle, explicabilidad, rendimiento del modelo,
registro de decisiones) en Streamlit, con Plotly para los gráficos. CLAUDE.md
§2 prohíbe que haya lógica solo en la capa de presentación.

## Decisión

- **Streamlit + Plotly** (dependencias nuevas, ya previstas en el stack de
  CLAUDE.md §2). Se descartan Dash (más verboso) y Grafana (orientado a series
  de infra, encaja peor con SHAP y tablas de ML) — SPEC §9 ya lo justifica.
- **Separación estricta lógica/presentación**: toda la preparación de datos
  vive en `src/alaves_predictor/dashboard/data.py` como funciones puras y
  testeables (clasificación real y proyectada, heatmap de posiciones,
  predicciones de jornada, serie del equipo foco, registro de modelos,
  historial de predicciones, índice de ADRs). `app/dashboard.py` solo llama a
  esas funciones y las pinta. Así el dashboard no tiene lógica sin test.
- **Reutilización**: la proyección Monte Carlo se extrajo a
  `simulation/project.py` (`project_standings`), usada por igual por el comando
  `alaves simulate` y por la página del dashboard — sin duplicar el ensamblado.
- **Waterfall por partido** (SPEC §7.2, aplazado de F5): se añade
  `explain.importance.match_contributions`, que da el SHAP de un partido
  concreto por clase; la página de explicabilidad lo pinta como barras
  (positivo empuja hacia esa clase). Reusa el TreeSHAP nativo de ADR-024.
- **Carga cacheada** (`@st.cache_resource`): el modelo y el feature store se
  construyen una vez por sesión, no en cada interacción.
- **Robustez ante datos ausentes**: cada página avisa con claridad si falta el
  modelo, no hay partidos programados (temporada en curso antes de la F7) o no
  hay predicciones persistidas, en vez de romper. Para temporadas históricas
  ofrece el modo demo (proyectar desde una jornada), como `alaves simulate`.

## Consecuencias

- El dashboard es fino y su lógica está cubierta por tests
  (`test_dashboard_data.py`); se verificó además que las seis páginas renderizan
  sin excepción con `streamlit.testing.AppTest` sobre una BD sintética.
- Se ejecuta con `uv run streamlit run app/dashboard.py`.
- La página de "próxima jornada" y la de proyección quedarán plenamente vivas
  cuando la F7 ingiera el calendario 2026-27; hasta entonces funcionan en modo
  demo sobre temporadas históricas.
- El gráfico de evolución jornada a jornada de P(descenso)/P(Europa) del Alavés
  (SPEC §8.4) se apoya en el mismo motor; queda como mejora incremental de la
  página del equipo foco.
