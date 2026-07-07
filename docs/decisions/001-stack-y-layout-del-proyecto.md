# ADR-001 — Stack, layout del proyecto y política de dependencias

- **Fecha**: 2026-07-07 (Fase 1)
- **Estado**: aceptada

## Contexto

Arranque del repositorio. CLAUDE.md fija el marco (Python 3.11+, `uv`, layout
`src/`, ruff, mypy, pytest), pero quedan decisiones de detalle: build backend,
qué dependencias instalar ya, y cómo estructurar el CLI.

## Opciones consideradas

1. **Instalar todo el stack desde el día 1** (pandas, lightgbm, shap, streamlit...).
2. **Instalar solo lo que usa la fase en curso** y añadir el resto cuando su fase lo necesite.

Para el build backend: `hatchling` (ligero, estándar de facto con `uv`),
`setuptools` (histórico, más verboso) o `poetry-core` (implicaría otro gestor).

## Decisión

- Opción 2: **dependencias por fase**. F1 (ETL) usa `httpx` (HTTP moderno con
  timeouts sanos), `pydantic` (validación de esquemas de fuentes) y `typer`
  (CLI). Dev: `pytest`, `ruff`, `mypy`. `pandas`, `lightgbm`, `shap`,
  `streamlit`, etc. entrarán en F2-F6, cada una justificada en su momento.
- Build backend **hatchling** con paquete en `src/alaves_predictor/`.
- `uv.lock` **se commitea** para reproducibilidad exacta del entorno.
- Parsers de F1 con `csv`/`re`/`json` de la stdlib: los volúmenes son pequeños
  y la validación fila a fila con pydantic es más clara que un DataFrame
  intermedio. `beautifulsoup4` no es necesaria aún (Understat embebe JSON, no
  hay que recorrer el DOM).

## Consecuencias

- `pyproject.toml` crece con el proyecto; cada dependencia estructural nueva
  requerirá justificarse (y ADR si procede).
- Quien clone el repo en cualquier fase instala solo lo necesario (`uv sync`).
- El código de F1 no depende de pandas; los módulos de features (F2) lo harán.
