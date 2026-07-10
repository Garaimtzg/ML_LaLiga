# ADR-012 — Feature set v1: alcance, corte temporal y dependencias de F2

- **Fecha**: 2026-07-10 (Fase 2)
- **Estado**: aceptada

## Contexto

SPEC §4.1 cataloga >150 features potenciales, pero exige (control de
dimensionalidad) que el modelo v1 entrene con un subconjunto curado de
~40-60, ampliable por forward selection. Además, el bloque técnico-táctico
depende de las estadísticas detalladas de FBref, cuyo anti-bot (ADR-009/010)
hace inviable hoy scrapear ~380 páginas de partido por temporada.

## Decisión

**Feature set v1** (~50 columnas), construido en `features/build.py` con
`as_of_date` = día anterior al partido:

- *Fuerza estructural*: Elo de ClubElo as-of (lookup `merge_asof` hacia
  atrás), Elo interno pre-partido (ADR-013) y sus diferencias; `promoted_flag`
  (equipo en temporada S ausente en S-1; 0 en la primera temporada de la BD).
- *Forma* (ventanas 5 y 10, configurables): puntos, goles a favor/en contra,
  xG a favor/en contra, goles−xG; cada una en versión general y por condición
  (local en casa / visitante fuera); rachas de victorias/derrotas; días de
  descanso (solo liga — la fatiga por Copa/Europa llega con API-Football, F7).
- *Contexto*: jornada oficial, mes, `no_crowd` (temporadas en config), derbi
  (pares en config), head-to-head (puntos/partido del local en los últimos 5
  cruces; peso bajo a propósito, que SHAP lo juzgue en F5).
- *Mercado*: probabilidades implícitas de las cuotas de APERTURA normalizadas
  (sin margen). Las de CIERRE se reservan como baseline (SPEC §4.1: la
  variante sin cuotas es la que se interpreta; la de apertura es la feature).

**Anti-leakage por construcción y por test**: todas las medias móviles operan
sobre valores desplazados (`shift(1)`); el Elo interno es secuencial; y un
test obligatorio verifica empíricamente que alterar un partido futuro no
cambia ninguna feature de partidos anteriores ni las del propio partido.

**Persistencia** (SPEC §12.4): tabla `features` con payload JSON por
(match_id, feature_set_version) + snapshot Parquet en `data/features/`.

**Aplazado**: bloque técnico-táctico de FBref (bloqueado por su anti-bot;
`match_stats` ya tiene las columnas y la forward selection de F3 lo
reevaluará si se materializa una vía de acceso) y valores de Transfermarkt.

**Dependencias nuevas** (todas del stack aprobado en CLAUDE.md §2):
`pandas` y `numpy` (transformaciones), `scikit-learn` (regresión logística
del baseline; base de F3), `pyarrow` (Parquet).

## Consecuencias

- `alaves features` materializa el feature store completo en segundos.
- El catálogo es ampliable sin tocar el esquema (payload JSON + Parquet).
- La ausencia del bloque técnico-táctico pone techo a la v1; se cuantificará
  contra los baselines en F3 y se documentará en el informe de importancia.
