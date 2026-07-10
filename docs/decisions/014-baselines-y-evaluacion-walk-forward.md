# ADR-014 — Baselines 1X2 y protocolo de evaluación walk-forward

- **Fecha**: 2026-07-10 (Fase 2)
- **Estado**: aceptada

## Contexto

CLAUDE.md §5.3: ningún modelo se acepta sin comparación contra tres baselines.
SPEC §6.1 los define; falta concretar la implementación y el protocolo.

## Decisión

**Protocolo**: walk-forward por temporada. Para cada temporada de test (las 3
últimas por defecto, `alaves baselines --seasons N`), se entrena SOLO con
temporadas estrictamente anteriores. Nunca splits aleatorios; con una sola
temporada disponible no se evalúa nada (verificado por test).

**Baselines** (`evaluation/baselines.py`):

1. *Frecuencias históricas*: vector [P(H), P(D), P(A)] del entrenamiento,
   constante para todos los partidos de test. El suelo absoluto.
2. *Elo logístico*: regresión logística multinomial (`scikit-learn`) sobre una
   única feature, `elo_clubelo_diff` pre-partido. La ventaja de campo no
   necesita feature: al ser constante, la capturan los interceptos por clase.
   Se usa el Elo de ClubElo (externo) y no el interno para que el baseline
   sea independiente de nuestras propias decisiones de cálculo.
3. *Cuotas de CIERRE normalizadas*: probabilidades implícitas 1/cuota,
   reescaladas a suma 1 (elimina el margen del bookmaker). Preferencia media
   de mercado > bet365; si no hay cierre (2018-19 no lo publica), apertura.
   Es el baseline exigente: la mejor estimación pública del partido.

**Métricas** (`evaluation/metrics.py`, con tests de valores resueltos a mano):
log-loss (principal), Brier multiclase, RPS (respeta el orden H>D>A: se
verifica por test que penaliza más equivocarse de lado que de empate) y
accuracy. Convención de columnas [H, D, A] en todo el proyecto.

**Informe**: `alaves baselines` imprime la tabla y escribe
`docs/reports/baselines_<fecha>.md` con métricas por temporada y medias.

## Consecuencias

- La vara de medir de F3 queda fijada ANTES de entrenar ningún modelo
  (criterios de aceptación de SPEC §12.1).
- El CLI gana dos comandos no listados en SPEC §10 (`features`, `baselines`);
  se consideran orquestación interna de F2 — SPEC §10 no se modifica.
