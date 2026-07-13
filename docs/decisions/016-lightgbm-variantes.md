# ADR-016 — LightGBM multiclase: hiperparámetros v1 y variantes con/sin cuotas

- **Fecha**: 2026-07-12 (Fase 3)
- **Estado**: aceptada

## Contexto

SPEC §6.3 fija el modelo B: `LGBMClassifier` multiclase sobre el feature set
v1, con búsqueda de hiperparámetros opcional en v1 ("valores razonables
documentados"). Hay que fijar esos valores, las variantes y las dependencias.

## Opciones consideradas

1. **Búsqueda con optuna desde el principio**: descartada en v1 — añade una
   dependencia y tiempo de cómputo antes de tener un backtest que diga si
   hace falta; SPEC la deja explícitamente como opcional.
2. **Una sola variante con cuotas**: descartada; SPEC §4.1 exige la variante
   sin cuotas como la interpretable (las cuotas enmascaran qué variables
   futbolísticas importan).

## Decisión

- **Dependencias nuevas**: `lightgbm` (el modelo) y `scipy` (ya llegaba
  transitivo con scikit-learn; se declara en pyproject porque el Dixon-Coles
  lo usa directamente — política de dependencias explícitas de ADR-001).
- **Dos variantes** entrenadas sobre las mismas features (`gbm_classifier.py`):
  `con_cuotas` (techo de rendimiento) y `sin_cuotas` (la que se interpreta
  con SHAP en F5). La única diferencia es excluir las columnas de mercado
  (`imp_home/imp_draw/imp_away`, cuotas de apertura).
- **Hiperparámetros v1** en `[models.lightgbm]` de settings.toml, pensados
  para ~3.000 filas (territorio de mucha regularización):
  `n_estimators=300`, `learning_rate=0.03`, `num_leaves=15`,
  `min_child_samples=50`, `feature_fraction=0.7`, `bagging_fraction=0.8`,
  `lambda_l2=1.0`, `random_state=42` (semilla fija del proyecto).
- **NaN se quedan como NaN**: LightGBM los enruta de forma nativa en cada
  split; imputar (media/mediana) destruiría la información de "dato ausente"
  (p. ej. cuotas sin publicar, primeras jornadas sin forma).
- El orden de clases de salida se reordena SIEMPRE al canónico [H, D, A] del
  proyecto (sklearn ordena alfabéticamente: A, D, H).

## Consecuencias

- Entrena en segundos: compatible con el reentrenado por jornada (SPEC §6.4)
  y con el backtest jornada a jornada (ADR-018).
- Si el backtest muestra que la v1 se queda corta, la búsqueda con optuna se
  añadirá con su propio ADR (y validando solo walk-forward).
