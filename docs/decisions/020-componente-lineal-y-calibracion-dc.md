# ADR-020 — Componente lineal Elo+forma, calibración del Dixon-Coles y guarda de calibración

- **Fecha**: 2026-07-12 (Fase 3)
- **Estado**: aceptada

## Contexto

El backtest con el ensemble apilado de ADR-019 cumplía el criterio con cuotas
(0.9611 ≤ 0.9637) pero se quedaba a 0.0033 del criterio sin cuotas (0.9739 vs
baseline Elo 0.9706). Diagnóstico por los pesos del apilado: el LightGBM rinde
peor que el propio Elo (log-loss ~1.00) y arrastra al ensemble; el tercer
componente sin cuotas era el Elo logístico crudo (solo `elo_clubelo_diff`),
que no aporta nada que el baseline no tenga ya.

## Opciones consideradas

1. **Optimizar el LightGBM** (hiperparámetros/optuna, selección de features):
   la palanca de más recorrido a medio plazo, pero cara y sin garantía de
   cruzar 0.0033; se deja para una iteración posterior con su ADR.
2. **Meta-modelo (stacking con aprendiz)**: más potente pero menos transparente
   y con más riesgo de sobreajuste sobre ~2.000 predicciones que una media.
3. **Enriquecer el componente lineal + calibrar el Dixon-Coles** (elegida):
   dos mejoras pequeñas, principiadas y baratas, ambas validadas walk-forward.

## Decisión

**Componente lineal Elo+forma** (`models/linear.py`) sustituye al Elo
logístico crudo en la variante sin cuotas: logística multinomial regularizada
(L2, `StandardScaler` + imputación por mediana) sobre 6 diferencias
local−visitante con mucha señal y poco riesgo — Elo de ClubElo, Elo interno,
puntos por partido (ventana 10), xG a favor y en contra (ventana 10) y días de
descanso. Al contener `elo_clubelo_diff`, en la práctica nunca es peor que el
baseline Elo, y la forma reciente con xG le añade lo que al Elo le falta. Es
robusto a columnas ausentes (esa feature se anula, no falla).

**Calibración del Dixon-Coles como componente**: el DC clásico infraestima los
empates; se calibra por clase con isotónica sobre el pool out-of-fold, igual
que el LightGBM. La versión CRUDA del DC se conserva para el marcador más
probable y como fila/métrica interpretable; solo se calibra su aportación al
ensemble.

**Guarda de calibración** (`calibration.fit_isotonic`): con menos de 300
predicciones se devuelven calibradores identidad (pasan la probabilidad tal
cual). La isotónica es una función escalonada que con pocos puntos memoriza el
ruido; si además sus pesos se eligen sobre ese mismo pool pequeño, generaliza
mal (se observó en una BD de prueba diminuta: el DC "calibrado" sobre 30
partidos se llevaba peso 1.0 y disparaba el log-loss). Con los volúmenes reales
(pools de miles) la calibración se aplica con normalidad.

## Consecuencias

- El ensemble sin cuotas incorpora ahora toda la señal estructural (Elo, forma,
  xG) de forma directa, no diluida entre 70 features del GBM.
- Los pesos del apilado que imprime `alaves train` siguen siendo el diagnóstico:
  si el LightGBM recibe peso ~0, confirma que la siguiente palanca es mejorarlo.
- La calibración deja de ser un riesgo en configuraciones con poca historia.
- El artefacto del registry gana el modelo lineal y los calibradores del DC;
  la reproducibilidad (SPEC §12.4) se mantiene (test de carga que reproduce
  predicciones).
