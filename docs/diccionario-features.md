# Diccionario de variables — feature set v1

Qué significa cada variable que ve el modelo. Todas se calculan con corte
temporal estricto: **solo con información anterior al día del partido**
(`as_of_date` = víspera; CLAUDE.md §5.1). La resolución programática de
cualquier nombre de columna está en
[`features/dictionary.py`](../src/alaves_predictor/features/dictionary.py)
(la usan el informe de importancia y el dashboard), con un test que garantiza
que ninguna feature queda sin documentar.

## Convenciones de nombres

| Fragmento | Significado |
|-----------|-------------|
| `home_` / `away_` | Del equipo **local** / **visitante** de ese partido |
| `_ma5` / `_ma10` | **Media móvil** de los últimos 5 / 10 partidos del equipo (anteriores al actual, nunca el propio) |
| `_venue_` | La media cuenta solo partidos en la **misma condición**: local en casa, visitante fuera |
| `_diff` | Diferencia local − visitante (positivo = ventaja del local) |
| `_pre` | Valor justo **antes** del partido |

## Fuerza estructural (Elo)

| Variable | Significado |
|----------|-------------|
| `elo_clubelo_home` / `elo_clubelo_away` | Rating Elo de [ClubElo.com](http://clubelo.com) de cada equipo, el último publicado antes del partido. Nivel absoluto del club (~1400 colista, ~1900 élite); serie externa con décadas de historia |
| `elo_clubelo_diff` | `elo_clubelo_home − elo_clubelo_away`: cuánta ventaja de nivel tiene el local. La variable más importante del modelo |
| `elo_internal_home_pre` / `elo_internal_away_pre` | Elo **interno** del proyecto (ADR-013: K=20, recalculable solo con partidos de LaLiga desde 2018), justo antes del partido |
| `elo_internal_diff` | Diferencia del Elo interno (local − visitante) |
| `promoted_home` / `promoted_away` | 1 si el equipo es recién ascendido (no jugó Primera la temporada anterior) |

Los dos Elo coexisten a propósito (SPEC §3.1): el de ClubElo aporta historia
larga y cobertura de Segunda; el interno es auditable y ajustable.

## Forma reciente (medias móviles de 5 y 10 partidos)

Cada estadística existe en 8 variantes: `{home|away}_{stat}[_venue]_ma{5|10}`.
Ejemplo: `away_xg_against_venue_ma10` = xG concedido por el visitante en sus
últimos 10 partidos **fuera de casa**.

| Estadística | Significado |
|-------------|-------------|
| `points` | Puntos por partido (3/1/0): resultados recientes |
| `goals_for` / `goals_against` | Goles marcados / encajados |
| `xg_for` | xG generado: calidad de las ocasiones **creadas** (mejor predictor que los goles, menos varianza) |
| `xg_against` | xG concedido: calidad de las ocasiones que **permite** al rival (solidez defensiva) |
| `g_minus_xg` | Goles reales − xG: sobre/infrarrendimiento. Un valor alto = está marcando más de lo que sus ocasiones justifican (pegada o suerte; **tiende a revertir**) |

## Rachas y descanso

| Variable | Significado |
|----------|-------------|
| `home_win_streak` / `away_win_streak` | Victorias consecutivas antes del partido |
| `home_loss_streak` / `away_loss_streak` | Derrotas consecutivas antes del partido |
| `home_rest_days` / `away_rest_days` | Días desde el partido de liga anterior del equipo (la fatiga por Copa/Europa llegará con API-Football, F7) |

## Contexto del partido

| Variable | Significado |
|----------|-------------|
| `matchday` | Jornada oficial (1–38): fase de la temporada |
| `month` | Mes del partido (estacionalidad) |
| `no_crowd` | 1 en temporadas sin público (2020-21, COVID; configurable) |
| `derby` | 1 si es un derbi (pares definidos en `config/settings.toml`) |
| `h2h_home_ppg` | Puntos/partido del local en los últimos 5 **enfrentamientos directos** contra ese rival (peso bajo a propósito: muestra pequeña) |

## Mercado (solo variante con cuotas)

| Variable | Significado |
|----------|-------------|
| `imp_home` / `imp_draw` / `imp_away` | Probabilidades implícitas de las cuotas de **apertura** (1/cuota, normalizadas para quitar el margen del bookmaker). Las de **cierre** nunca son feature: se reservan como baseline (SPEC §4.1) |

La variante **sin cuotas** del modelo excluye estas tres — es la que se
interpreta con SHAP, porque el mercado taparía al resto de variables.
