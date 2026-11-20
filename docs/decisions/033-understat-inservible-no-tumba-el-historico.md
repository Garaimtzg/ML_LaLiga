# ADR-033 — Una Understat inservible no tumba la ingesta histórica ni maquilla la proyección

Fecha: 2026-11-20 · Estado: aceptada · Fase: F7 (operación)

## Contexto

Con el aviso mejorado de ADR-032, una ingesta real dijo por fin qué pasaba:

```
ERROR de ingesta: getLeagueData de Understat no devolvió JSON válido (...): respondió
una página HTML (...). Empieza así: '<!DOCTYPE html> <html> <head>
<title>Understat.com</title> ...'
```

Understat responde **su web normal** en la ruta de la API interna
`getLeagueData` (ADR-011). No es un bloqueo anti-bot: lo más probable es que
haya vuelto a cambiar cómo carga los datos, como ya hizo en diciembre de 2025
(ADR-008/011). Averiguar la ruta nueva exige inspeccionar la web desde una
máquina con acceso, y mientras tanto había tres problemas:

1. **`alaves ingest --historical` abortaba entero.** Understat solo rellena el
   xG que FBref no trae (las temporadas recientes), pero su fallo tiraba las
   ocho temporadas: resultados, cuotas, xG de FBref y Elo. Sobre una BD nueva no
   había manera de seguir.
2. **`alaves sources` habría dicho "OK"**: solo comprobaba que la descarga
   respondiera, no que lo recibido se pudiera leer.
3. **La proyección mentía sin avisar.** Sin Understat, el calendario de la
   temporada en curso son solo los partidos inminentes de football-data. El
   Monte Carlo simulaba esos diez y presentaba el resultado como la
   clasificación final.

## Decisión

- **Ingesta histórica**: un fallo de descarga o de formato de Understat ya no
  aborta. La temporada queda con el xG que haya. Al final, la ingesta se declara
  **INCOMPLETA**: sale con código 1, dice qué temporadas faltan y por qué, y
  explica que relanzarla cuando Understat vuelva no repite lo ya descargado. El
  error incluye la ruta de la cache donde quedó guardada la respuesta, para
  poder abrirla. Un marcador discrepante entre fuentes (`SourceConsistencyError`)
  **sigue abortando**: eso no es una fuente caída, son datos en conflicto
  (CLAUDE.md §6). `alaves validate` sigue marcando la cobertura de xG que falte.
- **`alaves sources`** pasa cada respuesta por el mismo parser que usa la
  ingesta. Una fuente que responde algo ilegible sale como **NO SIRVE**, con el
  motivo.
- **Proyección**: `Projection.n_missing` cuenta los partidos de la temporada que
  no están en la BD. Si falta alguno, el CLI (`simulate` y el ciclo semanal) y el
  dashboard avisan de que la tabla no es la clasificación final. El ciclo deja
  de imprimir la "posición esperada" del equipo foco sobre una simulación a
  medias.
- **Medias móviles (corrige ADR-031)**: solo heredan la última media los
  partidos **por jugar**. ADR-031 hacía heredar también a los jugados sin el
  dato, y con la 2025-26 entera sin xG eso congelaba la media de xG de mayo de
  2025 en todos los partidos de la 2025-26 y la 2026-27: un valor con pinta de
  válido que no lo era. Un jugado vuelve a usar la ventana de sus últimos
  partidos jugados (vacía si ninguno tiene el dato), exactamente como antes de
  ADR-031. Verificado: con todo jugado, idéntico al original aun con un 30 % de
  partidos sin xG.

## Consecuencias

- Con Understat inservible se puede tener la BD histórica, entrenar y predecir
  la próxima jornada. Faltan el xG de las temporadas que dependan de ella y la
  proyección de la clasificación completa, y cada salida lo dice.
- **Pendiente**: adaptar el adaptador de Understat a su nueva forma de servir
  los datos, cuando se conozca. Eso será otro ADR; este solo hace que su caída
  no arrastre al resto.
