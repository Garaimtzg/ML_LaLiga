# ADR-011 — Understat vuelve vía su API interna como fuente de relleno de xG

- **Fecha**: 2026-07-08 (Fase 1, cierre)
- **Estado**: aceptada
- **Modifica**: ADR-008 (Understat estaba "en pausa"), ADR-010 (el barrido CDX
  no encontraba snapshots de la 2025-26 con xG — ahora se sabe por qué)

## Contexto

La validación de la BD dejó un único chequeo rojo: 0/760 filas con xG en la
temporada 2025-26. La investigación (snapshots de wayback + verificación
visual del usuario en la página en vivo) reveló la causa: **FBref eliminó las
columnas de xG de sus páginas de calendario en la versión 2026 del sitio**.
No era un snapshot desafortunado: el dato ya no está en esas páginas.

A la vez, el usuario capturó con DevTools cómo carga los datos el nuevo
Understat: un endpoint JSON interno, sin bloqueo anti-bot:

    GET https://understat.com/getLeagueData/La%20liga/<año>

## Opciones consideradas

1. **Páginas de partido de FBref** (380/temporada): bloqueadas para clientes
   automatizados y apenas archivadas. Descartada.
2. **API-Football**: su free tier no cubre estadísticas de temporadas
   recientes. Descartada.
3. **Understat vía getLeagueData**: mismo dato que daba su HTML antiguo,
   servido en JSON; sin desafíos anti-bot; una petición por temporada.

## Decisión

Opción 3, como **fuente de relleno** (no sustituye lo ya cargado):

- El adaptador de Understat se reescribe para el endpoint JSON. El parseo es
  tolerante con el sobre exterior (lista directa o dict con la lista bajo
  `datesData`/`matchesData`/...) porque es un endpoint interno sin contrato
  público; si cambia, el error lista las claves encontradas.
- En la ingesta, tras FBref: si una temporada tiene partidos sin xG, se pide
  el getLeagueData de esa temporada y se rellenan **solo** los huecos (nunca
  se pisa el xG de FBref ya almacenado). Cada relleno cruza el marcador con
  el almacenado antes de insertar y se registra con `source = "understat"`.
- El calendario de FBref se acepta ahora aunque no traiga xG: sigue aportando
  la jornada oficial (Wk) y una verificación adicional de marcadores.
- Consecuencia de coherencia: el histórico queda con xG de FBref (proveedor
  Opta) en 2018-25 y de Understat (modelo propio) en 2025-26. Ambos modelos
  están altamente correlacionados; aun así, la procedencia por fila permite
  a F2/F5 medir si la mezcla introduce sesgo (y corregirla si hiciera falta).

## Consecuencias

- La 2025-26 recupera su xG y la validación de F1 puede quedar en verde.
- **F7 gana su fuente de xG en vivo**: FBref seguirá amurallado para bots,
  pero getLeagueData de la temporada 2026-27 dará el xG post-jornada.
- Riesgo: el endpoint es interno y puede cambiar sin aviso; mitigado con el
  parseo tolerante, el fallo ruidoso con diagnóstico y el patrón de
  adaptadores que ya permitió pivotar dos veces sin tocar el resto.
