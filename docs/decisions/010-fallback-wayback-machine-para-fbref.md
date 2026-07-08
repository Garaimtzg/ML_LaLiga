# ADR-010 — Fallback a la Wayback Machine para el histórico de FBref

- **Fecha**: 2026-07-07 (Fase 1, tercera iteración de la ingesta real)
- **Estado**: aceptada
- **Modifica**: ADR-009 (su riesgo "desafío JavaScript" se materializó)

## Contexto

Ni las cabeceras de navegador (ADR-004-act.) ni la huella TLS de Chrome
(ADR-009) bastaron: FBref está sirviendo un **desafío JavaScript interactivo**
("Verificación de seguridad en curso...") que el usuario ve incluso en su
navegador real. Ningún cliente automatizado sin motor JS puede pasarlo, y es
posible que la IP del usuario esté además marcada temporalmente por los
intentos previos.

La observación clave: el histórico que necesita F1 (2018-19 → 2025-26) es
**estático** — esas páginas no cambian nunca — y la Wayback Machine de
Internet Archive las tiene archivadas y las sirve sin desafíos anti-bot.

## Opciones consideradas

1. Esperar a que el desafío/flag de IP expire y reintentar. Indeterminado.
2. Navegador headless (Playwright) para resolver el desafío. Dependencia
   enorme y frágil (Cloudflare detecta headless).
3. **Cascada de descargas**: cache → FBref directo (curl_cffi) → snapshot de
   la Wayback Machine → instrucciones de snapshot manual desde el navegador.

## Decisión

Opción 3, implementada en `_fetch_fbref_schedule`:

1. **Cache local** (`data/raw/fbref/schedule_<temporada>.html`): siempre
   primero; lo que entre una vez, no se vuelve a pedir.
2. **FBref directo** con curl_cffi: si algún día vuelve a funcionar, es la
   fuente primaria (necesaria en F7 para la temporada en curso).
3. **Wayback Machine**: los snapshots disponibles se listan con la **API CDX**
   de archive.org y se prueban del más reciente hacia atrás (máx. 8) hasta
   encontrar uno **con xG**; sufijo `id_` para el HTML original sin la barra
   de wayback. archive.org existe para esto y no bloquea bots respetuosos.

   *Refinamiento (mismo día)*: el snapshot único fijo (1 de agosto posterior
   al fin de temporada) resultó insuficiente — el de la 2025-26 tenía los 380
   marcadores pero las celdas de xG vacías (captura en mal momento). De ahí el
   barrido por CDX con verificación de xG; además, una cache local que parsea
   pero no contiene xG se descarta y se re-resuelve sola.
4. Si todo falla, el error explica el **snapshot manual**: abrir la URL en el
   navegador (que sí pasa el desafío) y guardar el HTML en la ruta de cache.

Salvaguardas de calidad de datos, idénticas por cualquier vía:
- el marcador de cada partido se cruza con football-data antes de insertar
  (una página archivada incompleta o corrupta no entra en silencio);
- la validación de cobertura de xG detecta snapshots parciales;
- la procedencia real se registra (`source = "fbref-wayback"`) y la ingesta
  avisa por temporada cuando usa el archivo en vez del directo.

## Consecuencias

- `alaves ingest --historical` vuelve a ser un solo comando sin intervención
  manual en el caso esperado.
- Para F7 (temporada 2026-27 en curso) la Wayback no sirve como fuente
  fresca post-jornada: si el desafío de FBref persiste entonces, habrá que
  decidir entre snapshot manual semanal (1 página/semana) u otra fuente.
  Queda anotado como riesgo abierto de F7.
- Dependencia de un tercero (archive.org) solo como fallback, nunca como vía
  primaria.
