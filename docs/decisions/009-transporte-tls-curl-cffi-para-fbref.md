# ADR-009 — Transporte con huella TLS de navegador (curl_cffi) para FBref

- **Fecha**: 2026-07-07 (Fase 1, segunda iteración de la ingesta real)
- **Estado**: aceptada
- **Modifica**: ADR-004 (capa de descargas), ADR-008 (riesgo materializado)

## Contexto

FBref devolvió `403 Forbidden` a la descarga del calendario **incluso con
cabeceras completas de navegador**. Su CDN (Cloudflare) no valida solo las
cabeceras: valida la **huella TLS/JA3 del cliente**. Cualquier cliente HTTP de
Python estándar (httpx, requests) tiene una huella TLS distinta a la de un
navegador real y es rechazado, se ponga el User-Agent que se ponga.

## Opciones consideradas

1. **curl_cffi**: cliente HTTP que imita byte a byte la huella TLS de Chrome
   (`impersonate="chrome"`). Wheels precompiladas, sin navegador, API tipo
   requests. Es la solución estándar del ecosistema de scraping de FBref.
2. **Playwright/navegador headless**: funciona pero añade una dependencia
   enorme (motor de navegador completo) para bajar 8 páginas estáticas.
3. **Cambiar de fuente de xG otra vez**: no hay tercera fuente gratuita de xG
   histórico comparable (Understat ya cayó, ADR-008).

## Decisión

Opción 1. `etl/http_cache.py` gana un segundo transporte:

- Por defecto, httpx con el User-Agent identificable y honesto (football-data
  y ClubElo siguen así).
- `fetch_text(..., impersonate=True)` usa `curl_cffi` con huella de Chrome;
  solo lo usa el adaptador de FBref.

Consideración ética, explícita: los datos son públicos, el acceso es mínimo
(1 página por temporada, 8 en total, cacheadas para siempre) y el rate limit
de 6 s respeta la petición de moderación de FBref hacia los bots. La huella
TLS solo evita un falso positivo de su filtro anti-scraping masivo; no se
elude ningún muro de pago ni autenticación.

## Consecuencias

- Dependencia nueva: `curl_cffi` (binaria, con wheels para Linux/WSL).
- Si Cloudflare endurece el desafío (JavaScript interactivo), curl_cffi
  dejaría de bastar; la señal sería otro 403 persistente y entonces habría
  que reevaluar (opción 2 o snapshot manual de páginas).
- El resto de fuentes no cambia de comportamiento.
