# ADR-007 — La ingesta real se ejecuta en local (WSL); el entorno remoto no tiene red hacia las fuentes

- **Fecha**: 2026-07-07 (Fase 1)
- **Estado**: aceptada

## Contexto

El desarrollo con Claude Code se hace en un entorno remoto cuyo proxy de red
**bloquea** las fuentes de datos (football-data.co.uk, understat.com,
api.clubelo.com devuelven 403 de política en el CONNECT). El entregable de F1
es "BD poblada y validada", pero la BD no puede poblarse desde ese entorno.
El usuario trabaja en local sobre WSL (Ubuntu en Windows), donde sí hay red.

## Opciones consideradas

1. Commitear una BD pre-poblada generada en otro sitio (viola CLAUDE.md §2:
   `data/` no se commitea).
2. Pedir excepciones de red al entorno remoto (fuera de nuestro control).
3. **Separar responsabilidades**: el repo entrega el pipeline completo,
   testeado con fixtures congelados que replican el formato real de cada
   fuente; la población real de la BD se ejecuta en la máquina local del
   usuario con un comando.

## Decisión

Opción 3:

- Los tests (26 en F1) cubren parsers, mapeo de nombres, BD, consistencia
  entre fuentes y el pipeline entero sobre una mini-liga sintética, sin red.
- El usuario puebla la BD en su WSL: `uv run alaves ingest --historical` y la
  certifica con `uv run alaves validate` (instrucciones en el README).
- Si una fuente real difiere del formato asumido (p. ej. un alias de ClubElo
  incorrecto), el pipeline falla con un mensaje que indica el arreglo exacto;
  esos arreglos se commitean como `fix:` y quedan cubiertos para siempre.

## Consecuencias

- El primer `ingest --historical` real es un pequeño acto de verificación:
  puede requerir 1-2 correcciones de alias/formato (esperado y barato).
- La BD y `data/raw/` viven solo en local (y en backups del usuario), como ya
  exigía CLAUDE.md.
- Este reparto (código y tests en remoto, ejecución con red en local) aplica a
  todas las fases siguientes.
