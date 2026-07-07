# ADR-002 — Esquema SQLite completo desde F1 e identificadores legibles

- **Fecha**: 2026-07-07 (Fase 1)
- **Estado**: aceptada

## Contexto

SPEC §3.2 define las tablas mínimas. Hay que decidir: (a) motor de BD,
(b) cuándo crear cada tabla, (c) tipo de identificadores, (d) cómo registrar
la procedencia cuando dos fuentes alimentan la misma fila.

## Opciones consideradas

1. **Motor**: SQLite vs PostgreSQL vs solo Parquet.
2. **Esquema**: crear tablas según se necesiten vs esquema completo desde F1.
3. **Ids**: enteros autoincrementales vs ids de texto deterministas.
4. **Procedencia en `match_stats`**: una fila por fuente vs una fila fusionada
   con etiqueta de fuentes combinada.

## Decisión

1. **SQLite** (`data/alaves.db`), como justifica SPEC §3.2: proyecto
   monousuario local, cero infraestructura, archivo único. La capa de acceso
   (`etl/db.py`) concentra el SQL por si algún día hay que migrar.
2. **Esquema completo desde F1**, incluidas tablas que se poblarán después
   (`features`, `predictions`, `model_registry`...): el contrato de datos queda
   fijado y visible desde el principio.
3. **Ids de texto deterministas y legibles**: `team_id` es un slug
   (`"alaves"`) y `match_id` es `"{season}_{home_id}_{away_id}"` (único porque
   cada emparejamiento ocurre una vez por temporada y dirección). Ventajas: la
   BD se audita a simple vista y la ingesta es idempotente (re-ejecutar hace
   upsert, nunca duplica). El coste (algo más de espacio que un entero) es
   irrelevante a esta escala (~3.000 partidos).
4. **Una fila por (partido, equipo)** en `match_stats`, con columna `source`
   que acumula las fuentes que aportaron columnas (`"football-data+understat"`).
   El upsert usa `COALESCE` para que una fuente nunca borre con NULL lo que
   otra rellenó. Alternativa descartada (fila por fuente): duplicaría claves y
   complicaría el feature engineering sin aportar trazabilidad práctica extra.

## Consecuencias

- Re-ejecutar `alaves ingest --historical` es seguro (idempotente).
- Las estadísticas de un partido pueden venir de varias fuentes sin conflicto;
  la discrepancia entre fuentes se detecta ANTES de insertar (ver ingest.py).
- La columna `source` a nivel de fila no dice qué columna concreta puso cada
  fuente; si eso importara en el futuro, se registraría por bloque de columnas.
