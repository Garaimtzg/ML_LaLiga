# ADR-030 — Una fuente caída degrada, y dice por qué

Fecha: 2026-09-05 · Estado: aceptada · Fase: F7

## Contexto

Tercera semana de operación real y tercer aviso del mismo tipo:

```
AVISO: ClubElo no responde para: alaves, almeria, athletic-club, atletico-madrid,
barcelona, betis, cadiz, celta-vigo, eibar, elche, espanyol, getafe, girona,
granada, huesca, las-palmas, leganes, levante, mallorca, osasuna, oviedo,
rayo-vallecano, real-madrid, real-sociedad, sevilla, valencia, valladolid,
villarreal, racing-santander, deportivo-la-coruna, malaga. (Todos conservan su
Elo ya almacenado en la BD.)
```

Ante la pregunta "¿y por qué no responde?", el sistema no tenía respuesta:

1. **El motivo se tiraba a la basura.** `ingest_clubelo` hacía
   `except SourceDownloadError:` sin capturar la excepción, así que el mensaje
   —que distingue un 404 de un bloqueo, de un timeout o de un DNS caído— se
   perdía. Es el mismo error que ya se corrigió en FBref (ADR-029): un aviso que
   no dice la causa no es accionable.
2. **La lista de 31 nombres tapaba lo único importante**, que es que fallaron
   *todos*. Que caiga la fuente entera y que caiga un alias mal escrito son
   diagnósticos opuestos, y el aviso los presentaba igual.
3. **El parseo quedaba fuera del `try`.** Solo se protegía la descarga, así que
   si ClubElo respondía HTTP 200 con algo que no es su CSV —una portada de
   error, un redirect servido como HTML— saltaba un `SourceFormatError` sin
   capturar que abortaba el ciclo semanal **entero**: ni reentrenamiento, ni
   predicciones, ni proyección. Justo lo contrario del principio que rige el
   modo temporada.

## Decisión

**El motivo se conserva.** `ingest_clubelo` devuelve `dict[team_id, motivo]` en
vez de una lista de nombres, y el aviso incluye uno representativo.

**El aviso describe la forma del fallo, no la enumera.** `_describe_teams`
resume: "los 31 equipos (la fuente entera, no un alias suelto)" cuando fallan
todos, los nombres cuando son pocos, y "N equipos (a, b, c...)" en medio.

**El parseo entra en el `try`.** Que la fuente devuelva basura es la fuente
caída, no un fallo del pipeline, y degrada igual que una descarga fallida.

**Es una nota, no un aviso, cuando no cuesta nada.** Si todos los equipos
conservan su Elo en la BD, no hay nada que mirar: el Elo se mueve poco de una
semana a otra y el almacenado sirve. Solo se convierte en aviso si algún equipo
se queda sin ningún Elo. Es la misma regla que ADR-029 aplicó a FBref.

**`base_url` pasa a HTTPS.** Es el default sano, y un `http://` que deja de
responder sin que se haya tocado el código apunta a un redirect a HTTPS o al
cierre del puerto 80. Es una hipótesis, no una comprobación: el entorno de
desarrollo no tiene red a la fuente (ADR-007). El motivo real lo dirá el propio
aviso en la siguiente ejecución, que es exactamente el punto de este ADR.

## Añadido: las jornadas del calendario no se agrupaban

Los datos de la BD real destaparon que la agrupación de jornadas de ADR-029 no
llegaba a ejecutarse nunca:

```
finished|1|8    scheduled|4|8
finished|2|12   scheduled|5|10
finished|3|10   scheduled|6|11
finished|4|1    scheduled|7|10
```

Una jornada de LaLiga tiene 10. Con bloques de 10 no puede salir un 11: la
prueba de que `assign_scheduled_matchdays` se estaba saliendo sin tocar nada.

**Causa.** `assign_matchdays` (la aproximación por conteo de ADR-006, pensada
para partidos jugados) no filtraba por estado, así que numeraba **también los
programados**. Después `assign_scheduled_matchdays` los encontraba todos con
jornada y se salía por su guarda de "todos traen jornada oficial". El calendario
futuro salía de una aproximación que para partidos no jugados no significa nada.

**Arreglos.**

1. `assign_matchdays` solo toca partidos `finished`.
2. La primera jornada programada **completa la jornada en curso** en vez de
   abrir la siguiente. Con un partido de la J4 adelantado al viernes, los 9
   restantes son J4, no J5; sin esto, todo el calendario iba desplazado.
3. `validate` gana un chequeo de **partidos por jornada**, que es el que caza
   esta clase de fallo de un vistazo.

**Lo que NO se arregla, y hay que decirlo.** Las jornadas de lo ya jugado
(8/12/10 arriba) siguen mal: son la aproximación de ADR-006 y los aplazamientos
la descuadran — dos partidos de la J1 jugados después de la J2 de sus equipos
cuentan como J2. Sin la Wk oficial no hay heurística que lo resuelva: agrupar
por fechas falla con las jornadas entre semana y por conteo falla con los
aplazamientos. Es el coste concreto de tener FBref inalcanzable, y por eso el
chequeo nuevo informa de ello sin tumbar la validación: no es un fallo que el
usuario pueda arreglar.

## Añadido: el ciclo semanal tardaba más de una hora

El diagnóstico de ClubElo llegó completo: el DNS resuelve (`37.128.134.74`)
pero **el puerto 443 no acepta conexiones** — `curl -I https://api.clubelo.com/...`
se queda 134 s y se rinde. El sitio solo sirve por el 80.

Eso invalida la hipótesis de HTTPS de este mismo ADR, y además **la empeoró**:
en HTTP la conexión falla rápido; en HTTPS se queda colgada hasta el timeout. Con
`_TIMEOUT_S = 60` y 3 reintentos son ~3 min por equipo, y ClubElo hace **una
petición por equipo**: 31 × 3 min ≈ hora y media de ciclo semanal.

Tres arreglos, en orden de importancia:

1. **Timeout de conexión separado del de lectura** (8 s vs 60 s). Leer una
   respuesta puede tardar; *abrir* la conexión no — un host que no acepta
   conexiones no va a aceptarlas en el segundo 59.
2. **Cortafuegos en ClubElo**: tras 3 fallos seguidos se da la fuente por caída
   y no se piden los 28 restantes. Insistir no aporta información y cuesta un
   timeout por equipo.
3. **Vuelta a HTTP** en `base_url`, con la evidencia escrita en el comentario
   para que nadie repita el intento.

## Añadido: `alaves sources`

La consecuencia incómoda de que las fuentes degraden bien es que cuesta ver qué
está pasando: el ciclo sigue, y el detalle se resume. `alaves sources` prueba
una URL representativa de **cada** fuente sin cache y muestra estado, tiempo y
error. Es el comando que responde "¿y por qué no va?" sin tener que leer código.

Y un fallo que ese diagnóstico dejó a la vista: el mensaje de error de FBref
propone guardar la página a mano en la cache, pero el ciclo semanal descarga con
`force=True` y **la ignoraba**, así que el arreglo manual documentado no
funcionaba. Ahora la cache es el último recurso incluso con `force`, etiquetado
como `fbref-cache-manual` para que quede claro que no es FBref en vivo.

## Consecuencias

- Un aviso de fuente caída ahora se puede accionar: dice qué falló, con qué
  forma y por qué.
- El ciclo semanal sobrevive a que ClubElo devuelva cualquier cosa, no solo a
  que no devuelva nada.
- La salida semanal se acorta: donde había 31 nombres, ahora hay una frase y
  una causa.
- Riesgo aceptado: se muestra **un** motivo representativo, no los 31. Si
  distintos equipos fallaran por razones distintas, solo se ve el primero. A
  cambio, la salida es legible; el detalle completo sigue en el `dict`.
