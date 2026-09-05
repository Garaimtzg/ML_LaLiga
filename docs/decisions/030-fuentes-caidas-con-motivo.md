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
