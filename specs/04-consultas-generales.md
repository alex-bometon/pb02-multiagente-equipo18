# 04 — Consultas generales y horarios del RAG de ReciclaTIA

**Proyecto:** PB02 Multiagente — ReciclaTIA  
**Método:** SDD (especificación anterior a los cambios)  
**Estado:** propuesta para implementación por archivos y validación posterior  
**Ámbito:** Madrid ciudad y corpus local ya indexado

## 1. Situación de partida

La implementación actual separa:

- `src/retrieve.py`: búsquedas semánticas, filtros de modalidad/distrito y
  `listar_destinos_residuos()`.
- `src/generate.py`: construcción de contexto, respuestas y validación de citas.
- `src/rag.py`: orquestación mediante `responder()` y métricas.
- `config.py`: parámetros compartidos, incluido `TOP_K = 5`.

Las **158 pruebas automatizadas** han sido confirmadas por el desarrollador.
La evaluación extremo a extremo de 14 preguntas obtuvo 10 respuestas en 11
consultas respondibles y tres abstenciones correctas fuera de dominio. La
pregunta 03 produjo una salida inválida de Gemini: se ha corregido después
el validador de citas, pero **falta repetir esa prueba real**.

Dos consultas continúan requiriendo un tratamiento específico:

- **P09:** «¿Qué horario tiene el punto limpio móvil?». El RAG enumeró
  horarios de paradas distintas sin identificar a cuáles correspondían.
- **P10:** «¿Qué tipos de residuos reconoce el Ayuntamiento de Madrid?».
  Se enumeraron cinco filas del CSV como si fuesen una respuesta global.

**Decisión:** conservar el retrieval semántico con K=5 para preguntas
concretas. Añadir vías estructuradas únicamente para consultas que requieren
una vista de conjunto o una parada identificada. No cambiar chunking,
embeddings, colección ChromaDB ni modelo por estos dos problemas.

## 2. Evidencia disponible en el índice

Se han revisado los registros del corpus de referencia utilizado para el RAG:

| Fuente | Registros | Campos útiles | Contenido |
| --- | ---: | --- | --- |
| `tipos_residuos.csv` | 177 | `source`, `destino`, `source_row` | Una relación residuo → lugar de depósito por chunk |
| `puntos_limpios_moviles.csv` | 351 | `document_type`, `district`, `day`, `source_row` | Dirección, día, turno y horario en el texto |
| `puntos_limpios_moviles_24h.csv` | 42 | `document_type`, `district`, `day` | Ubicación y día; no equivale al horario de una parada móvil ordinaria |
| `puntos_limpios_fijos.csv` | 16 | `document_type`, `district` | Dirección y horario del centro fijo |
| `puntos_limpios_proximidad.csv` | 95 | `document_type`, `district` | Dirección y horario del punto de proximidad |

Los 177 registros de `tipos_residuos.csv` contienen **19 valores distintos de
`destino`** en el corpus revisado. Estos valores son **lugares o servicios
de depósito**, no una taxonomía oficial exhaustiva de «tipos de residuos».
Las cantidades deberán calcularse a partir de la colección que ejecute el
proyecto, nunca asumirse como constantes inmutables.

## 3. Objetivos y límites

### Objetivos

1. Distinguir una consulta sobre un residuo concreto de una petición de
   resumen del catálogo.
2. Para el catálogo, consultar las relaciones pertinentes por vía
   estructurada en lugar de seleccionar cinco filas por similitud.
3. No asignar a todos los puntos limpios móviles un horario tomado de
   paradas diferentes.
4. Solicitar el dato que falta cuando no se pueda identificar la parada.
5. Mantener el contrato de respuesta, las fuentes verificables y las
   métricas de `responder()`.
6. Evitar consumo innecesario de Gemini cuando basta una aclaración
   determinista.

### Fuera de alcance

- Cambiar la cobertura geográfica más allá de Madrid ciudad.
- Crear rutas, geolocalización, mapas o seguimiento de horarios en tiempo real.
- Garantizar exhaustividad fuera de las filas existentes en el corpus.
- Introducir un sistema de búsqueda web o fuentes externas.
- Modificar índices o repetir embeddings.
- Implementar todavía agentes LangGraph, HITL o interfaces de usuario.

## 4. Responsabilidades por módulo

| Módulo | Responsabilidad prevista |
| --- | --- |
| `src/retrieve.py` | Consultar registros del catálogo y paradas móviles mediante `collection.get`, con filtros y normalización de metadatos; preparar datos estructurados sin redactar respuestas |
| `src/rag.py` | Clasificar de forma conservadora la intención explícita, elegir vía semántica o estructurada, solicitar aclaraciones y mantener el contrato público |
| `src/generate.py` | Redactar a partir de la evidencia entregada, delimitar el alcance y validar estructura y citas; no abrir ChromaDB |
| `config.py` | Centralizar valores globales compartidos; decidir cualquier parámetro nuevo antes de introducirlo en más de un módulo |
| `tests/` | Comprobar contratos, evidencia, filtros, aclaraciones, errores y evaluación real |

**No se moverá** la lógica de generación a `retrieve.py` ni la consulta
ChromaDB a `generate.py`. Los agentes posteriores consumirán `responder()`.

## 5. Vía estructurada: resumen de catálogo

### Detección conservadora

Activar esta vía ante peticiones claramente generales: «qué tipos de
residuos», «categorías del catálogo», «lista de destinos de residuos» o
«qué recoge el catálogo». Evitar activarla si la pregunta pide el destino
de **un residuo específico**, aunque contenga «tipo de residuo».

Si la petición es ambigua entre una clasificación y un listado exhaustivo,
la respuesta deberá explicar el alcance del archivo y no fingir una
taxonomía universal.

### Recuperación

- Consultar ChromaDB con `where={"source": "tipos_residuos.csv"}` y
  `include=["documents", "metadatas"]`.
- Validar cada registro, omitir filas incompletas y eliminar duplicados
  accidentales por `source_row` cuando exista.
- Agrupar por el valor **original** de `metadata["destino"]`; no fusionar
  destinos basándose en semejanzas inferidas.
- Calcular número de relaciones y destinos distintos con los datos leídos.
- Conservar `source`, `path` y, cuando proceda, `source_row` para auditoría.
- Producir un resumen compacto con destinos y, si se utilizan, ejemplos de
  residuos tomados literalmente de las filas. No inventar categorías.
- Si el conjunto no cabe en el límite de contexto, **indicar explícitamente
  el recorte** y ofrecer un alcance menor. Nunca etiquetar un resumen
  parcial como completo.

`listar_destinos_residuos()` ya devuelve los destinos distintos y debe
reutilizarse o ampliarse sin romper su contrato público. Un futuro lector
estructurado podrá devolver además agrupaciones y recuentos; **no se
confundirá** esa función con `buscar_chunks_relevantes()`.

### Respuesta y trazabilidad

La salida podrá resumir que el archivo municipal consultado contiene N
relaciones agrupadas en M destinos, mencionar dichos destinos y dar algunos
ejemplos representativos **solo cuando existan en los registros**.

No afirmar «estos son todos los tipos de residuos que reconoce el
Ayuntamiento». Los valores proceden de un CSV concreto del corpus, no de
una clasificación oficial universal.

Si se convierte la agrupación en un fragmento de evidencia para
`generar_respuesta()`, debe quedar **marcado como resumen derivado**, con
`source`/`path` verdaderos y el texto respaldado por los registros
consultados. La cita `[C1]` identificaría **ese resumen agregado**, no un
chunk original inexistente. No inventar identificadores ni URLs de origen.

El resultado público de `responder()` debe conservar `respuesta`,
`abstencion`, `motivo`, `fuentes`, `chunks` y `metricas`; si se añade un
indicador de vía estructurada, deberá ser adicional y probado. En la
respuesta de diagnóstico debe quedar claro qué evidencias son derivadas.

## 6. Vía estructurada: horarios de puntos limpios móviles

### Alcance de los datos

Los registros de `puntos_limpios_moviles.csv` contienen, por parada,
dirección, distrito, día, turno y franja horaria. No debe utilizarse un
horario encontrado en una parada para afirmar el horario de **todas**.
Los móviles 24 horas forman una modalidad distinta.

### Decisiones por intención

| Consulta | Comportamiento previsto |
| --- | --- |
| «¿Qué horario tiene el punto limpio móvil?» | Pedir calle/parada o distrito; no enumerar horas arbitrarias |
| «¿Qué horario tiene el punto limpio móvil de [distrito]?» | Si hay varias paradas, pedir una parada concreta o presentar opciones identificadas; no escoger la primera |
| «¿Qué horario tiene la parada de [dirección]?» | Buscar registros coincidentes y contestar con dirección, día y horario reales |
| «¿Qué horarios tienen los puntos limpios móviles en [distrito]?» | Permitir relación de paradas del distrito, cada una con dirección, día y horario, si la evidencia cabe |
| «¿Qué horarios tienen todos los móviles de Madrid?» | No afirmar exhaustividad con cinco chunks; pedir acotar por distrito o disponer de salida paginada en una fase posterior |
| Pregunta conceptual «¿Qué es un punto limpio móvil?» | Conservar búsqueda semántica normal |
| Móvil 24 horas / fijo / proximidad | Aplicar la modalidad solicitada; no mezclar horarios de fuentes diferentes |

### Recuperación y selección

- Usar `document_type="punto_limpio_movil"` y, cuando proceda,
  `district` con sus variantes originales de mayúsculas, tildes y guiones.
- No suponer que `district` identifica una parada única: hay múltiples
  registros por distrito.
- La dirección se extraerá del **texto del registro**, pues no aparece
  como campo `address` en los metadatos actuales de móviles.
- Leer también `day` y la línea `Horario:` del texto, conservando
  exactamente las franjas documentadas.
- Para coincidencias por dirección, normalizar para comparar, pero
  devolver nombres y horarios originales. Una coincidencia insuficiente
  no debe convertirse en una parada elegida por conjetura.
- Si hay varias paradas o días compatibles, incluirlos claramente o
  pedir precisión. Si no hay coincidencia, no buscar otra ubicación
  sustituta de forma silenciosa.
- No presentar la programación del CSV como disponibilidad en tiempo real.

### Aclaración sin Gemini

Cuando falte un dato decisivo, `responder()` podrá devolver:

```python
{
    "respuesta": (
        "¿En qué distrito o parada de Madrid quieres consultar "
        "el horario del punto limpio móvil?"
    ),
    "abstencion": True,
    "motivo": "ambiguedad",
    "fuentes": [],
    "chunks": [],
    "metricas": {...},
}
```

Se mantendrá la validación de preguntas y `top_k`, también en el camino
que no utiliza Gemini. Los límites compartidos deben provenir de
`config.py` cuando su uso abarque varios módulos.

## 7. Compatibilidad y errores

1. Mantener la firma actual de `responder(pregunta, *, top_k=TOP_K,
   query_embedding=None, client=None)`.
2. Conservar la semántica de `generar_respuesta()` y de su validador de
   citas, incluyendo `SalidaGeneracionInvalida`.
3. En preguntas de un residuo concreto y de un punto fijo concreto,
   mantener retrieval vectorial con K=5 como referencia.
4. Consultas estructuradas a ChromaDB: errores técnicos explícitos,
   distintos de `sin_contexto` y `evidencia_insuficiente`.
5. Si una consulta es una aclaración determinista, no generar embeddings
   ni consumir Gemini.
6. Si un resumen estructurado llega al generador, sus fuentes deben
   tener procedencia auditable; una salida inválida sigue siendo error.
7. Mantener los clientes y dependencias actuales: no añadir bibliotecas.
8. Evitar introducir reglas de detección dispersas o duplicadas en
   varios módulos.

## 8. Criterios de aceptación

- [ ] P09, sin ubicación, solicita el dato necesario y no mezcla paradas.
- [ ] Una parada única identificada devuelve dirección, día y horario
      sustentados por el registro correcto.
- [ ] Una consulta por distrito no atribuye a todas las paradas el
      horario de una sola.
- [ ] Móviles ordinarios, móviles 24 h, fijos y proximidad no se mezclan.
- [ ] P10 no se contesta usando arbitrariamente cinco filas; utiliza
      información agregada de la fuente apropiada.
- [ ] El resumen del catálogo diferencia destinos de categorías y
      declara con precisión su alcance.
- [ ] Las fuentes proceden de ChromaDB y las citas son verificables.
- [ ] `responder()` conserva la interfaz estable para el futuro agente.
- [ ] No se repiten embeddings si `query_embedding` se proporciona.
- [ ] Ningún nuevo flujo omite la validación de entradas o errores.
- [ ] Siguen superándose las 158 pruebas existentes, adaptando solamente
      las que deban incorporar los comportamientos nuevos.
- [ ] Existen pruebas nuevas sobre P09, P10, distritos y registros vacíos.
- [ ] Se repite la evaluación real con 14 preguntas y se revisa P03,
      cuyo fallo de salida se detectó antes de la corrección de citas.

## 9. Plan de implementación, un archivo por paso

1. Revisar si es necesario incorporar al `config.py` parámetros
   **compartidos** nuevos; modificarlo en un paso propio si procede.
2. Ampliar `src/retrieve.py` con lectura estructurada del catálogo y
   de paradas, sin alterar la búsqueda vectorial existente.
3. Ampliar `tests/test_retrieve.py` para proteger filtros, agrupaciones
   y procedencia.
4. Ampliar `src/rag.py` con selección conservadora de intención y
   petición de aclaraciones, reutilizando las funciones de retrieval.
5. Ampliar `tests/test_rag.py` y, si hiciera falta, el evaluador en
   pasos separados.
6. Ejecutar las 158 pruebas existentes, más las nuevas, y una
   reevaluación controlada de las 14 preguntas.

Cada paso requerirá revisión y confirmación antes de avanzar. No se
realizarán cambios en GitHub ni en archivos adicionales durante esta
fase de especificación.

## 10. Riesgos que deben comprobarse en la auditoría global

- Uso duplicado de constantes o límites entre `config.py`, retrieval y
  generación.
- Diferencias entre un chunk vectorial y evidencia estructurada derivada.
- Coherencia de métricas cuando se responde sin Gemini.
- Rendimiento de lecturas completas de ChromaDB.
- Criterios de pertinencia en consultas ambiguas y fuera de dominio.
- Exactitud, legibilidad y no duplicación de citas y fuentes.
- Posibles cambios en datos, horarios y cobertura del corpus.
- Cumplimiento integral del enunciado PB02, incluido supervisor,
  especialistas, herramientas autorizadas e HITL.
