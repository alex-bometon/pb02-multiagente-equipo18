# 02 — Mejora del retrieval de ReciclaTIA

**Proyecto:** PB02 Multiagente — ReciclaTIA  
**Estado:** propuesta de especificación (SDD), pendiente de implementar  
**Archivo de código previsto:** `src/retrieve.py`

## 1. Contexto y evidencia

La V1 dispone de una colección ChromaDB persistente, `residuos_madrid`, con 712 chunks, embeddings de `gemini-embedding-2` de 768 dimensiones y distancia coseno. `buscar_chunks_relevantes()` devuelve texto, distancia y metadatos, con `TOP_K=5` por defecto.

En la evaluación PB02 de 14 consultas (11 respondibles y tres fuera del dominio), los aciertos **por archivo fuente** fueron:

| K | Aciertos | Porcentaje |
| --- | ---: | ---: |
| 3 | 8/11 | 72,73 % |
| 5 | 9/11 | 81,82 % |
| 8 | 9/11 | 81,82 % |
| 10 | 9/11 | 81,82 % |

Fallos con K=5:

- **P08:** «¿Dónde hay un punto limpio fijo en el distrito de Arganzuela?» Se recuperaron puntos móviles o de proximidad, no la fuente de puntos fijos.
- **P10:** «¿Qué tipos de residuos reconoce el Ayuntamiento de Madrid?» No se recuperó `tipos_residuos.csv`.

La fuente `tipos_residuos.csv` no representa una taxonomía de tipos general: cada celda no vacía se transforma en una **relación residuo → lugar de depósito**, con metadatos `source`, `document_type="residuo_destino"` y `destino`.

## 2. Objetivo

Mejorar la precisión de la recuperación cuando el usuario menciona explícitamente el tipo de punto limpio, un distrito o el catálogo de residuos, sin cambiar el corpus, los embeddings ni el índice.

Se mantiene la búsqueda vectorial de Gemini + ChromaDB como estrategia por defecto. Los filtros serán **deterministas y conservadores**, basados en metadatos efectivamente presentes en la colección.

## 3. Datos disponibles y reglas de filtrado

Los metadatos se generaron durante la limpieza del corpus. Para esta mejora utilizaremos únicamente campos existentes:

| Intención explícita | Filtro ChromaDB propuesto |
| --- | --- |
| Punto limpio **fijo** | `{"document_type": "punto_limpio_fijo"}` |
| Punto limpio **móvil** | `{"document_type": "punto_limpio_movil"}` |
| Punto limpio **móvil 24 horas** | `{"document_type": "punto_limpio_movil_24h"}` |
| Punto limpio **de proximidad** | `{"document_type": "punto_limpio_proximidad"}` |
| Distrito municipal reconocido, expresamente indicado | `{"district": "ARGANZUELA"}` (ejemplo) |
| Consulta explícita sobre registros del catálogo de residuos | `{"source": "tipos_residuos.csv"}` |

**Combinación de filtros:** cuando se detecten tipo y distrito, aplicar una conjunción `where={"$and": [{"document_type": "punto_limpio_fijo"}, {"district": "ARGANZUELA"}]}` o equivalente soportado por ChromaDB.

**Detección:** normalizar solo el texto de la consulta para identificar expresiones (minúsculas, tildes y espacios), sin alterar los metadatos persistidos. Resolver primero las expresiones más específicas («móvil 24 horas» antes que «móvil»). Usar una lista de distritos válidos obtenida del corpus o explícitamente validada contra él. No interpretar automáticamente barrios como distritos.

**Aplicar filtros solo con evidencia clara en la pregunta.** «¿Dónde hay un punto limpio en Madrid?» seguirá usando búsqueda semántica general. Una coincidencia léxica débil no debe restringir la colección.

## 4. Tratamiento de resultados vacíos, ambiguos y fuera del ámbito

- Si un filtro explícito no produce registros, **no** reemplazarlo silenciosamente por resultados de otra modalidad o distrito. Devolver lista vacía; la posterior capa de respuesta solicitará precisión o reconocerá la ausencia de evidencia.
- Si la consulta mezcla tipos o contiene criterios contradictorios, evitar un filtro arbitrario; registrar internamente la ambigüedad y preservar una recuperación general para posterior validación, o pedir precisión cuando exista una capa conversacional.
- El ámbito del corpus continúa siendo **Madrid ciudad**. Las preguntas sobre Barcelona u otras ciudades no se consideran respondibles porque ChromaDB devuelva fragmentos cercanos.
- Las distancias no constituyen por sí solas un criterio suficiente de pertinencia. No fijar un umbral universal de abstención en `retrieve.py`.

## 5. Consulta de catálogo: alcance real de `tipos_residuos.csv`

Para P10, la recuperación filtrada por `source="tipos_residuos.csv"` permitirá encontrar fragmentos de esa fuente y mejorar la métrica por fuente. **Sin embargo, cinco relaciones individuales no bastan para enumerar de manera exhaustiva su contenido.**

Se propone además una función independiente del top-k, `listar_destinos_residuos()`, que:

1. Lea **solo metadatos** de los registros de `tipos_residuos.csv` mediante ChromaDB (`collection.get` con filtro por fuente).
2. Extraiga los valores no vacíos de `destino`, elimine duplicados y los devuelva ordenados.
3. Indique que se trata de **lugares/canales de depósito que figuran en el CSV**, no de una clasificación oficial exhaustiva de todos los tipos de residuos del Ayuntamiento.
4. No use Gemini ni genere respuestas en lenguaje natural.

Esta función quedará disponible para integrarla posteriormente en el RAG y en su herramienta LangGraph. No deberá confundirse con una búsqueda de un residuo concreto, para la que seguirá usándose top-k vectorial.

## 6. Contratos y compatibilidad

Mantener la firma y el resultado actuales:

```python
buscar_chunks_relevantes(
    pregunta: str,
    top_k: int = TOP_K,
    *,
    query_embedding: list[float] | None = None,
) -> list[dict]
```

Cada resultado seguirá incluyendo `text`, `distance` y `metadata`, en orden de distancia ascendente. `query_embedding` permitirá pruebas sin gasto de API. No cambiar el índice, `config.py`, el nombre de la colección ni la función pública de búsqueda.

La función de catálogo será una **nueva API pública** en `src/retrieve.py` con un tipo de retorno expresado claramente y pruebas específicas.

## 7. Pruebas y criterios de aceptación

**Compatibilidad y regresión**

1. Siguen superándose las 26 pruebas existentes de `tests/test_retrieve.py`.
2. Las consultas ordinarias sin intención de filtro mantienen el comportamiento semántico de la V1.
3. Se conservan los límites de pregunta y top-k y la validación de embeddings de 768 dimensiones.
4. No se hacen nuevas llamadas a Gemini cuando se facilita `query_embedding`.

**Casos dirigidos**

5. P08 devuelve un registro de `puntos_limpios_fijos.csv` correspondiente a **ARGANZUELA**, no uno móvil ni de proximidad. Se valida el **registro correcto**, además de la fuente.
6. P11, sobre el punto limpio fijo de Hortaleza, recupera un registro de `puntos_limpios_fijos.csv` del **distrito HORTALEZA**.
7. P10 identifica `tipos_residuos.csv` en el top-5; `listar_destinos_residuos()` recupera y deduplica los destinos disponibles en sus metadatos.
8. Una petición explícita de punto fijo en un distrito sin coincidencias no devuelve ubicaciones móviles como sustitución.
9. Las tres consultas fuera del dominio no se catalogan como respondibles por el mero hecho de recuperar resultados; la política final se probará con `generate.py`.
10. Añadir paráfrasis de P08, P10 y P11 para comprobar que las reglas no dependen de una única formulación.

**Evaluación**

11. Repetir las 14 preguntas con K=3, 5, 8 y 10. El objetivo en el conjunto fijo es **11/11 fuentes esperadas para K=5**, sin regresiones en las otras nueve preguntas respondibles. Este resultado es una meta de pruebas locales, **no una garantía de generalización ni una medida de corrección factual de respuestas**.
12. Informar por separado: `hit@k` por fuente, coincidencia del distrito y modalidad, resultados vacíos y consultas fuera de dominio.
13. Conservar las distancias y las fuentes para el diagnóstico, sin inferir un umbral de abstención a partir de tres ejemplos.

## 8. Fuera de alcance

- Reranking híbrido por pesos léxicos arbitrarios.
- Regeneración de embeddings, cambio del chunking o reconstrucción de ChromaDB.
- Generación de respuestas, prompts y criterios definitivos de abstención.
- Geolocalización, cálculo de distancias reales, rutas o información de otras ciudades.
- Añadir nuevas bibliotecas a `requirements.txt`.

## 9. Plan de implementación posterior (una modificación por paso)

1. Implementar reglas de identificación de intención y filtros `where` en `src/retrieve.py`, manteniendo su contrato.
2. Añadir en ese mismo módulo la función de agregación de destinos documentada, si se mantiene como requisito aprobado.
3. Ampliar las pruebas de `tests/test_retrieve.py` en un paso independiente.
4. Repetir `tests/evaluate_retrieve.py` y revisar el informe antes de avanzar a generación.

**Condición de cierre:** las mejoras solo se consideran completadas si los casos dirigidos, la batería de regresión y la evaluación empírica son satisfactorios. El código no debe modificarse antes de aceptar esta especificación.
