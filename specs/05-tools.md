
# 05 — Herramientas de ReciclaTIA PB02

## 1. Objetivo

Integrar cuatro herramientas independientes y reutilizables para que el sistema multiagente de ReciclaTIA pueda consultar información sobre gestión de residuos de Madrid ciudad.

Las herramientas deben funcionar sin LangGraph ni Streamlit y conservar contratos de entrada y salida verificables.

## 2. Arquitectura

El módulo `src/tools/` contiene:

| Archivo | Herramienta | Función |
|---|---|---|
| `rag_tool.py` | `consultar_rag` | Consultar el RAG |
| `hora_actual.py` | `hora_actual` | Obtener fecha y hora |
| `describir_imagen.py` | `describir_imagen` | Describir objetos mediante Gemini |
| `api_puntos_limpios.py` | `buscar_puntos_limpios_cercanos` | Localizar puntos limpios |

El RAG mantiene su implementación independiente en `src/rag.py`.

La herramienta geográfica reutiliza `src/retrieve.py` y `src/retrieve_structured.py` para acceder al corpus de ChromaDB.

Un registro central proporciona la allowlist y la ejecución controlada de herramientas.

## 3. Contratos de las herramientas

### 3.1. Consulta RAG

`consultar_rag(pregunta: str) -> dict`

- Delega en `src.rag.responder(pregunta)`.
- Conserva la respuesta estructurada, fuentes, citas y métricas.
- Respeta las abstenciones cuando no hay evidencia suficiente.
- No reproduce ni modifica la lógica de recuperación o generación.
- Propaga los errores técnicos sin transformarlos en respuestas inventadas.

### 3.2. Fecha y hora

`hora_actual() -> str`

- Devuelve fecha y hora ISO 8601 con zona horaria.
- Utiliza `Europe/Madrid` como zona configurada.
- No necesita HTTP ni Gemini.
- No deduce que una instalación está abierta únicamente a partir de la hora actual.

### 3.3. Descripción de imágenes

`describir_imagen(ruta_imagen: str) -> str`

- Utiliza Gemini para describir objetos y materiales visibles.
- Acepta archivos JPEG y PNG ubicados dentro de `assets/`.
- Valida ruta, extensión, formato real, tamaño y dimensiones.
- Admite imágenes de hasta 5 MiB y 16 millones de píxeles.
- Expresa incertidumbre cuando el material no puede identificarse.
- No determina el contenedor sin consultar evidencia del RAG.
- Reutiliza `src.gemini_client.crear_cliente_gemini()`.

### 3.4. Búsqueda geográfica

`buscar_puntos_limpios_cercanos(latitud, longitud, radio_metros=2000, tipo="todos") -> dict`

Fuente principal:

`https://sigma.madrid.es/hosted/rest/services/RESIDUOS/PUNTOS_LIMPIOS/MapServer/5/query`

- Recibe coordenadas WGS84 (EPSG:4326).
- Admite `todos`, `fijo`, `movil` y `proximidad`.
- Utiliza un radio predeterminado de 2.000 metros y máximo de 5.000.
- Consulta los registros mediante ArcGIS REST.
- Normaliza direcciones, modalidades, horarios y coordenadas.
- Calcula distancias geográficas con Haversine.
- Ordena los resultados de menor a mayor distancia.
- Devuelve hasta 10 puntos, conservando el total de coincidencias.

Ante un error técnico del servicio externo, utiliza los registros locales de ChromaDB. Una respuesta válida sin coincidencias no activa el respaldo.

El corpus local distingue también los puntos móviles de 24 horas. Cualquier diferencia de cobertura entre fuentes debe identificarse sin presentar modalidades o datos exclusivos del corpus como resultados de la API.

El resultado incluye:

```python
{
    "fuente": "api" | "corpus_local",
    "tipo_solicitado": str,
    "radio_metros": int,
    "total_encontrados": int,
    "puntos": list,
    "aviso": str | None,
}
```

Cada punto contiene coordenadas, distancia en metros e información disponible sobre su ubicación y funcionamiento.

No se calculan rutas, tiempos de desplazamiento ni disponibilidad en tiempo real.

## 4. Configuración

`config.py` es la fuente central de parámetros operativos del proyecto.

Incluye:

- Rutas y directorios.
- Modelos Gemini y parámetros de inferencia.
- Zona horaria.
- URLs, timeouts y reintentos.
- Radios y límites de resultados.
- Tamaño y formatos admitidos para imágenes.
- Límites y presupuestos de ejecución.

Los módulos importan estos parámetros sin duplicar sus valores.

Las reglas del dominio, algoritmos, esquemas de datos, correspondencias de campos y prompts permanecen en sus módulos correspondientes.

Las credenciales se obtienen mediante variables de entorno; no se almacenan en `config.py`.

## 5. Ejecución controlada

Las herramientas se registran mediante `@tool` de LangChain.

El archivo `src/tools/__init__.py` define:

- `TOOLS`: lista de herramientas disponibles.
- `TOOL_BY_NAME`: registro y allowlist por nombre.
- `ejecutar_tool(nombre, args)`: ejecución de herramientas autorizadas con validación de argumentos y gestión de errores.

Las herramientas desconocidas y los argumentos no permitidos se rechazan antes de ejecutar.

Los límites de iteraciones, el control de llamadas y las trazas corresponden al futuro grafo LangGraph, siguiendo el patrón de los Sprints 12 y 13.

Las herramientas funcionan independientemente del grafo y no dependen de Streamlit.

## 6. Alcance

Las herramientas proporcionan capacidades al futuro grafo multiagente, pero no contienen su lógica de orquestación.

La selección de especialistas, memoria conversacional, HITL y presentación de resultados corresponden a otras capas.

La geolocalización voluntaria del usuario se implementará en la interfaz y proporcionará coordenadas a la herramienta geográfica existente.

La cobertura de datos municipales se limita a Madrid ciudad.

## 7. Criterios de aceptación

1. Las cuatro herramientas funcionan independientemente del grafo.
2. El RAG conserva citas, abstenciones y errores técnicos.
3. La descripción de imágenes valida los archivos antes de invocar Gemini.
4. La API geográfica filtra, normaliza, calcula distancias y ordena correctamente.
5. Los fallos del servicio externo activan el respaldo local identificado.
6. Una consulta válida sin resultados no se confunde con un error técnico.
7. Los parámetros operativos se gestionan desde `config.py`.
8. La allowlist bloquea herramientas y argumentos no autorizados.
9. Los presupuestos impiden superar el máximo de ejecuciones permitido.
10. Las trazas identifican las operaciones sin exponer información sensible.
11. Los tests funcionan con mocks, sin depender de red ni cuotas externas.
12. La suite completa del RAG continúa funcionando sin regresiones.
13. Las pruebas de integración reales se realizan por separado.
14. No se modifican los embeddings ni el índice existente para integrar las herramientas.
