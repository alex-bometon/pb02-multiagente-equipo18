# 08. Plan de pruebas y criterios de aceptación

## Estrategia

Separamos las pruebas **unitarias**, la **integración interna** y la **comprobación real de servicios externos**. Una suite que utiliza mocks puede verificar contratos y límites, pero no acredita la disponibilidad actual de Gemini o ArcGIS ni la corrección factual de todas las respuestas.

No hay una cifra de pruebas superadas fija en esta especificación. Para cada entrega se registra el resultado obtenido al ejecutar los comandos sobre el código concreto que se presenta.

## Comandos de comprobación

```bash
python -m py_compile app.py
python -m pytest tests/ -q --tb=short
python tests/evaluate_retrieve.py
python tests/evaluate_rag.py --ids 9 10 14
```

Las dos evaluaciones finales requieren revisar sus opciones antes de ejecutarlas: pueden necesitar el índice local y consumir cuota de Gemini. Para pruebas puntuales también pueden utilizarse los comandos `python main.py rag query`, `rag ask` y `agente demo`.

## Matriz por componente

| Componente | Archivos de prueba | Qué debe acreditarse |
|---|---|---|
| Retrieval semántico | `test_retrieve.py`, `evaluate_retrieve.py` | Filtros, top-k, metadatos y compatibilidad de embeddings |
| Retrieval estructurado | `test_retrieve_structured.py`, `test_rag_structured.py` | Catálogo y horarios sin mezclar registros |
| Generación | `test_generate.py`, `test_rag.py` | Estructura, citas reales, abstención y errores |
| Dominio | `test_rag_domain.py` | No extrapolar Madrid a otros municipios |
| Tools | `test_tools_registry.py`, `test_api_puntos_limpios.py`, `test_describir_imagen.py`, `test_hora_actual.py`, `test_rag_tool.py` | Allowlist, argumentos, formatos, origen y fallback |
| Agente y HITL | `test_agent_regression.py` | Rutas, presupuestos, aislamiento, aprobación y rechazo |
| Horarios | `test_agent_schedule.py`, `test_agent_schedule_integration.py` | Interpretación de intervalos, priorización y límites de evidencia |
| Composición final | `test_agent_resultado.py` | Clasificación + instalaciones + condiciones disponibles |
| Terminal | `test_cli.py` | Comandos, parámetros y errores |
| Web | Comprobación manual de `app.py` | Carga, estado, permisos, plan, botones, fuentes y mapa |

## Escenarios de integración

| Caso | Entrada de ejemplo | Condición de aceptación |
|---|---|---|
| Residuo común | «¿Dónde tiro una botella de plástico?» | Respuesta sustentada y plan HITL |
| Fuera de dominio | «¿Dónde reciclo esto en Barcelona?» | No presentar reglas de Madrid como aplicables |
| Puntos sin ubicación | «Busca puntos limpios cerca de mí» | Solicitar coordenadas; no inventarlas |
| Puntos con ubicación | Coordenadas voluntarias + consulta de proximidad | Fuente API o respaldo identificada, distancia en línea recta |
| Pedido mixto | Residuo + punto limpio para depositarlo | Dos especialidades cuando corresponde; compatibilidad no inferida sin fuentes |
| Imagen | JPG/PNG válida en `assets/` | Descripción visual; recomendación documental solo si el RAG la respalda |
| Imagen inválida | Formato/archivo fuera de límites | Rechazo previo a la invocación visual |
| HITL aprobado | Aprobar plan existente | `done=True` solo después de ejecutar `aplicar` |
| HITL rechazado | Rechazar y reformular | `done=False`, plan retirado, nuevo turno en el mismo hilo |
| Límite de herramientas | Forzar llamadas superiores al presupuesto | Bloqueo de las que superan el límite y traza correspondiente |
| Datos horarios | Consulta de punto con horario publicado | Distinguir dentro/fuera/desconocido; no asegurar apertura real |
| Caída de ArcGIS | Respuesta HTTP fallida simulada | Identificar `corpus_local`, no hacerlo pasar por `api` |

## Casos de fallo que deben revisarse

**Catálogo incompleto.** Una búsqueda top-k puede devolver un subconjunto de `tipos_residuos.csv` que parezca exhaustivo. La vía estructurada evita confundir cinco resultados cercanos con todos los destinos del corpus.

**Horarios asociados a una parada equivocada.** Un resumen general puede mezclar paradas móviles distintas. La consulta estructurada debe asociar cada franja con su registro concreto, o pedir precisión.

**Texto generado sin citas válidas.** Un modelo puede devolver JSON mal formado o referencias inexistentes. La validación debe impedir una respuesta afirmativa sin soporte.

Estos casos son **riesgos y escenarios de regresión identificados**. El informe de entrega solo puede presentarlos como fallos efectivamente observados cuando se adjunten la entrada, la salida, la corrección y el resultado de una nueva prueba.

## Evidencia de ejecución

Al registrar un resultado de pruebas, incluir: comando utilizado, rama y commit evaluados, número de pruebas ejecutadas, fallidas y omitidas, disponibilidad del índice, si hubo llamadas externas, entrada del caso manual, salida significativa y traza de la herramienta. Nunca exponer la API key ni información privada dentro de los logs o capturas.

## Aceptación

La entrega se considera técnicamente revisada cuando los comandos de instalación, indexado, CLI y Streamlit funcionan en la misma rama; la suite automatizada no presenta fallos sin explicar; se han comprobado aprobación y rechazo; las fuentes se muestran correctamente; y las limitaciones externas se comunican sin simular pruebas que no se han ejecutado.
