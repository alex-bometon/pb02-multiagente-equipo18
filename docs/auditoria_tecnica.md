# Auditoría técnica del PB02 ReciclaTIA

**Fecha:** 20-10-2026. **Alcance:** código RAG y requisitos PB01/PB02
anteriores a Streamlit. **Contexto:** proyecto individual. Se consultaron los
enunciados del repositorio y la rama `feature/rag`, junto con las versiones
locales más recientes de `retrieve.py`, `generate.py`, `rag.py` y las pruebas.
No se han realizado commits, push ni modificaciones remotas.

## Dictamen

El RAG tiene una base funcional aprovechable: corpus municipal CSV/HTML,
712 chunks en evaluaciones previas, ChromaDB persistente, recuperación
semántica y estructurada, Gemini, citas, abstención y `responder()` sin UI.
Las evaluaciones previas recuperaron las fuentes esperadas de 11/11 preguntas
respondibles a K=5 y validaron las consultas estructuradas (177 relaciones,
19 destinos). Esta cifra **no equivale** a precisión factual universal.

PB02 como conjunto está **incompleto**, incluso excluyendo Streamlit.
No constan supervisor LangGraph, ≥2 especialistas, síntesis, aplicar,
`MemorySaver`/`thread_id`, HITL, API externa con fallback, auxiliar,
multimodal, allowlist, límites del grafo, CLI interactiva ni trazas del grafo.
No se deben declarar superados antes de su implementación y evaluación.

## Requisitos y verificabilidad

| Requisito | Estado | Evidencia o acción |
|---|---|---|
| Corpus ≥2 formatos | Cumplido en el RAG | CSV + HTML; documentar URLs, fechas y licencias |
| Pipeline load → clean/chunk → embed → index | Implementado | `src/ingest.py`, `embed.py`, `index.py` |
| ChromaDB persistente, metadatos `source` | Implementado | Verificado antes; revalidar índice local tras instalar cambios |
| Retrieval top-k y comparación K | Evaluado | K=3/5/8/10 y 14 consultas |
| Generación fundamentada, abstención | Evaluado | `generate.py`, tests y evaluación real previa |
| API sin UI | Implementada | `src.rag.responder()` |
| CLI del RAG (`--index`, `--query`, `--ask`) | Pendiente | Falta `main.py` |
| Tests del pipeline offline | Cobertura insuficiente | No se han confirmado tests aislados de `clean.py`, `embed.py`, `index.py` |
| Supervisor y ≥2 especialistas | Pendiente | LangGraph todavía no desarrollado |
| Síntesis + aplicar + HITL | Pendiente | Debe interrumpir ANTES de aplicar |
| `MemorySaver` + `thread_id` | Pendiente | Verificar continuidad de varios turnos |
| API externa + fallback, auxiliar, multimodal | Pendiente | Tools reales e implementadas |
| Allowlist y `max_steps`/límite Python | Pendiente | El LLM no puede decidir por sí solo lo ejecutable |
| Log de tools, supervisor, HITL, `done` | Pendiente | Debe ser visible en CLI y Streamlit |
| CLI multiagente con HITL | Pendiente | Aprobar/rechazar manualmente |
| Streamlit | Deliberadamente pendiente | Excluido de esta auditoría |
| `.env.example`, `.gitignore`, dependencias | Presentes | Verificar versiones en entorno de entrega |
| README | Incompleto antes de auditoría | Se propone reemplazo honesto y operativo |
| Informe de decisiones y presentación | Pendientes | Entregables obligatorios |
| PR por miembro y temática permitida | Pendiente de acuerdo docente | Adaptación explícita por trabajo individual |

## Complejidad y modularidad

| Archivo | Antes | Después | Evaluación |
|---|---:|---:|---|
| `src/retrieve.py` | 720 | ~409 | Separada consulta estructurada |
| `src/generate.py` | 409 | ~255 | Separadas validación/citas y acceso LLM |
| `src/rag.py` | 452 | ~277 | Separado el enrutamiento por intención |
| `tests/test_retrieve.py` | 964 | ~489 | Añadido test estructurado separado |
| `tests/test_rag.py` | 1153 | ~339 | Separadas rutas estructuradas y dominio |
| `src/clean.py` | 846 | Sin cambios | Candidato relevante para refactor posterior |
| `tests/test_generate.py` | 651 | Sin cambios | Extenso pero con casos parametrizados |

La longitud no es una métrica suficiente por sí sola. La refactorización ha
buscado **una responsabilidad por módulo** y compatibilidad pública, no
fragmentar cada función en un archivo.

### Nuevos módulos y razón de ser

- `retrieve_structured.py`: catálogo y paradas (ChromaDB sin embeddings).
- `generation_validation.py`: estructura JSON, citas, abstención legítima.
- `rag_intents.py`: detección, ubicación y dominio de consultas.
- `domain.py`: ciudades ajenas a Madrid reconocidas explícitamente.
- `text_utils.py`: normalizaciones compartidas de texto y distrito.
- `input_validation.py`: una única validación de pregunta común.
- `tests/conftest.py`: fixtures compartidas de pruebas RAG.
- Tests separados: semántica, tablas, orquestación y dominio.

**Contratos conservados:** `src.retrieve.buscar_chunks_relevantes`,
`consultar_catalogo_residuos`, `consultar_paradas_moviles`,
`src.generate.generar_respuesta` y `src.rag.responder`. Se conservan
alias necesarios para los tests anteriores. No se cambian embeddings,
metadatos persistidos, chunking, el esquema de respuesta ni el modelo.

### Duplicaciones eliminadas

- Validación idéntica de preguntas en tres módulos → `input_validation.py`.
- Normalización de distritos/texto del núcleo de retrieval → `text_utils.py`.
- Reglas de ciudades conocidas, antes alojadas dentro de generación →
  `domain.py` (sin importar el cliente Gemini para detectarlas).
- Composición de módulos mezclada con lectura estructurada → separación.

**No se han unificado indiscriminadamente** las dos detecciones de catálogo:
una decide la vía estructurada en `rag_intents.py` y la otra construye un
filtro conservador en retrieval semántico; tienen contratos diferentes.

## Verificación realizada

En un entorno aislado, con sustitutos para Gemini/ChromaDB y el código local
que refleja las versiones actuales:

```
268 passed, 3 skipped
```

La cifra comprende una regresión adicional para `source_row` repetido en
destinos diferentes. Se ha retirado el `xfail` del singular «celular».
**No se ha ejecutado aquí la integración con el índice local del usuario** ni
una evaluación de 14 consultas tras instalar todos los archivos. Eso debe
verificarse antes de dar por aceptada la migración.

## Hallazgos pendientes y riesgos

1. **Pipeline offline:** `clean.py` (846 líneas) concentra limpieza HTML y
   transformaciones de muchas variantes de CSV. Es el siguiente candidato
   a dividir en `clean_html.py`, `clean_csv.py` y un coordinador, **pero no
   debe tocarse sin tests de equivalencia de los 712 chunks**. Separarlo
   sin pruebas pondría en riesgo los embeddings y el índice existente.
2. **No hay tests confirmados del pipeline offline**, hashes ni snapshot
   de salida reproducible. Prioridad alta antes del cierre.
3. **Dominio:** detectar unas pocas ciudades externas no equivale a un
   clasificador geográfico completo. Deben mantenerse abstenciones y tests.
4. **Catálogo:** `tipos_residuos.csv` registra destinos; no es una taxonomía
   oficial exhaustiva. Los resúmenes deben describirse como derivados.
5. **Citas:** verifican referencias válidas, no todas las inferencias
   factuales. Revisar grounding con un conjunto de ejemplos difíciles.
6. **Persistencia/eficiencia:** la lista de distritos puede releerse del
   índice por consulta. Medir antes de introducir caché, sin romper la
   detección de cambios del corpus.
7. **Dependencias:** el fichero incluye paquetes para funcionalidades aún
   no implementadas. Validar instalación en una máquina limpia y eliminar
   dependencias no usadas tras terminar LangGraph y la UI.
8. **Secretos y datos:** `.env` está excluido, pero conviene escanear
   historial Git y evitar trazas con datos personales. `.gitignore` se
   endurece para excluir artefactos de embeddings/índice.
9. **Git:** ignorar un archivo no deja de seguirlo si estaba versionado.
   Revisar `git ls-files output` antes de limpiar artefactos locales;
   no borrar datos del usuario sin copia ni confirmación.
10. **Trabajo individual:** pedir al profesorado confirmación de
    evaluación de PRs por miembro y de la temática reciclaje.
11. **Software libre:** todavía no hay licencia explícita para el código;
    Gemini no es un modelo libre/local y puede tener costes o límites.
    Decidir licencia y estrategia de proveedor sin confundir ambos planos.
12. **Reproducibilidad:** el README original solo contenía el título del
    proyecto; se propone uno ajustado a lo implementado, sin anunciar
    funciones de agentes que todavía no existen.

## Siguientes hitos sin retrabajo

1. Instalar el paquete de refactorización completo como unidad, sustituir
   los archivos indicados y **eliminar** los tests que se hayan dividido
   únicamente cuando se confirme la copia (los nombres originales se
   conservan para las partes centrales).
2. Ejecutar `python -m pytest tests/ -q --tb=short` en el entorno local:
   deben ejecutarse los tres tests de integración con ChromaDB que aquí
   han quedado omitidos.
3. Repetir `python tests/evaluate_rag.py --ids 9 10 14` y un caso con
   Gemini de forma controlada para detectar problemas de imports/SDK.
4. Especificar el grafo antes de implementarlo: estado, supervisor,
   especialistas, síntesis, aplicar, allowlist y HITL real.
5. Crear CLI funcional sobre el mismo contrato que utilizará Streamlit.
6. Añadir API externa con fallback, herramienta auxiliar, multimodal,
   checkpointer/thread_id, límites Python y trazas comprobables.
7. Añadir pruebas/snapshots del pipeline offline y, entonces, decidir
   si dividir `clean.py` sin alterar los 712 chunks.
8. Preparar memoria, demo de HITL aprobado/rechazado, 3 fallos,
   presentación y revisión final de secretos/licencias.

## Conclusión

La refactorización es conservadora y facilita la próxima fase sin reescribir
el RAG. **No se debe etiquetar PB02 como finalizado** por haber superado
pruebas del RAG: los requisitos diferenciales de agentes, HITL, tools,
persistencia y CLI siguen pendientes. Es preferible implementar esas piezas
con contratos pequeños que seguir ampliando `rag.py` indiscriminadamente.
