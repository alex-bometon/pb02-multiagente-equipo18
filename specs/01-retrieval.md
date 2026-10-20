
# 01 — Retrieval de ReciclaTIA

## 1. Objetivo

Implementar desde cero la recuperación semántica
de documentos del corpus de ReciclaTIA.

El módulo debe ser independiente de la generación,
de LangGraph y de Streamlit.

## 2. Arquitectura

Pregunta del usuario
    ↓
Validación
    ↓
Embedding de la pregunta (Gemini)
    ↓
Consulta vectorial a ChromaDB
    ↓
Recuperación top-k
    ↓
Chunks + metadatos + distancias

## 3. Configuración

- Modelo: gemini-embedding-2
- Dimensiones: 768
- Colección: residuos_madrid
- Distancia: coseno
- TOP_K por defecto: 5
- Longitud máxima propuesta: 500 caracteres
- K máximo propuesto: 10

Los límites serán aplicados por Python.

## 4. Módulo

Archivo: src/retrieve.py

Función pública principal:

buscar_chunks_relevantes(
    pregunta: str,
    top_k: int = TOP_K,
    *,
    query_embedding: list[float] | None = None,
) -> list[dict]

El embedding precalculado es opcional y
permite evaluar sin repetir llamadas a Gemini.

## 5. Contrato de salida

Cada resultado contendrá:

{
    "text": str,
    "distance": float,
    "metadata": dict
}

Los resultados estarán ordenados desde
la menor distancia hasta la mayor.

Se conservarán los metadatos originales,
especialmente source y path.

## 6. Validaciones

- Rechazar preguntas vacías.
- Rechazar preguntas demasiado largas.
- Rechazar valores de K no válidos.
- Comprobar la dimensión de los vectores.
- Detectar colecciones inexistentes o vacías.
- Gestionar explícitamente errores de Gemini.
- No inventar resultados si falla ChromaDB.

## 7. Alcance

Incluye:
- Embeddings de consultas.
- Recuperación semántica top-k.
- Metadatos y distancias.
- Reutilización de embeddings precalculados.
- Pruebas independientes.

No incluye:
- Reranking híbrido.
- Generación de respuestas.
- Prompts del LLM.
- Integración con LangGraph.
- Interfaz Streamlit.

## 8. Criterios de aceptación

1. Recuperar documentos de ChromaDB.
2. Respetar TOP_K.
3. Devolver texto, distancia y metadatos.
4. Validar preguntas vacías y límites.
5. Permitir pruebas sin llamadas a Gemini
   mediante embeddings precalculados.
6. Mostrar fuentes recuperadas.
7. Probar preguntas dentro y fuera del corpus.
8. No modificar el índice persistente.
