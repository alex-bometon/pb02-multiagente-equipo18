"""Adaptador de consola del RAG; no contiene lógica de recuperación."""

import sys
from argparse import Namespace


def _mostrar_chunks(chunks: list[dict]) -> None:
    """Imprime evidencia, procedencia y distancia cuando existe."""
    if not chunks:
        print("No se encontraron chunks.")
        return
    for numero, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        fuente = metadata.get("source", "desconocida")
        ruta = metadata.get("path", "")
        distancia = chunk.get("distance")
        print(f"\n[{numero}] Fuente: {fuente}")
        if ruta:
            print(f"    Ruta: {ruta}")
        if distancia is not None:
            print(f"    Distancia coseno: {distancia:.4f}")
        print(f"    Texto: {chunk.get('text', '')}")


def _consultar(args: Namespace) -> int:
    from src.retrieve import buscar_chunks_relevantes

    chunks = buscar_chunks_relevantes(args.pregunta, top_k=args.top_k)
    print(f"Chunks recuperados: {len(chunks)} (top_k={args.top_k})")
    _mostrar_chunks(chunks)
    return 0


def _responder(args: Namespace) -> int:
    from src.rag import responder

    resultado = responder(args.pregunta, top_k=args.top_k)
    print(resultado["respuesta"])
    print(f"\nAbstención: {resultado['abstencion']}")
    if resultado.get("motivo"):
        print(f"Motivo: {resultado['motivo']}")
    fuentes = resultado.get("fuentes") or []
    if fuentes:
        print("\nFuentes:")
        for fuente in fuentes:
            print(
                f"  [{fuente.get('chunk_id', '?')}] "
                f"{fuente.get('source', 'desconocida')} "
                f"({fuente.get('path', '')})"
            )
    else:
        print("\nFuentes: ninguna")

    metricas = resultado.get("metricas") or {}
    print(
        "\nMétricas: "
        f"{metricas.get('chunks_recuperados', 0)} chunks; "
        f"{metricas.get('total_ms', '?')} ms; "
        f"modelo {metricas.get('modelo_configurado', '?')}"
    )
    if args.contexto:
        _mostrar_chunks(resultado.get("chunks") or [])
    return 0


def _indexar(args: Namespace) -> int:
    if not args.yes:
        print(
            "No se ha modificado el índice. Usa --yes para autorizar "
            "la reconstrucción; --full vuelve a generar embeddings "
            "con posible consumo de cuota Gemini.",
            file=sys.stderr,
        )
        return 2

    if args.full:
        from src.ingest import ejecutar_ingesta
        from src.embed import ejecutar_embeddings

        print("Preparando corpus y embeddings...")
        ejecutar_ingesta()
        ejecutar_embeddings()

    from src.index import ejecutar_indexacion

    print("Reconstruyendo ChromaDB desde embeddings.json...")
    ejecutar_indexacion()
    return 0


def ejecutar_rag(args: Namespace) -> int:
    """Despacha comandos del RAG; interfaz reutilizable desde main.py."""
    acciones = {
        "query": _consultar,
        "ask": _responder,
        "index": _indexar,
    }
    return acciones[args.accion](args)
