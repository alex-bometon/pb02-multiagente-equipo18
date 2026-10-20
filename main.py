"""Punto de entrada CLI de ReciclaTIA (RAG y futuros agentes)."""

import argparse
import sys

from config import TOP_K
from src.cli_rag import ejecutar_rag


def crear_parser() -> argparse.ArgumentParser:
    """Define la interfaz sin cargar ChromaDB ni Gemini."""
    parser = argparse.ArgumentParser(
        description="ReciclaTIA — asistente de residuos de Madrid"
    )
    secciones = parser.add_subparsers(dest="seccion", required=True)
    rag = secciones.add_parser("rag", help="Consultar o preparar el RAG")
    acciones = rag.add_subparsers(dest="accion", required=True)

    consulta = acciones.add_parser(
        "query", help="Mostrar los chunks recuperados sin generar respuesta"
    )
    consulta.add_argument("pregunta", help="Pregunta sobre residuos")
    consulta.add_argument("--top-k", type=int, default=TOP_K)

    respuesta = acciones.add_parser(
        "ask", help="Responder con el RAG y mostrar fuentes"
    )
    respuesta.add_argument("pregunta", help="Pregunta sobre residuos")
    respuesta.add_argument("--top-k", type=int, default=TOP_K)
    respuesta.add_argument(
        "--contexto", action="store_true", help="Mostrar los chunks usados"
    )

    indice = acciones.add_parser(
        "index", help="Reconstruir ChromaDB desde embeddings.json"
    )
    indice.add_argument(
        "--full", action="store_true",
        help="Rehacer además chunks y embeddings (usa la API de Gemini)",
    )
    indice.add_argument(
        "--yes", action="store_true",
        help="Autorizar expresamente la reconstrucción del índice",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Interpreta argumentos y delega; nunca implementa el RAG."""
    args = crear_parser().parse_args(argv)
    try:
        if args.seccion == "rag":
            return ejecutar_rag(args)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
