"""Punto de entrada CLI de ReciclaTIA: RAG y agentes LangGraph."""

from __future__ import annotations

import argparse
import sys

from config import (
    MAX_TURNS_DEFAULT,
    TOP_K,
)
from src.cli_rag import ejecutar_rag


def _agregar_entradas_opcionales(parser: argparse.ArgumentParser) -> None:
    """Define entradas estructuradas compartidas por los modos agente."""
    parser.add_argument(
        "--latitud",
        type=float,
        help="Latitud facilitada voluntariamente para búsqueda geográfica",
    )
    parser.add_argument(
        "--longitud",
        type=float,
        help="Longitud facilitada voluntariamente para búsqueda geográfica",
    )
    parser.add_argument(
        "--imagen",
        help="Ruta de una imagen permitida dentro de assets/",
    )


def crear_parser() -> argparse.ArgumentParser:
    """Define los comandos sin cargar ChromaDB ni Gemini."""
    parser = argparse.ArgumentParser(
        description="ReciclaTIA — asistente de residuos de Madrid"
    )
    secciones = parser.add_subparsers(dest="seccion", required=True)

    # Conservar la interfaz del RAG anterior.
    rag = secciones.add_parser(
        "rag",
        help="Consultar o preparar el RAG",
    )
    acciones_rag = rag.add_subparsers(dest="accion", required=True)

    consulta = acciones_rag.add_parser(
        "query",
        help="Mostrar los chunks recuperados sin generar respuesta",
    )
    consulta.add_argument("pregunta", help="Pregunta sobre residuos")
    consulta.add_argument("--top-k", type=int, default=TOP_K)

    respuesta = acciones_rag.add_parser(
        "ask",
        help="Responder con el RAG y mostrar fuentes",
    )
    respuesta.add_argument("pregunta", help="Pregunta sobre residuos")
    respuesta.add_argument("--top-k", type=int, default=TOP_K)
    respuesta.add_argument(
        "--contexto",
        action="store_true",
        help="Mostrar los chunks usados",
    )

    indice = acciones_rag.add_parser(
        "index",
        help="Reconstruir ChromaDB desde embeddings.json",
    )
    indice.add_argument(
        "--full",
        action="store_true",
        help="Rehacer también chunks y embeddings (usa Gemini)",
    )
    indice.add_argument(
        "--yes",
        action="store_true",
        help="Autorizar expresamente la reconstrucción del índice",
    )

    # Añadir un cliente del agente sin modificar su orquestador.
    agente = secciones.add_parser(
        "agente",
        help="Ejecutar el agente LangGraph con HITL",
    )
    acciones_agente = agente.add_subparsers(
        dest="accion",
        required=True,
    )

    demo = acciones_agente.add_parser(
        "demo",
        help="Ejecutar una consulta y revisar su plan",
    )
    demo.add_argument(
        "mensaje",
        nargs="?",
        default=None,
        help="Consulta de prueba (por defecto, botella de plástico)",
    )
    demo.add_argument(
        "--auto-aprobar",
        action="store_true",
        help="Aprobar automáticamente el plan de esta demostración",
    )
    _agregar_entradas_opcionales(demo)

    interactivo = acciones_agente.add_parser(
        "interactivo",
        help="Iniciar una conversación con aprobación y rechazo manuales",
    )
    interactivo.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS_DEFAULT,
        help="Máximo de consultas de usuario en el hilo",
    )
    interactivo.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Máximo de rondas LLM por especialista y consulta",
    )
    _agregar_entradas_opcionales(interactivo)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Despacha el modo solicitado sin implementar el RAG ni el grafo."""
    args = crear_parser().parse_args(argv)

    try:
        if args.seccion == "rag":
            return ejecutar_rag(args)

        if args.seccion == "agente":
            # Importación diferida para mantener independiente la CLI del RAG.
            from src.cli_agent import ejecutar_agente

            return ejecutar_agente(args)

    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
