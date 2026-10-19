"""Evalúa el RAG PB02 completo con las 14 consultas del retrieval.

Desde la raíz del proyecto::

    python tests/evaluate_rag.py
    python tests/evaluate_rag.py --ids 1 8 12

Requiere la caché creada por tests/evaluate_retrieve.py. No genera
embeddings ni modifica ChromaDB. Sí consume API de generación Gemini.
Las comprobaciones automáticas no sustituyen la revisión factual.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from config import (  # noqa: E402
    CHROMA_COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    OUTPUT_DIR,
    TOP_K,
)
from evaluate_retrieve import (  # noqa: E402
    CASOS,
    FORMATO_CONSULTA,
    RUTA_CACHE,
)
from src.gemini_client import crear_cliente_gemini  # noqa: E402
from src.rag import responder  # noqa: E402

RUTA_RESULTADOS = OUTPUT_DIR / "evaluation" / "rag_pb02.json"

# Son pistas léxicas, no pruebas de corrección factual.
PISTAS_RESPUESTA = {
    1: ("amarillo",),
    2: ("marron", "organico", "organica"),
    3: ("verde",),
    4: ("azul",),
    5: ("gris", "resto"),
    8: ("estrella denebola",),
    11: ("tomas redondo",),
}

CASOS_REVISION_ESPECIAL = {
    9: "Los horarios pueden depender de la parada o ubicación.",
    10: "Cinco chunks no permiten enumerar el catálogo completo.",
}


def _normalizar(texto: str) -> str:
    """Normaliza tildes y mayúsculas para comprobaciones orientativas."""
    texto = unicodedata.normalize("NFD", texto.casefold())
    return "".join(
        caracter for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )


def _clave_consulta(pregunta: str) -> str:
    """Reproduce la clave de la caché del evaluador de retrieval."""
    return hashlib.sha256(pregunta.encode("utf-8")).hexdigest()


def _validar_vector(vector: object) -> list[float]:
    """Comprueba dimensiones y valores finitos antes de recuperar."""
    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError("Vector ausente o de dimensión incorrecta.")

    if any(
        isinstance(valor, bool)
        or not isinstance(valor, (int, float))
        or not math.isfinite(valor)
        for valor in vector
    ):
        raise ValueError("El vector contiene valores no válidos.")

    return [float(valor) for valor in vector]


def _cargar_vectores(casos: list[tuple]) -> dict[int, list[float]]:
    """Exige embeddings guardados, sin llamadas de embedding a Gemini."""
    if not RUTA_CACHE.is_file():
        raise FileNotFoundError(
            "No existe la caché de consultas. Ejecuta primero: "
            "python tests/evaluate_retrieve.py"
        )

    with RUTA_CACHE.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    if not isinstance(datos, dict) or (
        datos.get("modelo") != EMBEDDING_MODEL
        or datos.get("dimensiones") != EMBEDDING_DIMENSIONS
        or datos.get("formato_consulta") != FORMATO_CONSULTA
    ):
        raise ValueError(
            "La caché no corresponde al modelo y formato actuales. "
            "Ejecuta primero: python tests/evaluate_retrieve.py"
        )

    embeddings = datos.get("embeddings")
    if not isinstance(embeddings, dict):
        raise ValueError("La caché no contiene embeddings válidos.")

    vectores = {}
    for identificador, _, pregunta, _ in casos:
        clave = _clave_consulta(pregunta)
        if clave not in embeddings:
            raise ValueError(
                f"Falta la pregunta {identificador} en la caché. "
                "Ejecuta primero: python tests/evaluate_retrieve.py"
            )
        vectores[identificador] = _validar_vector(embeddings[clave])

    return vectores


def _resumir_chunks(chunks: list[dict]) -> list[dict]:
    """Incluye evidencia recuperada para facilitar revisión humana."""
    resumen = []
    for posicion, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        resumen.append({
            "id_orientativo": f"C{posicion}",
            "source": metadata.get("source"),
            "document_type": metadata.get("document_type"),
            "district": metadata.get("district"),
            "distance": chunk.get("distance"),
            "text": str(chunk.get("text") or "")[:1_500],
        })
    return resumen


def _comprobar_respuesta(
    identificador: int,
    respondible: bool,
    resultado: dict,
) -> list[str]:
    """Detecta anomalías estructurales y pistas a revisar manualmente."""
    alertas = []
    abstencion = resultado["abstencion"]
    respuesta = resultado["respuesta"]
    fuentes = resultado["fuentes"]
    chunks = resultado["chunks"]

    if respondible and abstencion:
        alertas.append(
            "La pregunta es respondible en el corpus, pero hubo "
            "abstención: revisar suficiencia del contexto."
        )
    if not respondible and not abstencion:
        alertas.append("El sistema respondió a una consulta fuera de dominio.")

    if abstencion:
        if fuentes:
            alertas.append("La abstención incluye fuentes inesperadas.")
        if re.search(r"\[C\d+\]", respuesta):
            alertas.append("La abstención incluye citas en el texto.")
        return alertas

    if not fuentes:
        alertas.append("La respuesta afirmativa no tiene fuentes.")

    citas_texto = set(re.findall(r"\[C\d+\]", respuesta))
    citas_fuentes = {
        f"[{fuente.get('chunk_id')}]" for fuente in fuentes
    }
    if not citas_texto or citas_texto != citas_fuentes:
        alertas.append("Citas del texto y fuentes devueltas no coinciden.")

    fuentes_recuperadas = {
        (chunk.get("metadata") or {}).get("source")
        for chunk in chunks
    }
    for fuente in fuentes:
        if fuente.get("source") not in fuentes_recuperadas:
            alertas.append("Se cita una fuente ausente del retrieval.")
            break

    pistas = PISTAS_RESPUESTA.get(identificador)
    if pistas and not any(
        pista in _normalizar(respuesta) for pista in pistas
    ):
        alertas.append(
            "No aparece la pista léxica esperada en la respuesta: "
            + ", ".join(pistas)
        )

    if identificador in (8, 11):
        esperado = "arganzuela" if identificador == 8 else "hortaleza"
        evidencia = [
            chunk for chunk in chunks
            if (chunk.get("metadata") or {}).get("source")
            == "puntos_limpios_fijos.csv"
        ]
        if not any(
            esperado in _normalizar(
                str((chunk.get("metadata") or {}).get("district", ""))
            )
            for chunk in evidencia
        ):
            alertas.append("No se recuperó el distrito esperado.")
        if not any(
            fuente.get("source") == "puntos_limpios_fijos.csv"
            for fuente in fuentes
        ):
            alertas.append("Falta la cita del punto limpio fijo.")

    return alertas


def _evaluar_caso(caso: tuple, vector: list[float], cliente) -> dict:
    """Ejecuta la orquestación real y registra datos para auditoría."""
    identificador, categoria, pregunta, esperadas = caso
    resultado = responder(
        pregunta,
        top_k=TOP_K,
        query_embedding=vector,
        client=cliente,
    )
    alertas = _comprobar_respuesta(
        identificador, bool(esperadas), resultado
    )

    return {
        "id": identificador,
        "categoria": categoria,
        "pregunta": pregunta,
        "respondible_en_corpus": bool(esperadas),
        "fuentes_esperadas_retrieval": list(esperadas),
        "respuesta": resultado["respuesta"],
        "abstencion": resultado["abstencion"],
        "motivo": resultado["motivo"],
        "fuentes": resultado["fuentes"],
        "metricas": resultado["metricas"],
        "chunks": _resumir_chunks(resultado["chunks"]),
        "alertas_automaticas": alertas,
        "revision_especial": CASOS_REVISION_ESPECIAL.get(identificador),
        "revision_factual_manual": "pendiente",
    }


def _metricas(evaluaciones: list[dict], errores: list[dict]) -> dict:
    """Cuenta comportamientos observados, sin atribuir precisión factual."""
    respondibles = [
        caso for caso in evaluaciones if caso["respondible_en_corpus"]
    ]
    fuera = [
        caso for caso in evaluaciones if not caso["respondible_en_corpus"]
    ]
    return {
        "evaluadas": len(evaluaciones),
        "errores_tecnicos": len(errores),
        "respondibles_evaluadas": len(respondibles),
        "respondibles_con_respuesta": sum(
            not caso["abstencion"] for caso in respondibles
        ),
        "fuera_dominio_evaluadas": len(fuera),
        "fuera_dominio_con_abstencion": sum(
            caso["abstencion"] for caso in fuera
        ),
        "casos_con_alertas": sum(
            bool(caso["alertas_automaticas"]) for caso in evaluaciones
        ),
        "revision_factual_pendiente": len(evaluaciones),
    }


def _seleccionar_casos(ids: list[int] | None) -> list[tuple]:
    """Selecciona casos en su orden canónico y valida identificadores."""
    if ids is None:
        return list(CASOS)

    disponibles = {caso[0] for caso in CASOS}
    desconocidos = set(ids) - disponibles
    if desconocidos:
        raise ValueError(
            f"Identificadores desconocidos: {sorted(desconocidos)}"
        )
    return [caso for caso in CASOS if caso[0] in set(ids)]


def main() -> int:
    """Evalúa consultas, persiste el informe y muestra las respuestas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids", nargs="+", type=int, metavar="N",
        help="Evalúa solo los identificadores indicados; por defecto, 14.",
    )
    argumentos = parser.parse_args()
    casos = _seleccionar_casos(argumentos.ids)
    vectores = _cargar_vectores(casos)

    print("EVALUACIÓN EXTREMO A EXTREMO — PB02")
    print(f"Modelo de generación: {GENERATION_MODEL}")
    print(f"TOP_K: {TOP_K} | Casos seleccionados: {len(casos)}")
    print("Embeddings reutilizados; sin cambios en ChromaDB.")
    print("Se usarán llamadas reales a Gemini, con posibles reintentos.")
    print("La corrección factual requiere revisión humana.\n")

    evaluaciones = []
    errores = []
    cliente = crear_cliente_gemini()
    try:
        for caso in casos:
            identificador, _, pregunta, _ = caso
            print("=" * 70)
            print(f"{identificador:02d}. {pregunta}")
            try:
                resultado = _evaluar_caso(
                    caso, vectores[identificador], cliente
                )
                evaluaciones.append(resultado)
                print(f"Respuesta: {resultado['respuesta']}")
                print(
                    "Abstención: "
                    f"{resultado['abstencion']} "
                    f"| Motivo: {resultado['motivo']}"
                )
                print(
                    "Fuentes: "
                    f"{[f['source'] for f in resultado['fuentes']]}"
                )
                print(
                    "Tiempo total: "
                    f"{resultado['metricas']['total_ms']} ms"
                )
                for alerta in resultado["alertas_automaticas"]:
                    print(f"ALERTA: {alerta}")
                if resultado["revision_especial"]:
                    print(f"REVISAR: {resultado['revision_especial']}")
            except Exception as exc:
                errores.append({
                    "id": identificador,
                    "tipo": type(exc).__name__,
                    "mensaje": str(exc),
                })
                print(f"ERROR {type(exc).__name__}: {exc}")
            print()
    finally:
        cliente.close()

    informe = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "coleccion": CHROMA_COLLECTION_NAME,
        "modelo_embeddings": EMBEDDING_MODEL,
        "modelo_generacion": GENERATION_MODEL,
        "top_k": TOP_K,
        "preguntas_previstas": len(casos),
        "metricas": _metricas(evaluaciones, errores),
        "evaluaciones": evaluaciones,
        "errores": errores,
        "nota": (
            "Las alertas automáticas y las citas válidas no garantizan "
            "la corrección factual. Se requiere revisión manual."
        ),
    }
    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    with RUTA_RESULTADOS.open("w", encoding="utf-8") as archivo:
        json.dump(informe, archivo, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("RESUMEN (NO EQUIVALE A PRECISIÓN FACTUAL)")
    for nombre, valor in informe["metricas"].items():
        print(f"  {nombre}: {valor}")
    print(f"Informe JSON: {RUTA_RESULTADOS}")
    return 1 if errores else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"No se puede iniciar la evaluación: {error}")
        sys.exit(1)
