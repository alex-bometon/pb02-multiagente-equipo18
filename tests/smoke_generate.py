"""Prueba manual de integración entre retrieval, ChromaDB y Gemini.

Ejecutar desde la raíz del proyecto::

    python tests/smoke_generate.py

Reutiliza embeddings de evaluación y no modifica la base vectorial.
ATENCIÓN: prueba tres consultas con Gemini, más posibles reintentos.
"""

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from config import (  # noqa: E402
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    OUTPUT_DIR,
    TOP_K,
)
from src.generate import generar_respuesta  # noqa: E402
from src.retrieve import buscar_chunks_relevantes  # noqa: E402


CACHE = OUTPUT_DIR / "cache" / "pb02_embeddings_consultas.json"

CASOS = (
    {
        "nombre": "Envase de plástico",
        "pregunta": "¿En qué contenedor tiro un envase de plástico?",
        "abstencion_esperada": False,
        "texto_esperado": "amarillo",
        "fuente_esperada": None,
    },
    {
        "nombre": "Punto limpio fijo de Arganzuela",
        "pregunta": (
            "¿Dónde hay un punto limpio fijo en el distrito "
            "de Arganzuela?"
        ),
        "abstencion_esperada": False,
        "texto_esperado": "estrella denebola",
        "fuente_esperada": "puntos_limpios_fijos.csv",
    },
    {
        "nombre": "Piscinas municipales (fuera de dominio)",
        "pregunta": (
            "¿Cuál es el horario de las piscinas municipales de Madrid?"
        ),
        "abstencion_esperada": True,
        "texto_esperado": None,
        "fuente_esperada": None,
    },
    {
        "nombre": "Vidrio en Barcelona (fuera de dominio)",
        "pregunta": "¿Dónde reciclo el vidrio en Barcelona?",
        "abstencion_esperada": True,
        "texto_esperado": None,
        "fuente_esperada": None,
    },
)


def _normalizar(texto: str) -> str:
    """Elimina diferencias de tildes y mayúsculas para comprobar texto."""
    texto = unicodedata.normalize("NFD", texto.casefold())
    return "".join(
        caracter for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )


def _cargar_embeddings() -> dict:
    """Lee la caché previamente generada durante la evaluación."""
    if not CACHE.is_file():
        raise FileNotFoundError(
            f"No existe la caché: {CACHE}. "
            "Ejecuta primero: python tests/evaluate_retrieve.py"
        )

    with CACHE.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    if not isinstance(datos, dict) or (
        datos.get("modelo") != EMBEDDING_MODEL
        or datos.get("dimensiones") != EMBEDDING_DIMENSIONS
        or datos.get("formato_consulta") != "texto_sin_prefijo"
    ):
        raise ValueError(
            "La caché de consultas no corresponde al modelo "
            "y formato de embeddings actuales."
        )

    embeddings = datos.get("embeddings")
    if not isinstance(embeddings, dict):
        raise ValueError("La caché de embeddings está incompleta.")

    return embeddings


def _obtener_vector(pregunta: str, cache: dict) -> list[float]:
    """Recupera un embedding existente sin llamar a la API."""
    clave = hashlib.sha256(pregunta.encode("utf-8")).hexdigest()
    vector = cache.get(clave)
    if vector is None:
        raise ValueError(
            "No existe un embedding guardado para la pregunta: "
            f"{pregunta}. Ejecuta antes evaluate_retrieve.py."
        )
    return vector


def _comprobar_resultado(
    caso: dict,
    chunks: list[dict],
    resultado: dict,
) -> list[str]:
    """Devuelve observaciones sobre el cumplimiento del caso."""
    errores = []
    abstencion = resultado.get("abstencion")

    if abstencion is not caso["abstencion_esperada"]:
        errores.append("El estado de abstención es incorrecto.")

    if abstencion:
        if resultado.get("fuentes"):
            errores.append("Una abstención no debe atribuir fuentes.")
        return errores

    respuesta = _normalizar(resultado.get("respuesta") or "")
    texto_esperado = caso["texto_esperado"]
    if texto_esperado and texto_esperado not in respuesta:
        errores.append(
            f"La respuesta no menciona '{texto_esperado}'."
        )

    fuentes = resultado.get("fuentes") or []
    if not fuentes:
        errores.append("Una respuesta afirmativa debe citar fuentes.")

    fuentes_recuperadas = {
        chunk.get("metadata", {}).get("source") for chunk in chunks
    }
    for fuente in fuentes:
        if fuente.get("source") not in fuentes_recuperadas:
            errores.append("Se cita una fuente ajena al retrieval.")

    fuente_esperada = caso["fuente_esperada"]
    if fuente_esperada and not any(
        fuente.get("source") == fuente_esperada for fuente in fuentes
    ):
        errores.append(
            f"No se ha citado la fuente '{fuente_esperada}'."
        )

    return errores


def main() -> int:
    """Ejecuta casos reales y muestra los resultados para revisión."""
    print("PRUEBA REAL: CHROMADB → RETRIEVAL → GEMINI")
    print(f"Modelo de generación: {GENERATION_MODEL}")
    print(f"TOP_K: {TOP_K}")
    print("Se reutilizarán embeddings; no se generarán otros.")
    print("Tres consultas de generación, más posibles reintentos.\n")

    cache = _cargar_embeddings()
    fallidos = 0

    for posicion, caso in enumerate(CASOS, start=1):
        print("=" * 70)
        print(f"{posicion}. {caso['nombre']}")
        print(f"Pregunta: {caso['pregunta']}")

        try:
            vector = _obtener_vector(caso["pregunta"], cache)
            chunks = buscar_chunks_relevantes(
                caso["pregunta"],
                top_k=TOP_K,
                query_embedding=vector,
            )
            print(f"Chunks recuperados: {len(chunks)}")

            for indice, chunk in enumerate(chunks, start=1):
                metadata = chunk.get("metadata") or {}
                print(
                    f"  {indice}. {metadata.get('source', '?')} "
                    f"| distrito={metadata.get('district', '-')} "
                    f"| distancia={chunk['distance']:.4f}"
                )

            resultado = generar_respuesta(caso["pregunta"], chunks)
            print(f"Respuesta: {resultado.get('respuesta')}")
            print(f"Abstención: {resultado.get('abstencion')}")
            print(f"Motivo: {resultado.get('motivo')}")
            print(f"Fuentes: {resultado.get('fuentes')}")

            errores = _comprobar_resultado(caso, chunks, resultado)
        except Exception as exc:
            errores = [f"{type(exc).__name__}: {exc}"]

        if errores:
            fallidos += 1
            print("RESULTADO: REVISAR")
            for error in errores:
                print(f"  - {error}")
        else:
            print("RESULTADO: VALIDACIÓN BÁSICA SUPERADA")

        print()

    print("=" * 70)
    print(f"Casos sin incidencias automáticas: {len(CASOS) - fallidos}")
    print(f"Casos para revisar: {fallidos}")
    print(
        "Importante: la validación automática no demuestra por sí sola "
        "que todas las afirmaciones estén respaldadas por la evidencia."
    )
    return 1 if fallidos else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Preparación incompleta: {error}")
        sys.exit(1)
