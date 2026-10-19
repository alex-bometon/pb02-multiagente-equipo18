"""Evalúa el retrieval de ReciclaTIA con preguntas etiquetadas.

Ejecución desde la raíz del proyecto::

    python tests/evaluate_retrieve.py

Se genera un embedding por pregunta no cacheada y se reutiliza para
comparar K=3, 5, 8 y 10. No ejecuta generación ni modifica ChromaDB.
"""

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.genai import types

# Permite ejecutar este archivo directamente sin instalar el proyecto.
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from config import (  # noqa: E402
    CHROMA_COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OUTPUT_DIR,
)
from src.gemini_client import crear_cliente_gemini  # noqa: E402
from src.retrieve import buscar_chunks_relevantes  # noqa: E402


VALORES_K = (3, 5, 8, 10)
RUTA_CACHE = OUTPUT_DIR / "cache" / "pb02_embeddings_consultas.json"
RUTA_RESULTADOS = OUTPUT_DIR / "evaluation" / "retrieval_pb02.json"
FORMATO_CONSULTA = "texto_sin_prefijo"

# Referencia: queries/eval_queries.json del PB01, versión de 14 preguntas.
# Se incluyen aquí para no requerir ningún otro archivo de evaluación.
CASOS = (
    (
        1,
        "separacion_residuos",
        "¿En qué contenedor tiro un envase de plástico?",
        ("fraccion_plasticos_metales_briks.html",),
    ),
    (
        2,
        "separacion_residuos",
        "¿Dónde tiro los restos de comida y las cáscaras de fruta?",
        ("fraccion_organica.html",),
    ),
    (
        3,
        "separacion_residuos",
        "¿Qué puedo depositar en el contenedor verde de vidrio?",
        ("fraccion_vidrio.html",),
    ),
    (
        4,
        "separacion_residuos",
        "¿Qué se debe tirar en el contenedor azul de papel y cartón?",
        ("fraccion_papel.html",),
    ),
    (
        5,
        "separacion_residuos",
        "¿Qué va en el contenedor gris de restos?",
        ("fraccion_resto.html",),
    ),
    (
        6,
        "residuos_domesticos_especiales",
        "¿Dónde puedo tirar el aceite vegetal usado de la cocina?",
        ("recogida_aceite_vegetal.html",),
    ),
    (
        7,
        "servicios_recogida",
        "¿Cómo puedo deshacerme de un colchón o un mueble viejo?",
        ("recogida_muebles_enseres.html",),
    ),
    (
        8,
        "puntos_recogida",
        "¿Dónde hay un punto limpio fijo en el distrito de Arganzuela?",
        (
            "puntos_limpios_fijos.csv",
            "residuos_punto_limpio_fijo.html",
        ),
    ),
    (
        9,
        "puntos_recogida",
        "¿Qué horario tiene el punto limpio móvil?",
        (
            "puntos_limpios_moviles.csv",
            "residuos_punto_limpio_movil.html",
        ),
    ),
    (
        10,
        "destino_residuo",
        "¿Qué tipos de residuos reconoce el Ayuntamiento de Madrid?",
        ("tipos_residuos.csv",),
    ),
    (
        11,
        "puntos_recogida",
        "¿Cuál es la dirección del punto limpio fijo de Hortaleza?",
        ("puntos_limpios_fijos.csv",),
    ),
    (
        12,
        "fuera_de_dominio",
        "¿Cuál es el horario de las piscinas municipales de Madrid?",
        (),
    ),
    (
        13,
        "fuera_de_dominio",
        "¿Dónde reciclo el vidrio en Barcelona?",
        (),
    ),
    (
        14,
        "fuera_de_dominio",
        "¿Qué servicios de reparación de móviles hay cerca de Gran Vía?",
        (),
    ),
)


def _clave_consulta(pregunta: str) -> str:
    """Identifica de forma estable una consulta para la caché."""
    return hashlib.sha256(pregunta.encode("utf-8")).hexdigest()


def _cargar_cache() -> dict[str, list[float]]:
    """Lee solo embeddings compatibles con el modelo y formato actuales."""
    if not RUTA_CACHE.is_file():
        return {}

    with RUTA_CACHE.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    compatible = (
        datos.get("modelo") == EMBEDDING_MODEL
        and datos.get("dimensiones") == EMBEDDING_DIMENSIONS
        and datos.get("formato_consulta") == FORMATO_CONSULTA
    )
    if not compatible:
        print("Caché incompatible: se generarán embeddings nuevos.")
        return {}

    return datos.get("embeddings", {})


def _guardar_cache(embeddings: dict[str, list[float]]) -> None:
    """Guarda la caché de forma atómica tras cada nueva consulta."""
    RUTA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporal = RUTA_CACHE.with_suffix(".tmp")
    datos = {
        "modelo": EMBEDDING_MODEL,
        "dimensiones": EMBEDDING_DIMENSIONS,
        "formato_consulta": FORMATO_CONSULTA,
        "embeddings": embeddings,
    }
    with temporal.open("w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)
    temporal.replace(RUTA_CACHE)


def _validar_vector(vector: object) -> list[float]:
    """Comprueba que el vector es numérico, finito y de tamaño correcto."""
    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError("El embedding tiene dimensiones incorrectas.")

    if any(
        isinstance(valor, bool)
        or not isinstance(valor, (int, float))
        or not math.isfinite(valor)
        for valor in vector
    ):
        raise ValueError("El embedding contiene valores no válidos.")

    return [float(valor) for valor in vector]


def _generar_embedding(cliente, pregunta: str) -> list[float]:
    """Genera el vector con el mismo formato que usa retrieve.py."""
    respuesta = cliente.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=pregunta.strip(),
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    if not respuesta.embeddings or respuesta.embeddings[0].values is None:
        raise RuntimeError("Gemini no devolvió un embedding válido.")

    return _validar_vector(list(respuesta.embeddings[0].values))


def _fuente(chunk: dict) -> str:
    """Extrae el nombre del archivo de origen de los metadatos."""
    metadata = chunk.get("metadata") or {}
    origen = metadata.get("source") or metadata.get("path") or ""
    return Path(str(origen)).name


def _resumir_resultados(chunks: list[dict]) -> list[dict]:
    """Prepara resultados serializables para el informe de evaluación."""
    return [
        {
            "posicion": posicion,
            "fuente": _fuente(chunk),
            "distancia": float(chunk["distance"]),
            "metadata": chunk.get("metadata") or {},
            "fragmento": str(chunk["text"])[:300],
        }
        for posicion, chunk in enumerate(chunks, start=1)
    ]


def _evaluar_caso(
    identificador: int,
    categoria: str,
    pregunta: str,
    esperadas: tuple[str, ...],
    chunks: list[dict],
) -> dict:
    """Calcula acierto por fuente para cada valor de K."""
    fuentes = [_fuente(chunk) for chunk in chunks]
    aciertos = {
        str(k): bool(set(fuentes[:k]).intersection(esperadas))
        for k in VALORES_K
    } if esperadas else None

    return {
        "id": identificador,
        "categoria": categoria,
        "pregunta": pregunta,
        "respondible": bool(esperadas),
        "fuentes_esperadas": list(esperadas),
        "acierto_por_k": aciertos,
        "resultados": _resumir_resultados(chunks),
    }


def _calcular_metricas(evaluaciones: list[dict]) -> dict:
    """Calcula hit@k únicamente sobre las preguntas respondibles."""
    respondibles = [e for e in evaluaciones if e["respondible"]]
    total = len(respondibles)
    return {
        str(k): {
            "aciertos": sum(e["acierto_por_k"][str(k)] for e in respondibles),
            "total": total,
            "porcentaje": round(
                100 * sum(
                    e["acierto_por_k"][str(k)] for e in respondibles
                ) / total, 2,
            ) if total else None,
        }
        for k in VALORES_K
    }


def _imprimir_resultados(evaluaciones: list[dict], metricas: dict) -> None:
    """Muestra los resultados y los errores de recuperación por caso."""
    print("\nRESULTADOS POR PREGUNTA")
    print("=" * 65)

    for caso in evaluaciones:
        print(f"\n{caso['id']:02d}. {caso['pregunta']}")
        if caso["respondible"]:
            estado = "OK" if caso["acierto_por_k"]["5"] else "FALLO"
            print(f"   Fuente esperada en top-5: {estado}")
        else:
            print("   Fuera del dominio: sin puntuación de acierto")

        for chunk in caso["resultados"][:5]:
            print(
                f"   {chunk['posicion']}. {chunk['fuente']} "
                f"(distancia: {chunk['distancia']:.4f})"
            )

    print("\nACIERTO POR FUENTE (SOLO PREGUNTAS RESPONDIBLES)")
    print("=" * 65)
    for k in VALORES_K:
        m = metricas[str(k)]
        print(
            f"K={k:2d} | {m['aciertos']}/{m['total']} | "
            f"{m['porcentaje']:.2f}%"
        )

    print("\nFuera de dominio: la cercanía de un chunk NO implica que")
    print("la pregunta sea respondible. La abstención se evaluará después.")
    print("\nNota: acertar el archivo fuente no garantiza que el chunk")
    print("contenga el dato preciso; esta es una métrica de fuentes.")


def main() -> None:
    """Ejecuta las 14 consultas, guarda la caché y genera un informe JSON."""
    cache = _cargar_cache()
    cliente = None
    evaluaciones = []
    errores = []
    nuevos_embeddings = 0

    print(f"Evaluación PB02: {len(CASOS)} preguntas, K={VALORES_K}")
    print("La primera ejecución puede consumir API de embeddings.")

    try:
        for identificador, categoria, pregunta, esperadas in CASOS:
            try:
                clave = _clave_consulta(pregunta)
                vector_guardado = cache.get(clave)
                if vector_guardado is None:
                    if cliente is None:
                        cliente = crear_cliente_gemini()
                    vector = _generar_embedding(cliente, pregunta)
                    cache[clave] = vector
                    _guardar_cache(cache)
                    nuevos_embeddings += 1
                else:
                    vector = _validar_vector(vector_guardado)

                # Una única búsqueda con K=10 es suficiente para
                # evaluar también sus prefijos K=3, K=5 y K=8.
                chunks = buscar_chunks_relevantes(
                    pregunta,
                    top_k=max(VALORES_K),
                    query_embedding=vector,
                )
                evaluaciones.append(
                    _evaluar_caso(
                        identificador, categoria, pregunta, esperadas, chunks
                    )
                )
            except Exception as exc:
                errores.append({"id": identificador, "error": str(exc)})
                print(f"Error en pregunta {identificador}: {exc}")
    finally:
        if cliente is not None:
            cliente.close()

    metricas = _calcular_metricas(evaluaciones)
    informe = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "coleccion": CHROMA_COLLECTION_NAME,
        "modelo_embeddings": EMBEDDING_MODEL,
        "dimensiones": EMBEDDING_DIMENSIONS,
        "valores_k": list(VALORES_K),
        "preguntas_previstas": len(CASOS),
        "preguntas_evaluadas": len(evaluaciones),
        "nuevos_embeddings_generados": nuevos_embeddings,
        "metricas": metricas,
        "evaluaciones": evaluaciones,
        "errores": errores,
    }

    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    with RUTA_RESULTADOS.open("w", encoding="utf-8") as archivo:
        json.dump(informe, archivo, ensure_ascii=False, indent=2)

    if evaluaciones:
        _imprimir_resultados(evaluaciones, metricas)
    print(f"\nInforme guardado en: {RUTA_RESULTADOS}")
    print(f"Embeddings generados en esta ejecución: {nuevos_embeddings}")

    if errores:
        raise SystemExit(f"Evaluación incompleta: {len(errores)} error(es).")


if __name__ == "__main__":
    main()
