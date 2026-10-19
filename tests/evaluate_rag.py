"""Evalúa el RAG PB02 con 14 consultas, incluidas vías estructuradas.

Desde la raíz del proyecto::

    python tests/evaluate_rag.py
    python tests/evaluate_rag.py --ids 3 9 10

Reutiliza los embeddings de tests/evaluate_retrieve.py. Las consultas
estructuradas 9 y 10 no necesitan embeddings ni generación Gemini.
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
IDS_ESTRUCTURADOS = frozenset({9, 10})
FUENTE_CATALOGO = "tipos_residuos.csv"
FUENTE_PUNTOS_FIJOS = "puntos_limpios_fijos.csv"

# Pistas léxicas orientativas: nunca equivalen a exactitud factual.
PISTAS_RESPUESTA = {
    1: ("amarillo",),
    2: ("marron", "organico", "organica"),
    3: ("verde",),
    4: ("azul",),
    5: ("naranja", "resto"),
    8: ("estrella denebola",),
    11: ("tomas redondo",),
}

REVISION_MANUAL = {
    5: (
        "Verificar la explicación sobre fracción resto: "
        "cuerpo gris y tapa naranja, según el contexto."
    ),
    9: (
        "Se espera una aclaración sobre distrito o parada; "
        "no una selección arbitraria de horarios."
    ),
    10: (
        "El resumen enumera destinos del CSV, no una taxonomía "
        "oficial exhaustiva de residuos."
    ),
}

PATRON_CITAS = re.compile(r"\[(C\d+(?:\s*,\s*C\d+)*)\]")


def _normalizar(texto: str) -> str:
    """Normaliza tildes y mayúsculas para comprobaciones orientativas."""
    texto = unicodedata.normalize("NFD", texto.casefold())
    return "".join(
        caracter for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )


def _tipo_esperado(identificador: int, respondible: bool) -> str:
    """Distingue aclaración, resumen, respuesta y fuera de dominio."""
    if not respondible:
        return "fuera_de_dominio"
    if identificador == 9:
        return "aclaracion"
    if identificador == 10:
        return "resumen_catalogo"
    return "respuesta"


def _clave_consulta(pregunta: str) -> str:
    """Reproduce la clave de caché del evaluador de retrieval."""
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


def _cargar_vectores(
    casos: list[tuple],
) -> dict[int, list[float] | None]:
    """Carga vectores solo donde el flujo semántico los necesita."""
    vectores: dict[int, list[float] | None] = {
        identificador: None
        for identificador, _, _, _ in casos
        if identificador in IDS_ESTRUCTURADOS
    }
    casos_semanticos = [
        caso for caso in casos if caso[0] not in IDS_ESTRUCTURADOS
    ]
    if not casos_semanticos:
        return vectores

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

    for identificador, _, pregunta, _ in casos_semanticos:
        clave = _clave_consulta(pregunta)
        if clave not in embeddings:
            raise ValueError(
                f"Falta la pregunta {identificador} en la caché. "
                "Ejecuta primero: python tests/evaluate_retrieve.py"
            )
        vectores[identificador] = _validar_vector(embeddings[clave])

    return vectores


def _resumir_chunks(chunks: list[dict]) -> list[dict]:
    """Conserva evidencia, procedencia y señal de resumen derivado."""
    resumen = []
    for posicion, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        resumen.append({
            "id_orientativo": f"C{posicion}",
            "source": metadata.get("source"),
            "path": metadata.get("path"),
            "document_type": metadata.get("document_type"),
            "district": metadata.get("district"),
            "derived": metadata.get("derived", False),
            "total_relaciones": metadata.get("total_relaciones"),
            "total_destinos": metadata.get("total_destinos"),
            "distance": chunk.get("distance"),
            "text": str(chunk.get("text") or "")[:2_000],
        })
    return resumen


def _citas_en_texto(respuesta: str) -> set[str]:
    """Extrae identificadores de [C1] o [C1, C2] sin inventar citas."""
    citas = set()
    for grupo in PATRON_CITAS.findall(respuesta):
        citas.update(item.strip() for item in grupo.split(","))
    return citas


def _revisar_aclaracion(resultado: dict) -> list[str]:
    """Verifica la aclaración esperada en la pregunta nueve."""
    alertas = []
    if not resultado["abstencion"]:
        alertas.append("P09 debe pedir ubicación, no dar horarios.")
    if resultado["motivo"] != "ambiguedad":
        alertas.append("P09 debe indicar motivo 'ambiguedad'.")
    if resultado["fuentes"] or resultado["chunks"]:
        alertas.append("P09 no debe recuperar ni citar paradas arbitrarias.")
    texto = _normalizar(resultado["respuesta"])
    if not any(palabra in texto for palabra in (
        "distrito", "calle", "parada"
    )):
        alertas.append("P09 no solicita una ubicación identificable.")
    return alertas


def _revisar_catalogo(resultado: dict) -> list[str]:
    """Verifica trazabilidad y límites del resumen estructurado."""
    alertas = []
    chunks = resultado["chunks"]
    if resultado["abstencion"]:
        return ["P10 debería resumir el catálogo disponible."]
    if len(chunks) != 1:
        return ["P10 debe identificar un único resumen derivado."]

    metadata = chunks[0].get("metadata") or {}
    if (
        metadata.get("source") != FUENTE_CATALOGO
        or metadata.get("document_type") != "resumen_catalogo_derivado"
        or metadata.get("derived") is not True
    ):
        alertas.append("P10 no identifica el resumen como derivado.")
    if not metadata.get("path"):
        alertas.append("P10 no conserva la ruta de procedencia.")
    for campo in ("total_relaciones", "total_destinos"):
        valor = metadata.get(campo)
        if type(valor) is not int or valor <= 0:
            alertas.append(f"P10 no incluye {campo} válido.")

    texto = _normalizar(resultado["respuesta"])
    if not any(palabra in texto for palabra in (
        "destino", "lugar de deposito"
    )):
        alertas.append("P10 no diferencia destinos de tipos de residuos.")
    if "no constituyen" not in texto and "no es exhaustiv" not in texto:
        alertas.append("P10 no delimita el alcance del catálogo.")
    return alertas


def _revisar_citas(resultado: dict) -> list[str]:
    """Comprueba citas y metadatos tanto semánticos como derivados."""
    alertas = []
    respuesta = resultado["respuesta"]
    fuentes = resultado["fuentes"]
    chunks = resultado["chunks"]
    if not fuentes:
        return ["La respuesta afirmativa no tiene fuentes."]

    citas_texto = _citas_en_texto(respuesta)
    citas_fuentes = {f.get("chunk_id") for f in fuentes}
    if not citas_texto or citas_texto != citas_fuentes:
        alertas.append("Citas del texto y fuentes devueltas no coinciden.")

    evidencias = {
        f"C{posicion}": chunk.get("metadata") or {}
        for posicion, chunk in enumerate(chunks, start=1)
    }
    for fuente in fuentes:
        metadata = evidencias.get(fuente.get("chunk_id"))
        if metadata is None or any(
            fuente.get(campo) != metadata.get(campo)
            for campo in ("source", "path")
        ):
            alertas.append(
                "Se cita un chunk inexistente o con procedencia distinta."
            )
            break
    return alertas


def _revisar_punto_fijo(
    identificador: int, resultado: dict
) -> list[str]:
    """Comprueba la modalidad y el distrito de P08 y P11."""
    esperado = "arganzuela" if identificador == 8 else "hortaleza"
    chunks = resultado["chunks"]
    evidencia = [
        chunk for chunk in chunks
        if (chunk.get("metadata") or {}).get("source")
        == FUENTE_PUNTOS_FIJOS
    ]
    if not any(
        esperado in _normalizar(
            str((chunk.get("metadata") or {}).get("district", ""))
        )
        for chunk in evidencia
    ):
        return ["No se recuperó el distrito esperado en punto fijo."]
    if not any(
        fuente.get("source") == FUENTE_PUNTOS_FIJOS
        for fuente in resultado["fuentes"]
    ):
        return ["No se cita el punto limpio fijo."]
    return []


def _comprobar_respuesta(
    identificador: int,
    respondible: bool,
    resultado: dict,
) -> list[str]:
    """Distingue resultados esperados sin afirmar exactitud factual."""
    tipo = _tipo_esperado(identificador, respondible)
    if tipo == "aclaracion":
        return _revisar_aclaracion(resultado)

    abstencion = resultado["abstencion"]
    if tipo == "fuera_de_dominio":
        alertas = []
        if not abstencion or resultado["motivo"] != "fuera_de_dominio":
            alertas.append("Se esperaba abstención fuera de dominio.")
        if resultado["fuentes"]:
            # Una abstención semántica puede recuperar chunks, sin citarlos.
            alertas.append("La abstención no debe atribuir fuentes.")
        if abstencion and _citas_en_texto(resultado["respuesta"]):
            alertas.append("La abstención incluye citas en el texto.")
        return alertas

    if abstencion:
        return [
            "La pregunta admite respuesta, pero hubo abstención: "
            "revisar evidencia, formato o ambigüedad."
        ]

    alertas = _revisar_citas(resultado)
    if tipo == "resumen_catalogo":
        alertas.extend(_revisar_catalogo(resultado))
        return alertas

    pistas = PISTAS_RESPUESTA.get(identificador)
    if pistas and not any(
        pista in _normalizar(resultado["respuesta"]) for pista in pistas
    ):
        alertas.append(
            "Falta pista léxica orientativa: " + ", ".join(pistas)
        )

    if identificador in (8, 11):
        alertas.extend(_revisar_punto_fijo(identificador, resultado))
    return alertas


def _evaluar_caso(
    caso: tuple,
    vector: list[float] | None,
    cliente,
) -> dict:
    """Ejecuta la orquestación real y conserva los datos de auditoría."""
    identificador, categoria, pregunta, esperadas = caso
    resultado = responder(
        pregunta,
        top_k=TOP_K,
        query_embedding=vector,
        client=cliente,
    )
    tipo = _tipo_esperado(identificador, bool(esperadas))
    alertas = _comprobar_respuesta(
        identificador, bool(esperadas), resultado
    )

    return {
        "id": identificador,
        "categoria": categoria,
        "pregunta": pregunta,
        "respondible_en_corpus": bool(esperadas),
        "comportamiento_esperado": tipo,
        "fuentes_esperadas_retrieval": list(esperadas),
        "respuesta": resultado["respuesta"],
        "abstencion": resultado["abstencion"],
        "motivo": resultado["motivo"],
        "fuentes": resultado["fuentes"],
        "metricas": resultado["metricas"],
        "chunks": _resumir_chunks(resultado["chunks"]),
        "alertas_automaticas": alertas,
        "revision_especial": REVISION_MANUAL.get(identificador),
        "revision_factual_manual": "pendiente",
    }


def _metricas(evaluaciones: list[dict], errores: list[dict]) -> dict:
    """Cuenta rutas y alertas, sin presentarlas como precisión factual."""
    respondibles = [
        caso for caso in evaluaciones if caso["respondible_en_corpus"]
    ]
    fuera = [
        caso for caso in evaluaciones if not caso["respondible_en_corpus"]
    ]
    aclaraciones = [
        caso for caso in respondibles
        if caso["comportamiento_esperado"] == "aclaracion"
    ]
    return {
        "evaluadas": len(evaluaciones),
        "errores_tecnicos": len(errores),
        "respondibles_evaluadas": len(respondibles),
        "respondibles_con_respuesta": sum(
            not caso["abstencion"] for caso in respondibles
        ),
        "aclaraciones_esperadas": sum(
            caso["abstencion"]
            and caso["motivo"] == "ambiguedad"
            for caso in aclaraciones
        ),
        "respondibles_con_abstencion_no_esperada": sum(
            caso["abstencion"]
            and caso["comportamiento_esperado"] != "aclaracion"
            for caso in respondibles
        ),
        "fuera_dominio_evaluadas": len(fuera),
        "fuera_dominio_con_abstencion": sum(
            caso["abstencion"]
            and caso["motivo"] == "fuera_de_dominio"
            for caso in fuera
        ),
        "casos_sin_alertas_automaticas": sum(
            not caso["alertas_automaticas"] for caso in evaluaciones
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
    """Evalúa, persiste el informe y muestra los comportamientos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids", nargs="+", type=int, metavar="N",
        help="Evalúa los identificadores seleccionados; 14 por defecto.",
    )
    argumentos = parser.parse_args()
    casos = _seleccionar_casos(argumentos.ids)
    vectores = _cargar_vectores(casos)

    print("EVALUACIÓN EXTREMO A EXTREMO — PB02")
    print(f"Modelo configurado: {GENERATION_MODEL}")
    print(f"TOP_K: {TOP_K} | Casos seleccionados: {len(casos)}")
    print("Embeddings reutilizados cuando corresponde.")
    print("Las consultas estructuradas no consumen Gemini.")
    print("La corrección factual requiere revisión humana.\n")

    evaluaciones = []
    errores = []
    necesita_cliente = any(
        caso[0] not in IDS_ESTRUCTURADOS for caso in casos
    )
    cliente = crear_cliente_gemini() if necesita_cliente else None
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
                print(
                    "Comportamiento esperado: "
                    f"{resultado['comportamiento_esperado']}"
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
        if cliente is not None:
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
