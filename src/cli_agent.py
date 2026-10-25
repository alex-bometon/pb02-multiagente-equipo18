"""Interfaz de consola del agente multiagente ReciclaTIA.

Presenta las respuestas, los planes y el HITL usando exclusivamente
la API pública de src.agent. No ejecuta herramientas ni nodos.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from math import isfinite
from typing import Any

from config import MAX_TURNS_DEFAULT
from src.agent import (
    aprobar_plan,
    nuevo_thread_id,
    procesar_turno,
    run_demo,
)
from src.state import (
    EstadoProducto,
    Ubicacion,
    crear_estado_producto,
)


def _ubicacion_desde_valores(
    latitud: Any,
    longitud: Any,
) -> Ubicacion | None:
    """Convierte coordenadas aportadas expresamente por el usuario."""
    if latitud is None and longitud is None:
        return None

    if latitud is None or longitud is None:
        raise ValueError(
            "Debes indicar latitud y longitud conjuntamente."
        )

    try:
        lat = float(latitud)
        lon = float(longitud)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Latitud y longitud deben ser valores numéricos."
        ) from exc

    if not isfinite(lat) or not isfinite(lon):
        raise ValueError("Las coordenadas deben ser finitas.")

    return {"latitud": lat, "longitud": lon}


def _mostrar_fuentes(fuentes: list[dict[str, Any]]) -> None:
    """Presenta la procedencia declarada de las evidencias."""
    print("\nFuentes:")

    if not fuentes:
        print("  Ninguna fuente registrada.")
        return

    for numero, fuente in enumerate(fuentes, start=1):
        referencia = fuente.get("chunk_id") or str(numero)
        nombre = (
            fuente.get("source")
            or fuente.get("fuente")
            or fuente.get("tool")
            or "Fuente no identificada"
        )
        ruta = fuente.get("path")

        print(f"  [{referencia}] {nombre}")

        if ruta:
            print(f"      Ruta: {ruta}")


def _mostrar_traza(estado: EstadoProducto) -> None:
    """Expone los eventos y las herramientas ya registradas."""
    print("\nLog del grafo:")

    if not estado["log"]:
        print("  Sin eventos.")

    for evento in estado["log"]:
        print(
            f"  {evento.get('nodo', '?')}: "
            f"{evento.get('accion', '?')}"
        )

        if evento.get("detalle"):
            print(f"    {evento['detalle']}")

    print("\nTraza de herramientas:")

    if not estado["traza_tools"]:
        print("  No se ejecutaron herramientas.")

    for registro in estado["traza_tools"]:
        herramienta = registro.get("tool", "desconocida")
        estado_tool = registro.get("status", "desconocido")
        nodo = registro.get("nodo", "?")

        print(f"  {nodo} / {herramienta}: {estado_tool}")

        if registro.get("fuente"):
            print(f"    Fuente: {registro['fuente']}")

        if registro.get("preview"):
            print(f"    Vista previa: {registro['preview']}")


def _mostrar_estado(
    estado: EstadoProducto,
    *,
    mostrar_traza: bool = True,
) -> None:
    """Muestra el estado del turno sin reinterpretar el HITL."""
    print("\n" + "=" * 48)
    print("RECICLATIA - RESULTADO")
    print("=" * 48)

    print(f"Hilo: {estado['thread_id']}")
    print(
        "Turnos: "
        f"{estado['turnos_consumidos']}/{estado['max_turns']}"
    )
    print(f"Herramientas ejecutadas: {estado['tool_calls_turno']}")
    print(f"Rondas máximas por especialista: {estado['max_steps']}")
    print(f"Estado de evidencia: {estado['estado_evidencia']}")
    print(f"Abstención: {'sí' if estado['abstencion'] else 'no'}")

    if estado["error"]:
        print(f"\nError: {estado['error']}")

    if estado["respuesta"]:
        print(f"\nRespuesta:\n{estado['respuesta']}")

    if estado["motivo"]:
        print(f"\nMotivo: {estado['motivo']}")

    if estado["limitaciones"]:
        print("\nLimitaciones:")

        for limitacion in estado["limitaciones"]:
            print(f"  - {limitacion}")

    _mostrar_fuentes(estado["fuentes"])

    if estado["plan"]:
        print("\nPlan propuesto:")

        for numero, paso in enumerate(estado["plan"], start=1):
            print(f"  {numero}. {paso}")

    print("\nHITL pendiente:", estado["pendiente_hitl"])
    print("Finalizado:", estado["done"])

    if mostrar_traza:
        _mostrar_traza(estado)

    if estado["pendiente_hitl"] and not estado["error"]:
        print("\nDebes aprobar o rechazar el plan antes de continuar.")


def _pedir_texto(indicacion: str) -> str | None:
    """Lee una entrada o devuelve None al cerrar la terminal."""
    try:
        return input(indicacion).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nEntrada finalizada.")
        return None


def _decidir(
    estado: EstadoProducto,
    aprobar: bool,
) -> EstadoProducto:
    """Delega la decisión HITL en el orquestador compartido."""
    if estado["error"]:
        raise ValueError(
            "El turno contiene un error; no puede aprobarse."
        )

    if not estado["pendiente_hitl"]:
        raise ValueError("No hay ningún plan pendiente.")

    return aprobar_plan(
        estado["thread_id"],
        aprobar,
        estado,
    )


def _demo(args: Namespace) -> int:
    """Ejecuta una demo con HITL manual o aprobación explícita."""
    ubicacion = _ubicacion_desde_valores(
        getattr(args, "latitud", None),
        getattr(args, "longitud", None),
    )

    imagen = getattr(args, "imagen", None)
    mensaje = getattr(args, "mensaje", None)
    auto_aprobar = getattr(args, "auto_aprobar", False)

    argumentos: dict[str, Any] = {
        "auto_aprobar": False,
        "ubicacion": ubicacion,
        "ruta_imagen": imagen,
    }

    if mensaje is not None:
        argumentos["mensaje"] = mensaje

    estado = run_demo(**argumentos)

    _mostrar_estado(estado)

    if estado["error"]:
        return 1

    if not estado["pendiente_hitl"]:
        print("\nNo se alcanzó una pausa HITL válida.")
        return 1

    if auto_aprobar:
        print("\nAprobación automática solicitada para la demo.")

        estado = _decidir(estado, True)
        _mostrar_estado(estado)

        return 1 if estado["error"] or not estado["done"] else 0

    while True:
        decision = _pedir_texto(
            "\nDecisión [aprobar/rechazar/salir]: "
        )

        if decision is None or decision.lower() == "salir":
            print(
                "Plan sin aplicar. El checkpoint en memoria "
                "no persiste al cerrar el proceso."
            )
            return 0

        if decision.lower() not in {"aprobar", "rechazar"}:
            print("Escribe aprobar, rechazar o salir.")
            continue

        estado = _decidir(
            estado,
            decision.lower() == "aprobar",
        )

        _mostrar_estado(estado)

        return 1 if estado["error"] else 0


def _mostrar_ayuda() -> None:
    """Enumera las acciones que reconoce el modo interactivo."""
    print(
        "\nComandos:\n"
        "  aprobar            Aceptar el plan pendiente.\n"
        "  rechazar           Rechazarlo y permitir otra consulta.\n"
        "  /estado            Mostrar el estado actual.\n"
        "  /traza             Ver log y herramientas.\n"
        "  /ubicacion LAT LON Aportar coordenadas para próximos turnos.\n"
        "  /sin-ubicacion     Dejar de aportar coordenadas.\n"
        "  /imagen RUTA       Adjuntar imagen de assets/ al próximo turno.\n"
        "  /sin-imagen        Retirar la imagen preparada.\n"
        "  /nuevo             Iniciar otro hilo independiente.\n"
        "  /ayuda             Mostrar los comandos.\n"
        "  /salir             Terminar la sesión."
    )


def _interactivo(args: Namespace) -> int:
    """Permite realizar turnos y revisar planes de un mismo hilo."""
    max_turns = getattr(args, "max_turns", MAX_TURNS_DEFAULT)
    max_steps = getattr(args, "max_steps", None)

    ubicacion = _ubicacion_desde_valores(
        getattr(args, "latitud", None),
        getattr(args, "longitud", None),
    )

    imagen = getattr(args, "imagen", None)

    hilo = nuevo_thread_id()
    estado = crear_estado_producto(
        hilo,
        max_turns=max_turns,
    )

    print("ReciclaTIA - modo interactivo")
    print(f"Hilo: {hilo}")
    _mostrar_ayuda()

    while True:
        texto = _pedir_texto("\nReciclaTIA> ")

        if texto is None:
            return 0

        if not texto:
            continue

        comando = texto.lower()

        # -------------------------------------------------
        # SALIDA Y CONSULTA DEL ESTADO
        # -------------------------------------------------

        if comando in {"/salir", "salir"}:
            if estado["pendiente_hitl"]:
                print("Hay un plan pendiente, que no se aplicará.")

            print("Sesión terminada.")
            return 0

        if comando in {"/ayuda", "ayuda"}:
            _mostrar_ayuda()
            continue

        if comando == "/estado":
            _mostrar_estado(
                estado,
                mostrar_traza=False,
            )
            continue

        if comando == "/traza":
            _mostrar_traza(estado)
            continue

        # -------------------------------------------------
        # NUEVA CONVERSACIÓN
        # -------------------------------------------------

        if comando == "/nuevo":
            if estado["pendiente_hitl"]:
                print("El plan anterior quedará sin aplicar.")

            hilo = nuevo_thread_id()

            estado = crear_estado_producto(
                hilo,
                max_turns=max_turns,
            )

            ubicacion = None
            imagen = None

            print(f"Nueva conversación. Hilo: {hilo}")
            continue

        # -------------------------------------------------
        # UBICACIÓN PROPORCIONADA VOLUNTARIAMENTE
        # -------------------------------------------------

        if comando == "/sin-ubicacion":
            ubicacion = None
            print("Se ha retirado la ubicación de próximos turnos.")
            continue

        if comando.startswith("/ubicacion"):
            partes = texto.split()

            if (
                len(partes) != 3
                or partes[0].lower() != "/ubicacion"
            ):
                print("Uso: /ubicacion LATITUD LONGITUD")
                continue

            try:
                ubicacion = _ubicacion_desde_valores(
                    partes[1],
                    partes[2],
                )
            except ValueError as exc:
                print(f"Error: {exc}")
                continue

            print("Ubicación guardada para próximas consultas.")
            continue

        # -------------------------------------------------
        # IMAGEN PARA EL SIGUIENTE TURNO
        # -------------------------------------------------

        if comando == "/sin-imagen":
            imagen = None
            print("Imagen retirada.")
            continue

        if comando.startswith("/imagen"):
            partes = texto.split(maxsplit=1)

            if (
                len(partes) != 2
                or partes[0].lower() != "/imagen"
            ):
                print("Uso: /imagen assets/nombre.jpg")
                continue

            imagen = partes[1].strip()

            print(
                "Imagen preparada para el próximo turno: "
                f"{imagen}"
            )
            continue

        # -------------------------------------------------
        # DECISIÓN HUMANA SOBRE EL PLAN
        # -------------------------------------------------

        if comando in {"aprobar", "rechazar"}:
            try:
                estado = _decidir(
                    estado,
                    comando == "aprobar",
                )
            except (RuntimeError, ValueError) as exc:
                print(f"Error: {exc}")
                continue

            _mostrar_estado(estado)
            continue

        if comando.startswith("/"):
            print("Comando no reconocido. Consulta /ayuda.")
            continue

        # -------------------------------------------------
        # COMPROBACIONES PREVIAS AL TURNO
        # -------------------------------------------------

        if estado["pendiente_hitl"]:
            print("Primero debes aprobar o rechazar el plan.")
            continue

        if estado["done"]:
            print(
                "Conversación cerrada. Usa /nuevo para continuar."
            )
            continue

        if estado["turnos_consumidos"] >= estado["max_turns"]:
            print("Límite de turnos alcanzado. Usa /nuevo.")
            continue

        # -------------------------------------------------
        # PROCESAMIENTO MEDIANTE LA API PÚBLICA
        # -------------------------------------------------

        try:
            estado = procesar_turno(
                estado,
                texto,
                hilo,
                max_steps=max_steps,
                ubicacion=ubicacion,
                ruta_imagen=imagen,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Error: {exc}")
            continue

        # La imagen solo corresponde al turno procesado.
        imagen = None

        _mostrar_estado(estado)


def ejecutar_agente(args: Namespace) -> int:
    """Despacha los subcomandos de agente recibidos desde main.py."""
    accion = getattr(args, "accion", None)

    try:
        if accion == "demo":
            return _demo(args)

        if accion == "interactivo":
            return _interactivo(args)

    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Acción de agente no reconocida.",
        file=sys.stderr,
    )

    return 2
