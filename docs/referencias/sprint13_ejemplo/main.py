"""CLI — Live Review S13 multiagente · calidad del aire (alumno)."""

import argparse
import json

import config
from src.agent import aprobar_plan, nuevo_thread_id, procesar_turno, run_demo
from src.state import crear_estado


def _interactivo(max_turns: int, max_steps: int) -> dict:
    estado = crear_estado("")
    tid = nuevo_thread_id()
    print(f"Modo interactivo | thread_id={tid}")
    print("Vacío o 'salir' para terminar. Si hay HITL: 'aprobar' / 'rechazar'.\n")
    for _ in range(max_turns):
        if estado.get("pendiente_hitl"):
            print("HITL: plan pendiente. Escribe aprobar / rechazar")
            for item in estado.get("plan") or []:
                print(f"  - {item}")
        mensaje = input("Tú: ").strip()
        if not mensaje or mensaje.lower() in {"salir", "exit", "quit"}:
            break
        if mensaje.lower() in {"aprobar", "approve", "si", "sí", "yes"}:
            if estado.get("pendiente_hitl"):
                estado = aprobar_plan(tid, True, estado)
            else:
                print("(No hay plan pendiente)")
                continue
        elif mensaje.lower() in {"rechazar", "reject", "no"}:
            if estado.get("pendiente_hitl"):
                estado = aprobar_plan(tid, False, estado)
            else:
                print("(No hay plan pendiente)")
                continue
        else:
            estado = procesar_turno(estado, mensaje, tid, max_steps=max_steps)
        print("Agente:", estado.get("respuesta") or "(sin respuesta)")
        if estado.get("plan"):
            print("Plan:", estado["plan"])
        if estado.get("log"):
            print("Log:", estado["log"][-5:])
        print(
            "done:",
            estado.get("done"),
            "| hitl:",
            estado.get("pendiente_hitl"),
            "| error:",
            estado.get("error"),
        )
        print()
        if estado.get("done") or estado.get("error"):
            break
    estado["thread_id"] = tid
    return estado


def _resumen(estado: dict, max_turns: int, max_steps: int) -> None:
    traza = estado.get("traza") or []
    print("--- resumen ---")
    print(
        f"turnos: {len(traza)}/{max_turns} | done: {estado.get('done')} "
        f"| hitl: {estado.get('pendiente_hitl')} | error: {bool(estado.get('error'))}"
    )
    print(f"thread_id: {estado.get('thread_id')}")
    print(f"max_steps: {max_steps}")
    print("log:", estado.get("log"))
    print("---------------")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multiagente calidad del aire + LangGraph (alumno)"
    )
    parser.add_argument("--interactivo", action="store_true")
    parser.add_argument("--max-turns", type=int, default=config.MAX_TURNS_DEFAULT)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS_DEFAULT)
    parser.add_argument("--sin-auto-aprobar", action="store_true")
    parser.add_argument("--check", action="store_true", help="verificar TODOs (sin Gemini)")
    args = parser.parse_args()

    if args.check:
        from verificar import main as check_main

        check_main()
        return

    if args.interactivo:
        resultado = _interactivo(args.max_turns, args.max_steps)
    else:
        resultado = run_demo(
            max_turns=args.max_turns,
            max_steps=args.max_steps,
            auto_aprobar=not args.sin_auto_aprobar,
        )

    _resumen(resultado, args.max_turns, args.max_steps)
    print(json.dumps(resultado, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
