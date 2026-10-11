"""Comprueba que los TODOs de la Live Review S13 están rellenados (sin Gemini)."""

from __future__ import annotations

import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _codigo_fuente(obj) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ""


def _tiene_raise_not_implemented(codigo: str) -> bool:
    return "NotImplementedError" in codigo


def _fallback_ok() -> tuple[bool, str]:
    from src.graph import _fallback_siguiente

    codigo = _codigo_fuente(_fallback_siguiente)
    if _tiene_raise_not_implemented(codigo):
        return False, "Completa _fallback_siguiente (quita NotImplementedError)"

    casos = [
        ({"notas_diagnostico": "", "notas_practico": "", "respuesta": ""}, "diagnostico"),
        (
            {"notas_diagnostico": "hay datos", "notas_practico": "", "respuesta": ""},
            "practico",
        ),
        (
            {
                "notas_diagnostico": "diag",
                "notas_practico": "prac",
                "respuesta": "",
            },
            "sintetizar",
        ),
        (
            {
                "notas_diagnostico": "diag",
                "notas_practico": "prac",
                "respuesta": "plan",
            },
            "FINISH",
        ),
    ]
    for estado, esperado in casos:
        try:
            got = _fallback_siguiente(estado)  # type: ignore[arg-type]
        except NotImplementedError:
            return False, "Completa _fallback_siguiente"
        if got != esperado:
            return False, f"_fallback_siguiente({estado!r}) → {got!r}, esperaba {esperado!r}"
    return True, "_fallback_siguiente OK"


def _build_graph_ok() -> tuple[bool, str]:
    from src import graph as gmod

    codigo = _codigo_fuente(gmod.build_graph)
    if _tiene_raise_not_implemented(codigo):
        return False, "Completa build_graph (quita NotImplementedError)"

    # Sin invocar LLM: solo montar el grafo
    try:
        app = gmod.build_graph()
    except NotImplementedError:
        return False, "Completa build_graph"
    except Exception as e:  # noqa: BLE001
        return False, f"build_graph falló: {e}"

    # Heurística: el código debe mencionar interrupt y MemorySaver
    if "interrupt_before" not in codigo:
        return False, "build_graph debe usar interrupt_before=['aplicar']"
    if "MemorySaver" not in codigo:
        return False, "build_graph debe usar checkpointer=MemorySaver()"
    if "add_conditional_edges" not in codigo:
        return False, "build_graph debe usar add_conditional_edges"
    _ = app
    return True, "build_graph OK"


def _procesar_turno_ok() -> tuple[bool, str]:
    from src import agent as amod

    codigo = _codigo_fuente(amod.procesar_turno)
    if _tiene_raise_not_implemented(codigo):
        return False, "Completa procesar_turno (quita NotImplementedError)"

    # Mensaje vacío no debe lanzar
    try:
        out = amod.procesar_turno({"traza": []}, "   ", "check-thread")
    except NotImplementedError:
        return False, "Completa procesar_turno"
    except Exception as e:  # noqa: BLE001
        return False, f"procesar_turno(mensaje vacío) falló: {e}"

    if not (out.get("respuesta") or "").strip():
        return False, "procesar_turno con mensaje vacío debe devolver respuesta"

    # Debe llamar a invoke / get_app en el cuerpo (no solo comentarios)
    if "get_app" not in codigo or "invoke" not in codigo:
        return False, "procesar_turno debe usar get_app().invoke(...)"
    if "_marcar_hitl" not in codigo:
        return False, "procesar_turno debe llamar _marcar_hitl"
    return True, "procesar_turno esqueleto OK (sin llamar Gemini)"


def main() -> None:
    checks = [
        ("TODO [siguiente] _fallback_siguiente", _fallback_ok),
        ("TODO [edges] build_graph", _build_graph_ok),
        ("TODO [turno] procesar_turno", _procesar_turno_ok),
    ]
    ok_all = True
    for nombre, fn in checks:
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, str(e)
        tag = "[OK]" if ok else "[FAIL]"
        print(f"{tag} {nombre}: {msg}")
        ok_all = ok_all and ok

    if ok_all:
        print("\nTodos los checks estructurales OK. Prueba: python main.py")
    else:
        print("\nAún hay TODOs. Mira los [FAIL] de arriba.")


if __name__ == "__main__":
    main()
