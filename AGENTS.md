# AGENTS.md — PB02 Multiagente

## Contexto
Proyecto académico individual de AI Engineering.
Entrega: 25/10/2026.

## Fuentes de requisitos
- PB2: docs/referencias/pb02_enunciado.md
- PB1 RAG: docs/referencias/pb01_enunciado.md
- Referencia docente: docs/referencias/sprint13_ejemplo/

El ejemplo docente contiene implementaciones incompletas.
No copiarlo como solución final.

## Metodología SDD
Para cada funcionalidad:
1. Definir especificación y alcance.
2. Proponer diseño y plan.
3. Definir criterios de aceptación.
4. Implementar cambios acotados.
5. Ejecutar pruebas.
6. Revisar resultados y documentar.

No implementar funcionalidades cuya especificación
aún no haya sido aprobada.

## Arquitectura
- Python y LangGraph.
- RAG completo e independiente.
- Separación entre dominio, orquestación y UI.
- Supervisor y especialistas diferenciados.
- Herramientas controladas mediante allowlist.
- HITL obligatorio antes de aplicar.
- CLI y Streamlit sobre un contrato común.

## Calidad
- Código modular, tipado y mantenible.
- Tests para comportamientos importantes.
- Manejo explícito de errores.
- Límites impuestos mediante Python.
- Sin invención de evidencia del RAG.
- Sin secretos en código ni logs.

## Trabajo con Codex
- No asumir requisitos no documentados.
- Distinguir requisitos obligatorios de mejoras.
- Explicar decisiones arquitectónicas relevantes.
- No añadir dependencias sin justificación.
- No realizar commits ni push sin autorización.
- No introducir cambios ajenos a la tarea.
- Comunicar las pruebas ejecutadas y sus resultados.