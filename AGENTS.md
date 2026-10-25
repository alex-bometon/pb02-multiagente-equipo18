# Instrucciones de trabajo — ReciclaTIA

## Proyecto

ReciclaTIA es un proyecto académico individual de AI Engineering. La rama de referencia para el producto es la rama `main`; no se debe describir otra rama como si contuviera necesariamente el mismo código.

El ámbito es la gestión de residuos y puntos limpios de Madrid ciudad. El RAG es un componente independiente del multiagente. Las herramientas, el grafo, el orquestador y la interfaz tienen responsabilidades distintas.

## Referencias

- `docs/referencias/pb01_enunciado.md`: requisitos de la base RAG.
- `docs/referencias/pb02_enunciado.md`: requisitos del sistema multiagente.
- `docs/referencias/sprint13_ejemplo/`: material docente, no código del producto.
- `specs/`: contratos técnicos del proyecto.

No editar el material original de `docs/referencias/` para hacerlo pasar por trabajo propio.

## Trabajo con el código

1. Revisar la especificación del componente y el comportamiento real en la rama de trabajo.
2. Mantener el RAG reutilizable sin depender de Streamlit o LangGraph.
3. Ejecutar tools solo a través de su registro y sus permisos de rol.
4. Aplicar límites, validaciones y decisiones HITL desde Python, no confiarlos al modelo.
5. No duplicar lógica documental, ejecución de herramientas ni validación de estado en la UI.
6. Añadir pruebas para cambios de comportamiento y revisar regresiones.
7. Describir las pruebas realmente ejecutadas, sin inventar resultados.
8. No realizar commits, merges ni push sin autorización expresa.

## Documentación

- `README.md`: lenguaje cotidiano, explicación del producto y primeros pasos para cualquier lector.
- `docs/manual_uso.md`: instrucciones concretas para instalar, usar y resolver problemas.
- `specs/`: redacción técnica; contratos, invariantes, diagramas y criterios de prueba.
- `entregables/`: justificación de decisiones y evidencia de ejecución cuando exista.


Nunca incorporar claves API, datos personales innecesarios ni resultados inventados en documentación, pruebas o capturas.
