# 05 — Herramientas controladas de ReciclaTIA PB02

**Proyecto:** PB02 Multiagente — ReciclaTIA  
**Rama objetivo:** `feature/tools`  
**Fecha:** 20/10/2026  
**Método:** SDD (Specification-Driven Development)  
**Estado:** **BORRADOR PARA REVISIÓN. No implementado ni aprobado.**

## 1. Fuentes y situación comprobada

- Repositorio: https://github.com/alex-bometon/pb02-multiagente-equipo18 (lectura de `feature/tools`).
- A 20/10/2026, `feature/tools` y `develop` eran idénticas (0 ahead / 0 behind).
- Normas: `AGENTS.md`, `docs/referencias/LEER_ANTES.md`, `docs/referencias/pb02_enunciado.md`.
- `src.rag.responder(pregunta, *, top_k=TOP_K, query_embedding=None, client=None)` está operativo como interfaz de RAG según el código revisado; devuelve `pregunta`, `respuesta`, `abstencion`, `motivo`, `fuentes`, `chunks`, `metricas` y propaga excepciones técnicas.
- El RAG admite recuperación vectorial y consultas estructuradas. No modificarlo para crear la herramienta.
- `main.py` sigue siendo la CLI única. No modificarlo en esta fase.
- El ejemplo Sprint 13 es material didáctico: su mini-RAG no debe sustituir el sistema de ReciclaTIA.
- Las pruebas automatizadas indicadas en el traspaso anterior no se han reejecutado durante la elaboración de este documento.

## 2. Objetivo y alcance

**Objetivo:** exponer cuatro herramientas del dominio de reciclaje, controladas por una allowlist central de Python, con contratos de entrada y salida claros, presupuesto de ejecución, errores explícitos y trazabilidad para el futuro grafo LangGraph.

**Incluye:** adaptador RAG, fuente oficial externa HTTP + fallback, auxiliar determinista, análisis de imagen, registro y despacho seguros, tests sin depender de cuota de Gemini ni de red.

**No incluye:** supervisor, especialistas, LangGraph, HITL, Streamlit, cambio de `main.py`, cambios al RAG/ChromaDB/embeddings, geolocalización, búsqueda por proximidad, rutas ni predicción en tiempo real. No se generará un nuevo índice.

**Cobertura geográfica:** municipio de Madrid. La tool externa no debe aceptar consultas de otros municipios como si pertenecieran a Madrid.

## 3. Propuesta de estructura mínima

```text
src/
  tools/
    __init__.py            # Exportación de TOOLS, ejecutor y presupuesto; allowlist única
    rag_ask.py             # Adaptador fino a src.rag.responder
    api_puntos_limpios.py  # Lectura de servicio oficial + fallback
    hora_actual.py         # Función auxiliar determinista
    describir_imagen.py    # Lectura segura de asset + Gemini

data/
  api_puntos_limpios_fallback.json  # Snapshot documentado de fuente oficial
assets/
  imagen_residuos_demo.jpg          # Imagen apropiada para reciclaje
specs/
  05-tools.md                        # Este documento, al validarlo

tests/
  test_tools.py
  test_tools_api.py
  test_tools_multimodal.py
```

Se recomienda **no añadir `registry.py`** de entrada: `src/tools/__init__.py` puede mantener una allowlist pequeña y un ejecutor acotado. Si supera una responsabilidad manejable durante implementación, separar registro sin duplicar validaciones. No anticipar archivos vacíos.

## 4. Contratos públicos

### 4.1 `rag_ask(consulta: str)`

- Entrada: texto no vacío y acotado. Delegar la validación de longitud y dominio a `responder`; el adaptador puede verificar el tipo antes de invocarlo.
- Llamar **exactamente una vez** a `src.rag.responder(consulta)`, sin replicar retrieval, prompts ni generación.
- Devolver objeto serializable con `respuesta`, `abstencion`, `motivo`, `fuentes`, `metricas` y etiqueta `origen="rag_local"`. Mantener el texto y la procedencia originales; no inventar citas.
- Las abstenciones son resultados válidos, distintos de errores de ejecución. Un fallo de ChromaDB/Gemini se presenta al ejecutor como error técnico tipificado; no transformarlo en «no hay datos».
- No introducir un segundo cliente Gemini ni generar embeddings más allá de lo que `responder` efectúe.

### 4.2 `api_puntos_limpios(distrito: str = "", limite: int = 5)`

**Fuente propuesta:** Geoportal del Ayuntamiento de Madrid, servicio ESRI/ArcGIS REST «Puntos limpios». Su capa `MapServer/1` expone `TIPO_DATO`, `NOMBRE_DIS`, `Direccion`, `Ubicacion`, `ABIERTO`, `CERRADO`, `Residuos`, `PAGINA_WEB`, `OBJECTID_1`. Los metadatos publicados confirman compatibilidad con consultas y respuesta JSON.

- Ficha municipal: https://geoportal.madrid.es/IDEAM_WBGEOPORTAL/dataset.iam?id=2091e175-8561-4906-b103-8bce0d33bdf8
- Capa oficial: https://sigma.madrid.es/hosted/rest/services/RESIDUOS/PUNTOS_LIMPIOS/MapServer/1
- Endpoint a probar en integración: `GET https://sigma.madrid.es/hosted/rest/services/RESIDUOS/PUNTOS_LIMPIOS/MapServer/1/query`
- Parámetros fijos del cliente: `f=json`, `where="TIPO_DATO = 'Fijo'"`, `outFields` restringido a campos documentados, `returnGeometry=false`, paginación/límite explícitos. Usar `requests.get(..., params=..., timeout=...)`; no construir SQL mediante interpolación del usuario.

**Alcance v1:** **puntos limpios fijos** exclusivamente, para no mezclar horarios y modalidades. El nombre público de la tool debe aclarar esa restricción, por ejemplo `API_consultar_puntos_limpios_fijos`; mantener el nombre Python interno descriptivo si fuese necesario. Los móviles y los de proximidad siguen consultándose vía RAG cuando corresponda.

- `distrito`: opcional, normalizado únicamente para comparación local; no usarlo como filtro SQL del servicio ni como GPS. Distrito inexistente → resultado sin coincidencias, sin sustituirlo por otro.
- `limite`: entero genuino (rechazar booleanos), mínimo 1 y máximo de seguridad a fijar en `config.py` (propuesta: 10); resultados por orden estable, número total y señal `recortado` si procede.
- Devolver `puntos` con campos de texto documentados tal cual; no reinterpretar «ABIERTO» como abierto ahora, ni deducir una fecha de actualización que el servicio no aporte.
- Estructura de resultado: `origen` (`api_oficial` / `fallback_local`), `url_fuente`, `fecha_consulta`, `fecha_datos` (solo si consta; si no, `null`), `modalidad="fijo"`, `distrito_filtro`, `total_coincidencias`, `recortado`, `puntos`, `aviso`.
- Licencia y atribución del origen: Ayuntamiento de Madrid; comprobar/registrar condiciones de reutilización antes de publicar el producto. En los conjuntos públicos relacionados se declara CC BY 4.0.

**Fallos:** considerar HTTP no-2xx, timeout, JSON no válido, respuesta con `error` de ArcGIS o `features` ausente/mal formado. Si falla, usar un snapshot de puntos **realmente descargado de una fuente oficial**, situado en `data/api_puntos_limpios_fallback.json`, con `fecha_captura`, `fuente_original`, `modalidad`, `records` y nota inequívoca «no actualizado en tiempo real». **Prohibido fabricar puntos u horarios** para completar ese archivo. Si no existe snapshot verificado o está corrupto, devolver error explícito y no datos falsos.

**Estado de verificación:** se han contrastado el servicio, la capa y sus campos publicados. La respuesta real a `GET /query` **no se ha podido comprobar desde el entorno de revisión**; es una condición bloqueante para aprobar el endpoint definitivo. Si falla, evaluar la descarga JSON oficial del dataset «Puntos limpios fijos» (`200284-0-puntos-limpios-fijos`) como alternativa HTTP, con parser específico y una prueba equivalente.

### 4.3 `hora_actual()`

- Sin argumentos, sin llamadas HTTP ni LLM.
- Devolver ISO 8601 con zona `Europe/Madrid` de `zoneinfo.ZoneInfo`, con campo `zona_horaria`. No depender de la zona horaria implícita del sistema.
- Para tests, inyectar/congelar reloj; no exigir que dos invocaciones separadas coincidan en la hora exacta.
- Fecha/hora actual no equivale a garantía de que una instalación esté abierta.

### 4.4 `describir_imagen(nombre_imagen: str = "imagen_residuos_demo.jpg")`

- Entrada limitada a **nombres de assets expresamente autorizados**, no rutas arbitrarias, URLs, `..`, symlinks externos ni ficheros fuera de `assets/`.
- Admitir JPEG/PNG después de verificar tipo real de fichero (Pillow), tamaño máximo propuesto **5 MiB**, y límite de dimensiones/píxeles a fijar antes de codificar.
- Reutilizar `src.gemini_client.crear_cliente_gemini()` y modelo configurado en `config.py`; no crear otra gestión de claves API.
- Imagen→descripción breve en español sobre objetos/residuos observables; diferenciar objetos identificados de incertidumbre y evitar deducir automáticamente el contenedor correcto sin evidencia RAG.
- Devolver `descripcion`, `imagen` (nombre seguro), `origen="gemini_multimodal"`, y estado/error. Nunca devolver bytes ni clave API en logs.
- Asset de demostración nuevo y con licencia/autoría documentada; la imagen docente de calidad del aire no se reutiliza.

## 5. Allowlist, límites y trazabilidad

- Una **sola** definición de tools permitidas en Python (`TOOLS`, `TOOL_BY_NAME`) con nombres públicos estables. Exponer el ejecutor desde `src.tools` para su futuro consumo por LangGraph.
- Solo los nombres exactos registrados pueden ejecutarse. Rechazar la tool desconocida **antes de tocar** ninguna dependencia. Prohibir argumentos extra; validar tipos/cotas mediante firmas o esquemas de `@tool`.
- `ejecutar_tool(nombre, args, presupuesto)` recibe un presupuesto mutable explícito para el ciclo del futuro agente (sin contadores globales entre usuarios). Una reserva cuenta como operación consumida antes de invocar la función. `max_operaciones` se impone en Python, incluso si una respuesta del LLM pide varias tools simultáneas.
- Definir límites `max_steps` (iteraciones LLM en futuros especialistas) y `max_operaciones` (llamadas efectivamente despachadas a tools) como conceptos **distintos**. El presupuesto debe ser compartido por el bucle al que se aplique, no reiniciado por cada tool call.
- Trazas de cada intento: `nombre`, `estado` (`ok`, `rechazado`, `error`), `duracion_ms`, `argumentos_seguros`, `origen` cuando proceda y error no sensible. Las tools desconocidas y llamadas bloqueadas también generan evento de traza, sin ejecutarse.
- Evitar que errores de validación o de tool derriben el orquestador; devolver resultado estructurado compatible con serialización JSON y conservar causa técnica segura. No confundir error técnico con abstención RAG.
- No registrar claves, contenidos de imágenes, texto completo de prompts, ni respuestas íntegras por defecto. Acotar longitudes de respuesta para el futuro contexto del LLM **sin recortar silenciosamente evidencias**.
- No hacer una llamada HTTP ni a Gemini al importar `src.tools`.

## 6. Configuración y dependencias

- Centralizar exclusivamente los parámetros compartidos nuevos en `config.py`: URL base y tiempo de espera HTTP, rango de `limite`, ruta de fallback, ruta de assets, tamaño máximo de imágenes y presupuesto por defecto. No duplicar sus valores en módulos distintos.
- Preferir dependencias ya presentes: `requests`, `google-genai`, `langchain-core`, `Pillow`, `pytest`, `python-dotenv`. No agregar paquetes si no se demuestra necesidad.
- Las tools y sus tests no importarán Streamlit, ni grafo, ni módulos docentes de `docs/referencias/`.

## 7. Pruebas de aceptación (sin cuotas externas)

1. Mock de `responder`: **una** llamada; resultado válido, abstención, fuentes y error técnico conservados según contrato.
2. Rechazo previo de nombre de tool no autorizada, parámetros inesperados, `None`, cadenas vacías y booleanos donde se pide entero.
3. Presupuesto de una operación + dos tool calls: se ejecuta como máximo una, la segunda queda bloqueada y trazada. No se reinicia el contador de forma encubierta.
4. Mock HTTP: registros de puntos limpios fijos correctos, filtrado por distrito, tope y `recortado`, campos no presentes, sin URLs/consultas SQL proporcionadas por el modelo.
5. Mock HTTP: timeout, 500, JSON inválido, error ArcGIS y `features` mal formado → fallback etiquetado con fuente y fecha; fallback inexistente/corrupto → error explícito.
6. Fecha: hora ISO con zona Europe/Madrid y comportamiento reproducible mediante mock.
7. Mock Gemini: JPEG/PNG de assets aceptados, respuesta/errores serializables; ruta absoluta, traversal, symlink externo, formato prohibido, tamaño excesivo y archivo inexistente rechazados **sin invocar Gemini**.
8. Los imports de tools no requieren clave Gemini ni acceso a red. Todos los tests unitarios corren offline.
9. La suite previa de RAG continúa verde: `python -m pytest tests/ -q --tb=short`.
10. **Smoke test de integración real** (separado, opt-in): comprobar respuesta JSON/HTTP del endpoint oficial, campos esperados y la atribución. Si falla, no presentar como integración completada.

## 8. Plan de implementación tras aprobación

1. Resolver el bloqueo: prueba HTTP real de `MapServer/1/query`, verificación de esquema y snapshot oficial verificable para fallback; elegir endpoint definitivo documentando su procedencia.
2. Ajustar `config.py` solo con constantes justificadas; añadir el fichero de especificación validado.
3. Implementar adaptadores individuales RAG, hora, API y multimodal con contratos estables. Introducir imagen y snapshot únicamente cuando exista evidencia auténtica.
4. Implementar registro/dispatcher y presupuesto, reutilizando los contratos; evitar duplicaciones.
5. Añadir tests aislados y ejecutar suite completa. Corregir sin introducir cambios ajenos a esta fase.
6. Revisar trazas, atribución, instrucciones de uso y diff; después, incorporación manual a `feature/tools` y PR a `develop` por parte del desarrollador.

## 9. Decisiones pendientes de aprobación

- [ ] Confirmar la fuente externa: ArcGIS REST (preferida) tras smoke test real, o descarga JSON municipal si aquella falla.
- [ ] Validar que la API de la v1 abarque **solo puntos limpios fijos**; otras modalidades por RAG.
- [ ] Confirmar nombres exactos de la API pública y límites compartidos (`limite<=10`, 5 MiB, max_operaciones).
- [ ] Confirmar imagen de demostración y procedencia de snapshot fallback.
- [ ] Aprobar la especificación antes de implementar, de acuerdo con `AGENTS.md`.

**Cambios en GitHub: ninguno.** Este documento es un entregable local propuesto, no un commit ni evidencia de ejecución de tests.
