# Manual de uso de ReciclaTIA

Este manual explica cómo instalar ReciclaTIA y utilizarla sin necesitar conocimientos de agentes, embeddings ni bases de datos vectoriales. Al final también incluyo el modo de terminal y algunas comprobaciones útiles para revisar el proyecto.

## 1. Qué necesitas

- Un ordenador con Python 3.11 o 3.12.
- Conexión a Internet para utilizar Gemini.
- Una clave de API de [Google AI Studio](https://aistudio.google.com/apikey).
- El repositorio de ReciclaTIA, en la rama `main`.

La aplicación se ejecuta en tu ordenador. No necesitas desplegarla en un servidor ni crear una cuenta de usuario.

## 2. Instalación

Abre una terminal y ejecuta:

```bash
git clone https://github.com/alex-bometon/pb02-multiagente-equipo18.git
cd pb02-multiagente-equipo18
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

En Windows PowerShell, utiliza `.venv\Scripts\Activate.ps1` y `Copy-Item .env.example .env` en lugar de los dos comandos equivalentes de macOS/Linux.

Edita `.env` para indicar `GEMINI_API_KEY`. Mantén la clave fuera de Git. La opción de trazabilidad LangSmith del ejemplo es opcional; no necesitas configurarla para utilizar ReciclaTIA.

### Crear o reconstruir el índice

El asistente consulta un índice local de ChromaDB. Hay dos situaciones:

**Tienes `output/embeddings.json`, pero no `output/chroma_db/`:**

```bash
python main.py rag index --yes
```

**No tienes los embeddings preparados o has cambiado el corpus o el modelo de embeddings:**

```bash
python main.py rag index --full --yes
```

La segunda opción implica llamadas a Gemini. No cambies las dimensiones del modelo ni elimines los artefactos sin revisar antes cómo se construyó el índice: los vectores de las preguntas deben ser compatibles con los almacenados.

## 3. Abrir ReciclaTIA

Con el entorno virtual activo, ejecuta desde la carpeta principal:

```bash
python -m streamlit run app.py
```

La terminal mostrará una dirección local, habitualmente `http://localhost:8501`. Ábrela en el navegador si no se abre sola.

La pantalla tiene un **chat central** y una **barra lateral** con los ajustes. Los botones de ejemplos son consultas normales: puedes utilizarlos para familiarizarte con la aplicación.

## 4. Hacer tu primera consulta

Prueba con:

> ¿Dónde tiro una botella de plástico?

ReciclaTIA buscará información que respalde la respuesta. El especialista correspondiente consulta el RAG; la respuesta puede incluir referencias documentales. Después aparecerá un plan para que lo revises.

Otra consulta útil:

> ¿Qué es un punto limpio móvil?

Puedes consultar definiciones y horarios publicados, pero un horario documentado no equivale a confirmación de apertura o presencia física en ese momento.

### ¿Qué significa «Revisar el plan»?

El sistema se detiene antes de terminar. Esta pausa se llama **HITL** (*human in the loop*).

**Aprobar** significa que aceptas el plan y cierras esa conversación. **Rechazar** significa que no lo aceptas y quieres corregirlo: puedes escribir, por ejemplo, «me refería a un punto limpio fijo» o «quiero cambiar de distrito». No tienes que abrir otro hilo para reformular.

La aprobación **no ordena ninguna actuación real**. Solo registra tu decisión en la conversación.

## 5. Buscar puntos limpios cercanos

Abre la barra lateral y activa **Aportar coordenadas**. Introduce una latitud y una longitud de la ubicación que quieras consultar. Los valores que aparecen inicialmente son un ejemplo del centro de Madrid: **no son una lectura del GPS ni tu ubicación real**.

Después puedes preguntar:

> Busca puntos limpios de proximidad cerca de estas coordenadas.

Si hay resultados geográficos, la aplicación puede mostrar una tabla y un mapa. El resultado distingue si procede de la **API municipal** o del **corpus local de respaldo**. Este último puede contener información desactualizada.

El radio predeterminado de la herramienta geográfica es de 2.000 metros y su máximo permitido es de 5.000 metros. La distancia calculada es geográfica, en línea recta. ReciclaTIA no calcula itinerarios.

Si no proporcionas coordenadas, no debería iniciar una búsqueda de proximidad usando una ubicación supuesta. Te pedirá la información que falta.

### Horarios y disponibilidad

Para algunas consultas el agente compara la hora de Madrid con los horarios documentados de los puntos limpios. Si indica que uno está **dentro del horario publicado**, solo está comparando esos datos. Puede haber festivos, incidencias o cambios que no figuren en el corpus. Antes de desplazarte, compruébalo con el servicio municipal.

## 6. Consultar con una fotografía

En la barra lateral, utiliza **Adjuntar imagen de un residuo**. Se admiten JPG/JPEG o PNG, con un tamaño máximo de 5 MiB y una resolución de hasta 16 megapíxeles.

Puedes escribir:

> ¿Qué objeto aparece en esta imagen y dónde debería depositarlo?

Gemini describe lo que ve. Si un material no se distingue con claridad, esa incertidumbre debe mantenerse. Para recomendar un contenedor, el agente necesita documentación del RAG; la foto por sí sola no demuestra el destino correcto.

La imagen se guarda temporalmente en una subcarpeta de `assets/` para procesar el turno y se elimina cuando termina ese procesamiento. Aun así, su contenido se envía al proveedor del modelo para el análisis visual: evita incluir información personal o sensible.

## 7. Controles de la barra lateral

| Control | Para qué sirve |
|---|---|
| Nueva conversación | Crea otro `thread_id` y limpia el chat de esa sesión |
| Máximo de turnos | Limita las consultas del hilo; se ajusta antes del primer mensaje |
| Rondas LLM por especialista | Limita las rondas del modelo durante el procesamiento |
| Aportar coordenadas | Autoriza a utilizar las coordenadas que escribas |
| Adjuntar imagen | Envía una imagen válida a la siguiente consulta |
| Mostrar mapas | Activa o desactiva la representación geográfica |
| Mostrar depuración | Muestra el estado técnico, las decisiones del grafo y la traza de herramientas |
| Aprobar / Rechazar | Resuelve el plan que está esperando revisión humana |

El contador de turnos es un límite de uso del hilo. Aprobar o rechazar el plan no se considera una pregunta nueva.

## 8. Si no obtienes respuesta

**Falta la clave Gemini.** Revisa el archivo `.env`, su nombre y que `GEMINI_API_KEY` tenga un valor válido.

**No se encuentra ChromaDB o el índice está vacío.** Reconstruye el índice según el apartado 2. Si cambiaste de modelo o dimensiones de embeddings, regenera también los artefactos.

**Límite de cuota, red o error del proveedor.** El sistema depende de Gemini y de servicios externos. Reintentar repetidamente no resuelve una cuota agotada; revisa tu acceso al proveedor.

**La API de puntos limpios no responde.** La herramienta puede recurrir al corpus local e indicarlo en la fuente. Eso no convierte los datos locales en mediciones o horarios en directo.

**La conversación está bloqueada.** Puede haber un plan pendiente, haberse aprobado ya, haberse alcanzado el límite de turnos o existir un error técnico. Revisa el aviso y utiliza los botones HITL o **Nueva conversación** según corresponda.

**La aplicación no sabe qué contenedor corresponde.** A veces la fuente no permite identificar un material o un destino concreto. Reformula la pregunta indicando el tipo de objeto, su composición y cualquier característica relevante; el sistema no debe completar vacíos inventando información.

## 9. Uso desde la terminal

No necesitas Streamlit para comprobar el núcleo del proyecto.

```bash
# Buscar fragmentos relevantes
python main.py rag query "¿Dónde tiro una botella de plástico?"

# Recibir una respuesta documental con fuentes
python main.py rag ask "¿Dónde tiro una botella de plástico?"

# Abrir la demo multiagente y decidir manualmente
python main.py agente demo "¿Dónde van las pilas?"

# Mantener una conversación con el agente
python main.py agente interactivo --max-turns 6 --max-steps 4
```

En el modo interactivo funcionan `aprobar`, `rechazar`, `/estado`, `/traza`, `/nuevo`, `/ayuda` y `/salir`. Para facilitar coordenadas en ese modo, utiliza `/ubicacion LAT LON`; `/sin-ubicacion` las retira. Para preparar una foto, `/imagen assets/archivo.jpg`; `/sin-imagen` la cancela. Los comandos y sus rutas se comprueban desde Python.

También puedes pasar los datos a una demo directamente:

```bash
python main.py agente demo "Busca puntos limpios cercanos" \
  --latitud 40.4168 --longitud -3.7038

python main.py agente demo "Identifica el objeto de la foto" \
  --imagen assets/capsula.jpg
```

## 10. Alcance de la información

ReciclaTIA cubre **Madrid ciudad** con las fuentes incorporadas al proyecto. No debe usarse para atribuir reglas de reciclaje a otros municipios. No identifica objetos con certeza a partir de todas las fotografías, no informa de incidencias en tiempo real y no ejecuta recogidas, reservas, solicitudes ni cambios en sistemas municipales.

Si quieres profundizar en el diseño, los documentos de `specs/` explican el RAG, el grafo, los contratos de las herramientas, el estado y las pruebas.
