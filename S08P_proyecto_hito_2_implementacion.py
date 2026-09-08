# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# <div style="display: flex; justify-content: space-between; align-items: center; padding: 0 0; max-width: 100%;">
#   <a href="https://dcc.uchile.cl/" target="_blank">
# <img src="public/logos/logo_dcc.png" style="height: 140px; max-width: 100%; width: auto;" alt="Departamento de Ciencias de la Computación - Universidad de Chile" />
#   </a>
#   <a href="https://idia.uchile.cl/" target="_blank">
# <img src="public/logos/logo_idia.png" style="height: 140px; max-width: 100%; width: auto;" alt="Iniciativa de Datos e Inteligencia Artificial - Universidad de Chile" />
#   </a>
# </div>
#
# <div style="text-align: center; padding: 0.3rem 0 0.5rem;">
#   <p style="font-style: italic; color: #888; margin: 0 0 0.6rem; text-align: center;">Proyecto · Clase Práctica 8</p>
#   <h1 style="margin: 0 0 0.5rem; text-align: center;">Hito 2 — Primera Implementación del Agente</h1>
#   <p style="font-weight: bold; font-size: 1.05rem; margin: 0; text-align: center;">Automatización Inteligente con Agentes de IA</p>
# </div>
#
# <br>
#
# <div style="max-height: 500px; overflow: hidden; margin: 0 auto;">
#   <img src="public/stock/pexels-marek-piwnicki-3907296-23886059.jpg" style="width: 100%; margin-top: -100px;" alt="Cumbres nevadas cubiertas de nubes al atardecer" />
# </div>
# <p style="text-align: center; font-size: 0.75rem; color: #bbb; margin: 0.3rem 0 0;">
#   Foto: <a href="https://www.pexels.com/photo/clouds-covering-snowcapped-mountain-peaks-at-sunset-23886059/" target="_blank" style="color: #bbb;">Marek Piwnicki</a> en Pexels
# </p>

# %% [markdown]
# ## Objetivos de aprendizaje
#
# Al finalizar esta sesión, ustedes podrán:
#
# - [ ] Diseñar un prompt estructurado y específico para el agente de su proyecto.
# - [ ] Usar un documento propio mediante el módulo visual de RAG ya provisto.
# - [ ] Opcional: implementar una segunda tool relevante para el proyecto.
# - [ ] Probar que el agente responde con evidencia y reconoce cuando no la tiene.

# %% [markdown]
# <!-- IMPORTANT -->
# <div style="padding:16px 20px; margin:16px 0; border-left:4px solid #8957e5; background:rgba(137,87,229,.12); color:inherit; border-radius:4px;">
#   <strong>📌 Antes de empezar</strong><br>
#
#   <p style="margin:8px 0 0; line-height:1.6; color:inherit">
#
# 1. Verifiquen que descomprimieron el material dentro de `Curso-IA`, de modo que quede `Curso-IA\Proyecto_1\`.
# 2. Abran un terminal **PowerShell** en la carpeta `Curso-IA` (desde Visual Studio Code o desde donde prefieran) y activen el ambiente con `.venv\Scripts\activate.ps1`. Sabrán que quedó activo porque cada nueva línea del terminal parte con `(.venv)` en verde.
# 3. Con el ambiente ya activo, inicien Jupyter en ese mismo terminal con `jupyter-lab`. El orden importa: si lanzan Jupyter antes de activar, el notebook no encontrará las librerías del curso. Mantengan el terminal abierto mientras trabajan.
# 4. El navegador debería abrirse solo. Si no lo hace, copien el enlace que imprimió el terminal (parte con `http://localhost:8888/`) y péguenlo en el navegador.
# 5. En JupyterLab entren a `Proyecto_1` y abran este notebook desde ahí.
#
#   </p>
# </div>

# %% [markdown]
# ---
#
# ## 1. Encuadre y Alcance

# %% [markdown]
# En el Hito 1 definieron una tarea del trabajo del conocimiento y el proceso que quieren apoyar. 
#
# En este paso, deberán construir una primera versión de este: el prompt del agente principal, que entiende el propósito del proyecto, guarda conversaciones y memorias relevantes, consulta la documentación pertinente y responde de forma controlada.
#
# > **Importante**: No intenten implementar todo el proyecto. El objetivo de esta entrega es que el prompt y el uso de memoria/documentación funcione correctamente.

# %% [markdown]
# <!-- IMPORTANT -->
# <div style="padding:16px 20px; margin:16px 0; border-left:4px solid #8957e5; background:rgba(137,87,229,.12); color:inherit; border-radius:4px;">
#
#   <strong>📌 Primera entrega de código</strong><br>
#
#   <p style="margin:8px 0 0; line-height:1.6; color:inherit">
#
# En este hito, la infraestructura de la aplicación ya viene resuelta en `app_proyecto.py`: modelo, carga de documentos, RAG, memoria, trazas e interfaz en Streamlit.
#
# Para esta entrega deberán trabajar sobre esa base y completar tres elementos:
#
# * Diseñar el **prompt de su agente** (archivo `agente_proyecto/prompt.py`);
# * Agregar documentación y realizar consultas al **RAG** desde la interfaz gráfica;
# * Opcionalmente: agregar una **segunda tool** relevante para su proyecto (archivo `agente_proyecto/tools.py`).
#
# La entrega corresponde a `app_proyecto.py`, la carpeta `agente_proyecto/` y
# cualquier otro archivo necesario para ejecutar el proyecto.
#
#   </p>
# </div>
#

# %% [markdown]
# ---
# ## 2. El scaffolding (andamiaje) disponible

# %% [markdown]
# Antes de modificar código, identifiquen qué ya está resuelto. 
#
# La aplicación `app_proyecto.py` contiene una serie de módulos visuales para preparar documentos, agregarlos a la base de conocimiento, consultar las trazas y conversar con el agente.

# %% [markdown]
# | Componente | Estado | Qué deben hacer |
# |---|---|---|
# | **Configuración y cliente del modelo** | Ya implementado | No modificarlo. |
# | **Módulo para cargar documentos** | Ya implementado | Usarlo desde la interfaz. No construir otra ingesta. |
# | **Vector store y embeddings** | Ya implementado | Cargar al menos un documento propio y pertinente. |
# | **`buscar_vector_store`** | Ya implementado | Mantenerlo como la tool de consulta al RAG. |
# | **Memoria de corto plazo** | Ya implementada | No cambiar `checkpointer`, `thread_id` ni `st.session_state`. |
# | **Trazas del agente** | Ya implementadas | No agregar otra inicialización de tracing. |
# | **Interfaz gráfica en `Streamlit`** | Ya implementada | No reconstruir la navegación ni los módulos. |
# | **`agente_proyecto/prompt.py`** | Punto principal del hito | Reemplazar `SYSTEM_PROMPT` por el prompt del equipo. |
# | **`agente_proyecto/tools.py`** | RAG ya implementado; extensión opcional | Mantener `buscar_vector_store`. Opcionalmente, agregar otras tools relevantes y registrarlas en `TOOLS_PROYECTO`. |

# %% [markdown]
# ### 2.1 Trasladen el trabajo implementado hasta el momento
#
# En este hito continuarán el desarrollo del agente que han construido en las
# sesiones anteriores.
# Antes de ejecutar `app_proyecto.py`, trasladen su trabajo anterior a los nuevos archivos usando las siguientes instrucciones:
#
# > **Importante:** **No copien directamente `app_proyecto.py` ni `agente.py`** desde las versiones anteriores ya que las anteriores carecen de algunas de las funcionalidades presentes en la versión actual.
#
# 1. Si deciden conservar tools adicionales de su proceso, abran
#    `agente_proyecto/tools.py` y copien las funciones `@tool` propias desde los
#    archivos homónimos. Trasladen también los imports que esas funciones
#    necesiten y adapten las rutas al material de esta clase.
# 2. Copien el `SYSTEM_PROMPT` a `agente_proyecto/prompt.py`. Después agreguen las reglas de esta sesión sobre recuperación, selección de fuentes y límites.
# 3. Mantengan en `agente_proyecto/tools.py` las funciones
#    `configurar_base_de_conocimiento` y `buscar_vector_store`, ya que la
#    aplicación las utiliza para conectar LanceDB con la tool de RAG.
# 4. Si agregan tools adicionales, regístrenlas junto a `buscar_vector_store` en
#    `TOOLS_PROYECTO`, por ejemplo:
#
#    ```python
#    TOOLS_PROYECTO = [buscar_vector_store, tool_personalizada_q, tool_personalizada_2, ...]
#    ```
#
#    Usen los nombres reales de las tools que han construido hasta el momento.
#
#
#

# %% [markdown]
# ---
# ## 3. Diseñen el prompt como contrato

# %% [markdown]
# Recordemos que en la clase S03 trabajaron el prompt como un **contrato** que define el comportamiento esperado del agente. Entre otros elementos, consideraron:
#
# * **Rol:** quién es el agente y para qué existe.
# * **Tarea:** qué debe resolver ante una consulta.
# * **Contexto:** qué necesita saber sobre el dominio, proceso, organización y usuario.
# * **Instrucciones y restricciones:** qué puede hacer, qué no debe hacer y qué reglas debe respetar.
# * **Estrategia de resolución:** qué pasos generales debe seguir para abordar una consulta, sin necesidad de exponer su razonamiento interno.
# * **Formato de salida:** cómo debe estructurar y presentar la respuesta.
# * **Manejo de incertidumbre:** qué hacer cuando falta información o no existe evidencia suficiente para responder.
# * **Seguridad y prompt injection:** cómo actuar frente a instrucciones maliciosas, consultas fuera de alcance o instrucciones encontradas dentro de documentos.
# * **Ejemplos *few-shot*:** casos que muestran el comportamiento y las respuestas esperadas.
#
# Más recientemente incorporaron además:
#
# * **Definición y uso de tools:** qué herramientas tiene disponibles, para qué sirve cada una y cuándo corresponde utilizarlas.
# * **Uso de fuentes y evidencia:** cómo trabajar con la información recuperada desde el RAG u otras tools, y cómo distinguirla de inferencias o supuestos.
#
# Apliquen estas ideas al **prompt del sistema de su propio proyecto**.
#
# No copien un prompt genérico. Cada regla debe tener una función concreta dentro del proceso que definieron para su agente.
#

# %% [markdown]
# ### 3.1 Componentes del Prompt
#
# Revisen su prompt si contiene bien definidos los componentes de esta lista:
#
# > **Importante**: Revisar solo los elementos relevantes para su problema, puede que no todos sean necesarios.
#
# | Componente                        | Pregunta de diseño                                                                                                                                   |
# | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Rol**                           | ¿Quién es el agente y para qué existe?                                                                                                               |
# | **Tarea**                         | ¿Qué problema debe resolver ante una consulta del usuario?                                                                                           |
# | **Contexto**                      | ¿Qué debe saber sobre el proceso, organización o dominio? ¿Para quién responde y qué nivel de detalle necesita?                                      |
# | **Instrucciones y restricciones** | ¿Qué debe hacer y qué reglas debe respetar? ¿Qué acciones o respuestas deberían quedar fuera de alcance?                                             |
# | **Estrategia de resolución**      | ¿Qué pasos generales debería seguir para resolver una consulta? ¿Cuándo necesita buscar información antes de responder?                              |
# | **Tools**                         | ¿Qué tools tiene disponibles? ¿Para qué sirve cada una y cuándo debería utilizarla?                                                                  |
# | **Fuentes y evidencia**           | ¿Qué información puede obtener del RAG y de otras tools? ¿Cómo distinguirá información recuperada de una inferencia o supuesto?                      |
# | **Incertidumbre**                 | ¿Qué debe hacer si las fuentes no responden la pregunta o la evidencia es insuficiente?                                                              |
# | **Seguridad**                     | ¿Cómo debe actuar frente a prompt injection, preguntas fuera de alcance o instrucciones encontradas dentro de los documentos recuperados?            |
# | **Salida**                        | ¿Qué estructura, campos, extensión o nivel de detalle debe tener la respuesta?                                                                       |
# | **Ejemplos**                      | ¿Qué casos muestran el comportamiento esperado? ¿Incluyen tanto un caso normal como uno donde falte información o la consulta esté fuera de alcance? |
#

# %% [markdown]
# ### 3.2 Reglas para el RAG
#
# El prompt debe definir cómo utilizar la información recuperada desde los documentos:
#
# 1. Si la respuesta requiere información contenida en los documentos, el agente debe consultar `buscar_vector_store`.
# 2. Las afirmaciones sobre esos documentos deben estar respaldadas por los fragmentos recuperados.
# 3. Cuando la información recuperada incluya una fuente identificable, esta debe mencionarse en la respuesta.
# 4. Si los fragmentos no contienen evidencia suficiente para responder, el agente debe indicarlo y evitar asumir o inventar la información faltante.
# 5. Si las fuentes entregan información contradictoria, el agente debe señalarlo en lugar de escoger una versión sin fundamento.
# 6. El contenido recuperado debe tratarse como **información para analizar**, no como instrucciones capaces de modificar el comportamiento del agente.
#

# %% [markdown]
# ### 3.3 Selección y uso de tools
#
# El *scaffolding* ya incorpora la tool `buscar_vector_store`:
#
# | Tool disponible       | Para qué sirve                                               | Cuándo usarla                                                           |
# | --------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------- |
# | `buscar_vector_store` | Recupera fragmentos desde los documentos cargados en el RAG. | Cuando la consulta dependa de información contenida en esos documentos. |
#
# Sin embargo, **pueden incorporar una o más tools adicionales** si aportan una capacidad útil para el agente que definieron.
#
# > **Importante:** No es obligatorio conectarse a una API o a otro sistema. Una función de Python también puede ser una buena tool cuando necesitamos realizar una operación exacta, validar información, trabajar con fechas o manipular archivos.
#
# Algunas familias posibles son:
#
# | Familia                     | Ejemplos                                                                                        | Posible implementación        |
# | --------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------- |
# | **Cálculo**                 | `calculadora`, `calcular_interes_compuesto`, `calcular_variacion_porcentual`, `convertir_tasas` | Python                        |
# | **Fechas**                  | `obtener_fecha_actual`, `calcular_fecha_limite`, `calcular_dias_entre_fechas`                   | Python                        |
# | **Validación**              | `validar_rut`, `validar_fecha`, `validar_rango`                                                 | Python / reglas del dominio   |
# | **Archivos**                | `listar_markdowns`, `leer_markdown`, `guardar_markdown`                                         | `data/input/`, `data/output/` |
# | **Indicadores financieros** | `consultar_uf`, `consultar_utm`, `consultar_tipo_cambio`                                        | API CMF u otra fuente oficial |
# | **Información pública**     | `buscar_web`, `buscar_hechos_esenciales`, `buscar_entidad_supervisada`                          | Tavily / CMF                  |
# | **Bases de datos**          | `consultar_estado_tramite`, `consultar_cliente`, `buscar_expediente`                            | Oracle / SQL Server           |
# | **Acciones**                | `crear_ticket`, `registrar_solicitud`, `agendar_atencion`                                       | API o sistema externo         |
#

# %% [markdown]
# #### Nuevas tools para el agente
#
# Consideren el siguiente criterio: **Una buena tool debería resolver una necesidad concreta del proyecto.**
#
# El prompt debe dejar claro cuándo corresponde utilizar `buscar_vector_store` y cuándo utilizar alguna capacidad adicional. Y también puede haber consultas que necesiten encadenar tools:
#
# * consultar un valor en los documentos y luego calcular → RAG + tool de cálculo;
# * conocer los archivos disponibles y después leer uno → `listar_markdowns` + `leer_markdown`;
# * obtener información operacional → tool conectada a la base de datos;
# * validar un dato conocido → tool de validación;
# * ninguna tool tiene la información → explicar el límite.
#
# No llamen a las tools manualmente ni por defecto. El agente debe decidir cuándo
# utilizarlas según la consulta que necesite resolver.
#
#
# #### Definición de nuevas Tools
#
# Recuerden que el modelo utiliza el **nombre**, la **descripción** y los **argumentos** de una tool para decidir cuándo llamarla. Por ende, el `docstring` (es decir, la documentación de cada tool) debería indicar:
#
# * qué hace la tool;
# * cuándo corresponde utilizarla;
# * cuándo **no** corresponde utilizarla;
# * qué representa cada argumento;
# * qué formatos y restricciones deben respetarse;
# * qué significa el resultado.
#
#
# > **Importante:** las siguientes tools listadas son ejemplos, no requisitos. No agreguen una tool solamente para aumentar el número de herramientas del agente.
#
# También pueden utilizar Pydantic para validar los argumentos antes de ejecutar la función.
#
# ```python
# from langchain_core.tools import ToolException
# from pydantic import ValidationError
#
#
# def manejar_error_tool(error: ToolException) -> str:
#     return (
#         f"La operación no pudo completarse: {error}. "
#         "No inventes un resultado. Revisa los argumentos o informa "
#         "al usuario si falta información."
#     )
#
#
# def manejar_error_validacion(error: ValidationError) -> str:
#     return (
#         "Los argumentos enviados a la tool no cumplen su esquema. "
#         "Revisa los formatos y restricciones antes de volver a intentarlo."
#     )
#
#
# def configurar_manejo_errores(mi_tool):
#     """Configura el manejo común de errores de una tool."""
#     mi_tool.handle_tool_error = manejar_error_tool
#     mi_tool.handle_validation_error = manejar_error_validacion
#     return mi_tool
# ```
#
# Apliquen esta función a cada tool que pueda informar errores de ejecución o
# validación:
#
# ```python
# configurar_manejo_errores(mi_tool)
# ```
#
# ---
#
# ### Tools de cálculo
#
# Una tool puede encargarse de operaciones que conviene resolver de forma determinista.
#
# #### Calculadora
#
# ```python
# from typing import Literal
#
# from langchain.tools import tool
# from langchain_core.tools import ToolException
# from pydantic import BaseModel, Field
#
#
# class CalculadoraInput(BaseModel):
#     a: float = Field(description="Primer número de la operación.")
#     b: float = Field(description="Segundo número de la operación.")
#     operacion: Literal[
#         "sumar",
#         "restar",
#         "multiplicar",
#         "dividir",
#     ] = Field(
#         description="Operación que debe realizarse."
#     )
#
#
# @tool(args_schema=CalculadoraInput)
# def calculadora(a: float, b: float, operacion: str) -> dict:
#     """Realiza una operación aritmética básica.
#
#     Úsala cuando necesites calcular un resultado exacto a partir de dos
#     valores conocidos.
#
#     No la uses para estimar datos faltantes ni para responder preguntas
#     conceptuales sobre matemáticas.
#     """
#     if operacion == "sumar":
#         resultado = a + b
#     elif operacion == "restar":
#         resultado = a - b
#     elif operacion == "multiplicar":
#         resultado = a * b
#     elif operacion == "dividir":
#         if b == 0:
#             raise ToolException("No es posible dividir por cero.")
#         resultado = a / b
#     else:
#         raise ToolException("La operación solicitada no está soportada.")
#
#     return {
#         "operacion": operacion,
#         "resultado": resultado,
#     }
#
#
# configurar_manejo_errores(calculadora)
# ```
#
# #### Interés compuesto
#
# ```python
# from langchain.tools import tool
# from pydantic import BaseModel, Field
#
#
# class InteresCompuestoInput(BaseModel):
#     capital: float = Field(
#         gt=0,
#         description="Capital inicial.",
#     )
#     tasa: float = Field(
#         gt=-1,
#         description=(
#             "Tasa por período expresada como decimal. "
#             "Por ejemplo, 0.05 representa 5 %."
#         ),
#     )
#     periodos: int = Field(
#         ge=0,
#         description="Número de períodos durante los que se aplica la tasa.",
#     )
#
#
# @tool(args_schema=InteresCompuestoInput)
# def calcular_interes_compuesto(
#     capital: float,
#     tasa: float,
#     periodos: int,
# ) -> dict:
#     """Calcula el monto final utilizando interés compuesto.
#
#     Úsala cuando capital, tasa y número de períodos sean conocidos.
#
#     La tasa y los períodos deben usar la misma unidad temporal.
#     No asumas una tasa o periodicidad que no haya sido proporcionada.
#     """
#     monto_final = capital * (1 + tasa) ** periodos
#
#     return {
#         "capital": capital,
#         "tasa": tasa,
#         "periodos": periodos,
#         "monto_final": monto_final,
#         "interes_acumulado": monto_final - capital,
#     }
#
#
# configurar_manejo_errores(calcular_interes_compuesto)
# ```
#
# #### Variación porcentual
#
# ```python
# from langchain.tools import tool
# from langchain_core.tools import ToolException
#
#
# @tool
# def calcular_variacion_porcentual(
#     valor_inicial: float,
#     valor_final: float,
# ) -> dict:
#     """Calcula la variación porcentual entre dos valores.
#
#     Úsala cuando necesites comparar un valor inicial con uno posterior.
#
#     No la uses si el valor inicial es cero, porque la variación porcentual
#     no está definida en ese caso.
#     """
#     if valor_inicial == 0:
#         raise ToolException(
#             "No es posible calcular variación porcentual desde cero."
#         )
#
#     variacion = (valor_final - valor_inicial) / valor_inicial * 100
#
#     return {
#         "valor_inicial": valor_inicial,
#         "valor_final": valor_final,
#         "variacion_porcentual": variacion,
#     }
#
#
# configurar_manejo_errores(calcular_variacion_porcentual)
# ```

# %% [markdown]
#
# ---
#
# ### Tools de fechas
#
# #### Fecha actual
#
# ```python
# from datetime import datetime
# from zoneinfo import ZoneInfo
#
# from langchain.tools import tool
#
#
# @tool
# def obtener_fecha_actual() -> dict:
#     """Obtiene la fecha actual en Chile.
#
#     Úsala cuando la consulta dependa de saber cuál es la fecha de hoy,
#     por ejemplo para evaluar vencimientos o plazos relativos.
#
#     No la uses si la fecha relevante ya fue entregada explícitamente.
#     """
#     ahora = datetime.now(ZoneInfo("America/Santiago"))
#
#     return {
#         "fecha": ahora.date().isoformat(),
#         "zona_horaria": "America/Santiago",
#     }
# ```
#
# #### Fecha límite
#
# ```python
# from datetime import date, timedelta
#
# from langchain.tools import tool
# from pydantic import BaseModel, Field, field_validator
#
#
# class FechaLimiteInput(BaseModel):
#     fecha_inicio: str = Field(
#         description="Fecha inicial en formato YYYY-MM-DD."
#     )
#     dias: int = Field(
#         ge=0,
#         le=3650,
#         description="Cantidad de días corridos que deben sumarse.",
#     )
#
#     @field_validator("fecha_inicio")
#     @classmethod
#     def validar_fecha(cls, value: str) -> str:
#         date.fromisoformat(value)
#         return value
#
#
# @tool(args_schema=FechaLimiteInput)
# def calcular_fecha_limite(
#     fecha_inicio: str,
#     dias: int,
# ) -> dict:
#     """Calcula una fecha límite sumando días corridos.
#
#     Úsala cuando conozcas una fecha inicial y un plazo expresado
#     explícitamente en días corridos.
#
#     No la uses para días hábiles, feriados o reglas de cómputo especiales.
#     """
#     inicio = date.fromisoformat(fecha_inicio)
#     limite = inicio + timedelta(days=dias)
#
#     return {
#         "fecha_inicio": inicio.isoformat(),
#         "dias": dias,
#         "fecha_limite": limite.isoformat(),
#     }
#
#
# configurar_manejo_errores(calcular_fecha_limite)
# ```
#
# #### Diferencia entre fechas
#
# ```python
# from datetime import date
#
# from langchain.tools import tool
# from pydantic import BaseModel, Field, field_validator
#
#
# class DiasEntreFechasInput(BaseModel):
#     fecha_inicio: str = Field(
#         description="Fecha inicial en formato YYYY-MM-DD."
#     )
#     fecha_fin: str = Field(
#         description="Fecha final en formato YYYY-MM-DD."
#     )
#
#     @field_validator("fecha_inicio", "fecha_fin")
#     @classmethod
#     def validar_fecha(cls, value: str) -> str:
#         try:
#             date.fromisoformat(value)
#         except ValueError as exc:
#             raise ValueError(
#                 "La fecha debe ser válida y utilizar el formato YYYY-MM-DD."
#             ) from exc
#
#         return value
#
#
# @tool(args_schema=DiasEntreFechasInput)
# def calcular_dias_entre_fechas(
#     fecha_inicio: str,
#     fecha_fin: str,
# ) -> dict:
#     """Calcula la cantidad de días corridos entre dos fechas.
#
#     Ambas fechas deben utilizar el formato YYYY-MM-DD.
#
#     Úsala cuando necesites medir la duración entre dos fechas conocidas.
#     """
#     inicio = date.fromisoformat(fecha_inicio)
#     fin = date.fromisoformat(fecha_fin)
#
#     return {
#         "fecha_inicio": fecha_inicio,
#         "fecha_fin": fecha_fin,
#         "dias": (fin - inicio).days,
#     }
#
#
# configurar_manejo_errores(calcular_dias_entre_fechas)
# ```
#

# %% [markdown]
#
# ---
#
# ### Tools de validación
#
# También pueden convertir reglas conocidas del dominio en tools.
#
# #### Validar RUT
#
# ```python
# from langchain.tools import tool
# from langchain_core.tools import ToolException
# from pydantic import BaseModel, Field
#
#
# class ValidarRutInput(BaseModel):
#     rut: str = Field(
#         description=(
#             "RUT chileno que se quiere validar. "
#             "Puede incluir puntos y guion."
#         )
#     )
#
#
# @tool(args_schema=ValidarRutInput)
# def validar_rut(rut: str) -> dict:
#     """Comprueba el dígito verificador de un RUT chileno.
#
#     Úsala solamente para validar la estructura y dígito verificador.
#
#     Esta tool no comprueba que una persona o empresa exista ni consulta
#     registros del SII, CMF u otra institución.
#     """
#     limpio = rut.replace(".", "").replace("-", "").upper().strip()
#
#     if len(limpio) < 2:
#         raise ToolException("El RUT es demasiado corto.")
#
#     cuerpo, dv = limpio[:-1], limpio[-1]
#
#     if not cuerpo.isdigit():
#         raise ToolException("El cuerpo del RUT debe contener solo números.")
#
#     suma = 0
#     multiplicador = 2
#
#     for digito in reversed(cuerpo):
#         suma += int(digito) * multiplicador
#         multiplicador = 2 if multiplicador == 7 else multiplicador + 1
#
#     resultado = 11 - (suma % 11)
#
#     if resultado == 11:
#         esperado = "0"
#     elif resultado == 10:
#         esperado = "K"
#     else:
#         esperado = str(resultado)
#
#     return {
#         "rut": f"{int(cuerpo)}-{dv}",
#         "valido": dv == esperado,
#     }
#
#
# configurar_manejo_errores(validar_rut)
# ```
#

# %% [markdown]
# ---
#
# ### Tools para trabajar con el sistema de archivos y archivos Markdown
#
# Otra opción es permitir que el agente consulte o genere archivos del proyecto. Para asegurar la seguridad del agente, podemos imponer estas reglas:
#
# ```text
# data/
# ├── input/    ← listar y leer
# └── output/   ← escribir
# ```
#
# > **Importante:** No expongan una ruta arbitraria como argumento de la tool.
# > Una instrucción maliciosa, ya sea en la consulta del usuario o en contenido
# > procesado por el agente, podría intentar acceder a archivos fuera del alcance
# > esperado. Esta restricción debe implementarse en código y no depender
# > solamente del prompt.
#
# Las tools solo trabajarán con archivos `.md` y nunca recibirán rutas arbitrarias.
#
# #### Listar archivos disponibles
#
# ```python
# from pathlib import Path
#
# from langchain.tools import tool
#
#
# DATA_INPUT = Path("data/input").resolve()
#
#
# @tool
# def listar_markdowns() -> dict:
#     """Lista los archivos Markdown disponibles en data/input.
#
#     Úsala cuando necesites saber qué archivos puede consultar el agente
#     antes de intentar leer uno.
#
#     Solo muestra archivos .md directamente disponibles en data/input.
#     No entrega contenido ni permite acceder a otras carpetas.
#     """
#     archivos = sorted(
#         archivo.name
#         for archivo in DATA_INPUT.glob("*.md")
#         if archivo.is_file()
#     )
#
#     return {
#         "archivos": archivos,
#         "cantidad": len(archivos),
#     }
# ```
#
# #### Leer Markdown
#
# ```python
# from pathlib import Path
#
# from langchain.tools import tool
# from langchain_core.tools import ToolException
# from pydantic import BaseModel, Field, field_validator
#
#
# class LeerMarkdownInput(BaseModel):
#     nombre_archivo: str = Field(
#         description=(
#             "Nombre del archivo Markdown que se quiere leer, "
#             "por ejemplo 'normativa.md'. No incluyas rutas."
#         )
#     )
#
#     @field_validator("nombre_archivo")
#     @classmethod
#     def validar_nombre(cls, value: str) -> str:
#         if Path(value).name != value:
#             raise ValueError("No se permiten rutas ni subdirectorios.")
#
#         if not value.lower().endswith(".md"):
#             raise ValueError("Solo se permiten archivos .md.")
#
#         return value
#
#
# @tool(args_schema=LeerMarkdownInput)
# def leer_markdown(nombre_archivo: str) -> dict:
#     """Lee un archivo Markdown disponible en data/input.
#
#     Úsala cuando necesites consultar el contenido de un archivo previamente
#     identificado.
#
#     Solo permite leer archivos .md de data/input.
#     No permite modificar archivos ni acceder a otras carpetas.
#     """
#     archivo = (DATA_INPUT / nombre_archivo).resolve()
#
#     if archivo.parent != DATA_INPUT:
#         raise ToolException(
#             "No se permite acceder fuera de data/input."
#         )
#
#     if not archivo.is_file():
#         raise ToolException(
#             f"No existe '{nombre_archivo}' en data/input."
#         )
#
#     try:
#         contenido = archivo.read_text(encoding="utf-8")
#     except OSError as exc:
#         raise ToolException(
#             f"No fue posible leer '{nombre_archivo}'."
#         ) from exc
#
#     return {
#         "archivo": nombre_archivo,
#         "contenido": contenido,
#     }
#
#
# configurar_manejo_errores(leer_markdown)
# ```
#
# #### Guardar Markdown
#
# ```python
# from pathlib import Path
#
# from langchain.tools import tool
# from langchain_core.tools import ToolException
# from pydantic import BaseModel, Field, field_validator
#
#
# DATA_OUTPUT = Path("data/output").resolve()
# DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
#
#
# class GuardarMarkdownInput(BaseModel):
#     nombre_archivo: str = Field(
#         description=(
#             "Nombre del archivo que se quiere crear, "
#             "por ejemplo 'reporte.md'. No incluyas rutas."
#         )
#     )
#     contenido: str = Field(
#         min_length=1,
#         max_length=100_000,
#         description="Contenido completo que debe guardarse en Markdown.",
#     )
#
#     @field_validator("nombre_archivo")
#     @classmethod
#     def validar_nombre(cls, value: str) -> str:
#         if Path(value).name != value:
#             raise ValueError("No se permiten rutas ni subdirectorios.")
#
#         if not value.lower().endswith(".md"):
#             raise ValueError("El archivo debe tener extensión .md.")
#
#         return value
#
#
# @tool(args_schema=GuardarMarkdownInput)
# def guardar_markdown(
#     nombre_archivo: str,
#     contenido: str,
# ) -> dict:
#     """Crea un archivo Markdown dentro de data/output.
#
#     Úsala solamente cuando el usuario solicite guardar o generar
#     un resultado como archivo.
#
#     No permite escribir fuera de data/output.
#     No modifica archivos de data/input.
#     Tampoco sobrescribe archivos existentes.
#     """
#     archivo = (DATA_OUTPUT / nombre_archivo).resolve()
#
#     if archivo.parent != DATA_OUTPUT:
#         raise ToolException(
#             "No se permite escribir fuera de data/output."
#         )
#
#     if archivo.exists():
#         raise ToolException(
#             f"El archivo '{nombre_archivo}' ya existe."
#         )
#
#     try:
#         archivo.write_text(contenido, encoding="utf-8")
#     except OSError as exc:
#         raise ToolException(
#             f"No fue posible guardar '{nombre_archivo}'."
#         ) from exc
#
#     return {
#         "archivo": nombre_archivo,
#         "guardado": True,
#     }
#
#
# configurar_manejo_errores(guardar_markdown)
# ```
#

# %% [markdown]
#
# ---
#
# ### Tools conectadas a bases de datos
#
# También pueden utilizar una base de datos existente como fuente operacional.
#
# Una regla importante es **no entregar SQL arbitrario al modelo**:
#
# ```text
# ❌ ejecutar_sql(sql)
#
# ✅ consultar_estado_tramite(folio)
# ✅ consultar_cliente(rut)
# ✅ obtener_documentos_pendientes(folio)
# ```
#
# La consulta SQL debería quedar encapsulada dentro de la tool y utilizar parámetros.
#
# #### Oracle Database
#
# Pueden instalar:
#
# ```bash
# uv add oracledb
# ```
#
# Y definir la conexión fuera de la tool:
#
# ```python
# import os
#
# import oracledb
#
#
# def conectar_oracle():
#     return oracledb.connect(
#         user=os.environ["ORACLE_USER"],
#         password=os.environ["ORACLE_PASSWORD"],
#         dsn=os.environ["ORACLE_DSN"],
#     )
# ```
#
# Luego la tool implementa una operación concreta:
#
# ```python
# @tool
# def consultar_estado_tramite(folio: str) -> dict:
#     """Consulta el estado actual de un trámite almacenado en Oracle.
#
#     Úsala cuando el usuario entregue un folio y necesite conocer
#     información operacional actual.
#
#     No la uses para preguntas sobre normativa o procedimientos generales.
#     """
#     try:
#         with conectar_oracle() as conexion:
#             with conexion.cursor() as cursor:
#                 cursor.execute(
#                     """
#                     SELECT folio, estado, fecha_actualizacion
#                     FROM tramites
#                     WHERE folio = :folio
#                     """,
#                     {"folio": folio},
#                 )
#
#                 fila = cursor.fetchone()
#
#     except oracledb.Error as exc:
#         raise ToolException(
#             "No fue posible consultar la base de datos."
#         ) from exc
#
#     if fila is None:
#         raise ToolException(
#             f"No se encontró un trámite con folio '{folio}'."
#         )
#
#     return {
#         "folio": fila[0],
#         "estado": fila[1],
#         "fecha_actualizacion": str(fila[2]),
#     }
#
#
# configurar_manejo_errores(consultar_estado_tramite)
# ```
#
# #### Microsoft SQL Server
#
# Una alternativa es utilizar:
#
# ```bash
# uv add mssql-python
# ```
#
# y mantener la cadena de conexión en una variable de entorno:
#
# ```python
# import os
#
# from mssql_python import connect
#
#
# def conectar_sql_server():
#     return connect(
#         os.environ["SQL_CONNECTION_STRING"],
#         timeout=15,
#     )
# ```
#
# La tool puede seguir exactamente el mismo patrón: una operación concreta, consulta parametrizada y sin entregar acceso SQL libre al modelo.
#
# También pueden utilizar `pyodbc`:
#
# ```bash
# uv add pyodbc
# ```
#
# En ese caso, dependiendo del sistema, puede ser necesario instalar además **Microsoft ODBC Driver for SQL Server**.
#

# %% [markdown]
# ---
# ## 4. Implementen en `agente_proyecto/`

# %% [markdown]
# Ejecuten la aplicación abriendo un terminal apuntando al directorio de `Proyecto_1` (y activando su ambiente virtual viendo el cuadro morado del inicio) usando:
#
# ```bash
# streamlit run app_proyecto.py --server.runOnSave true
# ```
#
#
# Y luego trabajen sobre los archivos `agente_proyecto/prompt.py` y `agente_proyecto/tools.py`. 
#
# Gracias al _autoreload_ de `Streamlit`, las modificaciones deberían poder ser
# vistas en vivo, aunque en algunos casos deberán cortar la ejecución del
# proyecto usando **Control + C**.
#
# Solo modifiquen `app_proyecto.py` cuando sea estrictamente necesario (por
# ejemplo, para agregar nuevos módulos a la interfaz gráfica).
#

# %% [markdown]
# ### Paso 1: Carguen documentos
#
# 1. Abran el módulo visual para agregar Markdown o Convertir Documentos. En caso de necesitarlo, también pueden usar la interfaz gráfica de Gemini para convertir documentos o páginas web a markdown con un prompt simple (deben también elaborarlo o incluso, pedírselo al mismo Gemini).
# 2. Carguen un documento propio y pertinente al proceso del Hito 1.
# 3. Revisen la vista previa de los fragmentos y configuren la ingesta según lo que ustedes necesiten.
# 4. Agreguen el documento a la base de conocimiento.

# %% [markdown]
# ### Paso 2: Reemplacen y mejoren el System Prompt
#
# Sustituyan el contenido de `SYSTEM_PROMPT` en `agente_proyecto/prompt.py` por el prompt que diseñaron. Mantengan el resto del agente funcionando y prueben primero una consulta sencilla.
#
# Comprueben que el agente no responde con una explicación genérica cuando la pregunta necesita consultar el documento.

# %% [markdown]
# ### Paso 3 - Opcional: Implementen nuevas Tools
#
# Ver el punto 3.3. para esto. Recuerden, esto es opcional.

# %% [markdown]
# ---
# ## 5. Entrega y rúbrica

# %% [markdown]
# ### Entrega
#
# Entreguen los siguientes elementos:
#
# 1. **Código de la aplicación:** `app_proyecto.py`, la carpeta `agente_proyecto/` y cualquier otro archivo necesario para ejecutar el proyecto. No incluyan credenciales, API keys ni archivos `.env`.
#    Si agregaron dependencias, incluyan también los archivos de configuración
#    actualizados, como `pyproject.toml` y `uv.lock`. Si utilizan nuevas
#    variables de entorno, pueden incluir un `.env.example` con sus nombres,
#    pero nunca credenciales, API keys ni otros secretos.
#
# 2. **Evidencia de funcionamiento:** adjunten en U-Cursos capturas o conversaciones exportadas desde el agente. Deben incluir **al menos cinco conversaciones distintas**:
#
#    1. **Consulta respondida con RAG:** una pregunta cuya respuesta se encuentre en los documentos y que requiera usar `buscar_vector_store`.
#    2. **Información insuficiente:** una pregunta que los documentos no permitan responder. El agente debe reconocer el límite y evitar completar la respuesta con supuestos.
#    3. **Conversación con seguimiento:** una consulta inicial y al menos una pregunta posterior que dependa del contexto de la conversación.
#    4. **Seguridad o límite del agente:** una consulta fuera de alcance, un intento de *prompt injection* o un caso equivalente definido para su proyecto.
#    5. **Caso desafiante:** una consulta ambigua, fuentes contradictorias, información incompleta o alguna otra situación en que el agente deba decidir cómo continuar sin asumir información que no tiene.
#
# 3. **Tool adicional, opcional:** si implementaron una o más tools adicionales, incluyan al menos una conversación que permita revisar su uso. La evidencia debe mostrar que el agente selecciona la tool cuando corresponde, respeta sus argumentos y restricciones, y maneja adecuadamente sus resultados o errores.
#
# Las conversaciones deben permitir identificar **la consulta, la respuesta del agente y las tools utilizadas**. Cuando corresponda, también debe ser posible reconocer la fuente o el resultado en que se apoyó la respuesta.
#
# **No es necesario entregar este notebook.**
#

# %% [markdown]
# ### Rúbrica grupal
#
# Escala chilena de 1 a 7. La aplicación debe utilizar `buscar_vector_store` como tool base. Las tools adicionales son opcionales y se consideran una extensión cuando aportan una capacidad relevante para el proyecto.
#
# | Criterio                                     | 1–3 Insuficiente                                                                                                                                                                      | 4–5 Suficiente                                                                                                         | 6–7 Sobresaliente                                                                                                                                                                    | Peso |
# | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---: |
# | **Diseño del prompt**                        | El prompt es genérico, ambiguo o no define adecuadamente la tarea y los límites del agente.                                                                                           | Define rol, tarea, contexto, formato de salida e instrucciones básicas.                                                | Define con claridad los componentes relevantes para el proyecto, incluyendo rol, tarea, contexto, restricciones y, cuando corresponda, seguridad, uso de evidencia, incertidumbre y comportamiento esperado ante distintos tipos de consulta. |  30% |
# | **Uso del RAG y evidencia**                  | No usa `buscar_vector_store` cuando corresponde, responde sin respaldo o atribuye información que no aparece en los documentos.                                                       | Consulta el RAG y responde preguntas respaldadas por los fragmentos recuperados.                                       | Decide correctamente cuándo consultar el RAG, utiliza la evidencia recuperada, conserva la fuente cuando está disponible y reconoce cuando los documentos no permiten responder.     |  30% |
# | **Comportamiento del agente**                | Pierde el contexto de la conversación, inventa información faltante o responde de forma inadecuada ante consultas ambiguas, fuera de alcance o intentos de cambiar sus instrucciones. | Mantiene el contexto básico de la conversación y maneja de forma razonable consultas sin respuesta o fuera de alcance. | Mantiene el contexto entre turnos y responde de forma consistente ante información insuficiente, ambigüedad, fuentes contradictorias, consultas fuera de alcance y prompt injection. |  25% |
# | **Aplicación y evidencia de funcionamiento** | La aplicación no ejecuta correctamente o la evidencia entregada no permite revisar su comportamiento.                                                                                 | La aplicación funciona y se incluyen las cinco conversaciones solicitadas con evidencia identificable.                 | Las conversaciones permiten verificar con claridad las decisiones del agente, las llamadas a tools, las fuentes utilizadas y su comportamiento en los distintos casos evaluados.     |  15% |
#
# #### Tools adicionales
#
# Las tools adicionales **no son requisito de la entrega** y su ausencia no limita la nota máxima.
#
# Si incorporan una, se considerará como parte de la calidad técnica del proyecto siempre que:
#
# * resuelva una necesidad real del agente;
# * tenga un contrato claro para el modelo;
# * valide sus argumentos cuando corresponda;
# * maneje errores esperables;
# * no entregue al agente permisos más amplios de los necesarios;
# * se integre de forma coherente con el prompt y con las demás capacidades del agente.
#
# Una tool adicional mal integrada o innecesaria no mejora por sí sola la evaluación.
#
