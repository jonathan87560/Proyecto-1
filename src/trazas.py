"""Reconstrucción de trazas a partir del checkpointer de LangGraph.

El checkpointer guarda un checkpoint por cada paso del grafo, con su marca de
tiempo y los mensajes acumulados. De ahí salen los pasos de la conversación,
sus tokens y la latencia de cada uno: la diferencia entre la marca de tiempo de
un checkpoint y la del anterior. No son spans de OpenTelemetry, así que miden
el paso completo del grafo y no solo la llamada al modelo.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

TIPOS_MENSAJE = ("human", "ai", "tool")

# Precios en USD por millón de tokens, los mismos que se usan en S02.
PRECIOS: dict[str, dict[str, float]] = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.5},
    "gemini-3.5-flash": {"input": 1.5, "output": 9},
    "gemini-3.1-pro-preview": {"input": 2, "output": 12},
}
USD_A_CLP = 920.0


def estimar_costo_clp(
    tokens_entrada: int,
    tokens_salida: int,
    modelo: str,
    usd_a_clp: float = USD_A_CLP,
) -> float:
    """Estima el costo en CLP de una llamada según precios por millón.

    Un modelo desconocido cuesta 0: la traza sigue siendo útil aunque el
    equipo cambie a un modelo que no está en la tabla.
    """
    precios = PRECIOS.get(modelo)
    if precios is None:
        return 0.0
    return (
        tokens_entrada / 1_000_000 * precios["input"] * usd_a_clp
        + tokens_salida / 1_000_000 * precios["output"] * usd_a_clp
    )


def extraer_texto(contenido: Any) -> str:
    """Aplana el contenido de un mensaje, que Gemini devuelve como lista."""

    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(
            parte.get("text", "") if isinstance(parte, dict) else str(parte)
            for parte in contenido
        )
    return str(contenido)


@dataclass(frozen=True)
class PasoTraza:
    """Un mensaje de la conversación con sus tiempos y su consumo."""

    orden: int
    tipo: str
    titulo: str
    contenido: str
    entrada: dict[str, Any]
    salida: dict[str, Any]
    metadata: dict[str, Any]
    tokens_entrada: int = 0
    tokens_salida: int = 0
    tokens_razonamiento: int = 0
    duracion_segundos: float | None = None

    @property
    def tokens_totales(self) -> int:
        return self.tokens_entrada + self.tokens_salida

    @property
    def costo_clp(self) -> float:
        return estimar_costo_clp(
            self.tokens_entrada,
            self.tokens_salida,
            self.metadata.get("modelo", ""),
        )


@dataclass(frozen=True)
class Conversacion:
    """Una conversación completa, lista para mostrarse o descargarse."""

    thread_id: str
    inicio: str
    fin: str
    pasos: list[PasoTraza]

    @property
    def mensajes(self) -> int:
        return sum(paso.tipo in TIPOS_MENSAJE for paso in self.pasos)

    @property
    def llamadas_tool(self) -> int:
        return sum(paso.tipo == "tool" for paso in self.pasos)

    @property
    def tokens_totales(self) -> int:
        return sum(paso.tokens_totales for paso in self.pasos)

    @property
    def costo_clp(self) -> float:
        return sum(paso.costo_clp for paso in self.pasos)

    @property
    def duracion_segundos(self) -> float:
        if not self.inicio or not self.fin:
            return 0.0
        return (
            datetime.fromisoformat(self.fin)
            - datetime.fromisoformat(self.inicio)
        ).total_seconds()


def listar_hilos(checkpointer: Any) -> list[dict[str, Any]]:
    """Devuelve los hilos guardados, del más reciente al más antiguo."""

    hilos: dict[str, dict[str, Any]] = {}
    for tupla in checkpointer.list(None):
        thread_id = tupla.config["configurable"]["thread_id"]
        marca = tupla.checkpoint["ts"]
        mensajes = tupla.checkpoint["channel_values"].get("messages", [])
        conocido = hilos.get(thread_id)
        if conocido is None or marca > conocido["ultima_actividad"]:
            hilos[thread_id] = {
                "thread_id": thread_id,
                "ultima_actividad": marca,
                "mensajes": len(mensajes),
            }
    return sorted(
        hilos.values(),
        key=lambda hilo: hilo["ultima_actividad"],
        reverse=True,
    )


def etiquetas_de_hilos(hilos: list[dict[str, Any]]) -> dict[str, str]:
    """Rótulos de los hilos para los selectores del chat y de las trazas."""

    return {
        hilo["thread_id"]: (
            f"{hilo['thread_id']} · {hilo['mensajes']} mensajes · "
            f"{datetime.fromisoformat(hilo['ultima_actividad']):%Y-%m-%d %H:%M}"
        )
        for hilo in hilos
    }


def _resumir_mensaje(mensaje: Any) -> dict[str, Any]:
    """Versión compacta de un mensaje, para el JSON de entrada del modelo."""

    resumen: dict[str, Any] = {
        "tipo": mensaje.type,
        "contenido": extraer_texto(mensaje.content),
    }
    if getattr(mensaje, "tool_calls", None):
        resumen["tool_calls"] = mensaje.tool_calls
    if mensaje.type == "tool":
        resumen["tool_call_id"] = getattr(mensaje, "tool_call_id", "")
    return resumen


def _duraciones_por_mensaje(tuplas: list[Any]) -> dict[str, float]:
    """Asigna a cada mensaje el tiempo del paso del grafo que lo produjo."""

    duraciones: dict[str, float] = {}
    vistos: set[str] = set()
    marca_anterior: datetime | None = None
    for tupla in tuplas:
        marca = datetime.fromisoformat(tupla.checkpoint["ts"])
        for mensaje in tupla.checkpoint["channel_values"].get("messages", []):
            if mensaje.id in vistos:
                continue
            vistos.add(mensaje.id)
            if marca_anterior is not None:
                duraciones[mensaje.id] = (
                    marca - marca_anterior
                ).total_seconds()
        marca_anterior = marca
    return duraciones


def _paso_de_mensaje(
    mensaje: Any,
    orden: int,
    previos: list[Any],
    system_prompt: str,
    duracion: float | None,
) -> PasoTraza:
    contenido = extraer_texto(mensaje.content)
    uso = getattr(mensaje, "usage_metadata", None) or {}
    respuesta = getattr(mensaje, "response_metadata", None) or {}
    tool_calls = getattr(mensaje, "tool_calls", None) or []

    if mensaje.type == "human":
        titulo = "Mensaje del usuario"
        entrada: dict[str, Any] = {"contenido": contenido}
        salida: dict[str, Any] = {}
    elif mensaje.type == "tool":
        nombre = getattr(mensaje, "name", "") or "herramienta"
        titulo = f"Herramienta · {nombre}"
        entrada = {
            "nombre": nombre,
            "argumentos": _argumentos_de_tool(mensaje, previos),
        }
        salida = {
            "contenido": contenido,
            "status": getattr(mensaje, "status", "success"),
        }
    else:
        titulo = (
            "Llamada a herramientas" if tool_calls else "Respuesta del modelo"
        )
        # El modelo recibe el prompt de sistema más todo el historial previo.
        entrada = {
            "system": system_prompt,
            "mensajes": [_resumir_mensaje(previo) for previo in previos],
        }
        salida = {"contenido": contenido, "tool_calls": tool_calls}

    return PasoTraza(
        orden=orden,
        tipo=mensaje.type,
        titulo=titulo,
        contenido=contenido,
        entrada=entrada,
        salida=salida,
        metadata={
            "message_id": mensaje.id,
            "modelo": respuesta.get("model_name", ""),
            "finish_reason": respuesta.get("finish_reason", ""),
            "tool_call_id": getattr(mensaje, "tool_call_id", ""),
        },
        tokens_entrada=uso.get("input_tokens", 0),
        tokens_salida=uso.get("output_tokens", 0),
        tokens_razonamiento=uso.get("output_token_details", {}).get(
            "reasoning",
            0,
        ),
        duracion_segundos=None if mensaje.type == "human" else duracion,
    )


def _argumentos_de_tool(mensaje: Any, previos: list[Any]) -> dict[str, Any]:
    """Busca los argumentos con que se invocó esta herramienta."""

    tool_call_id = getattr(mensaje, "tool_call_id", "")
    for previo in reversed(previos):
        for llamada in getattr(previo, "tool_calls", None) or []:
            if llamada.get("id") == tool_call_id:
                return llamada.get("args", {})
    return {}


def leer_conversacion(
    checkpointer: Any,
    thread_id: str,
    system_prompt: str = "",
) -> Conversacion:
    """Arma la traza de un hilo con sus pasos, tokens y latencias."""

    tuplas = sorted(
        checkpointer.list({"configurable": {"thread_id": thread_id}}),
        key=lambda tupla: tupla.checkpoint["ts"],
    )
    if not tuplas:
        return Conversacion(thread_id=thread_id, inicio="", fin="", pasos=[])

    duraciones = _duraciones_por_mensaje(tuplas)
    mensajes = tuplas[-1].checkpoint["channel_values"].get("messages", [])

    pasos: list[PasoTraza] = []
    if system_prompt.strip():
        # El prompt de sistema no viaja en el estado: create_agent lo inyecta
        # en cada llamada al modelo, así que se muestra como paso reconstruido.
        pasos.append(
            PasoTraza(
                orden=0,
                tipo="system",
                titulo="Prompt de sistema (reconstruido)",
                contenido=system_prompt,
                entrada={},
                salida={},
                metadata={"persistido": False},
            )
        )

    for posicion, mensaje in enumerate(mensajes):
        pasos.append(
            _paso_de_mensaje(
                mensaje,
                orden=len(pasos),
                previos=mensajes[:posicion],
                system_prompt=system_prompt,
                duracion=duraciones.get(mensaje.id),
            )
        )

    return Conversacion(
        thread_id=thread_id,
        inicio=tuplas[0].checkpoint["ts"],
        fin=tuplas[-1].checkpoint["ts"],
        pasos=pasos,
    )


def historial_de_chat(conversacion: Conversacion) -> list[dict[str, str]]:
    """Convierte una conversación guardada al historial que muestra el chat."""

    roles = {"human": "user", "ai": "assistant"}
    return [
        {"role": roles[paso.tipo], "content": paso.contenido}
        # Las respuestas que solo pidieron herramientas no llevan texto: en el
        # chat se verían como burbujas vacías.
        for paso in conversacion.pasos
        if paso.tipo in roles and paso.contenido.strip()
    ]


def traza_a_json(conversacion: Conversacion) -> dict[str, Any]:
    """Serializa la conversación completa para inspeccionarla o descargarla."""

    return {
        "thread_id": conversacion.thread_id,
        "inicio": conversacion.inicio,
        "fin": conversacion.fin,
        "duracion_segundos": round(conversacion.duracion_segundos, 3),
        "mensajes": conversacion.mensajes,
        "llamadas_tool": conversacion.llamadas_tool,
        "tokens_totales": conversacion.tokens_totales,
        "costo_clp": round(conversacion.costo_clp, 2),
        "pasos": [
            {
                "orden": paso.orden,
                "tipo": paso.tipo,
                "titulo": paso.titulo,
                "duracion_segundos": paso.duracion_segundos,
                "tokens_entrada": paso.tokens_entrada,
                "tokens_salida": paso.tokens_salida,
                "tokens_razonamiento": paso.tokens_razonamiento,
                "costo_clp": round(paso.costo_clp, 2),
                "entrada": paso.entrada,
                "salida": paso.salida,
                "metadata": paso.metadata,
            }
            for paso in conversacion.pasos
        ],
    }


ICONOS_MARKDOWN = {
    "system": "Prompt de sistema",
    "human": "Usuario",
    "ai": "Agente",
    "tool": "Herramienta",
}


def traza_a_markdown(conversacion: Conversacion) -> str:
    """Rinde la conversación como Markdown pegable en el informe del hito."""

    lineas = [
        f"# Traza de la conversación `{conversacion.thread_id}`",
        "",
        f"- **Inicio:** {conversacion.inicio}",
        f"- **Duración:** {conversacion.duracion_segundos:.1f} s",
        f"- **Mensajes:** {conversacion.mensajes}",
        f"- **Llamadas a herramientas:** {conversacion.llamadas_tool}",
        f"- **Tokens acumulados:** {conversacion.tokens_totales:,}",
        f"- **Costo estimado:** ${conversacion.costo_clp:,.1f} CLP",
        "",
    ]

    for paso in conversacion.pasos:
        etiqueta = ICONOS_MARKDOWN.get(paso.tipo, paso.tipo)
        lineas.append(f"## {paso.orden + 1}. {etiqueta} — {paso.titulo}")
        detalles = []
        if paso.duracion_segundos is not None:
            detalles.append(f"{paso.duracion_segundos:.2f} s")
        if paso.tokens_totales:
            detalles.append(f"{paso.tokens_totales:,} tokens")
            detalles.append(f"${paso.costo_clp:,.1f} CLP")
        if paso.metadata.get("modelo"):
            detalles.append(paso.metadata["modelo"])
        if detalles:
            lineas.append(f"_{' · '.join(detalles)}_")
        lineas.append("")

        if paso.tipo == "tool":
            lineas.append(f"**Argumentos:** `{paso.entrada.get('argumentos')}`")
            lineas.append("")
            lineas.append("```text")
            lineas.append(paso.contenido)
            lineas.append("```")
        elif paso.tipo == "system":
            lineas.append("```text")
            lineas.append(paso.contenido)
            lineas.append("```")
        else:
            lineas.append(paso.contenido or "_(sin texto: solo pidió tools)_")
            for llamada in paso.salida.get("tool_calls") or []:
                lineas.append("")
                lineas.append(
                    f"> Llama a `{llamada['name']}` con `{llamada['args']}`"
                )
        lineas.append("")

    return "\n".join(lineas)
