"""Pantalla del chat: identidad de la conversación, streaming y herramientas."""

import re
from collections.abc import Iterator
from datetime import datetime
from typing import Any
from uuid import uuid4

import streamlit as st
from src.documentos import extraer_texto
from src.trazas import (
    etiquetas_de_hilos,
    historial_de_chat,
    leer_conversacion,
    listar_hilos,
)


@st.cache_resource
def id_de_arranque() -> str:
    """Identificador de la conversación con que arranca este servidor.

    `st.cache_resource` vive en el proceso del servidor: se calcula una vez, lo
    comparten todas las pestañas y recargas, y muere al cortar el CLI. Por eso
    cada `streamlit run` abre una conversación nueva que sobrevive a los F5.
    """
    return f"conv-{datetime.now():%Y%m%d-%H%M%S}"


def conversacion_activa() -> str:
    """Devuelve el hilo en uso: el de la URL o, si no hay, el del arranque."""

    return st.query_params.get("conv") or id_de_arranque()


def cambiar_de_conversacion(thread_id: str) -> None:
    """Deja la conversación en la URL para que sobreviva a una recarga."""

    st.query_params["conv"] = thread_id
    st.rerun()


def _tokens_de_respuesta(
    agente: Any,
    entrada: dict[str, Any],
    configuracion: dict[str, Any],
    estado: Any,
    fuentes: list[str],
) -> Iterator[str]:
    """Cede el texto del modelo y anota en el estado cada herramienta usada."""

    flujo = agente.stream(
        entrada,
        config=configuracion,
        stream_mode=["updates", "messages"],
    )
    for modo, dato in flujo:
        if modo == "updates":
            for mensajes in dato.values():
                for mensaje in (mensajes or {}).get("messages", []):
                    _anotar_herramienta(estado, mensaje, fuentes)
            continue

        chunk, metadata = dato
        # El nodo `tools` también emite mensajes; su contenido se muestra
        # dentro del estado, no como respuesta del asistente.
        if metadata.get("langgraph_node") != "model":
            continue
        texto = extraer_texto(chunk.content)
        if texto:
            yield texto


def _anotar_herramienta(estado: Any, mensaje: Any, fuentes: list[str]) -> None:
    for llamada in getattr(mensaje, "tool_calls", None) or []:
        estado.update(label=f"Consultando {llamada['name']}…")
        estado.write(f"**{llamada['name']}** · argumentos: `{llamada['args']}`")
    if getattr(mensaje, "name", None) and mensaje.type == "tool":
        resultado = extraer_texto(mensaje.content)
        estado.write(f"Resultado ({len(resultado)} caracteres):")
        estado.code(resultado[:600], language="markdown")
        fuentes.extend(fuentes_citadas(resultado))


def fuentes_citadas(resultado: str) -> list[str]:
    """Extrae los nombres de archivo que la búsqueda marcó como FUENTE."""

    encontradas = re.findall(r"^FUENTE:\s*(.+)$", resultado, flags=re.MULTILINE)
    return list(dict.fromkeys(fuente.strip() for fuente in encontradas))


def _responder(agente: Any, consulta: str, thread_id: str) -> None:
    """Ejecuta un turno mostrando el avance y sin tumbar la página si falla."""

    entrada = {"messages": [{"role": "user", "content": consulta}]}
    configuracion = {"configurable": {"thread_id": thread_id}}
    fuentes: list[str] = []
    with st.chat_message("assistant"):
        estado = st.status("Pensando…")
        try:
            st.write_stream(
                _tokens_de_respuesta(
                    agente,
                    entrada,
                    configuracion,
                    estado,
                    fuentes,
                )
            )
        except Exception as error:
            estado.update(label="Falló la respuesta", state="error")
            st.error(
                f"El agente no pudo responder: {error}\n\n"
                "Revisen las credenciales, la cuota del proyecto y que el "
                "modelo esté disponible en la región configurada."
            )
        else:
            estado.update(label="Listo", state="complete")
            if fuentes:
                st.caption("Fuentes: " + " · ".join(f"`{f}`" for f in fuentes))


def _selector_de_conversaciones(checkpointer: Any) -> None:
    """Permite volver a un hilo guardado; el historial sale del checkpointer."""

    hilos = listar_hilos(checkpointer)
    if not hilos:
        st.caption("Todavía no hay conversaciones guardadas.")
        return

    etiquetas = etiquetas_de_hilos(hilos)
    elegido = st.selectbox(
        "Conversación guardada",
        options=list(etiquetas),
        format_func=lambda thread_id: etiquetas[thread_id],
        key="conversacion_a_restaurar",
    )
    if st.button(
        "Cargar conversación",
        key="cargar_conversacion",
        use_container_width=True,
    ):
        cambiar_de_conversacion(elegido)


def render_chat(
    agente: Any,
    checkpointer: Any,
    titulo: str,
    subtitulo: str,
) -> None:
    """Renderiza el chat del agente sobre la conversación activa."""

    st.title(titulo)
    st.caption(subtitulo)

    thread_id = conversacion_activa()

    if st.button("Nueva conversación"):
        cambiar_de_conversacion(f"conv-{uuid4().hex[:8]}")

    with st.expander("Conversaciones anteriores"):
        _selector_de_conversaciones(checkpointer)

    st.caption(f"Conversación activa: `{thread_id}`")

    # El historial visible sale del checkpointer, que ya es la memoria del
    # agente: así una recarga del navegador no vacía el chat.
    conversacion = leer_conversacion(checkpointer, thread_id)
    for mensaje in historial_de_chat(conversacion):
        st.chat_message(mensaje["role"]).write(mensaje["content"])

    consulta = st.chat_input("Escribe la pregunta")
    if consulta:
        st.chat_message("user").write(consulta)
        _responder(agente, consulta, thread_id)
