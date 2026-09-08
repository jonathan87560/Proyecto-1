"""Pantalla Streamlit para inspeccionar las trazas del agente."""

import json
from typing import Any
from uuid import uuid4

import streamlit as st
from src.modulo_chat import cambiar_de_conversacion, conversacion_activa
from src.trazas import (
    Conversacion,
    PasoTraza,
    etiquetas_de_hilos,
    leer_conversacion,
    listar_hilos,
    traza_a_json,
    traza_a_markdown,
)

ICONOS = {"system": "⚙️", "human": "🧑", "ai": "🤖", "tool": "🔧"}


def _titulo_paso(paso: PasoTraza) -> str:
    partes = [f"{ICONOS.get(paso.tipo, '•')} {paso.orden + 1}. {paso.titulo}"]
    if paso.duracion_segundos is not None:
        partes.append(f"{paso.duracion_segundos:.2f} s")
    if paso.tokens_totales:
        partes.append(f"{paso.tokens_totales:,} tokens")
    return " · ".join(partes)


def _render_paso(paso: PasoTraza) -> None:
    with st.expander(
        _titulo_paso(paso),
        expanded=paso.tipo in ("human", "ai"),
    ):
        if paso.tipo == "system":
            st.warning(
                "El checkpointer no guarda el prompt de sistema: LangGraph lo "
                "inyecta en cada llamada. Este es el prompt vigente de la app, "
                "que puede diferir del que se usó en esta conversación."
            )

        contenido, entrada, salida, metadata = st.tabs(
            ["Contenido", "Entrada", "Salida", "Metadata"]
        )
        with contenido:
            if paso.tipo in ("tool", "system"):
                st.code(paso.contenido, language="markdown")
            elif paso.contenido:
                st.markdown(paso.contenido)
            else:
                st.caption("Sin texto: el modelo solo pidió herramientas.")
        with entrada:
            st.json(paso.entrada)
        with salida:
            st.json(paso.salida)
        with metadata:
            st.json(
                {
                    **paso.metadata,
                    "duracion_segundos": paso.duracion_segundos,
                    "tokens_entrada": paso.tokens_entrada,
                    "tokens_salida": paso.tokens_salida,
                    "tokens_razonamiento": paso.tokens_razonamiento,
                    "costo_clp": round(paso.costo_clp, 2),
                }
            )


def _render_metricas(conversacion: Conversacion) -> None:
    metricas = st.columns(4)
    metricas[0].metric("Mensajes", conversacion.mensajes)
    metricas[1].metric("Llamadas a tools", conversacion.llamadas_tool)
    metricas[2].metric(
        "Tokens acumulados",
        f"{conversacion.tokens_totales:,}",
        help=(
            "Suma de todas las llamadas al modelo. El historial se reenvía "
            "completo en cada turno, así que los tokens de entrada se repiten."
        ),
    )
    metricas[3].metric(
        "Costo estimado",
        f"${conversacion.costo_clp:,.1f}",
        help=(
            "En pesos chilenos, con los precios por millón de tokens que "
            "vieron en la clase 2. Un modelo fuera de esa tabla cuenta 0."
        ),
    )
    st.caption(f"Duración total: {conversacion.duracion_segundos:.1f} s")


def _boton_eliminar(checkpointer: Any, thread_id: str) -> None:
    """Borra el hilo del checkpointer tras confirmar la pérdida de memoria."""

    with st.popover("Eliminar esta conversación"):
        st.warning(
            "El checkpointer es la memoria del agente: al borrar la traza, la "
            "conversación también desaparece del chat y el agente pierde ese "
            "contexto. No se puede deshacer."
        )
        if st.button("Confirmar", key=f"confirmar_borrar_{thread_id}"):
            checkpointer.delete_thread(thread_id)
            st.toast(f"Se eliminó la conversación {thread_id}.")
            if conversacion_activa() == thread_id:
                # El chat quedaría apuntando a un hilo inexistente.
                # `cambiar_de_conversacion` ya reejecuta la app.
                cambiar_de_conversacion(f"conv-{uuid4().hex[:8]}")
            st.rerun()

        st.caption(
            "Si borran la conversación activa, el chat abrirá una nueva."
        )


def render_trazas(checkpointer: Any, system_prompt: str) -> None:
    """Renderiza las conversaciones guardadas y el detalle de cada paso."""

    st.title("Trazas del agente")
    st.markdown(
        "Cada conversación guardada aparece paso a paso: lo que envía el "
        "usuario, lo que responde el modelo, qué herramientas llama y con qué "
        "argumentos."
    )
    st.caption(
        "Los tiempos se calculan con las marcas de los checkpoints de "
        "LangGraph, así que miden el paso completo del grafo y no solo la "
        "llamada al modelo."
    )

    hilos = listar_hilos(checkpointer)
    if not hilos:
        st.info(
            "Todavía no hay conversaciones guardadas. Hablen con el agente y "
            "vuelvan a esta pantalla."
        )
        return

    etiquetas = etiquetas_de_hilos(hilos)
    identificadores = list(etiquetas)
    thread_activo = conversacion_activa()
    activo = (
        identificadores.index(thread_activo)
        if thread_activo in etiquetas
        else 0
    )
    thread_id = st.selectbox(
        "Conversación",
        options=identificadores,
        index=activo,
        format_func=lambda thread_id: etiquetas[thread_id],
    )

    conversacion = leer_conversacion(checkpointer, thread_id, system_prompt)
    # Va antes de la salida temprana: un hilo sin mensajes es justo el que
    # conviene poder borrar.
    _boton_eliminar(checkpointer, thread_id)

    if not conversacion.pasos:
        st.info("Esta conversación todavía no tiene mensajes.")
        return

    _render_metricas(conversacion)
    for paso in conversacion.pasos:
        _render_paso(paso)

    descargas = st.columns([1, 1, 2])
    descargas[0].download_button(
        "Descargar traza (JSON)",
        data=json.dumps(
            traza_a_json(conversacion),
            ensure_ascii=False,
            indent=2,
        ),
        file_name=f"traza_{thread_id}.json",
        mime="application/json",
        use_container_width=True,
    )
    descargas[1].download_button(
        "Descargar traza (Markdown)",
        data=traza_a_markdown(conversacion),
        file_name=f"traza_{thread_id}.md",
        mime="text/markdown",
        use_container_width=True,
        help="Pegable directo en el workbook del hito como evidencia.",
    )
