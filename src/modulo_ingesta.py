"""Piezas de UI que comparten el convertidor y el módulo de Markdown."""

from typing import Any, cast
from uuid import uuid4

import streamlit as st
from langchain_core.documents import Document
from src.conocimiento import indexar_documentos
from src.documentos import (
    NIVELES_ENCABEZADO,
    TAMANOS_CHUNK,
    ConfiguracionTroceado,
    DocumentoConvertido,
    ModoSplitter,
    obtener_overlaps_validos,
    preparar_fragmentos_markdown,
)

ETIQUETAS_SPLITTER = {
    "largo_fijo": "Largo fijo",
    "secciones": "Secciones Markdown (# a ####)",
}
NIVELES_DISPONIBLES = [nivel for nivel, _, _ in NIVELES_ENCABEZADO]
MARCAS_NIVEL = {nivel: marca for nivel, marca, _ in NIVELES_ENCABEZADO}


def seleccionar_configuracion_troceado(prefijo: str) -> ConfiguracionTroceado:
    """Muestra los controles de troceado y devuelve la configuración elegida."""

    modo_splitter = cast(
        ModoSplitter,
        st.selectbox(
            "Splitter",
            options=list(ETIQUETAS_SPLITTER),
            format_func=lambda modo: ETIQUETAS_SPLITTER[modo],
            key=f"{prefijo}_modo_splitter",
            help=(
                "El modo por secciones conserva la jerarquía de encabezados y "
                "subdivide las secciones largas."
            ),
        ),
    )
    nivel_maximo = NIVELES_DISPONIBLES[-1]
    if modo_splitter == "secciones":
        nivel_maximo = st.selectbox(
            "Cortar hasta el nivel",
            options=NIVELES_DISPONIBLES,
            format_func=lambda nivel: f"{MARCAS_NIVEL[nivel]} (nivel {nivel})",
            index=len(NIVELES_DISPONIBLES) - 1,
            key=f"{prefijo}_nivel_maximo",
            help=(
                "Los encabezados más profundos que este no abren un bloque "
                "nuevo: quedan dentro del bloque de su sección."
            ),
        )
    chunk_size = st.selectbox(
        "Tamaño máximo del bloque (caracteres)",
        options=TAMANOS_CHUNK,
        index=TAMANOS_CHUNK.index(600),
        key=f"{prefijo}_chunk_size",
    )
    overlaps_validos = obtener_overlaps_validos(chunk_size)
    overlap_key = f"{prefijo}_chunk_overlap"
    # El overlap depende del tamaño elegido: si el valor guardado dejó de ser
    # válido hay que corregirlo antes de instanciar el widget.
    if st.session_state.get(overlap_key) not in overlaps_validos:
        st.session_state[overlap_key] = overlaps_validos[0]
    chunk_overlap = st.selectbox(
        "Overlap (caracteres)",
        options=overlaps_validos,
        key=overlap_key,
        help="Solo se muestran overlaps menores al tamaño del bloque.",
    )
    return ConfiguracionTroceado(
        modo=modo_splitter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        nivel_maximo=nivel_maximo,
    )


def render_fragmentos(
    fragmentos: list[Document],
    configuracion: ConfiguracionTroceado,
) -> None:
    """Muestra la vista previa de los bloques que se van a indexar."""

    nivel = (
        f" hasta {MARCAS_NIVEL[configuracion.nivel_maximo]} ·"
        if configuracion.modo == "secciones"
        else ""
    )
    st.caption(
        f"{len(fragmentos)} bloques · "
        f"{ETIQUETAS_SPLITTER[configuracion.modo]}{nivel} · "
        f"máximo {configuracion.chunk_size} caracteres · "
        f"overlap {configuracion.chunk_overlap}"
    )
    texto_anterior: str | None = None
    for posicion, fragmento in enumerate(fragmentos, start=1):
        ruta = fragmento.metadata.get("section_path", "Documento completo")
        with st.expander(
            f"Bloque {posicion} · {len(fragmento.page_content)} "
            f"caracteres · {ruta}",
            expanded=posicion == 1,
        ):
            render_detalle_fragmento(
                fragmento.page_content,
                fragmento.metadata,
                texto_anterior,
            )
        texto_anterior = fragmento.page_content


def texto_solapado(
    texto_anterior: str | None,
    texto_actual: str,
    maximo: int,
) -> str:
    """Devuelve el prefijo del bloque actual que también cierra el anterior."""

    if not texto_anterior or maximo <= 0:
        return ""

    limite = min(len(texto_anterior), len(texto_actual), maximo)
    for longitud in range(limite, 0, -1):
        if texto_anterior.endswith(texto_actual[:longitud]):
            return texto_actual[:longitud]
    return ""


def render_detalle_fragmento(
    texto: str,
    metadata: dict[str, Any],
    texto_anterior: str | None = None,
) -> None:
    """Muestra el texto, su solapamiento real y la metadata del fragmento."""

    seccion = metadata.get("section_path", "Documento completo")
    overlap_configurado = int(metadata.get("chunk_overlap", 0) or 0)
    overlap = texto_solapado(texto_anterior, texto, overlap_configurado)
    contenido, solapamiento, metadata_tab = st.tabs(
        ["Texto", "Overlap", "Metadata"]
    )
    with contenido:
        st.caption(f"Sección: {seccion}")
        if overlap:
            st.caption(
                "El siguiente texto también aparece al final del bloque "
                "anterior:"
            )
            st.code(overlap, language="markdown")
        st.markdown(texto)
    with solapamiento:
        if texto_anterior is None:
            st.caption("Es el primer bloque: no tiene overlap anterior.")
        elif overlap:
            st.caption(
                f"{len(overlap)} caracteres compartidos con el bloque "
                f"anterior · configuración: {overlap_configurado}."
            )
            st.code(overlap, language="markdown")
        else:
            st.caption(
                "No se detectó texto compartido con el bloque anterior. "
                f"La configuración de overlap es {overlap_configurado}."
            )
    with metadata_tab:
        st.json(metadata)


def boton_indexar(
    documento: DocumentoConvertido,
    configuracion: ConfiguracionTroceado,
    vector_store: Any,
    clave: str,
) -> None:
    """Agrega una versión independiente del documento a la vector store."""

    if not st.button(
        "Agregar Documento a la Base de Datos",
        key=f"index-{clave}",
        type="primary",
        use_container_width=True,
        help=(
            "Divide el Markdown con la configuración actual y genera una "
            "nueva versión independiente en LanceDB."
        ),
    ):
        return

    ingestion_id = uuid4().hex
    fragmentos = preparar_fragmentos_markdown(
        documento,
        configuracion,
        ingestion_id=ingestion_id,
    )
    try:
        with st.spinner("Generando embeddings y actualizando LanceDB..."):
            indexar_documentos(vector_store, fragmentos)
    except Exception as error:
        st.error(f"No se pudo actualizar la vector store: {error}")
    else:
        st.toast(
            f"Nueva versión agregada: {len(fragmentos)} fragmentos "
            f"indexados ({ingestion_id[:8]}…)."
        )
        # Redibuja para que los contadores lean la base ya actualizada.
        st.rerun()
