"""Pantalla Streamlit para indexar Markdown sin pasar por la conversión."""

from dataclasses import replace
from typing import Any

import streamlit as st
from src.documentos import (
    DocumentoConvertido,
    decodificar_archivo_texto,
    documento_desde_markdown,
    preparar_fragmentos_markdown,
    resumir_markdown,
)
from src.modulo_ingesta import (
    boton_indexar,
    render_fragmentos,
    seleccionar_configuracion_troceado,
)

EXTENSIONES_MARKDOWN = ["md", "markdown", "txt"]


def _agregar_archivos(
    archivos: list[Any],
    documentos: dict[str, DocumentoConvertido],
) -> None:
    for archivo in archivos:
        try:
            texto, _ = decodificar_archivo_texto(
                archivo.getvalue(),
                limite_caracteres=None,
            )
        except ValueError as error:
            st.error(f"{archivo.name}: {error}")
            continue
        try:
            documento = documento_desde_markdown(archivo.name, texto)
        except ValueError as error:
            st.error(f"{archivo.name}: {error}")
        else:
            documentos[documento.huella] = documento


def _panel_resumen(
    documento: DocumentoConvertido,
    modelo: Any,
) -> str:
    """Muestra el resumen editable y lo devuelve tal como quedó."""

    clave_resumen = f"markdown_resumen_{documento.huella}"
    # El botón se instancia antes que el campo de texto: Streamlit no permite
    # escribir en el estado de un widget que ya se dibujó en esta pasada.
    if st.button(
        "Generar resumen con Gemini",
        key=f"markdown_resumir_{documento.huella}",
    ):
        try:
            with st.spinner("Gemini está resumiendo el documento..."):
                st.session_state[clave_resumen] = resumir_markdown(
                    documento.markdown,
                    modelo,
                )
        except Exception as error:
            st.error(f"No se pudo generar el resumen: {error}")

    return st.text_area(
        "Resumen del documento",
        value=documento.resumen,
        key=clave_resumen,
        height=100,
        help=(
            "Se guarda como contexto de cada fragmento. Pueden editarlo o "
            "escribirlo a mano."
        ),
    )


def render_markdown(modelo: Any, vector_store: Any) -> None:
    """Renderiza la ingesta de Markdown con los recursos de la app."""

    st.title("Agregar Markdown")
    st.markdown(
        "Suban o peguen Markdown ya listo, revisen cómo queda troceado y "
        "agréguenlo al conocimiento de su agente sin pasar por Gemini."
    )

    documentos: dict[str, DocumentoConvertido] = st.session_state.setdefault(
        "markdown_documentos",
        {},
    )

    subir, pegar = st.tabs(["Subir archivos", "Pegar texto"])
    with subir:
        archivos = st.file_uploader(
            "Seleccionen uno o varios archivos Markdown",
            type=EXTENSIONES_MARKDOWN,
            accept_multiple_files=True,
            key="markdown_archivos",
        )
        if archivos and st.button(
            f"Agregar {len(archivos)} archivo(s) a la lista",
            type="primary",
            use_container_width=True,
        ):
            _agregar_archivos(archivos, documentos)

    with pegar:
        nombre = st.text_input(
            "Nombre del documento",
            key="markdown_nombre",
            placeholder="politica_de_devoluciones.md",
        )
        texto = st.text_area(
            "Contenido Markdown",
            key="markdown_texto",
            height=260,
            placeholder="# Título\n\nContenido del documento...",
        )
        if st.button(
            "Agregar a la lista",
            type="primary",
            use_container_width=True,
            key="markdown_agregar_texto",
        ):
            if not nombre.strip():
                st.error("Escriban un nombre para el documento.")
            else:
                try:
                    documento = documento_desde_markdown(nombre.strip(), texto)
                except ValueError as error:
                    st.error(str(error))
                else:
                    documentos[documento.huella] = documento
                    st.success(f"{documento.nombre} quedó en la lista.")

    if not documentos:
        st.info("Suban o peguen un documento para comenzar.")
        return

    st.subheader("Configuración de ingesta")
    configuracion = seleccionar_configuracion_troceado("markdown")

    st.subheader("Documentos por indexar")
    for huella, documento in list(documentos.items()):
        with st.expander(f"📄 {documento.nombre}", expanded=True):
            fragmentos = preparar_fragmentos_markdown(documento, configuracion)
            detalles = st.columns([3, 1, 1])
            detalles[0].caption(documento.mime_type)
            detalles[1].metric("Tamaño", f"{documento.tamano / 1024:.1f} KB")
            detalles[2].metric("Bloques", len(fragmentos))

            resumen = _panel_resumen(documento, modelo)
            if resumen != documento.resumen:
                documento = replace(documento, resumen=resumen)
                documentos[huella] = documento

            vista, fuente = st.tabs(["Vista previa", "Markdown"])
            with vista:
                render_fragmentos(fragmentos, configuracion)
            with fuente:
                st.code(documento.markdown, language="markdown")

            acciones = st.columns([1, 3])
            if acciones[0].button(
                "Quitar de la lista",
                key=f"markdown_quitar_{huella}",
                use_container_width=True,
            ):
                documentos.pop(huella, None)
                st.rerun()
            boton_indexar(
                documento,
                configuracion,
                vector_store,
                clave=f"markdown-{huella}",
            )
