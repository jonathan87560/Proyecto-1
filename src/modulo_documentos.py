"""Pantalla Streamlit reutilizable para convertir e indexar documentos."""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import streamlit as st
from src.conocimiento import listar_ingestas
from src.documentos import (
    EXTENSIONES_DOCUMENTOS,
    DocumentoConvertido,
    convertir_archivo_a_markdown,
    preparar_fragmentos_markdown,
)
from src.modulo_ingesta import (
    boton_indexar,
    render_fragmentos,
    seleccionar_configuracion_troceado,
)


def _identificador(archivo: Any) -> str:
    return hashlib.sha256(archivo.getvalue()).hexdigest()


def _render_resultados(
    archivos: Sequence[Any],
    resultados: dict[str, DocumentoConvertido],
    vector_store: Any,
    prefijo: str,
) -> None:
    """Muestra, previsualiza e indexa los resultados de una pestaña."""

    ingestas = listar_ingestas(vector_store)
    huellas = [_identificador(archivo) for archivo in archivos]
    resumen = st.columns(3)
    resumen[0].metric("Archivos seleccionados", len(archivos))
    resumen[1].metric(
        "Disponibles",
        sum(huella in resultados for huella in huellas),
    )
    resumen[2].metric(
        "Versiones indexadas",
        0 if ingestas.empty else int(ingestas["file_hash"].isin(huellas).sum()),
    )

    st.subheader("Configuración de ingesta")
    configuracion = seleccionar_configuracion_troceado(prefijo)

    st.subheader("Resultados")
    resultados_visibles = 0
    for archivo in archivos:
        archivo_id = _identificador(archivo)
        resultado = resultados.get(archivo_id)
        if resultado is None:
            continue

        resultados_visibles += 1
        with st.expander(f"📄 {resultado.nombre}", expanded=True):
            fragmentos_preview = preparar_fragmentos_markdown(
                resultado,
                configuracion,
            )
            versiones = (
                0
                if ingestas.empty
                else int((ingestas["file_hash"] == archivo_id).sum())
            )
            detalles = st.columns([3, 1, 1])
            detalle_formato = resultado.mime_type
            if resultado.codificacion:
                detalle_formato += f" · {resultado.codificacion}"
            detalles[0].caption(detalle_formato)
            detalles[1].metric("Tamaño", f"{resultado.tamano / 1024:.1f} KB")
            detalles[2].metric(
                "Estado",
                f"{versiones} versión(es)" if versiones else "Listo",
            )

            st.info(
                f"**Resumen:** {resultado.resumen or 'Sin resumen disponible.'}"
            )

            vista, fuente = st.tabs(["Vista previa", "Markdown"])
            with vista:
                render_fragmentos(fragmentos_preview, configuracion)
            with fuente:
                st.code(resultado.markdown, language="markdown")

            acciones = st.columns([1, 3])
            acciones[0].download_button(
                "Descargar .md",
                data=resultado.markdown,
                file_name=f"{Path(resultado.nombre).stem}.md",
                mime="text/markdown",
                key=f"download-{prefijo}-{archivo_id}",
                use_container_width=True,
            )
            boton_indexar(
                resultado,
                configuracion,
                vector_store,
                clave=f"{prefijo}-{archivo_id}",
            )

    if resultados_visibles == 0:
        st.info("Presionen el botón para procesar los archivos seleccionados.")


def _render_importacion_documentos(modelo: Any, vector_store: Any) -> None:
    """Renderiza la conversión de PDF y formatos de texto a Markdown."""

    st.caption(
        "Gemini conserva la estructura visual de los PDF y normaliza los "
        "formatos de texto a Markdown."
    )
    archivos = st.file_uploader(
        "Seleccionen uno o varios documentos",
        type=EXTENSIONES_DOCUMENTOS,
        accept_multiple_files=True,
        help=(
            "PDF de hasta 20 MB; TXT, Markdown, JSON, CSV, TSV, HTML, XML y "
            "YAML de hasta 100.000 caracteres por conversión."
        ),
        key="archivos_documentos",
    )
    if not archivos:
        st.info("Elijan uno o varios documentos para comenzar.")
        return

    resultados: dict[str, DocumentoConvertido] = st.session_state.setdefault(
        "documentos_convertidos",
        {},
    )
    errores: dict[str, str] = st.session_state.setdefault(
        "errores_conversion",
        {},
    )
    if st.button(
        f"Convertir {len(archivos)} archivo(s) a Markdown",
        type="primary",
        use_container_width=True,
        key="convertir_con_gemini",
    ):
        progreso = st.progress(0, text="Preparando archivos...")
        with st.spinner(
            "Gemini está leyendo los archivos y reconstruyendo su estructura..."
        ):
            for posicion, archivo in enumerate(archivos, start=1):
                archivo_id = _identificador(archivo)
                errores.pop(archivo_id, None)
                try:
                    resultado = convertir_archivo_a_markdown(archivo, modelo)
                except Exception as error:
                    errores[archivo_id] = str(error)
                else:
                    resultados[resultado.huella] = resultado
                progreso.progress(
                    posicion / len(archivos),
                    text=f"Procesados {posicion} de {len(archivos)} archivos",
                )
        progreso.empty()

    for archivo in archivos:
        archivo_id = _identificador(archivo)
        if archivo_id in errores:
            st.error(f"{archivo.name}: {errores[archivo_id]}")

    _render_resultados(archivos, resultados, vector_store, "convertidor")


def render_convertidor(modelo: Any, vector_store: Any) -> None:
    """Renderiza el convertidor usando los recursos de la app anfitriona."""

    st.title("Importar documentos")
    st.markdown(
        "Conviertan archivos PDF o de texto a Markdown y agreguen solo los resultados que "
        "necesiten al conocimiento de su agente."
    )
    st.info(
        "📌 **Formatos admitidos:** PDF, TXT, Markdown, JSON, CSV, TSV, HTML, "
        "XML y YAML. Los PDF conservan su maquetación visual; los archivos de "
        "texto se normalizan desde su contenido. Para indexar Markdown sin "
        "modificarlo, usen la pantalla **Agregar Markdown**."
    )
    _render_importacion_documentos(modelo, vector_store)
