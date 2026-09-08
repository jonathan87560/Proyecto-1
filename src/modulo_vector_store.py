"""Pantalla Streamlit para explorar y depurar la base de conocimiento."""

from typing import Any, cast

import pandas as pd
import streamlit as st
from src.conocimiento import (
    buscar_con_detalle,
    eliminar_documento,
    eliminar_ingesta,
    fragmentos_de_ingesta,
    listar_ingestas,
)
from src.modulo_ingesta import (
    ETIQUETAS_SPLITTER,
    MARCAS_NIVEL,
    render_detalle_fragmento,
)

COLUMNAS_VISIBLES = [
    "file_name",
    "origin",
    "splitter_mode",
    "nivel",
    "chunk_size",
    "chunk_overlap",
    "fragmentos",
    "indexed_at",
]
CONFIGURACION_COLUMNAS = {
    "file_name": st.column_config.TextColumn("Archivo"),
    "origin": st.column_config.TextColumn("Origen"),
    "splitter_mode": st.column_config.TextColumn("Splitter"),
    "nivel": st.column_config.TextColumn("Nivel"),
    "chunk_size": st.column_config.NumberColumn("Tamaño"),
    "chunk_overlap": st.column_config.NumberColumn("Overlap"),
    "fragmentos": st.column_config.NumberColumn("Fragmentos"),
    "indexed_at": st.column_config.DatetimeColumn(
        "Indexado",
        format="YYYY-MM-DD HH:mm",
    ),
}


def _render_fragmentos_indexados(fragmentos: list[dict[str, Any]]) -> None:
    texto_anterior: str | None = None
    for fragmento in fragmentos:
        titulo = (
            f"Bloque {fragmento['chunk_index']} de {fragmento['chunk_count']}"
            f" · {len(fragmento['text'])} caracteres"
            f" · {fragmento['section_path']}"
        )
        with st.expander(titulo):
            metadata = {
                clave: valor
                for clave, valor in fragmento.items()
                if clave not in {"id", "text"}
            }
            render_detalle_fragmento(
                fragmento["text"],
                metadata,
                texto_anterior,
            )
        texto_anterior = fragmento["text"]


def _render_resultados(resultados: list[dict[str, Any]]) -> None:
    if not resultados:
        st.warning("La búsqueda no devolvió fragmentos.")
        return

    for resultado in resultados:
        score = resultado["score"]
        titulo = (
            f"{resultado['posicion']}. {resultado['file_name']} · "
            f"{resultado['section_path']}"
        )
        if score is not None:
            titulo += f" · distancia {score:.3f}"
        with st.expander(titulo, expanded=resultado["posicion"] == 1):
            st.caption(
                f"Versión {str(resultado['ingestion_id'])[:8]}… · "
                f"{resultado['splitter_mode']} · "
                f"bloque {resultado['chunk_index']}"
            )
            render_detalle_fragmento(
                resultado["texto"],
                resultado["metadata"],
                resultado["texto_anterior"],
            )


@st.cache_data(show_spinner=False)
def _buscar_cacheado(
    _vector_store: Any,
    consulta: str,
    n: int,
    usar_mmr: bool,
    ingestion_id: str | None,
    fragmentos_en_la_base: int,
) -> list[dict[str, Any]]:
    """Cachea la búsqueda porque embeber la consulta es una llamada a la API.

    Sin esto, cualquier interacción de la pantalla (seleccionar una fila, abrir
    un popover) volvería a embeber la misma consulta. `_vector_store` no entra
    en la clave por no ser hasheable; `fragmentos_en_la_base` la invalida cuando
    se indexa o se borra algo.
    """
    return buscar_con_detalle(
        _vector_store,
        consulta,
        n=n,
        usar_mmr=usar_mmr,
        ingestion_id=ingestion_id,
    )


def _probar_busqueda(vector_store: Any, ingestas: pd.DataFrame) -> None:
    """Deja lanzar consultas sueltas para ver qué recupera el agente."""

    st.subheader("Probar la búsqueda")
    st.markdown(
        "Lancen la consulta que le harían al agente y miren qué fragmentos "
        "recupera. Si la respuesta del agente falla, aquí se ve si el problema "
        "está en la recuperación o en el prompt."
    )

    consulta = st.text_input(
        "Consulta",
        key="consulta_de_prueba",
        placeholder="¿Cuánto demora un despacho a regiones?",
    )
    controles = st.columns([1, 1, 2])
    n = controles[0].number_input(
        "Fragmentos (k)",
        min_value=1,
        max_value=20,
        value=3,
        key="k_de_prueba",
    )
    usar_mmr = controles[1].toggle(
        "MMR",
        key="mmr_de_prueba",
        help=(
            "Maximal Marginal Relevance prioriza la diversidad entre los "
            "resultados; sin él manda la similitud pura. LanceDB no expone el "
            "score en este modo."
        ),
    )
    versiones = {"": "Todas las versiones"} | {
        fila["ingestion_id"]: (
            f"{fila['file_name']} · {fila['splitter_mode']} · "
            f"{fila['chunk_size']} caracteres"
        )
        for _, fila in ingestas.iterrows()
    }
    ingestion_id = controles[2].selectbox(
        "Buscar solo en",
        options=list(versiones),
        format_func=lambda version: versiones[version],
        key="version_de_prueba",
    )

    if not consulta:
        st.info("Escriban una consulta para ver qué recupera la base.")
        return

    with st.spinner("Generando el embedding de la consulta y buscando..."):
        resultados = _buscar_cacheado(
            vector_store,
            consulta,
            int(n),
            usar_mmr,
            ingestion_id or None,
            int(ingestas["fragmentos"].sum()),
        )
    _render_resultados(resultados)


def _acciones_de_borrado(
    vector_store: Any,
    ingesta: pd.Series,
) -> None:
    columnas = st.columns([1, 1, 2])
    with columnas[0].popover("Eliminar esta versión", use_container_width=True):
        st.caption(
            f"Se borrarán los {ingesta['fragmentos']} fragmentos de esta "
            "versión. Las otras versiones del documento se conservan."
        )
        if st.button("Confirmar", key="confirmar_borrar_version"):
            borrados = eliminar_ingesta(vector_store, ingesta["ingestion_id"])
            st.toast(f"Se eliminaron {borrados} fragmentos.")
            st.rerun()

    with columnas[1].popover(
        "Eliminar el documento",
        use_container_width=True,
    ):
        st.caption(
            f"Se borrarán todas las versiones de {ingesta['file_name']}, "
            "incluidas las indexadas con otra configuración."
        )
        if st.button("Confirmar", key="confirmar_borrar_documento"):
            borrados = eliminar_documento(vector_store, ingesta["file_hash"])
            st.toast(f"Se eliminaron {borrados} fragmentos.")
            st.rerun()


def render_vector_store(vector_store: Any) -> None:
    """Renderiza la tabla de versiones indexadas y su detalle."""

    st.title("Base de conocimiento")
    st.markdown(
        "Cada fila es una **versión de ingesta**: el mismo documento troceado "
        "con una configuración distinta aparece más de una vez, y el agente "
        "busca sobre todas."
    )

    ingestas = listar_ingestas(vector_store)
    if ingestas.empty:
        st.info(
            "La base está vacía. Agreguen documentos desde el convertidor o "
            "desde el módulo de Markdown."
        )
        return

    metricas = st.columns(3)
    metricas[0].metric("Documentos", ingestas["file_hash"].nunique())
    metricas[1].metric("Versiones", len(ingestas))
    metricas[2].metric("Fragmentos", int(ingestas["fragmentos"].sum()))

    _probar_busqueda(vector_store, ingestas)

    st.divider()
    st.subheader("Documentos indexados")
    tabla = ingestas.copy()
    tabla["splitter_mode"] = tabla["splitter_mode"].map(
        lambda modo: ETIQUETAS_SPLITTER.get(modo, modo)
    )
    tabla["nivel"] = tabla["splitter_max_level"].map(
        lambda nivel: MARCAS_NIVEL.get(nivel, "—")
    )
    tabla["indexed_at"] = pd.to_datetime(tabla["indexed_at"], errors="coerce")

    seleccion = st.dataframe(
        tabla,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_order=COLUMNAS_VISIBLES,
        column_config=CONFIGURACION_COLUMNAS,
    )

    # Streamlit expone `selection.rows` cuando el dataframe está en modo
    # interactivo, pero el tipo publicado por la versión instalada no lo
    # declara todavía.
    filas = cast(Any, seleccion).selection.rows
    if not filas:
        st.info("Seleccionen una fila para ver su contenido y borrarla.")
        return

    ingesta = ingestas.iloc[filas[0]]
    st.subheader(ingesta["file_name"])
    nivel = MARCAS_NIVEL.get(ingesta["splitter_max_level"])
    st.caption(
        f"Versión {ingesta['ingestion_id'][:8]}… · {ingesta['origin']} · "
        f"{ETIQUETAS_SPLITTER.get(ingesta['splitter_mode'], ingesta['splitter_mode'])}"
        f"{f' hasta {nivel}' if nivel else ''}"
        f" · {ingesta['chunk_size']} caracteres · overlap "
        f"{ingesta['chunk_overlap']}"
    )
    if ingesta["document_summary"]:
        st.info(f"**Resumen:** {ingesta['document_summary']}")

    fragmentos = fragmentos_de_ingesta(vector_store, ingesta["ingestion_id"])
    st.subheader("Fragmentos")
    _render_fragmentos_indexados(fragmentos)

    _acciones_de_borrado(vector_store, ingesta)
