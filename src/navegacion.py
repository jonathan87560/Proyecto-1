"""Menú de la app: una página por módulo, cada una con su propia URL."""

from functools import partial
from typing import Any

import streamlit as st
from src.modulo_chat import render_chat
from src.modulo_documentos import render_convertidor
from src.modulo_markdown import render_markdown
from src.modulo_trazas import render_trazas
from src.modulo_vector_store import render_vector_store


def construir_navegacion(
    modelo: Any,
    vector_store: Any,
    checkpointer: Any,
    agente: Any,
    system_prompt: str,
    titulo: str,
    subtitulo: str,
) -> Any:
    """Arma el menú lateral y devuelve la página que toca renderizar.

    Cada pantalla recibe por parámetro los recursos que necesita. `st.Page`
    acepta el `partial` siempre que se le pase `title` y `url_path`, porque un
    partial no tiene `__name__` del que deducirlos.
    """
    paginas = [
        st.Page(
            partial(render_chat, agente, checkpointer, titulo, subtitulo),
            title="Agente Conversacional",
            icon="📰",
            url_path="agente",
            default=True,
        ),
        st.Page(
            partial(render_convertidor, modelo, vector_store),
            title="Convertir Documentos",
            icon="📄",
            url_path="convertir",
        ),
        st.Page(
            partial(render_markdown, modelo, vector_store),
            title="Agregar Markdown",
            icon="📝",
            url_path="markdown",
        ),
        st.Page(
            partial(render_vector_store, vector_store),
            title="Base de Conocimiento",
            icon="🗂️",
            url_path="conocimiento",
        ),
        st.Page(
            partial(render_trazas, checkpointer, system_prompt),
            title="Trazas del Agente",
            icon="🔍",
            url_path="trazas",
        ),
    ]

    return st.navigation(paginas)
