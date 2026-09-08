"""Aplicación base S08 para el proyecto del equipo."""

import pathlib
from typing import Literal

import streamlit as st
from agente_proyecto.agente import construir_agente
from agente_proyecto.prompt import SYSTEM_PROMPT
from agente_proyecto.tools import configurar_base_de_conocimiento
from langchain_google_genai import ChatGoogleGenerativeAI
from phoenix.otel import register
from src.configuracion import (
    GoogleGenerativeAIEmbeddingsWrapper,
    cargar_configuracion,
)
from src.conocimiento import abrir_vector_store
from src.navegacion import construir_navegacion

st.set_page_config(
    page_title="Agente del equipo · S08",
    page_icon="📰",
    layout="wide",
)

# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------

MODO: Literal["estudiante", "docente"] = "estudiante"

# Relativo a este archivo: la app usa la misma base aunque Streamlit se
# levante desde otro directorio.
ARTEFACTOS_DIR = pathlib.Path(__file__).parent / ".artifacts"
ARTEFACTOS_DIR.mkdir(exist_ok=True)

TABLA_DOCUMENTOS = "documentos"
DIMENSION_EMBEDDINGS = 3072

# Phoenix es opcional y está pensado para el entorno docente.
USAR_PHOENIX = False


@st.cache_resource
def instrumentar_phoenix():
    """Conecta la app con Phoenix cuando el equipo docente lo solicita."""
    return register(
        project_name="s08-proyecto-rag",
        endpoint="http://localhost:6006/v1/traces",
        auto_instrument=True,
    )


if USAR_PHOENIX:
    try:
        instrumentar_phoenix()
    except Exception as error:
        st.warning(
            "Phoenix está activado, pero no responde. "
            f"La app seguirá sin enviar trazas: {error}"
        )


# -------------------------------------------------------------------
# Modelo y recursos
# -------------------------------------------------------------------


@st.cache_resource
def obtener_recursos(modo: Literal["estudiante", "docente"]):
    """Crea una sola vez el modelo, embeddings y vector store."""
    config = cargar_configuracion(modo)
    modelo = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        project=config.project_id,
        location=config.location,
        credentials=config.target_credentials,
        vertexai=config.vertexai,
    )
    embeddings = GoogleGenerativeAIEmbeddingsWrapper(
        model="gemini-embedding-2",
        project=config.project_id,
        location=config.location,
        credentials=config.target_credentials,
        vertexai=config.vertexai,
    )
    vector_store = abrir_vector_store(
        uri=ARTEFACTOS_DIR / "lancedb_proyecto",
        tabla=TABLA_DOCUMENTOS,
        embeddings=embeddings,
        dimension=DIMENSION_EMBEDDINGS,
    )
    return modelo, vector_store


modelo, vector_store = obtener_recursos(MODO)
# La app abre LanceDB; la tool RAG usa esta conexión para buscar.
configurar_base_de_conocimiento(vector_store)


TITULO_APP = "Agente del equipo · S08"
SUBTITULO = "Integren recuperación documental y una segunda tool"

# El agente, el prompt y las tools viven en `agente_proyecto/`.
agente, checkpointer = construir_agente(modelo)


construir_navegacion(
    modelo=modelo,
    vector_store=vector_store,
    checkpointer=checkpointer,
    agente=agente,
    system_prompt=SYSTEM_PROMPT,
    titulo=TITULO_APP,
    subtitulo=SUBTITULO,
).run()
