"""Construcción del agente del proyecto, con memoria SQLite."""

import pathlib
import sqlite3
from typing import Any

import streamlit as st
from agente_proyecto.prompt import SYSTEM_PROMPT
from agente_proyecto.tools import TOOLS_PROYECTO
from langchain.agents import create_agent
from src.vendor import activar_vendor

activar_vendor()

from langgraph.checkpoint.sqlite import SqliteSaver

ARTEFACTOS_DIR = pathlib.Path(__file__).parents[1] / ".artifacts"
ARTEFACTOS_DIR.mkdir(exist_ok=True)


@st.cache_resource
def obtener_checkpointer() -> SqliteSaver:
    """Abre la memoria persistente de las conversaciones del proyecto."""
    conexion = sqlite3.connect(
        ARTEFACTOS_DIR / "conversaciones_proyecto.sqlite",
        check_same_thread=False,
    )
    return SqliteSaver(conexion)


def construir_agente(modelo: Any) -> tuple[Any, SqliteSaver]:
    """Arma el agente con las tools, prompt y memoria del proyecto."""
    checkpointer = obtener_checkpointer()
    agente = create_agent(
        model=modelo,
        tools=TOOLS_PROYECTO,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agente, checkpointer
