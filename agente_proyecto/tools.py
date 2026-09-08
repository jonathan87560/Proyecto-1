"""Tools editables del agente del equipo en S08."""

from typing import Any

from langchain_core.tools import tool
from src.conocimiento import buscar_en_vector_store

VECTOR_STORE: Any | None = None


def configurar_base_de_conocimiento(vector_store: Any) -> None:
    """Conecta la tool RAG con la base de conocimiento abierta por la app."""
    global VECTOR_STORE
    VECTOR_STORE = vector_store


@tool
def buscar_vector_store(pregunta: str, n: int = 3) -> str:
    """Recupera fragmentos y fuentes relacionados con una pregunta."""
    if VECTOR_STORE is None:
        return "ERROR: la base de conocimiento todavía no está disponible."
    return buscar_en_vector_store(VECTOR_STORE, pregunta, n)


# Definan aquí la segunda tool del proyecto con argumentos tipados y `@tool`.
# Si usa una API externa, lean la credencial desde una variable de entorno.
TOOLS_PROYECTO: list[Any] = [buscar_vector_store]
