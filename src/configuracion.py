"""Carga de configuración y clientes de embeddings compartidos."""

import runpy
from pathlib import Path
from typing import Any, Literal

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel


class GoogleGenerativeAIEmbeddingsWrapper(GoogleGenerativeAIEmbeddings):
    """Adaptador para modelos cuyo endpoint acepta un contenido por llamada."""

    def embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 1,
        task_type: str | None = None,
        titles: list[str] | None = None,
        output_dimensionality: int | None = None,
    ) -> list[list[float]]:
        return super().embed_documents(
            texts,
            batch_size=1,
            task_type=task_type,
            titles=titles,
            output_dimensionality=output_dimensionality,
        )


class Config(BaseModel):
    project_id: str
    location: str
    target_credentials: Any | None = None
    vertexai: bool


def cargar_configuracion(
    modo: Literal["estudiante", "docente"] = "estudiante",
) -> Config:
    if modo not in ["estudiante", "docente"]:
        raise ValueError("modo debe ser 'estudiante' o 'docente'.")

    if modo == "docente":
        print("Configuración cargada en modo docente.")
        return Config(
            project_id="idia-general",
            location="global",
            target_credentials=None,
            vertexai=True,
        )

    cwd = Path.cwd()
    rutas_configuracion = [
        cwd / "01_configuracion_jupyter.py",
        cwd / "03-proyecto" / "01_configuracion_jupyter.py",
        cwd.parent / "03-proyecto" / "01_configuracion_jupyter.py",
        cwd.parent.parent / "03-proyecto" / "01_configuracion_jupyter.py",
        cwd.parent.parent.parent
        / "03-proyecto"
        / "01_configuracion_jupyter.py",
    ]
    archivo = next(
        (ruta for ruta in rutas_configuracion if ruta.exists()),
        None,
    )
    if archivo is None:
        raise FileNotFoundError(
            "No se encontró 01_configuracion_jupyter.py. "
            "Verificar si el archivo existe en las siguientes rutas:\n\t"
            + "\n\t".join(map(str, rutas_configuracion))
        )

    config_cargadas = runpy.run_path(str(archivo))
    print(
        "Configuración cargada en modo estudiante:\n"
        f"\t- PROJECT_ID: {config_cargadas['PROJECT_ID']}\n"
        f"\t- LOCATION: {config_cargadas['LOCATION']}"
    )
    try:
        return Config(
            project_id=config_cargadas["PROJECT_ID"],
            location=config_cargadas["LOCATION"],
            target_credentials=config_cargadas["target_credentials"],
            vertexai=True,
        )
    except KeyError as error:
        raise KeyError(
            f"Faltan variables de configuración: {error!r}"
        ) from error
