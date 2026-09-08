"""Operaciones explícitas sobre la vector store y el retriever."""

from pathlib import Path
from typing import Any

import lancedb
import pandas as pd
import pyarrow as pa
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document

# Contrato de metadatos de la tabla. LanceDB congela el esquema en la primera
# inserción, por lo que cada equipo debe declarar aquí el nombre y tipo de sus
# campos propios antes de indexarlos. Tras agregar uno, se debe reconstruir la
# tabla para que tenga el nuevo esquema.
CAMPOS_METADATA: dict[str, pa.DataType] = {
    "source": pa.string(),
    "file_name": pa.string(),
    "file_hash": pa.string(),
    "mime_type": pa.string(),
    "origin": pa.string(),
    "document_summary": pa.string(),
    "splitter_mode": pa.string(),
    "chunk_size": pa.int64(),
    "chunk_overlap": pa.int64(),
    "splitter_max_level": pa.int64(),
    "ingestion_id": pa.string(),
    "indexed_at": pa.string(),
    "section_path": pa.string(),
    "chunk_index": pa.int64(),
    "chunk_count": pa.int64(),
    "h1": pa.string(),
    "h2": pa.string(),
    "h3": pa.string(),
    "h4": pa.string(),
}


def esquema_documentos(dimension: int) -> pa.Schema:
    """Construye el esquema de la tabla para embeddings de un tamaño dado."""

    return pa.schema(
        [
            ("vector", pa.list_(pa.float32(), dimension)),
            ("id", pa.string()),
            ("text", pa.string()),
            ("metadata", pa.struct(list(CAMPOS_METADATA.items()))),
        ]
    )


def normalizar_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Completa campos faltantes y rechaza metadatos fuera del esquema."""

    desconocidos = set(metadata) - set(CAMPOS_METADATA)
    if desconocidos:
        raise ValueError(
            "Hay metadatos sin declarar: "
            f"{', '.join(sorted(desconocidos))}. Agréguenlos a "
            "CAMPOS_METADATA con su tipo PyArrow y reconstruyan la tabla."
        )

    normalizada = {}
    for campo, tipo in CAMPOS_METADATA.items():
        valor = metadata.get(campo)
        if valor is None:
            normalizada[campo] = 0 if pa.types.is_integer(tipo) else ""
        else:
            normalizada[campo] = valor
    return normalizada


def _verificar_esquema(tabla_lance: Any, uri: Path | str, tabla: str) -> None:
    """Avisa temprano si la tabla quedó con un esquema de metadata antiguo."""

    metadata = tabla_lance.schema.field("metadata").type
    guardados = {metadata.field(i).name for i in range(metadata.num_fields)}
    faltantes = set(CAMPOS_METADATA) - guardados
    if faltantes:
        raise ValueError(
            f"La tabla '{tabla}' se creó con un esquema anterior y le faltan "
            f"estos campos de metadata: {', '.join(sorted(faltantes))}. "
            "LanceDB no permite cambiar el esquema de una tabla existente: "
            f"borren {Path(uri) / f'{tabla}.lance'} y vuelvan a indexar."
        )


def abrir_vector_store(
    uri: Path | str,
    tabla: str,
    embeddings: Any,
    dimension: int,
) -> LanceDB:
    """Abre la tabla de documentos y la crea vacía si todavía no existe."""

    conexion = lancedb.connect(str(uri))
    # `list_tables()` devuelve una respuesta paginada: el `in` hay que hacerlo
    # sobre `.tables`, porque sobre el objeto siempre da False.
    if tabla not in conexion.list_tables().tables:
        conexion.create_table(tabla, schema=esquema_documentos(dimension))
    else:
        _verificar_esquema(conexion.open_table(tabla), uri, tabla)

    # El `mode` por defecto del wrapper de LangChain es "overwrite": cada
    # add_documents borraría todo lo indexado antes.
    return LanceDB(
        connection=conexion,
        table_name=tabla,
        embedding=embeddings,
        mode="append",
    )


def buscar_en_vector_store(
    vector_store: LanceDB,
    pregunta: str,
    n: int = 3,
) -> str:
    """Recupera fragmentos y fuentes para una tool del agente."""

    n = max(1, min(n, 20))
    if vector_store.get_table().count_rows() == 0:
        return (
            "ERROR: la base de conocimiento está vacía. "
            "Agreguen documentos desde el convertidor o el módulo de Markdown."
        )

    documentos = vector_store.similarity_search(query=pregunta, k=n)
    if not documentos:
        return "ERROR: no se encontraron noticias relacionadas."

    bloques = []
    for posicion, documento in enumerate(documentos, start=1):
        fuente = Path(documento.metadata["source"]).name
        bloques.append(
            f"RESULTADO {posicion}\n\n"
            f"FUENTE: {fuente}\n\n"
            f"{documento.page_content}\n"
        )
    return "\n\n".join(bloques)


def buscar_con_detalle(
    vector_store: LanceDB,
    consulta: str,
    n: int = 3,
    usar_mmr: bool = False,
    ingestion_id: str | None = None,
) -> list[dict[str, Any]]:
    """Recupera fragmentos con su score para inspeccionar la búsqueda.

    Es la versión de laboratorio de `buscar_en_vector_store`: devuelve la
    metadata de cada resultado en vez del texto que lee el agente. Con `mmr`
    LanceDB no expone el score, así que queda en `None`.
    """
    if vector_store.get_table().count_rows() == 0:
        return []

    filtro = {"metadata.ingestion_id": ingestion_id} if ingestion_id else None
    if usar_mmr:
        encontrados = vector_store.max_marginal_relevance_search(
            query=consulta,
            k=n,
            fetch_k=max(n * 4, 20),
            filter=filtro,
        )
        resultados = [(documento, None) for documento in encontrados]
    else:
        resultados = vector_store.similarity_search_with_score(
            query=consulta,
            k=n,
            filter=filtro,
        )

    textos_anteriores = _textos_anteriores_por_fragmento(vector_store)
    return [
        {
            "posicion": posicion,
            "score": score,
            "texto": documento.page_content,
            "texto_anterior": textos_anteriores.get(
                (
                    str(documento.metadata.get("ingestion_id", "")),
                    int(documento.metadata.get("chunk_index", 0) or 0),
                )
            ),
            "metadata": dict(documento.metadata),
            **{
                campo: documento.metadata.get(campo)
                for campo in (
                    "file_name",
                    "section_path",
                    "ingestion_id",
                    "splitter_mode",
                    "chunk_size",
                    "chunk_index",
                )
            },
        }
        for posicion, (documento, score) in enumerate(resultados, start=1)
    ]


def _textos_anteriores_por_fragmento(
    vector_store: LanceDB,
) -> dict[tuple[str, int], str | None]:
    """Relaciona cada bloque con el que lo precede en su misma ingesta."""

    fragmentos = leer_fragmentos(vector_store)
    textos_anteriores: dict[tuple[str, int], str | None] = {}
    for _, grupo in fragmentos.groupby("ingestion_id"):
        texto_anterior: str | None = None
        for _, fragmento in grupo.sort_values("chunk_index").iterrows():
            identificador = (
                str(fragmento["ingestion_id"]),
                int(fragmento["chunk_index"]),
            )
            textos_anteriores[identificador] = texto_anterior
            texto_anterior = str(fragmento["text"])
    return textos_anteriores


def indexar_documentos(
    vector_store: LanceDB,
    documentos: list[Document],
) -> None:
    """Agrega una versión de fragmentos a la tabla LanceDB existente.

    La identidad de cada versión vive en la metadata de los documentos, por lo
    que esta operación conserva también indexaciones repetidas del mismo
    archivo y configuración.
    """
    for documento in documentos:
        documento.metadata = normalizar_metadata(documento.metadata)
    vector_store.add_documents(documentos)


def leer_fragmentos(vector_store: LanceDB) -> pd.DataFrame:
    """Devuelve una fila por fragmento indexado, con la metadata expandida."""

    tabla = vector_store.get_table()
    columnas = ["id", "text", *CAMPOS_METADATA]
    if tabla.count_rows() == 0:
        return pd.DataFrame(columns=columnas)

    filas = tabla.to_pandas().drop(columns=["vector"])
    metadata = pd.DataFrame(
        [normalizar_metadata(valor) for valor in filas["metadata"]]
    )
    return pd.concat(
        [filas.drop(columns=["metadata"]).reset_index(drop=True), metadata],
        axis=1,
    )[columnas]


def listar_ingestas(vector_store: LanceDB) -> pd.DataFrame:
    """Agrupa los fragmentos en una fila por versión de ingesta."""

    columnas = [
        "file_name",
        "origin",
        "splitter_mode",
        "splitter_max_level",
        "chunk_size",
        "chunk_overlap",
        "fragmentos",
        "indexed_at",
        "document_summary",
        "file_hash",
        "ingestion_id",
    ]
    fragmentos = leer_fragmentos(vector_store)
    if fragmentos.empty:
        return pd.DataFrame(columns=columnas)

    ingestas = fragmentos.groupby("ingestion_id", as_index=False).agg(
        file_name=("file_name", "first"),
        origin=("origin", "first"),
        splitter_mode=("splitter_mode", "first"),
        splitter_max_level=("splitter_max_level", "first"),
        chunk_size=("chunk_size", "first"),
        chunk_overlap=("chunk_overlap", "first"),
        fragmentos=("id", "count"),
        indexed_at=("indexed_at", "first"),
        document_summary=("document_summary", "first"),
        file_hash=("file_hash", "first"),
    )
    ordenadas = ingestas.sort_values(
        ["indexed_at", "file_name"],
        ascending=[False, True],
    )
    return ordenadas.reset_index(drop=True)[columnas]


def fragmentos_de_ingesta(
    vector_store: LanceDB,
    ingestion_id: str,
) -> list[dict[str, Any]]:
    """Devuelve los fragmentos de una versión, en su orden de troceado."""

    fragmentos = leer_fragmentos(vector_store)
    if fragmentos.empty:
        return []

    seleccion = fragmentos[fragmentos["ingestion_id"] == ingestion_id]
    # pandas tipa las claves como Hashable; aquí son los nombres de columna.
    # pyrefly: ignore[bad-return]
    return seleccion.sort_values("chunk_index").to_dict("records")


def _eliminar_por_campo(
    vector_store: LanceDB,
    campo: str,
    valor: str,
) -> int:
    tabla = vector_store.get_table()
    antes = tabla.count_rows()
    # El filtro es SQL: una comilla simple en el valor rompería el predicado.
    escapado = valor.replace("'", "''")
    tabla.delete(f"metadata.{campo} = '{escapado}'")
    return antes - tabla.count_rows()


def eliminar_ingesta(vector_store: LanceDB, ingestion_id: str) -> int:
    """Borra una versión de ingesta y devuelve cuántos fragmentos eliminó."""

    return _eliminar_por_campo(vector_store, "ingestion_id", ingestion_id)


def eliminar_documento(vector_store: LanceDB, file_hash: str) -> int:
    """Borra todas las versiones de un documento y devuelve los fragmentos."""

    return _eliminar_por_campo(vector_store, "file_hash", file_hash)
