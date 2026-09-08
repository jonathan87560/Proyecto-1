# Dependencias vendorizadas de S08

Este directorio conserva únicamente las dependencias necesarias para la memoria
SQLite de los agentes de S08:

```text
langgraph/checkpoint/sqlite/ # langgraph-checkpoint-sqlite 3.1.1
aiosqlite/                   # aiosqlite 0.22.1
```

La app importa `src.vendor.activar_vendor()` antes de cargar
`langgraph.checkpoint.sqlite.SqliteSaver`, porque el ambiente estudiantil no
incluye `langgraph-checkpoint-sqlite`.

No se incluyen dependencias vendorizadas para importar documentos: la
aplicación admite exclusivamente archivos PDF y los convierte con Gemini.

La versión 3.1.1 del checkpointer corrige el aviso de seguridad publicado para
versiones anteriores. Para bases de datos no confiables, configuren
`LANGGRAPH_STRICT_MSGPACK=true` antes de iniciar la app.
