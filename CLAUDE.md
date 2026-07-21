# Proyecto: RAG de papers de arXiv con ingesta incremental automática

Quiero que construyas un sistema RAG end-to-end, de cero, en Python. 
Es un proyecto de portfolio: priorizá código claro, bien estructurado y 
comentado sobre atajos. Construilo en FASES; al terminar cada fase, pará, 
explicame qué hiciste y cómo probarlo antes de seguir.

## Qué hace el sistema (visión general)

Tres procesos INDEPENDIENTES que se comunican SOLO a través de la base de 
datos (nunca entre sí directamente):

1. INGESTA (automática): un scheduler consulta la API pública de arXiv cada 
   cierto tiempo, detecta papers nuevos, los chunkea, genera embeddings con un 
   modelo LOCAL gratuito, y los guarda en pgvector. Es INCREMENTAL: nunca 
   reprocesa lo que ya vio.

2. API WEB (pública): un usuario pregunta en lenguaje natural desde una web. 
   El sistema hace retrieval híbrido + rerank, y genera la respuesta con un 
   modelo gratuito de OpenRouter, citando las fuentes.

3. BOT DE TELEGRAM (privado): la misma capacidad de consulta, accesible desde 
   Telegram, reusando EXACTAMENTE el mismo motor de consulta que la web.

Principio de diseño central: los tres procesos están DESACOPLADOS. Solo 
comparten la base de datos. Si uno se cae, los otros siguen funcionando. El 
motor de consulta es UN SOLO módulo compartido; la web y Telegram son solo 
"puertas de entrada" que lo invocan, sin duplicar lógica.

## Stack técnico

- Python 3.12+
- PostgreSQL con extensión pgvector como ÚNICA base de datos (guarda los 
  chunks vectorizados, la tabla de estado de ingesta, y la tabla de categorías 
  activas — todo junto, para tener transacciones consistentes).
- Embeddings: modelo local gratuito (sentence-transformers, ej. BAAI/bge-m3), 
  corriendo en CPU. NADA de embeddings pagos.
- Reranking: cross-encoder local gratuito (ej. BAAI/bge-reranker-v2-m3)
- Generación (LLM): OpenRouter, modelos con ID terminado en ":free". 
  IMPORTANTE: los modelos free rotan y desaparecen. Implementá una lista de 
  fallback: si el primer modelo falla, probá el siguiente automáticamente.
- API web: FastAPI
- Bot: python-telegram-bot
- Scheduler: para empezar, un script invocable por cron. Dejá la lógica de 
  ingesta separada del scheduler para poder migrar a un worker persistente después.
- Config vía variables de entorno (.env). NADA hardcodeado (claves, categorías, 
  intervalos, IDs de modelo, todo en config).

## Arquitectura y estructura de carpetas

Respetá esta estructura. El principio clave: core/ y query/ son el "cerebro" 
compartido y NO deben saber nada de web ni de telegram. web/ y bot/ son solo 
puertas de entrada que llaman a query/pipeline.py.

arxiv-rag/
├── docker-compose.yml          # levanta Postgres + pgvector
├── .env.example                # plantilla de config
├── README.md
├── pyproject.toml
├── sql/
│   └── schema.sql              # active_categories, ingested_papers, chunks
├── src/
│   ├── config.py               # carga y valida variables de entorno
│   ├── db.py                   # conexión a Postgres, helpers de query
│   ├── core/                   # motor compartido (agnóstico de interfaz)
│   │   ├── embeddings.py       # texto → vector (modelo local)
│   │   ├── chunking.py         # texto → fragmentos
│   │   ├── retrieval.py        # búsqueda híbrida + RRF
│   │   ├── rerank.py           # reordenar candidatos por relevancia
│   │   └── generation.py       # OpenRouter + fallback de modelos
│   ├── ingest/                 # PROCESO 1: llenar la base
│   │   ├── arxiv_client.py     # cliente de la API de arXiv
│   │   ├── incremental.py      # lógica "solo lo nuevo" (hash, estado, upsert)
│   │   ├── backfill.py         # carga histórica de una categoría
│   │   └── run_ingest.py       # script CLI que dispara la ingesta
│   ├── query/                  # motor de consulta compartido
│   │   └── pipeline.py         # orquesta: retrieval → rerank → generation
│   ├── web/                    # PROCESO 2: puerta pública
│   │   ├── api.py              # FastAPI: endpoint /ask
│   │   └── static/index.html   # buscador que muestra la evidencia
│   └── bot/                    # PROCESO 3: puerta privada
│       └── telegram_bot.py     # comando /ask reusando query/pipeline.py
└── scripts/
    └── cron_ingest.sh          # lo que ejecuta el scheduler

## Requisitos de diseño clave (esto es lo que importa)

1. CATEGORÍAS EN CONFIG, NO EN CÓDIGO: las categorías de arXiv a ingestar (ej. 
   cs.CL, cs.IR) viven en la tabla active_categories, no clavadas en el código. 
   Agregar una categoría debe ser un INSERT, sin tocar lógica ni redesplegar.

2. INGESTA INCREMENTAL con tabla de estado: la tabla ingested_papers registra 
   cada paper ya procesado (arxiv_id, content_hash, last_indexed_at). Antes de 
   procesar, consultá esa tabla. Solo procesá papers nuevos o cuyo hash cambió.

3. UPSERT correcto: cada chunk guarda metadata con su arxiv_id. Si un paper 
   cambia (hash distinto), BORRÁ todos sus chunks viejos (DELETE WHERE 
   arxiv_id=X) ANTES de insertar los nuevos. El DELETE de chunks y el UPDATE de 
   ingested_papers deben ir en la MISMA transacción, para no quedar en estado 
   inconsistente. Nunca dupliques chunks.

4. BACKFILL vs INCREMENTAL separados: dos procesos distintos. 
   - Backfill: carga histórica de una categoría (últimos N papers de un saque). 
     Se corre a mano cuando agregás una categoría nueva.
   - Incremental: mantenimiento continuo (solo lo nuevo desde la última corrida).

5. RETRIEVAL HÍBRIDO: combiná búsqueda vectorial (semántica) con búsqueda por 
   keywords (full-text de Postgres), fusionando con Reciprocal Rank Fusion 
   (RRF). No solo vectores.

6. RESPUESTAS CON FUENTES: toda respuesta cita de qué papers salió (título + 
   arxiv_id + link). El endpoint web debe poder devolver también los fragmentos 
   recuperados y sus scores de rerank, para exponer la evidencia.

7. RESILIENCIA: manejá con gracia los rate limits de OpenRouter (20 req/min, 
   50 req/día en tier gratis) y de arXiv. Reintentos con backoff donde 
   corresponda. Respetá los rate limits de arXiv (no la martilles).

## Fases de construcción (respetá este orden, pará al final de cada una)

FASE 0 — Scaffolding: estructura de carpetas de arriba, dependencias, 
.env.example, README con setup, y schema.sql completo (active_categories, 
ingested_papers, chunks con columna vector de pgvector). Docker Compose para 
Postgres+pgvector local.

FASE 1 — Ingesta incremental (el núcleo, PROCESO 1): 
  - Cliente de la API de arXiv (fetch por categoría, parseo del Atom feed).
  - Chunking. Embeddings local. Lógica incremental (estado + hash + upsert 
    transaccional). Scripts de backfill e incremental, separados. CLI para 
    correr la ingesta manualmente y verla funcionar.

FASE 2 — Retrieval + rerank (parte del cerebro): función que toma una pregunta, 
hace retrieval híbrido con RRF, aplica rerank, y devuelve los top-K fragmentos 
con scores. CLI para probarlo sin LLM todavía.

FASE 3 — Generación (completa el motor de consulta): integración con OpenRouter 
(:free + fallback). Arma el prompt con los fragmentos, genera respuesta con 
citas. query/pipeline.py queda como el motor compartido completo. CLI para 
preguntar end-to-end por terminal.

FASE 4 — API web (PROCESO 2): endpoint POST /ask que llama a query/pipeline.py 
y devuelve respuesta + fuentes + fragmentos con scores. Página HTML mínima con 
campo de búsqueda que consume ese endpoint y muestra la evidencia.

FASE 5 — Bot de Telegram (PROCESO 3): comando para preguntar al RAG, reusando 
el MISMO query/pipeline.py de la web (sin duplicar lógica).

FASE 6 — Scheduling: configuración de cron para correr la ingesta incremental 
automáticamente. Documentá cómo dejarlo corriendo en un VPS.

## Cómo quiero trabajar

- Una fase a la vez. Al terminar cada una, explicá qué construiste, cómo 
  probarlo, y esperá mi OK antes de seguir.
- Explicá las decisiones técnicas no obvias cuando las tomes.
- Preguntame si algo es ambiguo en vez de asumir.

Empezá por la FASE 0.