# Maki – Multi-Agent RAG template (LangGraph + Gradio)

Sistema base para orquestar **multi-agentes** con **RAG híbrido (BM25 + embeddings)**, **bucles de verificación**, **fallback a LLM** y **trazabilidad**. Incluye una **UI en Gradio** para pruebas locales y tests básicos.

---

## Estructura

```
.
├─ .env                         # opcional (variables de entorno)
├─ .env.example                 # opcional (variables de entorno)
├─ requirements.txt
├─ README.md
└─ maki/
   ├─ __init__.py
   ├─ app.py                    # UI (Gradio)
   ├─ main.py                   # clase Maki (importable)
   ├─ agents/
   │  ├─ __init__.py
   │  ├─ relevance_checker.py   # LLM/embeddings/heurística → GYM|CLIENTS|UNKNOWN
   │  ├─ research_agent.py      # síntesis (LLM si hay) o fallback local
   │  ├─ verification_agent.py  # verificación (LLM si hay) o fallback local
   │  ├─ router.py              # enruta por etiqueta
   │  ├─ tools.py               # tool de bienvenida (mock)
   │  └─ workflow.py            # LangGraph: welcome→retrieve→route→research→verify→{redraft|fallback}
   ├─ retriever/
   │  ├─ __init__.py
   │  └─ builder.py             # retriever híbrido (BM25 + embeddings OpenAI/ST)
   ├─ config/
   │  ├─ __init__.py
   │  ├─ constants.py           # mensajes/constantes (IRRELEVANT_MSG, etc.)
   │  └─ settings.py            # carga .env (robusta), thresholds y rutas de logs
   ├─ utils/
   │  ├─ __init__.py            # expone logger & datos mock
   │  ├─ data.py                # GYM_DOCS / CLIENTS_DOCS (mock JSON)
   │  └─ logging.py             # logger con rotación + helpers de trace
   └─ tests/
      └─ test_agent.py
```

---

## Requisitos

```txt
# --- UI ---
gradio

# --- Orquestación (grafo) ---
langgraph

# --- Retriever híbrido ---
rank-bm25
numpy

# --- LLMs (opcional: habilita clasificación/síntesis/verificación y fallback) ---
openai

# --- Utilidades ---
python-dotenv

# --- Tests (para ejecutar test_agent) ---
pytest
```
---

## Configuración (.env)

Crea un `.env` en la **raíz del repo**:

```env
# LLM (opcional para fallback y agentes con IA)
OPENAI_API_KEY=sk-xxxx
MODEL_NAME=gpt-4o-mini
MAX_OUTPUT_TOKENS=400

# Embeddings
EMBEDDING_MODEL=text-embedding-3-small

# Logging
LOG_LEVEL=INFO
LOG_JSON=1
LOG_DIR=./logs
LOG_FILE=maki.log
```

> `config/settings.py` carga el `.env` **relativo al proyecto**, no al directorio desde donde ejecutes.

---

## Instalación

```bash
# activar venv (ejemplo Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Ejecutar la UI (Gradio)

Desde la **raíz del repo**:

```bash
python -m maki.app
```

Esto abre `http://127.0.0.1:7860` (o puerto libre).
La pestaña “QA del análisis” permite preguntar:

* “¿Qué canal de contacto prefieren los clientes?” → dominio **CLIENTS** (RAG).
* “¿Cuál es el plan más barato del gym?” → dominio **GYM** (RAG).
* “¿qué versiones de gpt hay hasta el momento?” → **UNKNOWN** → **fallback** (LLM si hay `OPENAI_API_KEY`; si no, `IRRELEVANT_MSG`).

---

## Usar como módulo (backend/script)

```python
from maki.main import Maki

mk = Maki()
print(mk.ask("¿Qué planes de gimnasio tienen?"))
# {'final_answer': '...', 'domain': 'GYM', 'trace': [...]}
```

---

## Arquitectura (resumen)

* **LangGraph**: `welcome → retrieve → route → research → verify → {redraft|fallback} → format_output`.
* **Retrieval híbrido**: `rank_bm25 + embeddings` (OpenAI si hay API key, si no Sentence-Transformers; si no, solo BM25).
* **Agentes con IA**:

  * `RelevanceChecker`: LLM → embeddings → heurística. Detecta términos de IA (gpt, openai…) y devuelve **UNKNOWN**.
  * `ResearchAgent`: sintetiza **solo** con fragmentos (LLM si hay; fallback local sin “eco” de la pregunta).
  * `VerificationAgent`: valida soporte y relevancia (LLM si hay; fallback local ignora “eco”).
* **Fallback**: si no hay respuesta o dominio `UNKNOWN`, intenta **completion** (OpenAI) y lo anota en el **trace**.
* **Tracing & logging**: `utils/logging.py` guarda consola + archivo rotativo (`logs/maki.log`) con helpers:

  * `start_run(session_id=...)`
  * `append_and_log(state, key, value, **campos)`
  * `persist_agent_trace(state, **campos)`

---

## Trazabilidad

Cada ejecución crea un `run_id` y (opcional) `session_id`.
En `workflow.run()` ya se hace:

* `start_run()` al inicio
* `persist_agent_trace(final_state, domain=domain)` al final

El `trace` también vuelve en la respuesta (`out["trace"]`).

---

## Tests

```bash
pytest -q
```

`tests/test_agent.py` valida rutas básicas y que haya trace.

---

## Personalización rápida

* **Thresholds de relevancia**: en `agents/relevance_checker.py` (umbrales de embeddings y cobertura).
* **Fallback prompt**: en `main._openai_fallback_callable()` (texto que recibe el LLM cuando no hay datos).
* **Datos**: edita `utils/data.py` (mock de GYM\_DOCS/CLIENTS\_DOCS).
* **Modelo de embeddings**: cambia `EMBEDDING_MODEL` (OpenAI) o `ST_EMBEDDING_MODEL` (local) en `.env`.

---

## Problemas comunes (FAQ)

* **“No module named 'maki'”**
  Ejecuta desde la raíz: `python -m maki.app`. Evita nombres de paquete con guiones.

* **No toma el `OPENAI_API_KEY`**
  Asegura `.env` en raíz. `settings.py` ya lo carga relativo al proyecto. Reinicia el proceso.
  Verifica:

  ```bash
  python - << "PY"
  ```

from maki.config.settings import settings
import os; print('settings:', bool(settings.OPENAI\_API\_KEY), 'os:', bool(os.getenv("OPENAI\_API\_KEY")))
PY

---

## Roadmap corto

- Nodo **human-in-the-loop** real con `graph.interrupt()` (LangGraph).
- Persistir trazas en base de datos / observabilidad.
- RAG con vector store real (FAISS/PGVector/Chroma) y loaders.

