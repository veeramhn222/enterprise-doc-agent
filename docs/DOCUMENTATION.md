# Documentation - Enterprise Document Agent

## 1. Purpose
A Generative-AI knowledge assistant that lets users upload enterprise documents (PDF, TXT, MD, CSV, XLSX) and ask
natural-language questions. It combines **Retrieval-Augmented Generation (RAG)** with a small **multi-agent workflow**
that plans, retrieves, reasons with tools and validates the answer before it is shown.

## 2. System setup
See `README.md` (quick start, manual install, configuration, troubleshooting). Stack:
Python 3.10-3.13 (the launcher provisions 3.12 via `uv`), FastAPI (REST API + HTML UI) or Streamlit (`streamlit_app.py`), **LangGraph** (agent orchestration), ChromaDB (vector DB; numpy fallback), ONNX MiniLM-L6-v2
embeddings (local, no key), LiteLLM (any LLM), pypdf, pandas/openpyxl, vanilla JS UI.

## 3. Architecture
```
 Browser UI (Ask | Documents | Admin)
        |  REST (JSON / multipart)
 FastAPI  app/main.py
   |-- /api/admin/*     -> config.py  -> data/settings.json   (provider, model, key, base URL)
   |-- /api/documents*  -> guardrails.validate_upload -> ingestion.py -> vectorstore.py (Chroma / numpy)
   '-- /api/query       -> guardrails.validate_question -> agents.answer_question
                                  |
        +-------------------------+-------------------------------------------+
        | PLANNER --plan--> RETRIEVER --chunks--> REASONER <--tools--> VALIDATOR |
        |  (LLM)           (tools, no LLM)     (LLM, ReAct loop)        (LLM)     |
        |                                          ^                      |       |
        |                                          '---- one revision ----'       |
        +-------------------------------------------------------------------------+
                 tools: search_documents (vector store), list_documents, calculator
                 LLM calls: llm.py -> LiteLLM -> OpenAI / Anthropic / Gemini / Groq / Ollama / ...
```

### Data flow - ingestion
1. Upload validated (extension allow-list, non-empty, <= 25 MB).
2. Parser per type: PDF text per page; TXT/MD; CSV and Excel (every sheet) -> one **summary chunk** (row count, columns,
   sum/mean/min/max of numeric columns) + **row-group chunks** (12 rows each, `column: value | ...`).
3. Text is split into ~1000-character paragraph-aware chunks with 150-character overlap.
4. Chunks are embedded locally and stored in the vector store (cosine space) with metadata: file, page/sheet/rows.
5. A small JSON registry (`data/documents.json`) tracks uploaded documents for listing/deleting.

### Document descriptions (catalog)
After indexing, `app/describe.py` creates a catalog entry for each file: a short summary, topic tags and suggested questions.
- **Offline (always):** built from the parsed content. Text/PDF: kind, page and word count, title and detected section headings.
  CSV/Excel: row counts, column names per sheet, and questions built from the numeric/label columns.
- **AI (when an LLM is configured):** a sample of the document (evenly spaced chunks, sanitised, wrapped in `<document>` tags and
  treated as untrusted data) is sent to the LLM, which returns JSON `{summary, topics, questions}`. Output is validated and
  length-limited; any failure silently keeps the offline entry, so uploads never fail because of this step.
- `POST /api/documents/{id}/describe` regenerates the AI entry (e.g. after saving an API key). The entry is stored in the document registry.
- The UI shows it on the Documents tab, in the Ask tab's "Knowledge base" panel, and as clickable suggestions on the welcome screen.

### Data flow - question answering
1. Input guardrails -> 2. Planner -> 3. Retriever -> 4. Reasoner (with tools) -> 5. Validator (+ optional revision)
-> 6. Output redaction -> response with answer, citations `[S#]`, sources, validation score and full agent trace.

## 4. Agent orchestration (LangGraph)
The four agents are coordinated by a **LangGraph** state machine (`app/graph.py`; LangGraph is part of the LangChain ecosystem).
Each agent function in `app/agents.py` is wrapped as a graph node, and LangGraph owns the control flow, including conditional routing:

```
START -> precheck -> planner -> retriever -> reasoner -> validator --(weak answer, first time)--> reviser -> validator
            |                      |                          \--(otherwise)--> finish -> END
            '-- no documents       '-- nothing relevant  => END (refusal; the LLM is never called)
```
- **State:** a typed dict carrying the question, the run object (trace + retrieved sources), plan, answer, validator verdict and a `revised` flag.
- **Conditional edges:** early exit when nothing is indexed or nothing relevant is retrieved; a bounded self-correction loop (validator -> reviser -> validator, at most once).
- **Fallback:** if `langgraph` is missing or incompatible, the app automatically uses an equivalent sequential orchestrator
  (`ORCHESTRATOR=plain` forces it). Both use the same helper functions, so results are identical; the trace's first line says which one ran.
- The LLM layer stays on LiteLLM so that one API-key field works for every provider.

### Why LangGraph (and not LangChain agents or CrewAI)?
| Option | Fit for this project |
|---|---|
| **LangGraph** | Explicit graph of steps with conditional edges and loops: a natural match for plan -> retrieve -> reason -> validate -> revise. Few dependencies, no lock-in to a model vendor. **Chosen.** |
| LangChain (agents / chains) | Large API surface that changes often; model wrappers are separate packages per provider (`langchain-openai`, `langchain-anthropic`, ...), which conflicts with "one key field for any LLM". Not needed for this flow. |
| CrewAI | Role/task paradigm with a heavy dependency tree (pins chromadb, litellm and others, so it can clash with this project's own pins). Better for open-ended multi-agent "crews" than for a deterministic pipeline with a validation loop. |
| Hand-written loop | Used as the fallback; easy to test, but control flow is implicit. |

## 5. Agent roles
| Agent | Uses LLM | Responsibility |
|---|---|---|
| **Planner** | yes | Reads the question and returns a JSON plan (1-3 tool steps). Splits multi-part questions into several focused search queries; may choose `list_documents`. Falls back to a plain semantic search if the plan cannot be parsed. |
| **Retriever** | no | Executes the plan with the tools, de-duplicates chunks, and keeps only chunks under the relevance threshold (cosine distance <= 0.85). |
| **Reasoner** | yes | ReAct-style loop (max 4 steps). Each turn returns JSON: `calculator`, `search_documents` (extra retrieval) or `final`. Must answer only from the documents and cite `[S#]`. |
| **Validator** | yes | Fact-checks the answer against the retrieved context; returns `grounded`, `confidence`, `issues`. If not grounded or confidence < 0.5, the Reasoner is asked for **one revision** with the issues; if still weak, a low-confidence warning is appended. |

Tools: `search_documents(query)`, `list_documents()`, `calculator(expression)` (AST-based, no `eval`).

## 6. Reliability and safety controls
- **Input validation:** question length 3-2000 chars; file type allow-list; size and empty-file checks.
- **Prompt-injection screening:** regex screening of user questions (blocked) and of retrieved text (neutralised);
  retrieved text is wrapped in `<document>` tags and the Reasoner is told it is data, not instructions.
- **Harmful-request filter:** small deny-list for obviously harmful requests.
- **Hallucination reduction:** relevance threshold (no relevant chunk -> the LLM is not even called and the app says it could not find an answer);
  "answer only from context" prompt; mandatory citations; independent Validator agent; revision loop; low-confidence banner.
- **Secret protection:** the Admin tab lists every saved key (masked) with a delete icon (`DELETE /api/admin/keys/{provider}`) that erases that key on demand; once the active key is gone every question is refused up front with "The model is not configured" and nothing is sent to the LLM. API key is stored server-side only, never returned by the API, masked in error messages, and redacted from outputs; `settings.json` is chmod 600.
- **Error handling:** friendly messages for missing config, bad key, rate limit, wrong model name, timeouts, unreadable/password-protected/scanned PDFs, empty files; a global handler prevents stack traces reaching the UI; LLM calls retry twice.
- **Graceful degradation:** planner/validator parse failures fall back to a single search / neutral verdict; missing chromadb or ONNX falls back to a numpy store / hash embeddings.

## 7. Deployment
**Local:** `run.bat` or `./run.sh` (README).

**Docker:**
```bash
docker build -t doc-agent .
docker run -p 8000:8000 -v docagent-data:/app/data doc-agent
# optionally: -e LLM_PROVIDER=openai -e LLM_MODEL=gpt-4o-mini -e LLM_API_KEY=...
```
**Cloud (Render / Railway / Fly.io / Azure Web App for Containers):** deploy the Dockerfile as a web service, expose
port 8000 (the `PORT` env var is honoured), and attach a persistent disk at `/app/data` so uploads, the vector index and
`settings.json` survive restarts. Reviewers then open the public URL and enter their key in **Admin**.

**Streamlit Community Cloud:** `streamlit_app.py` provides the same Ask / Documents / Admin pages on top of the same backend modules
(`app/service.py`, `app/agents.py`, ...). Deploy from a GitHub repo with main file `streamlit_app.py` and Python 3.12; steps are in the README.
Limits of this host: temporary file system (uploads, index and saved settings reset on restart), one shared knowledge base and API key
for all visitors, limited memory (`EMBEDDINGS=hash` is available as a lightweight mode), and an automatic numpy fallback if ChromaDB cannot start.

**Hugging Face Spaces (Docker SDK):** push the repo, set `app_port: 8000` in the Space README header.

## 8. Limitations
- **No authentication:** anyone who can reach the URL can open Admin, change the key or delete documents. Put it behind a login/reverse proxy or a private network before real use; add per-user keys and tenant isolation.
- **Single shared knowledge base and API key** for all users; no per-user document permissions.
- **Scanned PDFs / images** are not supported (no OCR). Tables inside PDFs are extracted as flat text.
- **Spreadsheets are searched semantically**, not queried with SQL. Aggregations over *all* rows rely on the summary chunk, retrieved rows and the calculator; tables over 20,000 rows are truncated and may give incomplete totals.
- **Embedding model is English-centric** (MiniLM) and needs a one-time download; chunks beyond ~256 tokens are truncated by the model.
- **Guardrails are heuristic** (regex + LLM validator); they reduce but do not eliminate hallucination or prompt-injection risk. The validator uses the same LLM as the reasoner, so it can share its blind spots.
- **Latency and cost:** one question costs 3-6 LLM calls.
- The LangGraph integration was verified offline with a stand-in for the library; run it once against the installed package (`pip install -r requirements.txt`) to confirm. If it fails to load, the fallback orchestrator is used automatically.
- Legacy `.xls`, `.docx`, `.pptx` are not supported. No conversation memory - each question is independent.
- Automated tests use a fake LLM; answer quality with a real model depends on that model.

## 9. Challenges and design decisions
- **One key format for every LLM:** providers differ in endpoints, parameters and error types. LiteLLM gives a single `provider/model` call; unsupported parameters are dropped automatically, and errors are mapped to plain-language messages. Embeddings are local and independent of the chosen LLM, so changing provider never requires re-indexing.
- **Structured agent output from arbitrary LLMs:** not all models support JSON mode, so agents are prompted for JSON and parsed leniently (code fences and surrounding text tolerated), with fallbacks on failure.
- **Tabular data in a text retriever:** raw rows embed poorly, so each row is serialised as `column: value` text, grouped, and complemented by a statistics summary chunk; the calculator tool handles arithmetic the LLM would otherwise get wrong.
- **Hallucination control:** answering "I don't know" when nothing is relevant required a distance threshold and an independent validation pass.
- **Adopting an agent framework:** frameworks add dependency weight and version churn (LangChain/CrewAI in particular). To keep the project installable on any machine, LangGraph is used only for orchestration, the agents remain plain, individually testable functions, and a built-in fallback orchestrator takes over if the library cannot be imported. Offline tests verify that both orchestrators produce identical answers and traces.
- **Portable tool use:** native function-calling differs between LLM providers and is missing in some models, so the Reasoner uses a JSON action protocol that works with any model instead of framework-specific tool binding.
- **Python-version portability:** scientific packages (numpy, onnxruntime) often have no prebuilt wheels for the newest Python, so pip tries to compile them and fails on machines without a C compiler. The launchers use `uv` to provision Python 3.12 automatically, and the code falls back to a pure-numpy vector store when chromadb is unavailable.
- **Secret handling in a demo setting:** the key lives in a server-side file and is write-only from the UI (leave blank to keep the stored key).
- *(Add your own experience here: problems you hit while running the project, e.g. model download, rate limits, chunk-size tuning.)*

## 10. Possible extensions
Authentication and roles, OCR for scans, SQL/pandas agent for spreadsheets, hybrid (BM25 + vector) search with re-ranking,
streaming responses, conversation memory, evaluation set with RAGAS-style metrics.
