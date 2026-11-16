# Enterprise Document Agent - RAG + Agentic AI

Upload PDF / TXT / CSV / Excel files, ask questions in natural language, and get **grounded, cited answers**
produced by a team of AI agents (Planner -> Retriever -> Reasoner -> Validator) orchestrated with **LangGraph**, on top of a vector store.

The LLM is **provider-agnostic** (OpenAI, Anthropic, Gemini, Groq, Mistral, DeepSeek, OpenRouter, Together, Ollama,
or any OpenAI-compatible endpoint) through [LiteLLM](https://github.com/BerriAI/litellm). Reviewers open the
**Admin** tab, pick a provider, paste *their own* API key and press **Submit**.

---
## 1. Quick start (any Python version)

**Requirements:** any Python 3.9+ already installed, and internet on first run (packages + a one-time ~80 MB embedding model).
The launcher installs a small helper called `uv`, which **downloads Python 3.12 automatically** and builds the
environment, so it does not matter which Python version your machine has.

```bash
unzip rag-agent-app.zip && cd rag-agent-app

run.bat          # Windows (Command Prompt or double-click)
./run.sh         # macOS / Linux / Git Bash
```
Wait for `Uvicorn running on http://0.0.0.0:8000`, then open **http://localhost:8000**.
The first run takes a few minutes; progress is printed at every step.

Manual alternative (needs Python 3.10-3.13):
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

### Troubleshooting
| Symptom | Fix |
|---|---|
| pip tries to compile numpy / "Unknown compiler" | Your Python is too new for prebuilt packages. Use `run.bat` / `run.sh` (they use Python 3.12), or delete `.venv` and retry |
| `python` opens the Microsoft Store (Windows) | Install Python from python.org, or turn off the "App execution aliases" for python.exe |
| Nothing prints for minutes | Normal on first run (large downloads). Let it finish |
| Port 8000 busy | `python -m uvicorn app.main:app --port 8010` |
| No internet / want a quick smoke test | set `EMBEDDINGS=hash` before starting (lower search quality) |

If `chromadb` cannot be installed on a very new Python, the app automatically falls back to a built-in numpy vector store.

## 1b. Streamlit version (and deploying on Streamlit Community Cloud)
`streamlit_app.py` is a Streamlit front end with the same three pages (Ask, Documents, Admin), the same agents and the same
document descriptions. It imports the backend directly, so no FastAPI server is needed.

**Run locally**
```bash
.venv\Scripts\python -m streamlit run streamlit_app.py      # Windows (after run.bat created .venv)
.venv/bin/python -m streamlit run streamlit_app.py           # macOS / Linux
```
**Deploy on Streamlit Community Cloud (free)**
1. Put the project in a GitHub repository, with `streamlit_app.py` and `requirements.txt` in the repo root
   (`data/` is git-ignored, so no keys or uploads are committed).
2. Go to https://share.streamlit.io, sign in with GitHub and choose **Create app / New app**.
3. Select the repository and branch, and set **Main file path** to `streamlit_app.py`.
4. Open **Advanced settings** and choose **Python 3.12**. Optionally add defaults under **Secrets**:
   ```toml
   LLM_PROVIDER = "anthropic"
   LLM_MODEL = "claude-sonnet-5-5"
   LLM_API_KEY = "sk-ant-..."      # leave this out if reviewers should enter their own key
   # EMBEDDINGS = "hash"           # lightweight mode if the free tier runs out of memory (lower search quality)
   ```
5. Click **Deploy**. The first build takes several minutes. Open the app, go to **Admin**, pick the LLM, paste the key and press **Submit**.
6. Share the app URL. In the app's **Sharing** settings you can restrict viewers by email.

**Things to know about Streamlit Community Cloud**
- The file system is temporary: uploaded documents, the vector index and `settings.json` are lost when the app restarts or redeploys. Re-upload (or click *Load sample documents*) after a restart.
- All visitors share one knowledge base and one saved API key, so restrict access if you do not want that.
- Memory and CPU are limited; the first upload downloads an ~80 MB embedding model. If the app runs out of memory, set `EMBEDDINGS = "hash"` in Secrets.
- If ChromaDB cannot start on the host (for example an old `sqlite3`), the app falls back to its built-in numpy vector store automatically and prints a warning in the logs.

## 2. Using the app (reviewer walkthrough)
The UI has three tabs: **Ask** (chat + a "Knowledge base" side panel), **Documents** and **Admin**.

1. **Admin tab** -> choose *Provider*, type/select *Model*, paste *API key* -> **Submit**
   (stored in `data/settings.json`; the key is never returned to the browser) -> **Test connection**.
   - Examples: OpenAI `gpt-4o-mini` · Anthropic `claude-sonnet-5-5` · Gemini `gemini-2.0-flash` · Groq `llama-3.3-70b-versatile`
   - Ollama / custom endpoints: also fill *Base URL* (e.g. `http://localhost:11434`). Ollama needs no key.
   - Model names are free text, so any model your key can access works.
2. **Documents tab** -> drag & drop files (or **Load sample documents**). Every file gets a **high-level description**:
   a summary, topic tags and **"You could ask"** suggestions. Click a suggestion to ask it straight away.
   Descriptions are built offline from the file content ("Auto summary"). When an LLM key is saved, new uploads get an
   "AI summary" instead; for files uploaded earlier, click **Regenerate description**.
3. **Ask tab** -> try the questions in `docs/TEST_QUESTIONS.md`. Expand **Agent trace** and **Sources** under each answer.

## 3. Sample documents (`sample_docs/`)
| File | Format | Content |
|---|---|---|
| `company_policy.txt` | TXT | Remote work, leave, security, expense approvals |
| `employee_handbook.pdf` | PDF (2 pages) | Hours, performance reviews, L&D budget, conduct |
| `products.csv` | CSV | 15 products with price, stock, supplier |
| `sales_q1.xlsx` | Excel, 2 sheets | Monthly sales by region + regional targets |

Questions with expected answers: `docs/TEST_QUESTIONS.md`.

## 4. Automated tests
```bash
pytest -q tests
```
They run **offline** (fake LLM, hash embeddings): ingestion of all 4 formats, retrieval, calculator safety, guardrails,
the full agent loop, validator-triggered revision, settings persistence and the numpy fallback store.

## 5. Configuration
| Setting | Where | Default |
|---|---|---|
| LLM provider / model / key / base URL | Admin tab -> `data/settings.json` (or env `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`) | none |
| Chunk size / overlap, top-k, relevance threshold | `app/config.py` | 1000 / 150 / 5 / 0.85 |
| Embeddings | env `EMBEDDINGS=onnx` (MiniLM) or `hash` (tiny offline fallback) | onnx |
| Data directory | env `DATA_DIR` | `./data` |
| Agent orchestrator | env `ORCHESTRATOR=langgraph` or `plain` (automatic fallback if LangGraph cannot load) | langgraph |

## 6. Project layout
```
app/main.py          FastAPI routes (admin, documents, query) + static UI
streamlit_app.py     Streamlit UI (same features) for Streamlit Community Cloud
app/service.py       framework-independent upload / sample / describe services used by the Streamlit UI
app/config.py        settings file, providers list, tunables
app/llm.py           LiteLLM wrapper with friendly error mapping
app/ingestion.py     PDF/TXT/CSV/Excel parsing + chunking
app/describe.py      per-document summary, topics and suggested questions (offline + AI)
app/vectorstore.py   ChromaDB (or numpy fallback) store, embeddings, document registry
app/tools.py         search_documents, list_documents, calculator
app/agents.py        Planner, Retriever, Reasoner, Validator + plain orchestrator (fallback)
app/graph.py         LangGraph state machine wiring the agents (nodes, conditional edges, revision loop)
app/guardrails.py    input validation, prompt-injection screening, redaction
static/              index.html, app.js, util.js, style.css (Ask / Documents / Admin)
sample_docs/         test documents
tests/               offline test-suite
docs/                DOCUMENTATION.md, TEST_QUESTIONS.md
```

See **docs/DOCUMENTATION.md** for architecture, agent roles, deployment and limitations.
