"""FastAPI application: REST API + static UI."""
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agents, config, describe, guardrails, ingestion, llm, vectorstore

app = FastAPI(title="Enterprise Document Agent (RAG + Agentic AI)")
STATIC = config.BASE_DIR / "static"


class SettingsIn(BaseModel):
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""


class QueryIn(BaseModel):
    question: str


@app.exception_handler(guardrails.GuardrailViolation)
async def _guard(_, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(llm.LLMError)
async def _llm(_, exc):
    code = 400 if isinstance(exc, llm.LLMNotConfigured) else 502
    return JSONResponse(status_code=code, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unhandled(_, exc):
    return JSONResponse(status_code=500, content={"detail": f"Unexpected server error: {type(exc).__name__}: {str(exc)[:300]}"})


# ------------------------------------------------------------------ admin
@app.get("/api/admin/settings")
def get_settings():
    return {"settings": config.public_settings(),
            "providers": {k: {"label": v["label"], "models": v["models"], "hint": v["hint"], "key_url": v.get("key_url", "")}
                          for k, v in config.PROVIDERS.items()}}


@app.post("/api/admin/settings")
def post_settings(body: SettingsIn):
    if body.provider not in config.PROVIDERS:
        raise HTTPException(400, "Unknown provider.")
    if not body.model.strip():
        raise HTTPException(400, "Model name is required.")
    config.save_settings(body.provider, body.model, body.api_key, body.base_url)
    return {"ok": True, "settings": config.public_settings()}


@app.post("/api/admin/test")
def test_connection():
    reply = llm.chat([{"role": "user", "content": "Reply with the single word: OK"}], max_tokens=10)
    return {"ok": True, "reply": reply[:100]}


# -------------------------------------------------------------- documents
def _ingest(path: Path, display_name: str) -> dict:
    chunks = ingestion.parse_file(path, display_name)
    info = vectorstore.add_document(display_name, path, chunks)
    desc = describe.describe(display_name, chunks)      # never raises; AI summary if an LLM is configured
    return vectorstore.update_document(info["doc_id"], description=desc) or info


@app.get("/api/documents")
def documents():
    return {"documents": vectorstore.list_documents()}


@app.post("/api/documents")
def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        name = Path(f.filename or "file").name
        try:
            data = f.file.read()
            guardrails.validate_upload(name, len(data))
            dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{name}"
            dest.write_bytes(data)
            try:
                info = _ingest(dest, name)
            except Exception:
                dest.unlink(missing_ok=True)
                raise
            results.append({"filename": name, "ok": True, "chunks": info["chunks"]})
        except (guardrails.GuardrailViolation, ingestion.IngestionError) as e:
            results.append({"filename": name, "ok": False, "error": str(e)})
        except Exception as e:
            results.append({"filename": name, "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return {"results": results}


@app.post("/api/documents/load-samples")
def load_samples():
    existing = {d["filename"] for d in vectorstore.list_documents()}
    results = []
    for p in sorted(config.SAMPLE_DIR.iterdir()):
        if p.suffix.lower() not in config.ALLOWED_EXT or p.name in existing:
            continue
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{p.name}"
        shutil.copy(p, dest)
        try:
            info = _ingest(dest, p.name)
            results.append({"filename": p.name, "ok": True, "chunks": info["chunks"]})
        except Exception as e:
            dest.unlink(missing_ok=True)
            results.append({"filename": p.name, "ok": False, "error": str(e)})
    return {"results": results}


@app.post("/api/documents/{doc_id}/describe")
def redescribe(doc_id: str):
    """(Re)generate the AI description of a document, e.g. after saving an API key."""
    info = next((d for d in vectorstore.list_documents() if d["doc_id"] == doc_id), None)
    if not info:
        raise HTTPException(404, "Document not found.")
    chunks = ingestion.parse_file(config.UPLOAD_DIR / info["stored_as"], info["filename"])
    ai = describe.ai_description(info["filename"], chunks)       # LLMError -> friendly message via handler
    if not ai:
        raise HTTPException(502, "The model returned an unusable description. Please try again.")
    base = describe.heuristic_description(info["filename"], chunks)
    base.update({k: v for k, v in ai.items() if v})
    base["described_by"] = "ai"
    return {"description": vectorstore.update_document(doc_id, description=base)["description"]}


@app.delete("/api/documents/{doc_id}")
def delete_doc(doc_id: str):
    if not vectorstore.delete_document(doc_id):
        raise HTTPException(404, "Document not found.")
    return {"ok": True}


# ------------------------------------------------------------------ query
@app.post("/api/query")
def query(body: QueryIn):
    return agents.answer_question(body.question)


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_configured": config.public_settings()["configured"],
            "documents": len(vectorstore.list_documents())}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
