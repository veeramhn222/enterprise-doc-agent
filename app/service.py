"""Framework-independent application services (used by the Streamlit UI; mirrors the FastAPI routes)."""
import shutil
import uuid
from pathlib import Path

from . import config, describe, guardrails, ingestion, vectorstore


def ingest_path(path: Path, display_name: str) -> dict:
    chunks = ingestion.parse_file(path, display_name)
    info = vectorstore.add_document(display_name, path, chunks)
    desc = describe.describe(display_name, chunks)          # never raises; AI summary if an LLM is configured
    return vectorstore.update_document(info["doc_id"], description=desc) or info


def ingest_upload(name: str, data: bytes) -> dict:
    """Validate, store and index one uploaded file. Always returns a result dict (never raises)."""
    name = Path(name or "file").name
    try:
        guardrails.validate_upload(name, len(data))
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{name}"
        dest.write_bytes(data)
        try:
            info = ingest_path(dest, name)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        return {"filename": name, "ok": True, "chunks": info["chunks"]}
    except (guardrails.GuardrailViolation, ingestion.IngestionError) as e:
        return {"filename": name, "ok": False, "error": str(e)}
    except Exception as e:
        return {"filename": name, "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def load_samples() -> list[dict]:
    existing = {d["filename"] for d in vectorstore.list_documents()}
    results = []
    for p in sorted(config.SAMPLE_DIR.iterdir()):
        if p.suffix.lower() not in config.ALLOWED_EXT or p.name in existing:
            continue
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{p.name}"
        shutil.copy(p, dest)
        try:
            info = ingest_path(dest, p.name)
            results.append({"filename": p.name, "ok": True, "chunks": info["chunks"]})
        except Exception as e:
            dest.unlink(missing_ok=True)
            results.append({"filename": p.name, "ok": False, "error": str(e)})
    return results


def regenerate_description(doc_id: str) -> dict:
    """AI description for one document. Raises KeyError (unknown id), LLMError (provider problem), ValueError (unusable output)."""
    info = next((d for d in vectorstore.list_documents() if d["doc_id"] == doc_id), None)
    if not info:
        raise KeyError(doc_id)
    chunks = ingestion.parse_file(config.UPLOAD_DIR / info["stored_as"], info["filename"])
    ai = describe.ai_description(info["filename"], chunks)
    if not ai:
        raise ValueError("The model returned an unusable description. Please try again.")
    base = describe.heuristic_description(info["filename"], chunks)
    base.update({k: v for k, v in ai.items() if v})
    base["described_by"] = "ai"
    return vectorstore.update_document(doc_id, description=base)["description"]
