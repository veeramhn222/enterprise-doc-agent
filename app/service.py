"""Framework-independent application services (used by the Streamlit UI; mirrors the FastAPI routes)."""
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, describe, guardrails, ingestion, vectorstore


def _index(path: Path, display_name: str):
    """Parse + embed + store. Returns (registry info, chunks). The description is added separately."""
    chunks = ingestion.parse_file(path, display_name)
    return vectorstore.add_document(display_name, path, chunks), chunks


def _describe_many(items: list[tuple[str, str, list]]):
    """items: (doc_id, display_name, chunks). LLM descriptions run in parallel; failures keep the offline text."""
    def work(item):
        doc_id, name, chunks = item
        vectorstore.update_document(doc_id, description=describe.describe(name, chunks))
    if len(items) == 1:
        work(items[0])
    elif items:
        with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
            list(pool.map(work, items))


def ingest_path(path: Path, display_name: str) -> dict:
    info, chunks = _index(path, display_name)
    _describe_many([(info["doc_id"], display_name, chunks)])
    return next((d for d in vectorstore.list_documents() if d["doc_id"] == info["doc_id"]), info)


def ingest_uploads(files: list[tuple[str, bytes]]) -> list[dict]:
    """Validate, store and index uploaded files; describe them all in parallel. Never raises."""
    results, pending = [], []
    for name, data in files:
        name = Path(name or "file").name
        try:
            guardrails.validate_upload(name, len(data))
            dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{name}"
            dest.write_bytes(data)
            try:
                info, chunks = _index(dest, name)
            except Exception:
                dest.unlink(missing_ok=True)
                raise
            pending.append((info["doc_id"], name, chunks))
            results.append({"filename": name, "ok": True, "chunks": info["chunks"]})
        except (guardrails.GuardrailViolation, ingestion.IngestionError) as e:
            results.append({"filename": name, "ok": False, "error": str(e)})
        except Exception as e:
            results.append({"filename": name, "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    _describe_many(pending)
    return results


def ingest_upload(name: str, data: bytes) -> dict:
    return ingest_uploads([(name, data)])[0]


def load_samples() -> list[dict]:
    existing = {d["filename"] for d in vectorstore.list_documents()}
    results, pending = [], []
    for p in sorted(config.SAMPLE_DIR.iterdir()):
        if p.suffix.lower() not in config.ALLOWED_EXT or p.name in existing:
            continue
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{p.name}"
        shutil.copy(p, dest)
        try:
            info, chunks = _index(dest, p.name)
            pending.append((info["doc_id"], p.name, chunks))
            results.append({"filename": p.name, "ok": True, "chunks": info["chunks"]})
        except Exception as e:
            dest.unlink(missing_ok=True)
            results.append({"filename": p.name, "ok": False, "error": str(e)})
    _describe_many(pending)
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
