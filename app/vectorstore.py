"""Vector knowledge store (ChromaDB, persistent) + document registry.

Works on any Python version: if chromadb or its ONNX embedder cannot be imported (e.g. a very new
Python without prebuilt wheels), the app falls back to a built-in numpy store / hash embedder."""
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config

_lock = threading.Lock()
_collection = None
_embedder = None


# ------------------------------------------------------------- embeddings
class _HashEmbedder:
    """Tiny offline embedder (hashed bag-of-words). Used for tests / no-download fallback."""
    DIM = 384

    def __call__(self, texts):
        import re
        import zlib
        import numpy as np
        out = []
        for t in texts:
            v = np.zeros(self.DIM, dtype="float32")
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[zlib.crc32(w.encode()) % self.DIM] += 1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


_embedder_lock = threading.Lock()


def _get_embedder():
    global _embedder
    with _embedder_lock:
        return _init_embedder()


def _init_embedder():
    global _embedder
    if _embedder is None:
        if config.EMBEDDINGS == "hash":
            _embedder = _HashEmbedder()
        else:
            try:
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                ef = DefaultEmbeddingFunction()      # all-MiniLM-L6-v2 via ONNX, downloaded once (~80 MB)

                def _embed(texts, _ef=ef):
                    return [list(map(float, v)) for v in _ef(list(texts))]
                _embedder = _embed
            except Exception as e:   # chromadb/onnx unavailable on this Python -> keyword-style fallback
                print(f"[WARN] ONNX embeddings unavailable ({type(e).__name__}); using built-in hash embeddings "
                      "(lower search quality). Use Python 3.10-3.12 for best results.")
                _embedder = _HashEmbedder()
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    return _get_embedder()(texts)


class NumpyCollection:
    """Pure-numpy persistent vector store with the subset of the Chroma API we use."""

    def __init__(self, folder: Path):
        import numpy as np
        self.np = np
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_file, self.vec_file = self.dir / "meta.json", self.dir / "vectors.npy"
        self.ids, self.docs, self.metas = [], [], []
        self.vecs = np.zeros((0, 0), dtype="float32")
        if self.meta_file.exists() and self.vec_file.exists():
            d = json.loads(self.meta_file.read_text(encoding="utf-8"))
            self.ids, self.docs, self.metas = d["ids"], d["docs"], d["metas"]
            self.vecs = np.load(self.vec_file)

    def _save(self):
        self.np.save(self.vec_file, self.vecs)
        self.meta_file.write_text(json.dumps({"ids": self.ids, "docs": self.docs, "metas": self.metas}), encoding="utf-8")

    def count(self):
        return len(self.ids)

    def add(self, ids, documents, embeddings, metadatas):
        np = self.np
        v = np.asarray(embeddings, dtype="float32")
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        self.vecs = v if self.vecs.size == 0 else np.vstack([self.vecs, v])
        self.ids += list(ids)
        self.docs += list(documents)
        self.metas += list(metadatas)
        self._save()

    def delete(self, where):
        keep = [i for i, m in enumerate(self.metas) if m["doc_id"] != where["doc_id"]]
        self.ids = [self.ids[i] for i in keep]
        self.docs = [self.docs[i] for i in keep]
        self.metas = [self.metas[i] for i in keep]
        self.vecs = self.vecs[keep] if keep else self.np.zeros((0, 0), dtype="float32")
        self._save()

    def query(self, query_embeddings, n_results, include=None):
        np = self.np
        q = np.asarray(query_embeddings[0], dtype="float32")
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        dist = 1.0 - self.vecs @ q
        order = np.argsort(dist)[:n_results]
        return {"ids": [[self.ids[i] for i in order]], "documents": [[self.docs[i] for i in order]],
                "metadatas": [[self.metas[i] for i in order]], "distances": [[float(dist[i]) for i in order]]}


def _get_collection():
    global _collection
    if _collection is None:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            _collection = client.get_or_create_collection(
                name="enterprise_docs", embedding_function=None, metadata={"hnsw:space": "cosine"})
        except Exception as e:   # chromadb missing or unusable here (e.g. old sqlite) -> built-in numpy store
            print(f"[WARN] ChromaDB unavailable ({type(e).__name__}: {str(e)[:120]}); using built-in numpy vector store.")
            _collection = NumpyCollection(config.DATA_DIR / "numpy_store")
    return _collection


# --------------------------------------------------------------- registry
def _read_registry() -> dict:
    if config.REGISTRY_FILE.exists():
        try:
            return json.loads(config.REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_registry(reg: dict):
    config.REGISTRY_FILE.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def list_documents() -> list[dict]:
    reg = _read_registry()
    return sorted(reg.values(), key=lambda d: d["uploaded_at"], reverse=True)


# ------------------------------------------------------------- operations
def add_document(filename: str, stored_path: Path, chunks: list[tuple[str, str]]) -> dict:
    doc_id = uuid.uuid4().hex[:12]
    texts = [c[0] for c in chunks]
    with _lock:
        col = _get_collection()
        B = 64
        for i in range(0, len(texts), B):
            batch = texts[i:i + B]
            col.add(
                ids=[f"{doc_id}-{i + j}" for j in range(len(batch))],
                documents=batch,
                embeddings=embed(batch),
                metadatas=[{"doc_id": doc_id, "filename": filename, "location": chunks[i + j][1]}
                           for j in range(len(batch))],
            )
        info = {"doc_id": doc_id, "filename": filename, "chunks": len(texts),
                "stored_as": stored_path.name,
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        reg = _read_registry()
        reg[doc_id] = info
        _write_registry(reg)
    return info


def update_document(doc_id: str, **fields) -> dict | None:
    """Merge extra fields (e.g. description) into a document's registry entry."""
    with _lock:
        reg = _read_registry()
        if doc_id not in reg:
            return None
        reg[doc_id].update(fields)
        _write_registry(reg)
        return reg[doc_id]


def delete_document(doc_id: str) -> bool:
    with _lock:
        reg = _read_registry()
        info = reg.pop(doc_id, None)
        if not info:
            return False
        _get_collection().delete(where={"doc_id": doc_id})
        _write_registry(reg)
        p = config.UPLOAD_DIR / info["stored_as"]
        if p.exists():
            p.unlink()
    return True


def search(query: str, k: int = config.TOP_K) -> list[dict]:
    """Semantic similarity search. Returns chunks sorted by distance (lower = closer)."""
    col = _get_collection()
    n = col.count()
    if n == 0:
        return []
    res = col.query(query_embeddings=embed([query]), n_results=min(k, n),
                    include=["documents", "metadatas", "distances"])
    out = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"id": cid, "text": doc, "filename": meta["filename"],
                    "location": meta["location"], "distance": float(dist)})
    return out
