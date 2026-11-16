"""Streamlit front end for the Enterprise Document Agent (RAG + agents).

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py (see README)."""
import os
import sys
import tempfile
from pathlib import Path

try:    # chromadb needs sqlite >= 3.35; use the bundled one if installed (some hosts ship an old sqlite)
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

_data = Path(__file__).parent / "data"
try:    # on read-only hosts fall back to a temp folder
    _data.mkdir(exist_ok=True)
    (_data / ".probe").write_text("x")
    (_data / ".probe").unlink()
except OSError:
    os.environ.setdefault("DATA_DIR", str(Path(tempfile.gettempdir()) / "docagent"))

import streamlit as st

from app import agents, config, guardrails, llm, service, vectorstore

ASK, DOCS, ADMIN = "💬 Ask", "📁 Documents", "⚙️ Admin"
ICONS = {".pdf": "📕", ".txt": "📄", ".md": "📄", ".csv": "📊", ".xlsx": "📗", ".xlsm": "📗"}


def md_safe(text) -> str:
    """Streamlit renders $...$ as LaTeX; escape dollar signs so amounts like $750 display normally."""
    return str(text).replace("$", "\\$")


def _load_secrets():
    """Optional defaults from Streamlit secrets (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL)."""
    try:
        for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"):
            if k in st.secrets:
                os.environ.setdefault(k, str(st.secrets[k]))
    except Exception:
        pass


def ask_cb(question: str):
    """Button callback: jump to the Ask page and run this question."""
    st.session_state.pending_q = question
    st.session_state.page = ASK


def flash(results: list[dict]):
    st.session_state.flash = results


def show_flash():
    for r in st.session_state.pop("flash", []):
        if r["ok"]:
            st.success(f"✔ {r['filename']} – {r['chunks']} chunks indexed")
        else:
            st.error(f"✘ {r['filename']} – {r['error']}")


# ----------------------------------------------------------------- sidebar
def sidebar():
    with st.sidebar:
        st.title("📚 Document Agent")
        st.caption("Ask questions · Get cited answers")
        st.radio("Navigate", [ASK, DOCS, ADMIN], key="page", label_visibility="collapsed")
        s = config.public_settings()
        if s["configured"]:
            st.success(f"{s['provider']} · {s['model']}", icon="🟢")
        else:
            st.warning("LLM not configured. Open the Admin page.", icon="🟡")
        st.caption(f"{len(vectorstore.list_documents())} document(s) indexed")


# --------------------------------------------------------------------- ask
def render_result(r: dict):
    v = r.get("validation")
    if v:
        ok = v["grounded"] and v["confidence"] >= 0.5
        st.caption(("✔ grounded" if ok else "⚠ unverified") + f" · {v['confidence']:.0%} confidence")
    with st.expander(f"Agent trace ({len(r['trace'])} steps)"):
        for t in r["trace"]:
            secs = f" _({t['seconds']}s)_" if t.get("seconds") is not None else ""
            st.markdown(f"**{t['agent']}** — {md_safe(t['detail'])}{secs}")
    if r["sources"]:
        with st.expander(f"Sources ({len(r['sources'])})"):
            for s in r["sources"]:
                st.markdown(f"**[{s['id']}] {s['filename']}** · {s['location']} · distance {s['distance']}"
                            + (" · ✅ cited" if s["cited"] else ""))
                st.text(s["text"])


def page_ask():
    st.header("Ask your documents")
    typed = st.chat_input("Ask about your documents…")
    q = st.session_state.pop("pending_q", None) or typed
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Agents are working: plan, retrieve, reason, validate…"):
            try:
                r = agents.answer_question(q)
                st.session_state.messages.append({"role": "assistant", "content": r["answer"], "result": r})
            except (guardrails.GuardrailViolation, llm.LLMError) as e:
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {e}", "error": True})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "error": True,
                                                  "content": f"⚠️ Unexpected error: {type(e).__name__}: {str(e)[:300]}"})
    if not st.session_state.messages:
        docs = vectorstore.list_documents()
        if docs:
            st.write(f"{len(docs)} document(s) indexed. Pick a suggestion or type your own question below.")
            n = 0
            for d in docs:
                for question in ((d.get("description") or {}).get("questions") or [])[:2]:
                    if n < 6:
                        st.button(question, key=f"sugg_{n}", on_click=ask_cb, args=(question,))
                        n += 1
        else:
            st.info("No documents yet. Open the **Documents** page to upload files or load the samples.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(md_safe(m["content"]))
            if m.get("result"):
                render_result(m["result"])


# --------------------------------------------------------------- documents
def page_docs():
    st.header("Documents")
    show_flash()
    n = st.session_state.get("uploader_n", 0)
    files = st.file_uploader("Upload PDF, TXT, MD, CSV or XLSX (max 25 MB each)", accept_multiple_files=True,
                             type=["pdf", "txt", "md", "csv", "xlsx", "xlsm"], key=f"uploader_{n}")
    c1, c2 = st.columns(2)
    if c1.button("Index uploaded files", type="primary", disabled=not files):
        with st.spinner("Indexing and describing… the first run downloads the embedding model"):
            flash([service.ingest_upload(f.name, f.getvalue()) for f in files])
        st.session_state.uploader_n = n + 1
        st.rerun()
        return
    if c2.button("✨ Load sample documents"):
        with st.spinner("Loading samples… the first run downloads the embedding model"):
            flash(service.load_samples() or [{"filename": "samples", "ok": False, "error": "already loaded"}])
        st.rerun()
        return

    docs = vectorstore.list_documents()
    st.subheader(f"Your documents ({len(docs)})")
    if not docs:
        st.info("No documents yet. Upload files above or load the samples to get started.")
    for d in docs:
        ds = d.get("description") or {"summary": "No description yet. Click “Regenerate description”.",
                                      "topics": [], "questions": [], "described_by": "none", "kind": ""}
        with st.container(border=True):
            icon = ICONS.get(Path(d["filename"]).suffix.lower(), "📄")
            st.markdown(f"**{icon} {d['filename']}**")
            badge = {"ai": "✨ AI summary", "auto": "Auto summary"}.get(ds.get("described_by"), "")
            st.caption(f"{ds.get('kind', '')} · {d['chunks']} chunks · added {d['uploaded_at'][:10]}" + (f" · {badge}" if badge else ""))
            st.markdown(md_safe(ds["summary"]))
            if ds.get("topics"):
                st.caption("Topics: " + " · ".join(ds["topics"]))
            if ds.get("questions"):
                st.markdown("**You could ask:**")
                for i, question in enumerate(ds["questions"]):
                    st.button(md_safe(question), key=f"q_{d['doc_id']}_{i}", on_click=ask_cb, args=(question,))
            b1, b2 = st.columns(2)
            if b1.button("✨ Regenerate description", key=f"re_{d['doc_id']}"):
                try:
                    with st.spinner("Asking the LLM…"):
                        service.regenerate_description(d["doc_id"])
                    st.toast("Description updated")
                except Exception as e:
                    st.error(str(e))
                else:
                    st.rerun()
                    return
            if b2.button("🗑 Delete", key=f"del_{d['doc_id']}"):
                vectorstore.delete_document(d["doc_id"])
                st.rerun()
                return


# ------------------------------------------------------------------- admin
def page_admin():
    st.header("Admin – LLM settings")
    st.info("Pick any provider, paste an API key and press **Submit**. Settings are saved to `data/settings.json` on the server "
            "and apply to everyone using this deployment. On Streamlit Community Cloud this storage is temporary and "
            "is cleared when the app restarts.")
    if st.session_state.get("admin_msg"):
        kind, text = st.session_state.pop("admin_msg")
        (st.success if kind == "ok" else st.error)(text)
    cur = config.load_settings()
    keys = list(config.PROVIDERS)
    prov = st.selectbox("Provider", keys, index=keys.index(cur["provider"]) if cur["provider"] in keys else 0,
                        format_func=lambda k: config.PROVIDERS[k]["label"], key="adm_provider")
    p = config.PROVIDERS[prov]
    default_model = cur["model"] if (prov == cur["provider"] and cur["model"]) else (p["models"][0] if p["models"] else "")
    model = st.text_input("Model name", value=default_model, key=f"adm_model_{prov}",
                          help="Examples: " + (", ".join(p["models"]) or "any model name your endpoint serves"))
    kn = st.session_state.get("adm_key_n", 0)
    api_key = st.text_input("API key", type="password", key=f"adm_key_{kn}", placeholder="Paste API key")
    st.caption("✔ A key is saved. Leave blank to keep it." if cur["api_key"] else "No key saved yet.")
    if p.get("key_url"):
        st.caption(f"[Get an API key]({p['key_url']})")
    base_url = st.text_input("Base URL (optional – Ollama / custom endpoints)", value=cur["base_url"], key="adm_base")
    if p["hint"]:
        st.caption(p["hint"])
    c1, c2 = st.columns(2)
    if c1.button("Submit", type="primary"):
        if not model.strip():
            st.error("Model name is required.")
        else:
            config.save_settings(prov, model, api_key, base_url)
            st.session_state.adm_key_n = kn + 1
            st.session_state.admin_msg = ("ok", "Saved to data/settings.json. Click “Test connection” to verify.")
            st.rerun()
            return
    if c2.button("Test connection"):
        with st.spinner("Testing the connection…"):
            try:
                reply = llm.chat([{"role": "user", "content": "Reply with the single word: OK"}], max_tokens=10)
                st.success(f"✔ Connected. Model replied: {reply[:100]}")
            except llm.LLMError as e:
                st.error(f"✘ {e}")


def main():
    st.set_page_config(page_title="Document Agent", page_icon="📚", layout="wide")
    _load_secrets()
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("page", ASK)
    sidebar()
    {ASK: page_ask, DOCS: page_docs, ADMIN: page_admin}.get(st.session_state.page, page_ask)()


if __name__ == "__main__":
    main()
