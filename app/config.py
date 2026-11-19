"""Central configuration + persistent LLM settings (stored in data/settings.json)."""
import json
import os
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
SETTINGS_FILE = DATA_DIR / "settings.json"
REGISTRY_FILE = DATA_DIR / "documents.json"
SAMPLE_DIR = BASE_DIR / "sample_docs"

APP_VERSION = "2.4-model-dropdown"      # shown in the UI footer / sidebar so you can tell which build is running

ALLOWED_EXT = {".pdf", ".txt", ".md", ".csv", ".xlsx", ".xlsm"}
MAX_UPLOAD_MB = 25
CHUNK_SIZE = 1000          # characters
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_DISTANCE = 0.85        # cosine distance; chunks farther than this are "not relevant"
MAX_QUESTION_CHARS = 2000
MAX_TABLE_ROWS = 20000
ROWS_PER_CHUNK = 12
EMBEDDINGS = os.getenv("EMBEDDINGS", "onnx").lower()   # "onnx" | "hash"
ORCHESTRATOR = os.getenv("ORCHESTRATOR", "langgraph").lower()   # "langgraph" | "plain"

for _p in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# provider id -> UI label, LiteLLM prefix, example models, hint
PROVIDERS = {
    "openai":     {"label": "OpenAI",                      "prefix": "openai",     "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], "hint": "", "key_url": "https://platform.openai.com/api-keys"},
    "anthropic":  {"label": "Anthropic (Claude)",          "prefix": "anthropic",  "models": ["claude-sonnet-5-5", "claude-haiku-5-5"], "hint": "", "key_url": "https://console.anthropic.com/settings/keys"},
    "gemini":     {"label": "Google Gemini",               "prefix": "gemini",     "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro"], "hint": "", "key_url": "https://aistudio.google.com/apikey"},
    "groq":       {"label": "Groq",                        "prefix": "groq",       "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], "hint": "", "key_url": "https://console.groq.com/keys"},
    "mistral":    {"label": "Mistral",                     "prefix": "mistral",    "models": ["mistral-small-latest", "mistral-large-latest"], "hint": "", "key_url": "https://console.mistral.ai/api-keys"},
    "deepseek":   {"label": "DeepSeek",                    "prefix": "deepseek",   "models": ["deepseek-chat"], "hint": "", "key_url": "https://platform.deepseek.com/api_keys"},
    "openrouter": {"label": "OpenRouter (any model)",      "prefix": "openrouter", "models": ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku", "meta-llama/llama-3.3-70b-instruct"], "hint": "Model format: vendor/model-name", "key_url": "https://openrouter.ai/keys"},
    "together_ai":{"label": "Together AI",                 "prefix": "together_ai","models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"], "hint": "", "key_url": "https://api.together.ai/settings/api-keys"},
    "ollama":     {"label": "Ollama (local, no key)",      "prefix": "ollama",     "models": ["llama3.1", "mistral"], "hint": "Base URL e.g. http://localhost:11434 - API key not needed"},
    "custom":     {"label": "Custom OpenAI-compatible API","prefix": "openai",     "models": [], "hint": "Provide Base URL (e.g. https://my-host/v1) and model name"},
}

_lock = threading.Lock()
NO_KEY_PROVIDERS = {"ollama", "custom"}      # providers that can work without an API key


def needs_key(provider: str) -> bool:
    return provider not in NO_KEY_PROVIDERS


def mask_key(key: str) -> str:
    """Safe-to-display form of a key (first and last 4 characters only)."""
    if not key:
        return ""
    return "\u2022\u2022\u2022\u2022" if len(key) <= 8 else f"{key[:4]}\u2026{key[-4:]}"


# The settings file holds one profile per provider, so several keys can be saved side by side:
#   {"active": "openai", "profiles": {"openai": {"model": ..., "api_key": ..., "base_url": ...}, ...}}
def _profile(model="", api_key="", base_url="") -> dict:
    return {"model": str(model or "").strip(), "api_key": str(api_key or "").strip(), "base_url": str(base_url or "").strip()}


def _read_store() -> dict:
    """Current store. Reads data/settings.json (migrating the old single-key format); only when that file does not
    exist yet does it fall back to the LLM_PROVIDER / LLM_MODEL / LLM_API_KEY / LLM_BASE_URL environment variables."""
    raw = None
    if SETTINGS_FILE.exists():
        try:
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            raw = None
    if isinstance(raw, dict):
        if isinstance(raw.get("profiles"), dict):
            profiles = {prov: _profile(v.get("model"), v.get("api_key"), v.get("base_url"))
                        for prov, v in raw["profiles"].items() if prov in PROVIDERS and isinstance(v, dict)}
            active = raw.get("active") if raw.get("active") in profiles else ""
            return {"active": active, "profiles": profiles}
        if raw.get("provider") in PROVIDERS:                                   # legacy single-key file
            prov = raw["provider"]
            return {"active": prov, "profiles": {prov: _profile(raw.get("model"), raw.get("api_key"), raw.get("base_url"))}}
    prov = os.getenv("LLM_PROVIDER", "")
    if prov in PROVIDERS:
        return {"active": prov, "profiles": {prov: _profile(os.getenv("LLM_MODEL"), os.getenv("LLM_API_KEY"), os.getenv("LLM_BASE_URL"))}}
    return {"active": "", "profiles": {}}


def _write_store(store: dict):
    tmp = SETTINGS_FILE.with_name(SETTINGS_FILE.name + ".tmp")
    tmp.write_text(json.dumps(store, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    os.replace(tmp, SETTINGS_FILE)


def load_settings() -> dict:
    """The ACTIVE provider's settings as a flat dict (what the LLM layer uses)."""
    store = _read_store()
    p = store["profiles"].get(store["active"])
    if not p:
        return {"provider": "", "model": "", "api_key": "", "base_url": ""}
    return {"provider": store["active"], "model": p["model"], "api_key": p["api_key"], "base_url": p["base_url"]}


def is_configured(s: dict) -> bool:
    """True when the given (flat) settings are enough to call the LLM."""
    prov = s.get("provider")
    if prov not in PROVIDERS or not s.get("model"):
        return False
    if prov == "custom" and not s.get("base_url"):
        return False
    return bool(s.get("api_key")) or not needs_key(prov)


def has_saved_key(provider: str) -> bool:
    return bool(_read_store()["profiles"].get(provider, {}).get("api_key"))


def save_settings(provider: str, model: str, api_key: str, base_url: str) -> dict:
    """Save (or update) this provider's profile and make it the active one.
    A blank api_key keeps the key already saved for the SAME provider; keys of other providers are untouched."""
    provider = provider.strip()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'.")
    with _lock:
        store = _read_store()
        old = store["profiles"].get(provider, {})
        store["profiles"][provider] = _profile(model, api_key.strip() or old.get("api_key", ""), base_url)
        store["active"] = provider
        _write_store(store)
    return load_settings()


def set_active(provider: str) -> dict:
    """Switch the active provider to an already saved profile. Raises KeyError if there is none."""
    with _lock:
        store = _read_store()
        if provider not in store["profiles"]:
            raise KeyError(provider)
        store["active"] = provider
        _write_store(store)
    return public_settings()


def delete_key(provider: str) -> dict:
    """Delete one provider's saved entry completely (key, model and base URL). If it was the active one, no model
    is active afterwards, so questions answer "The model is not configured". Raises KeyError if the provider has no saved entry. Because the file is written, a key coming from the
    LLM_API_KEY environment variable / Streamlit secret is ignored from now on (until the file is lost on restart)."""
    with _lock:
        store = _read_store()
        if provider not in store["profiles"]:
            raise KeyError(provider)
        del store["profiles"][provider]
        if store["active"] == provider:
            store["active"] = ""
        _write_store(store)
    return public_settings()


def public_settings() -> dict:
    """Settings safe to send to the browser. Keys are only ever included in masked form."""
    store, s = _read_store(), load_settings()
    return {
        "provider": s["provider"], "model": s["model"], "base_url": s["base_url"],
        "has_api_key": bool(s["api_key"]),
        "configured": is_configured(s),
        "env_key_present": bool(os.getenv("LLM_API_KEY")),
        "profiles": [{"provider": prov, "label": PROVIDERS[prov]["label"], "model": p["model"], "base_url": p["base_url"],
                      "has_key": bool(p["api_key"]), "key_hint": mask_key(p["api_key"]),
                      "needs_key": needs_key(prov), "active": prov == store["active"]}
                     for prov, p in store["profiles"].items()],
    }
