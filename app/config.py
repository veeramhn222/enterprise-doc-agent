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
    "gemini":     {"label": "Google Gemini",               "prefix": "gemini",     "models": ["gemini-2.0-flash", "gemini-1.5-pro"], "hint": "", "key_url": "https://aistudio.google.com/apikey"},
    "groq":       {"label": "Groq",                        "prefix": "groq",       "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], "hint": "", "key_url": "https://console.groq.com/keys"},
    "mistral":    {"label": "Mistral",                     "prefix": "mistral",    "models": ["mistral-small-latest", "mistral-large-latest"], "hint": "", "key_url": "https://console.mistral.ai/api-keys"},
    "deepseek":   {"label": "DeepSeek",                    "prefix": "deepseek",   "models": ["deepseek-chat"], "hint": "", "key_url": "https://platform.deepseek.com/api_keys"},
    "openrouter": {"label": "OpenRouter (any model)",      "prefix": "openrouter", "models": ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku", "meta-llama/llama-3.3-70b-instruct"], "hint": "Model format: vendor/model-name", "key_url": "https://openrouter.ai/keys"},
    "together_ai":{"label": "Together AI",                 "prefix": "together_ai","models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"], "hint": "", "key_url": "https://api.together.ai/settings/api-keys"},
    "ollama":     {"label": "Ollama (local, no key)",      "prefix": "ollama",     "models": ["llama3.1", "mistral"], "hint": "Base URL e.g. http://localhost:11434 - API key not needed"},
    "custom":     {"label": "Custom OpenAI-compatible API","prefix": "openai",     "models": [], "hint": "Provide Base URL (e.g. https://my-host/v1) and model name"},
}

_lock = threading.Lock()


def load_settings() -> dict:
    s = {}
    if SETTINGS_FILE.exists():
        try:
            s = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            s = {}
    # environment fallback (handy for Docker / CI)
    s.setdefault("provider", os.getenv("LLM_PROVIDER", ""))
    s.setdefault("model", os.getenv("LLM_MODEL", ""))
    s.setdefault("api_key", os.getenv("LLM_API_KEY", ""))
    s.setdefault("base_url", os.getenv("LLM_BASE_URL", ""))
    return s


def save_settings(provider: str, model: str, api_key: str, base_url: str) -> dict:
    """Persist settings to disk. A blank api_key keeps the previously stored key."""
    with _lock:
        current = load_settings()
        new = {
            "provider": provider.strip(),
            "model": model.strip(),
            "api_key": api_key.strip() or current.get("api_key", ""),
            "base_url": base_url.strip(),
        }
        SETTINGS_FILE.write_text(json.dumps(new, indent=2), encoding="utf-8")
        try:
            os.chmod(SETTINGS_FILE, 0o600)
        except Exception:
            pass
        return new


def public_settings() -> dict:
    """Settings safe to send to the browser (never contains the key)."""
    s = load_settings()
    return {
        "provider": s["provider"],
        "model": s["model"],
        "base_url": s["base_url"],
        "has_api_key": bool(s["api_key"]),
        "configured": bool(s["provider"] and s["model"]),
    }
