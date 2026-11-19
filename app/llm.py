"""Provider-agnostic LLM access via LiteLLM (100+ providers behind one call)."""
from . import config


class LLMError(Exception):
    """Raised with a user-friendly message."""


class LLMNotConfigured(LLMError):
    pass


def model_string(s: dict) -> str:
    prov = config.PROVIDERS.get(s["provider"])
    if not prov:
        raise LLMError(f"Unknown provider '{s['provider']}'. Choose one in the Admin tab.")
    return f"{prov['prefix']}/{s['model']}"


def not_configured_message(s: dict) -> str:
    prov = s.get("provider")
    if prov not in config.PROVIDERS or not s.get("model"):
        reason = "no provider and model are selected"
    elif prov == "custom" and not s.get("base_url"):
        reason = "the custom provider needs a Base URL"
    else:
        reason = f"no API key is saved for {config.PROVIDERS[prov]['label']}"
    return f"The model is not configured: {reason}. Open the Admin page, choose a provider and model, and add an API key."


def ensure_configured(settings: dict | None = None) -> dict:
    """Return the settings in use, or raise LLMNotConfigured with a message that tells the user what to do."""
    s = settings if settings is not None else config.load_settings()
    if not config.is_configured(s):
        raise LLMNotConfigured(not_configured_message(s))
    return s


def chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1200,
         settings: dict | None = None, timeout: int = 90, retries: int = 2) -> str:
    s = ensure_configured(settings)

    import litellm  # imported lazily so the app starts fast
    litellm.drop_params = True       # silently drop params a given model does not support
    litellm.suppress_debug_info = True

    kwargs = dict(model=model_string(s), messages=messages, temperature=temperature,
                  max_tokens=max_tokens, timeout=timeout, num_retries=retries)
    if s.get("api_key"):
        kwargs["api_key"] = s["api_key"]
    if s.get("base_url"):
        kwargs["api_base"] = s["base_url"]

    try:
        resp = litellm.completion(**kwargs)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # map provider errors to friendly messages
        name = type(e).__name__
        key = s.get("api_key") or "\0"
        msg = str(e).replace(key, "***")[:400]
        if "Authentication" in name:
            raise LLMError("The LLM provider rejected the API key (authentication failed). Check the key in the Admin tab.")
        if "RateLimit" in name:
            raise LLMError("Rate limit or quota reached on the LLM provider. Wait a moment or check your plan/credits.")
        if "NotFound" in name:
            raise LLMError(f"Model '{s['model']}' was not found for provider '{s['provider']}'. Check the model name.")
        if "Timeout" in name or "Connection" in name:
            raise LLMError("Could not reach the LLM provider (timeout/connection). Check network or Base URL.")
        raise LLMError(f"LLM call failed ({name}): {msg}")
