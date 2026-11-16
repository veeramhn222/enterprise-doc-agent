"""Reliability & safety controls: input validation, injection screening, output hygiene."""
import re

from . import config

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |any |the )?(previous|prior|above|system) (instructions|prompts|rules)",
    r"reveal (your |the )?(system|hidden|developer) (prompt|instructions|message)",
    r"(print|show|repeat|output) (your |the )?(system prompt|initial instructions)",
    r"you are now (dan|in developer mode|jailbroken)",
    r"(show|give|tell|print|reveal) me (the |your |any )?(api[ -]?key|secret|password|credentials?)",
]
_HARMFUL = [
    r"\b(build|make|create|synthesi[sz]e)\b.{0,30}\b(bomb|explosive|nerve agent|bioweapon)\b",
    r"\b(write|create|generate)\b.{0,30}\b(malware|ransomware|keylogger)\b",
]


class GuardrailViolation(Exception):
    pass


def validate_question(q: str) -> str:
    q = (q or "").strip()
    if len(q) < 3:
        raise GuardrailViolation("Please enter a question (at least 3 characters).")
    if len(q) > config.MAX_QUESTION_CHARS:
        raise GuardrailViolation(f"Question is too long (max {config.MAX_QUESTION_CHARS} characters).")
    low = q.lower()
    for pat in _INJECTION:
        if re.search(pat, low):
            raise GuardrailViolation("This looks like an attempt to override the assistant's instructions or extract secrets, so it was blocked.")
    for pat in _HARMFUL:
        if re.search(pat, low):
            raise GuardrailViolation("This request involves potentially harmful content and cannot be processed.")
    return q


def validate_upload(filename: str, size: int):
    from pathlib import Path
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise GuardrailViolation(f"'{filename}': unsupported type. Allowed: {', '.join(sorted(config.ALLOWED_EXT))}")
    if size == 0:
        raise GuardrailViolation(f"'{filename}' is empty.")
    if size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise GuardrailViolation(f"'{filename}' exceeds {config.MAX_UPLOAD_MB} MB.")


def sanitize_for_context(text: str) -> str:
    """Neutralise obvious prompt-injection phrases inside retrieved document text."""
    for pat in _INJECTION:
        text = re.sub(pat, "[removed-instruction-like-text]", text, flags=re.I)
    return text


def redact_output(answer: str, secrets: list[str]) -> str:
    for s in secrets:
        if s and len(s) > 6:
            answer = answer.replace(s, "***")
    return answer
