"""Redaction helpers for Telegram bot logs and user-facing diagnostics."""

from __future__ import annotations

import re
from typing import Any

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|api[_-]?secret|secret|password|token|credential|cst|x-security-token)",
    re.IGNORECASE,
)
_PEM_RE = re.compile(
    r"-----BEGIN [^-]+-----.*?-----END [^-]+-----",
    re.DOTALL,
)
_ASSIGNMENT_RE = re.compile(
    r"(?P<key>(?:api[_-]?key|api[_-]?secret|secret|password|token|credential|cst|x-security-token))"
    r"(?P<sep>\s*[=:]\s*)"
    r"(?P<value>[^\s,;]+)",
    re.IGNORECASE,
)
_BOT_TOKEN_RE = re.compile(r"\b\d{5,}:[A-Za-z0-9_-]{20,}\b")
_LONG_SECRET_RE = re.compile(r"\b[A-Za-z0-9_./+=-]{32,}\b")


def redact(value: Any) -> str:
    """Return a conservative redacted string for logs/replies."""

    text = str(value)
    text = _PEM_RE.sub("[REDACTED_PEM]", text)
    text = _ASSIGNMENT_RE.sub(lambda match: f"{match.group('key')}{match.group('sep')}[REDACTED]", text)
    text = _BOT_TOKEN_RE.sub("[REDACTED_TOKEN]", text)
    text = _LONG_SECRET_RE.sub("[REDACTED_SECRET]", text)
    return text


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Redact secret-looking keys recursively while preserving safe structure."""

    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if _SECRET_KEY_RE.search(str(key)):
            redacted[key] = "[REDACTED]"
        elif isinstance(value, dict):
            redacted[key] = redact_mapping(value)
        elif isinstance(value, list):
            redacted[key] = [
                redact_mapping(item) if isinstance(item, dict) else redact(item)
                for item in value
            ]
        else:
            redacted[key] = redact(value)
    return redacted


def safe_exception_message(exc: BaseException) -> str:
    """Sanitized exception summary for logs only, not public replies."""

    return f"{exc.__class__.__name__}: {redact(exc)}"
