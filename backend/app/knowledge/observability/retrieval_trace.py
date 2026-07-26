from __future__ import annotations

import hashlib
import re
from typing import Any


DEBUG_SAMPLE_MAX_LENGTH = 200
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)")


def controlled_debug_sample(value: str) -> str:
    redacted = _EMAIL.sub("[email]", value)
    redacted = _PHONE.sub("[phone]", redacted)
    return redacted[:DEBUG_SAMPLE_MAX_LENGTH]


def redacted_message_summary(message: str) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "length": len(message),
    }


def sanitize_trace_payload(payload: dict[str, Any], debug: bool = False) -> dict[str, Any]:
    sanitized = dict(payload)
    if debug:
        return sanitized
    for key in ("current_message", "user_message", "raw_messages", "prompt"):
        if key in sanitized:
            sanitized[key] = redacted_message_summary(str(sanitized[key]))
    return sanitized
