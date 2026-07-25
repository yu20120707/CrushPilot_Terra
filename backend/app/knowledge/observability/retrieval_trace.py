from __future__ import annotations

import hashlib
from typing import Any


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
