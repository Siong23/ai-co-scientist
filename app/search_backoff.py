"""Share provider cooldowns across retrieval rounds and concurrent agents."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class _ProviderState:
    lock: object = field(default_factory=Lock)
    retry_at: float = 0.0


_states: dict[tuple, _ProviderState] = {}
_states_lock = Lock()


def guarded_search(source, search, *, provider_name=""):
    """Return (results, cooling_down); serialize calls to the same provider.

    Credentials distinguish accounts without retaining plaintext keys. Method
    identity isolates injected providers as well as different implementations.
    Zero results alone never trigger a cooldown, and query-specific HTTP 404
    responses do not prevent a different query from succeeding.
    """
    method = getattr(source, "search_papers", None) or source.search
    implementation = getattr(method, "__func__", method)
    credential = getattr(source, "api_key", "")
    credential = credential if isinstance(credential, str) else ""
    key = (provider_name, type(source), implementation, hashlib.sha256(credential.encode()).digest())
    with _states_lock:
        state = _states.setdefault(key, _ProviderState())
    with state.lock:
        if time.monotonic() < state.retry_at:
            return [], True
        try:
            results = search()
        except Exception:
            state.retry_at = time.monotonic() + 30
            raise
        status = getattr(source, "last_error_status", None)
        kind = getattr(source, "last_error_kind", "")
        delay = 0
        if status in (401, 402, 403, 432, 433) or kind == "quota_or_plan_rejection":
            delay = 300
        elif status in (429, 503) or kind == "rate_limited":
            delay = 60
        elif kind in ("timeout", "provider_error") and status not in (400, 404):
            delay = 30
        state.retry_at = time.monotonic() + delay
        return results, False
