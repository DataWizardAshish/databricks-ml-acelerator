"""SHA-256 hash computation for AuditEvent tamper-evidence."""

import hashlib
import json


def compute_event_hash(event: dict) -> str:
    """
    SHA-256 of canonical JSON serialization of the event.
    All fields except 'event_hash' itself, keys sorted alphabetically.
    Deterministic across Python versions via ensure_ascii=True and fixed separators.
    """
    fields = {k: v for k, v in event.items() if k != "event_hash"}
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
