"""AuditEvent and VerificationResult data models."""

from typing import Optional
from typing_extensions import TypedDict


class AuditEvent(TypedDict):
    event_id: str           # UUID4
    run_id: str
    sequence_number: int    # 1-based, monotonically increasing per run
    event_type: str         # controlled vocabulary — see design.md
    actor: str              # "agent" | "user"
    node_name: str
    timestamp_utc: str      # ISO-8601 UTC e.g. "2024-01-15T10:30:00.000000+00:00"
    payload: dict           # event-specific structured data
    prev_hash: str          # SHA-256 hex or "GENESIS" for first event
    event_hash: str         # SHA-256 hex of canonical serialization


class VerificationResult(TypedDict):
    valid: bool
    event_count: int
    tampered_at_sequence: Optional[int]   # None when valid=True
    reason: Optional[str]                 # "hash_mismatch" | "chain_broken" | "no_events" | None
