"""ChainVerifier — validates SHA-256 hash chain integrity for a run's audit trail."""

from audit.hashing import compute_event_hash
from audit.models import VerificationResult
from audit.store import AuditStore


class ChainVerifier:
    def __init__(self, store: AuditStore) -> None:
        self._store = store

    def verify(self, run_id: str) -> VerificationResult:
        events = self._store.get_events(run_id)

        if not events:
            return VerificationResult(valid=False, event_count=0, tampered_at_sequence=None, reason="no_events")

        for i, event in enumerate(events):
            # 1. Recompute hash and compare
            recomputed = compute_event_hash(event)
            if recomputed != event["event_hash"]:
                return VerificationResult(
                    valid=False,
                    event_count=len(events),
                    tampered_at_sequence=event["sequence_number"],
                    reason="hash_mismatch",
                )

            # 2. Check prev_hash linkage
            expected_prev = "GENESIS" if i == 0 else events[i - 1]["event_hash"]
            if event["prev_hash"] != expected_prev:
                return VerificationResult(
                    valid=False,
                    event_count=len(events),
                    tampered_at_sequence=event["sequence_number"],
                    reason="chain_broken",
                )

        return VerificationResult(valid=True, event_count=len(events), tampered_at_sequence=None, reason=None)
