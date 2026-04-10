"""
AuditWriter — fire-and-forget audit event emitter.

Constructs AuditEvent records, computes SHA-256 hash chain, and dispatches
writes to the AuditStore as background threads. Never raises into the caller.
"""

import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from audit.hashing import compute_event_hash
from audit.models import AuditEvent
from audit.store import AuditStore, get_audit_store

logger = logging.getLogger(__name__)

_writer_instance: Optional["AuditWriter"] = None
_writer_lock = threading.Lock()

_MAX_BUFFER = 100
_RETRY_DELAYS = [1, 2, 4, 8, 16, 32, 60]  # exponential backoff cap 60s


class AuditWriter:
    """
    Process-level singleton. Emits audit events without blocking the pipeline.
    - Sequence numbers and prev_hashes are tracked per run_id in memory.
    - Writes dispatched as daemon threads — caller returns immediately.
    - Failed writes buffered (max 100) with exponential backoff retry.
    """

    def __init__(self, store: AuditStore) -> None:
        self._store = store
        self._seq: dict[str, int] = {}          # run_id → current sequence number
        self._last_hash: dict[str, str] = {}    # run_id → last event_hash (for chain)
        self._run_lock: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._buffer: deque = deque(maxlen=_MAX_BUFFER)
        self._start_retry_loop()

    def _get_run_lock(self, run_id: str) -> threading.Lock:
        with self._global_lock:
            if run_id not in self._run_lock:
                self._run_lock[run_id] = threading.Lock()
            return self._run_lock[run_id]

    def emit(
        self,
        run_id: str,
        event_type: str,
        actor: str,
        node_name: str,
        payload: dict,
        langsmith_run_id: Optional[str] = None,
    ) -> None:
        """
        Construct an AuditEvent, compute hashes, and dispatch a background write.
        Never raises. Returns immediately.
        """
        try:
            if langsmith_run_id:
                payload = {**payload, "langsmith_run_id": langsmith_run_id}

            # Sequence + prev_hash assignment is serialised per run_id
            with self._get_run_lock(run_id):
                seq = self._next_sequence(run_id)
                prev_hash = self._last_hash.get(run_id, "GENESIS")

                event: AuditEvent = {
                    "event_id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "sequence_number": seq,
                    "event_type": event_type,
                    "actor": actor,
                    "node_name": node_name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "payload": payload,
                    "prev_hash": prev_hash,
                    "event_hash": "",
                }
                event["event_hash"] = compute_event_hash(event)
                # Cache the hash immediately so the next emit gets the right prev_hash
                self._last_hash[run_id] = event["event_hash"]

            t = threading.Thread(target=self._write, args=(event,), daemon=True)
            t.start()

        except Exception as e:
            logger.error("AuditWriter.emit failed (run_id=%s, event_type=%s): %s", run_id, event_type, e)

    def _next_sequence(self, run_id: str) -> int:
        # Called inside per-run lock — no additional locking needed
        if run_id not in self._seq:
            # Seed from store on first emit for this run (handles restarts)
            self._seq[run_id] = self._store.get_max_sequence(run_id)
            # Also seed prev_hash from store if events already exist
            if self._seq[run_id] > 0 and run_id not in self._last_hash:
                existing = self._store.get_events(run_id)
                if existing:
                    self._last_hash[run_id] = existing[-1]["event_hash"]
        self._seq[run_id] += 1
        return self._seq[run_id]

    def _write(self, event: AuditEvent) -> None:
        try:
            self._store.write(event)
        except Exception as e:
            logger.error(
                "AuditStore write failed (run_id=%s, event_type=%s): %s — buffering for retry",
                event["run_id"], event["event_type"], e,
            )
            if len(self._buffer) >= _MAX_BUFFER:
                dropped = self._buffer[0]
                logger.error(
                    "Retry buffer full — dropping oldest event (run_id=%s, event_type=%s)",
                    dropped["run_id"], dropped["event_type"],
                )
            self._buffer.append(event)

    def _start_retry_loop(self) -> None:
        t = threading.Thread(target=self._retry_loop, daemon=True)
        t.start()

    def _retry_loop(self) -> None:
        attempt = 0
        while True:
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            time.sleep(delay)
            if not self._buffer:
                attempt = 0
                continue
            event = self._buffer[0]
            try:
                self._store.write(event)
                self._buffer.popleft()
                attempt = 0
                logger.info("Retry succeeded for event_id=%s", event["event_id"])
            except Exception:
                attempt += 1


def get_audit_writer() -> AuditWriter:
    """Return the process-level AuditWriter singleton."""
    global _writer_instance
    if _writer_instance is not None:
        return _writer_instance
    with _writer_lock:
        if _writer_instance is None:
            _writer_instance = AuditWriter(get_audit_store())
    return _writer_instance
