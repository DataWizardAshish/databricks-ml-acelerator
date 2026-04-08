# Implementation Plan: Audit Trail

## Overview

Implement an immutable, hash-chained audit event log for the ML Accelerator. The `audit/` package is created first, then instrumentation is added to existing nodes and API endpoints, and finally the `GET /runs/{run_id}/audit` endpoint is wired in.

All audit writes are fire-and-forget — the pipeline must never be blocked or broken by audit instrumentation.

## Tasks

- [ ] 1. Create the `audit/` package with core data models and hashing
  - [ ] 1.1 Create `audit/__init__.py` and `audit/models.py`
    - Define `AuditEvent` TypedDict with all 10 required fields: `event_id`, `run_id`, `sequence_number`, `event_type`, `actor`, `node_name`, `timestamp_utc`, `payload`, `prev_hash`, `event_hash`
    - Define `VerificationResult` TypedDict with `valid`, `event_count`, `tampered_at_sequence`, `reason`
    - _Requirements: 1.1_

  - [ ] 1.2 Create `audit/hashing.py`
    - Implement `compute_event_hash(event: dict) -> str`: SHA-256 of canonical JSON (all fields except `event_hash`, keys sorted alphabetically, `separators=(",", ":")`, `ensure_ascii=True`)
    - _Requirements: 1.2_

  - [ ]* 1.3 Write property test for hash computation determinism (Property 2)
    - **Property 2: Hash computation determinism and correctness**
    - Generate random `AuditEvent` dicts via `st.fixed_dictionaries()`. Call `compute_event_hash()` twice. Assert results are identical and match the stored hash.
    - **Validates: Requirements 1.2**

- [ ] 2. Implement `AuditStore` — SQLite and Delta backends
  - [ ] 2.1 Create `audit/store.py` with `AuditStore` Protocol and `SqliteAuditStore`
    - Define `AuditStore` Protocol: `write(event)`, `get_events(run_id)`, `get_max_sequence(run_id)`, `ensure_table()`
    - Implement `SqliteAuditStore` using the existing `data/checkpoints.db` connection (reuse `_get_db_conn()` from `agent/graph.py` or accept a connection parameter)
    - `ensure_table()` creates the `audit_trail` table with schema from design (including `UNIQUE(run_id, sequence_number)` and index `idx_audit_run_seq`)
    - `write()` uses INSERT OR IGNORE to be idempotent on duplicate `event_id`
    - `get_events()` returns events ordered by `sequence_number ASC`
    - `get_max_sequence()` returns 0 if no events exist for the run
    - _Requirements: 1.4, 5.5, 5.6_

  - [ ]* 2.2 Write unit tests for `SqliteAuditStore`
    - Test `write()` + `get_events()` round-trip with in-memory SQLite (`:memory:`)
    - Test `get_max_sequence()` returns 0 for unknown run, correct value after writes
    - Test `ensure_table()` is idempotent (call twice, no error)
    - _Requirements: 5.5_

  - [ ] 2.3 Implement `DeltaAuditStore` in `audit/store.py`
    - Uses `spark.sql` INSERT INTO for writes (payload JSON-serialized as string)
    - `ensure_table()` issues `CREATE TABLE IF NOT EXISTS ml_accelerator.audit_trail` with `TBLPROPERTIES ('delta.appendOnly' = 'true')`
    - `get_events()` uses `spark.sql` SELECT ordered by `sequence_number`
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ] 2.4 Implement `get_audit_store()` factory in `audit/store.py`
    - Returns `DeltaAuditStore` when `os.getenv("DATABRICKS_HOST")` is set, else `SqliteAuditStore`
    - Result is cached at module level (checked once at import time)
    - _Requirements: 5.5, 5.6_

  - [ ]* 2.5 Write property test for append-only immutability (Property 5)
    - **Property 5: Append-only immutability**
    - Write N events to in-memory SQLite store. Snapshot their hashes. Write M more events. Assert original N events are byte-for-byte identical.
    - **Validates: Requirements 1.4, 5.4**

- [ ] 3. Implement `AuditWriter` with fire-and-forget dispatch
  - [ ] 3.1 Create `audit/writer.py` with `AuditWriter` class and `get_audit_writer()` singleton
    - `emit()` constructs the `AuditEvent`: generates `event_id` (UUID4), sets `timestamp_utc` (ISO-8601 UTC), assigns `sequence_number` (from per-run in-memory counter seeded from `get_max_sequence()` on first emit), computes `prev_hash` and `event_hash` synchronously before dispatch
    - Dispatches write via `asyncio.create_task()` if event loop is running, else uses `threading.Thread(daemon=True)` as fallback
    - Wraps all exceptions — `emit()` never raises; logs ERROR with `run_id` and `event_type` on failure
    - Implements bounded retry buffer (deque, max 100): failed writes are enqueued; a background retry loop uses exponential backoff (1s, 2s, 4s, … cap 60s); events beyond 100 are dropped with ERROR log
    - `get_audit_writer()` returns the process-level singleton
    - _Requirements: 1.1, 1.3, 1.5, 4.1, 4.2, 4.4_

  - [ ]* 3.2 Write property test for AuditEvent schema completeness (Property 1)
    - **Property 1: AuditEvent schema completeness**
    - Generate random (run_id, event_type, actor, node_name, payload) via `st.text()` and `st.dictionaries()`. Call `emit()` with mock store. Assert all 10 fields present with correct types.
    - **Validates: Requirements 1.1**

  - [ ]* 3.3 Write property test for monotonic sequence numbers (Property 4)
    - **Property 4: Monotonic sequence numbers**
    - Generate N random payloads for the same run_id. Write all via `AuditWriter` with mock store. Assert `sequence_numbers == list(range(1, N+1))`.
    - **Validates: Requirements 1.5**

  - [ ]* 3.4 Write property test for hash chain linkage (Property 3)
    - **Property 3: Hash chain linkage**
    - Generate a random list of N payloads (N drawn from 1..20). Write them sequentially via `AuditWriter` with mock store. Assert `prev_hash` of event N equals `event_hash` of event N-1, and event 1 has `prev_hash="GENESIS"`.
    - **Validates: Requirements 1.3**

  - [ ]* 3.5 Write property test for error isolation (Property 7)
    - **Property 7: Error isolation — write failures never propagate**
    - Mock store raises random exception types (`st.sampled_from([ValueError, IOError, RuntimeError, Exception])`). Assert `emit()` does not raise and ERROR was logged with `run_id` and `event_type`.
    - **Validates: Requirements 4.2**

- [ ] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement `ChainVerifier`
  - [ ] 5.1 Create `audit/verifier.py` with `ChainVerifier` class
    - `verify(run_id)` retrieves events from store ordered by `sequence_number`
    - Recomputes `event_hash` for each event using `compute_event_hash()` and compares to stored value; returns `{"valid": false, "tampered_at_sequence": n, "reason": "hash_mismatch"}` on mismatch
    - Checks `prev_hash` linkage for each consecutive pair; returns `{"valid": false, "tampered_at_sequence": n, "reason": "chain_broken"}` on mismatch
    - Returns `{"valid": true, "event_count": n, "tampered_at_sequence": None, "reason": None}` when all pass
    - Returns `{"valid": false, "event_count": 0, "reason": "no_events"}` for empty event list
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 5.2 Write property test for chain verification detects tampering (Property 8)
    - **Property 8: Chain verification detects any single-field tampering**
    - Generate a valid event sequence. Pick a random event index and a random field (excluding `event_hash`) to modify with a random value. Assert `ChainVerifier.verify()` returns `valid=False` with the correct `tampered_at_sequence`.
    - **Validates: Requirements 6.2, 6.3, 6.4, 6.5**

- [ ] 6. Add `run_id` to `AgentState` workspace dict and instrument agent nodes
  - [ ] 6.1 Propagate `run_id` into the workspace dict in `agent/graph.py`
    - In `run_discovery()`, add `"run_id": run_id` to the `initial_state["workspace"]` dict so nodes can access it via `state["workspace"].get("run_id", "")`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ] 6.2 Instrument `rank_opportunities` in `agent/nodes.py`
    - After building `new_state`, call `get_audit_writer().emit(run_id=..., event_type="opportunity_ranked", actor="agent", node_name="rank_opportunities", payload={"opportunities": opportunities, "estate_summary_digest": hashlib.sha256(...).hexdigest()})`
    - Wrap in `try/except Exception: pass`
    - _Requirements: 2.1_

  - [ ] 6.3 Instrument `dry_run_explain` in `agent/trust_nodes.py`
    - After building updated state, emit `event_type="dry_run_planned"` with `payload={"dry_run_plan": plan}`
    - Wrap in `try/except Exception: pass`
    - _Requirements: 2.2_

  - [ ] 6.4 Instrument `generate_code` in `agent/code_gen_nodes.py`
    - After building artifacts, emit `event_type="code_generated"` with `payload={"artifacts": [{"filename": a["filename"], "content_hash": hashlib.sha256(a["content"].encode()).hexdigest()} for a in artifacts]}`
    - Wrap in `try/except Exception: pass`
    - _Requirements: 2.3, 8.4_

  - [ ] 6.5 Instrument `compute_risk_scorecard` in `agent/trust_nodes.py`
    - After building scorecard, emit `event_type="risk_scorecard_computed"` with `payload={"risk_scorecard": scorecard}`
    - Wrap in `try/except Exception: pass`
    - _Requirements: 2.4, 8.3_

  - [ ] 6.6 Instrument `write_bundle` in `agent/code_gen_nodes.py`
    - After successful write, emit `event_type="bundle_written"` with `payload={"written_paths": written, "bundle_written": True}`
    - Wrap in `try/except Exception: pass`
    - _Requirements: 2.5_

  - [ ] 6.7 Instrument `generate_exec_summary` in `agent/trust_nodes.py`
    - After building summary, emit `event_type="exec_summary_generated"` with `payload={"exec_summary_hash": hashlib.sha256(summary_md.encode()).hexdigest()}`
    - Wrap in `try/except Exception: pass`
    - _Requirements: 2.6_

  - [ ]* 6.8 Write unit tests for node instrumentation
    - Mock `get_audit_writer()`. For each instrumented node, assert `emit()` is called with the correct `event_type`, `actor`, and key payload fields.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [ ] 7. Instrument FastAPI approval endpoints in `agent/graph.py`
  - [ ] 7.1 Instrument `approve_opportunity()` in `agent/graph.py`
    - Before `graph.invoke(Command(resume=...))`, emit `event_type="opportunity_approved"`, `actor="user"`, `node_name="human_checkpoint"`, `payload={"selected_rank": selected_rank, "approved_opportunity": <full MLOpportunity dict>}`
    - _Requirements: 3.1, 3.5, 8.2_

  - [ ] 7.2 Instrument `confirm_dry_run()` in `agent/graph.py`
    - Before `graph.invoke(Command(resume=...))`, emit `event_type="dry_run_confirmed"`, `actor="user"`, `node_name="dry_run_checkpoint"`, `payload={"confirmed": True}`
    - _Requirements: 3.2, 3.5_

  - [ ] 7.3 Instrument `approve_code()` in `agent/graph.py`
    - Before `graph.invoke(Command(resume=...))`, emit `event_type="code_approved"` (action=approve) or `event_type="code_regeneration_requested"` (action=regenerate), `actor="user"`, `node_name="human_checkpoint_code"`, payload includes `action` and `instructions` when regenerating
    - _Requirements: 3.3, 3.4, 3.5_

  - [ ]* 7.4 Write property test for opportunity approval payload completeness (Property 10)
    - **Property 10: Opportunity approval payload completeness**
    - Generate random `MLOpportunity` dicts via `st.fixed_dictionaries()`. Emit `opportunity_approved` event with mock store. Assert all required fields (`use_case`, `target_table`, `target_column`, `ml_type`, `business_value`, `financial_impact`, `confidence`) are present in payload unchanged.
    - **Validates: Requirements 3.1, 8.2**

  - [ ]* 7.5 Write property test for regeneration instructions round-trip (Property 11)
    - **Property 11: Regeneration instructions round-trip**
    - Generate random instruction strings via `st.text()`. Emit `code_regeneration_requested` event with mock store. Assert `payload["instructions"]` equals the input string exactly.
    - **Validates: Requirements 3.4**

- [ ] 8. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Add `GET /runs/{run_id}/audit` endpoint to `api/main.py`
  - [ ] 9.1 Add `AuditTrailResponse` Pydantic model and the endpoint
    - Define `AuditTrailResponse(BaseModel)` with `run_id`, `event_count`, `chain_valid`, `events`
    - Implement `GET /runs/{run_id}/audit`: call `get_audit_store().get_events(run_id)`, raise 404 if empty, call `ChainVerifier(store).verify(run_id)`, return `AuditTrailResponse`
    - Import `get_audit_store` from `audit.store` and `ChainVerifier` from `audit.verifier`
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 9.2 Write unit tests for the audit endpoint
    - Test 200 response with valid chain for a run with events in mock store
    - Test 404 for unknown run_id
    - Test `chain_valid=False` when verifier returns invalid
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 9.3 Write property test for audit response schema invariant (Property 9)
    - **Property 9: Audit response schema invariant**
    - Generate N random events for a run_id. Write to mock store. Call the audit endpoint handler directly. Assert `run_id` matches, `event_count == N`, `chain_valid` is bool, `events` has length N sorted by `sequence_number` ascending.
    - **Validates: Requirements 7.1, 7.3, 7.4**

  - [ ]* 9.4 Write property test for artifact content hash correctness (Property 6)
    - **Property 6: Artifact content hash correctness**
    - Generate random artifact content strings via `st.text()`. Build the `code_generated` payload. Assert each filename maps to `hashlib.sha256(content.encode()).hexdigest()`.
    - **Validates: Requirements 2.3, 8.4**

- [ ] 10. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- All `emit()` calls in nodes are wrapped in `try/except Exception: pass` — the pipeline must never be broken by audit instrumentation
- Property tests use [Hypothesis](https://hypothesis.readthedocs.io/) with a minimum of 100 iterations each
- `SqliteAuditStore` reuses the existing `data/checkpoints.db` connection — no new database files
- `DeltaAuditStore` uses `spark.sql` INSERT INTO — never OVERWRITE or UPDATE
- `run_id` is propagated into `state["workspace"]` in `run_discovery()` so all nodes can access it without state schema changes
