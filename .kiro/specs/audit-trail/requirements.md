# Requirements Document

## Introduction

The Audit Trail feature provides an immutable, append-only log of every significant event in a Databricks ML Accelerator run. It is the primary enterprise trust signal: a CTO or compliance officer must be able to ask "what did the AI decide, who approved it, and when?" and receive a complete, tamper-evident, chronologically ordered answer.

Events captured include: LLM decisions (opportunity ranking, dry run planning, code generation), human approvals at each checkpoint, risk scorecard results, and artifact writes. Each event is hash-chained using SHA-256 so the integrity of the full log can be verified. Storage is an append-only Delta table in Unity Catalog (`ml_accelerator` schema), with a SQLite fallback for local development. A new FastAPI endpoint exposes the full ordered log per run.

## Glossary

- **Audit_Trail**: The complete, ordered sequence of `AuditEvent` records for a given run.
- **AuditEvent**: A single immutable record capturing one significant occurrence in a run (see schema below).
- **Actor**: The entity responsible for an event — either `"user"` (human approval via API) or `"agent"` (LangGraph node decision).
- **Event_Type**: A controlled vocabulary string classifying the event (e.g. `opportunity_ranked`, `opportunity_approved`, `dry_run_planned`, `dry_run_confirmed`, `code_generated`, `risk_scorecard_computed`, `code_approved`, `bundle_written`, `exec_summary_generated`).
- **Hash_Chain**: A tamper-evidence mechanism where each `AuditEvent` includes a SHA-256 hash computed over its own content plus the hash of the immediately preceding event in the same run.
- **Audit_Writer**: The internal component responsible for constructing and persisting `AuditEvent` records.
- **Audit_Store**: The durable storage backend — Delta table in Unity Catalog for production, SQLite table for local development.
- **Chain_Verifier**: The internal component that validates the integrity of a run's hash chain.
- **Run_ID**: The UUID that uniquely identifies a single end-to-end pipeline execution, already present in the system.
- **LangGraph_Node**: A discrete processing step in the agent pipeline (e.g. `rank_opportunities`, `dry_run_explain`, `generate_code`).
- **Human_Checkpoint**: A LangGraph interrupt node where execution pauses for user approval via a FastAPI endpoint.
- **Delta_Table**: An ACID-compliant table format in Databricks Unity Catalog used as the production Audit_Store.
- **Payload**: A JSON-serializable dict attached to an `AuditEvent` containing event-specific structured data (e.g. the approved opportunity, the risk scorecard result).

---

## Requirements

### Requirement 1: AuditEvent Schema and Immutability

**User Story:** As a compliance officer, I want every audit event to have a consistent, well-defined structure, so that I can reliably query and interpret the audit trail across all runs.

#### Acceptance Criteria

1. THE Audit_Writer SHALL produce `AuditEvent` records containing the following fields: `event_id` (UUID), `run_id` (string), `sequence_number` (integer, 1-based, monotonically increasing per run), `event_type` (string from controlled vocabulary), `actor` (string: `"user"` or `"agent"`), `node_name` (string: the LangGraph node that emitted the event), `timestamp_utc` (ISO-8601 UTC string), `payload` (JSON object), `prev_hash` (SHA-256 hex string or `"GENESIS"` for the first event), and `event_hash` (SHA-256 hex string).
2. THE Audit_Writer SHALL compute `event_hash` as the SHA-256 digest of the canonical JSON serialization of all fields except `event_hash` itself, with keys sorted alphabetically.
3. THE Audit_Writer SHALL set `prev_hash` to the `event_hash` of the immediately preceding event in the same run, ordered by `sequence_number`, or to the string `"GENESIS"` when `sequence_number` is 1.
4. THE Audit_Store SHALL be append-only — THE Audit_Writer SHALL never update or delete existing `AuditEvent` records.
5. WHEN an `AuditEvent` is written, THE Audit_Writer SHALL assign a `sequence_number` that is exactly one greater than the highest existing `sequence_number` for that `run_id`, or 1 if no prior events exist for that run.

---

### Requirement 2: Event Coverage — Agent Decisions

**User Story:** As a CTO, I want every AI decision recorded in the audit trail, so that I can understand exactly what the agent decided and why at each step.

#### Acceptance Criteria

1. WHEN the `rank_opportunities` node completes, THE Audit_Writer SHALL emit an `AuditEvent` with `event_type="opportunity_ranked"`, `actor="agent"`, and a `payload` containing the full ranked opportunities list and the estate summary digest.
2. WHEN the `dry_run_explain` node completes, THE Audit_Writer SHALL emit an `AuditEvent` with `event_type="dry_run_planned"`, `actor="agent"`, and a `payload` containing the full `DryRunPlan` dict.
3. WHEN the `generate_code` node completes, THE Audit_Writer SHALL emit an `AuditEvent` with `event_type="code_generated"`, `actor="agent"`, and a `payload` containing the list of generated artifact filenames and their SHA-256 content hashes (not the raw content).
4. WHEN the `compute_risk_scorecard` node completes, THE Audit_Writer SHALL emit an `AuditEvent` with `event_type="risk_scorecard_computed"`, `actor="agent"`, and a `payload` containing the full `RiskScorecard` dict.
5. WHEN the `write_bundle` node completes, THE Audit_Writer SHALL emit an `AuditEvent` with `event_type="bundle_written"`, `actor="agent"`, and a `payload` containing the list of written file paths and the `bundle_written` boolean.
6. WHEN the `generate_exec_summary` node completes, THE Audit_Writer SHALL emit an `AuditEvent` with `event_type="exec_summary_generated"`, `actor="agent"`, and a `payload` containing the SHA-256 hash of the executive summary content.

---

### Requirement 3: Event Coverage — Human Approvals

**User Story:** As a compliance officer, I want every human approval recorded with the actor identity and the exact choice made, so that I can demonstrate human oversight of all AI-generated outputs.

#### Acceptance Criteria

1. WHEN `POST /runs/{run_id}/approve` is called, THE Audit_Writer SHALL emit an `AuditEvent` with `event_type="opportunity_approved"`, `actor="user"`, `node_name="human_checkpoint"`, and a `payload` containing `selected_rank` and the full approved `MLOpportunity` dict.
2. WHEN `POST /runs/{run_id}/confirm-dry-run` is called, THE Audit_Writer SHALL emit an `AuditEvent` with `event_type="dry_run_confirmed"`, `actor="user"`, `node_name="dry_run_checkpoint"`, and a `payload` containing `{"confirmed": true}`.
3. WHEN `POST /runs/{run_id}/approve-code` is called with `action="approve"`, THE Audit_Writer SHALL emit an `AuditEvent` with `event_type="code_approved"`, `actor="user"`, `node_name="human_checkpoint_code"`, and a `payload` containing `{"action": "approve"}`.
4. WHEN `POST /runs/{run_id}/approve-code` is called with `action="regenerate"`, THE Audit_Writer SHALL emit an `AuditEvent` with `event_type="code_regeneration_requested"`, `actor="user"`, `node_name="human_checkpoint_code"`, and a `payload` containing `{"action": "regenerate", "instructions": "<the provided instructions>"}`.
5. THE Audit_Writer SHALL record human approval events before the corresponding LangGraph `Command(resume=...)` is issued, so that the approval is captured even if the downstream graph execution fails.

---

### Requirement 4: Non-Blocking Writes

**User Story:** As a developer, I want audit writes to never block the agent pipeline or API response, so that audit trail instrumentation does not degrade run performance or reliability.

#### Acceptance Criteria

1. THE Audit_Writer SHALL execute all Audit_Store writes asynchronously using `asyncio` background tasks or a fire-and-forget pattern, so that the calling LangGraph node or FastAPI handler returns without waiting for the write to complete.
2. IF an Audit_Store write fails, THEN THE Audit_Writer SHALL log the failure at ERROR level including the `run_id`, `event_type`, and exception message, and SHALL NOT raise an exception to the calling node or endpoint.
3. THE Audit_Writer SHALL not add more than 50 milliseconds of latency to any LangGraph node execution in the p99 case when the Audit_Store is available.
4. WHILE the Audit_Store is unavailable, THE Audit_Writer SHALL buffer up to 100 pending events in memory and retry writes with exponential backoff, without blocking the pipeline.

---

### Requirement 5: Storage — Delta Table in Unity Catalog

**User Story:** As a Databricks platform engineer, I want audit events stored in a Delta table in Unity Catalog, so that the audit trail is durable, queryable with SQL, and governed by existing Unity Catalog access controls.

#### Acceptance Criteria

1. THE Audit_Store SHALL persist `AuditEvent` records to a Delta table named `ml_accelerator.audit_trail` in the configured Unity Catalog.
2. WHEN the Audit_Store is initialized and the `ml_accelerator.audit_trail` table does not exist, THE Audit_Store SHALL create it with the schema defined in Requirement 1, with `event_id` as the primary identifier and `(run_id, sequence_number)` as a composite index.
3. THE Audit_Store SHALL use `spark.sql` or the Databricks SDK to write to the Delta table, compatible with the Databricks Apps execution environment.
4. THE Audit_Store SHALL append records using Delta's `INSERT INTO` semantics, never using `OVERWRITE` or `UPDATE` operations on existing rows.
5. WHERE the `DATABRICKS_HOST` environment variable is not set (local development), THE Audit_Store SHALL fall back to a SQLite table named `audit_trail` in the existing `data/checkpoints.db` database, using the same schema.
6. THE Audit_Store SHALL expose a single `write(event: AuditEvent) -> None` interface that abstracts over the Delta and SQLite backends, so that the Audit_Writer is decoupled from the storage implementation.

---

### Requirement 6: Hash Chain Integrity Verification

**User Story:** As a compliance officer, I want to verify that the audit trail has not been tampered with, so that I can present it as evidence in an audit or regulatory review.

#### Acceptance Criteria

1. THE Chain_Verifier SHALL accept a `run_id` and retrieve all `AuditEvent` records for that run ordered by `sequence_number`.
2. WHEN verifying a run's audit trail, THE Chain_Verifier SHALL recompute the `event_hash` for each event using the same canonical serialization algorithm used by the Audit_Writer, and SHALL compare it to the stored `event_hash`.
3. WHEN verifying a run's audit trail, THE Chain_Verifier SHALL confirm that each event's `prev_hash` equals the `event_hash` of the preceding event (or `"GENESIS"` for `sequence_number=1`).
4. IF any `event_hash` does not match the recomputed value, THEN THE Chain_Verifier SHALL return a verification result with `{"valid": false, "tampered_at_sequence": <n>, "reason": "hash_mismatch"}`.
5. IF any `prev_hash` does not match the preceding event's `event_hash`, THEN THE Chain_Verifier SHALL return a verification result with `{"valid": false, "tampered_at_sequence": <n>, "reason": "chain_broken"}`.
6. WHEN all events pass verification, THE Chain_Verifier SHALL return `{"valid": true, "event_count": <n>}`.

---

### Requirement 7: Audit Trail API Endpoint

**User Story:** As a CTO, I want a single API endpoint that returns the complete ordered audit trail for a run, so that I can retrieve a full decision history without querying the database directly.

#### Acceptance Criteria

1. THE System SHALL expose a `GET /runs/{run_id}/audit` endpoint that returns the complete ordered list of `AuditEvent` records for the specified run, sorted by `sequence_number` ascending.
2. WHEN `GET /runs/{run_id}/audit` is called for a `run_id` that does not exist in the Audit_Store, THE System SHALL return HTTP 404 with a JSON body `{"detail": "No audit trail found for run <run_id>"}`.
3. WHEN `GET /runs/{run_id}/audit` is called for a valid run, THE System SHALL include a `chain_valid` boolean field in the response, computed by the Chain_Verifier, indicating whether the hash chain is intact.
4. THE System SHALL return the audit response in the following structure: `{"run_id": "<id>", "event_count": <n>, "chain_valid": <bool>, "events": [<AuditEvent>, ...]}`.
5. THE System SHALL return the `GET /runs/{run_id}/audit` response within 2 seconds for runs with up to 50 audit events.

---

### Requirement 8: Run Reconstruction from Audit Log

**User Story:** As a developer, I want the audit trail to contain enough information to reconstruct the key decisions of a run, so that I can diagnose issues or replay a run's logic without access to the LangGraph checkpoint.

#### Acceptance Criteria

1. THE Audit_Trail for a completed run SHALL contain at minimum one event for each of the following milestones: opportunity ranking, opportunity approval, dry run planning, dry run confirmation, code generation, risk scorecard computation, code approval, and bundle write.
2. WHEN the `opportunity_approved` event payload is present, THE Audit_Trail SHALL contain the full `MLOpportunity` dict including `use_case`, `target_table`, `target_column`, `ml_type`, `business_value`, `financial_impact`, and `confidence`.
3. WHEN the `risk_scorecard_computed` event payload is present, THE Audit_Trail SHALL contain the full `RiskScorecard` dict including all `items`, `overall` status, and `summary`.
4. WHEN the `code_generated` event payload is present, THE Audit_Trail SHALL contain the filenames and content hashes of all generated artifacts, sufficient to verify whether the written bundle matches what was approved.

---

### Requirement 9: LangSmith Integration Alignment

**User Story:** As a developer, I want the audit trail to complement (not duplicate) LangSmith traces, so that the two observability layers serve distinct purposes without redundant instrumentation.

#### Acceptance Criteria

1. THE Audit_Trail SHALL record business-level events (human approvals, AI decisions, artifact hashes) while LangSmith traces record LLM-level events (token counts, latency, prompt/response pairs) — THE Audit_Writer SHALL not log raw LLM prompts or responses.
2. WHEN an `AuditEvent` is emitted for a node that also produces a LangSmith trace, THE Audit_Writer SHALL include the LangSmith `run_id` in the event `payload` under the key `"langsmith_run_id"` if it is available in the LangGraph execution context.
3. THE Audit_Writer SHALL not make any calls to the LangSmith API — it SHALL only read the LangSmith run ID from the LangGraph execution context if already present.
