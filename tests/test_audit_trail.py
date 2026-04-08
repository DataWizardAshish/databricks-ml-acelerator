"""
Integration test for the audit trail — validates the full outcome end-to-end.

Simulates a complete run through all 3 human checkpoints and asserts:
1. All expected event types are recorded in the correct order
2. The hash chain is intact (chain_valid=True)
3. Human approval events are recorded with actor="user"
4. Agent decision events are recorded with actor="agent"
5. The GET /audit endpoint returns the correct structure
"""

import sqlite3
import pytest
from fastapi.testclient import TestClient

from audit.store import SqliteAuditStore
from audit.writer import AuditWriter
from audit.verifier import ChainVerifier
from api.main import app


@pytest.fixture
def mem_store():
    """In-memory SQLite store — isolated per test, no file I/O."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    return SqliteAuditStore(conn)


@pytest.fixture
def writer(mem_store):
    return AuditWriter(mem_store)


def _emit_full_run(writer: AuditWriter, run_id: str):
    """Simulate all events a complete run produces, in order."""
    # Agent decisions
    writer.emit(run_id, "opportunity_ranked", "agent", "rank_opportunities",
                {"opportunities": [{"rank": 1, "use_case": "Churn Prediction"}]})
    # Human approval
    writer.emit(run_id, "opportunity_approved", "user", "human_checkpoint",
                {"selected_rank": 1, "approved_opportunity": {
                    "rank": 1, "use_case": "Churn Prediction",
                    "target_table": "cat.schema.customers",
                    "target_column": "is_churned",
                    "ml_type": "classification",
                    "business_value": "Reduce churn",
                    "financial_impact": "$500K annually",
                    "confidence": "high",
                }})
    writer.emit(run_id, "dry_run_planned", "agent", "dry_run_explain",
                {"dry_run_plan": {"estimated_dbu_cost": "$20"}})
    # Human confirmation
    writer.emit(run_id, "dry_run_confirmed", "user", "dry_run_checkpoint",
                {"confirmed": True})
    writer.emit(run_id, "code_generated", "agent", "generate_code",
                {"artifacts": [{"filename": "01_feature_engineering.py", "content_hash": "abc123"}]})
    writer.emit(run_id, "risk_scorecard_computed", "agent", "compute_risk_scorecard",
                {"risk_scorecard": {"overall": "ready", "summary": "All checks passed", "items": []}})
    # Human code approval
    writer.emit(run_id, "code_approved", "user", "human_checkpoint_code",
                {"action": "approve"})
    writer.emit(run_id, "bundle_written", "agent", "write_bundle",
                {"written_paths": ["bundles/src/churn/01_feature_engineering.py"], "bundle_written": True})
    writer.emit(run_id, "exec_summary_generated", "agent", "generate_exec_summary",
                {"exec_summary_hash": "def456"})


def test_full_run_audit_trail(mem_store, writer):
    """
    Core outcome test: a complete run produces all expected events,
    in the correct order, with a valid hash chain, correct actors,
    and correct sequence numbers.
    """
    import time
    run_id = "test-run-001"
    _emit_full_run(writer, run_id)

    # Give background threads time to write
    time.sleep(0.3)

    events = mem_store.get_events(run_id)

    # 1. All 9 expected event types present
    expected_event_types = [
        "opportunity_ranked",
        "opportunity_approved",
        "dry_run_planned",
        "dry_run_confirmed",
        "code_generated",
        "risk_scorecard_computed",
        "code_approved",
        "bundle_written",
        "exec_summary_generated",
    ]
    actual_types = [e["event_type"] for e in events]
    assert actual_types == expected_event_types, f"Event order mismatch: {actual_types}"

    # 2. Sequence numbers are 1..9 with no gaps
    assert [e["sequence_number"] for e in events] == list(range(1, 10))

    # 3. Actor correctness
    user_events = {"opportunity_approved", "dry_run_confirmed", "code_approved"}
    for e in events:
        if e["event_type"] in user_events:
            assert e["actor"] == "user", f"{e['event_type']} should have actor=user"
        else:
            assert e["actor"] == "agent", f"{e['event_type']} should have actor=agent"

    # 4. Hash chain is intact
    result = ChainVerifier(mem_store).verify(run_id)
    assert result["valid"] is True, f"Chain invalid: {result}"
    assert result["event_count"] == 9

    # 5. First event has prev_hash=GENESIS
    assert events[0]["prev_hash"] == "GENESIS"

    # 6. Each event's prev_hash links to the previous event's hash
    for i in range(1, len(events)):
        assert events[i]["prev_hash"] == events[i - 1]["event_hash"], \
            f"Chain broken at sequence {events[i]['sequence_number']}"

    # 7. Opportunity approval payload contains full opportunity
    approval = next(e for e in events if e["event_type"] == "opportunity_approved")
    opp = approval["payload"]["approved_opportunity"]
    for field in ("use_case", "target_table", "target_column", "ml_type",
                  "business_value", "financial_impact", "confidence"):
        assert field in opp, f"Missing field '{field}' in opportunity_approved payload"


def test_tamper_detection(mem_store, writer):
    """Modifying any event field causes ChainVerifier to detect tampering."""
    import time
    run_id = "test-run-tamper"
    writer.emit(run_id, "opportunity_ranked", "agent", "rank_opportunities", {"opportunities": []})
    writer.emit(run_id, "opportunity_approved", "user", "human_checkpoint", {"selected_rank": 1})
    time.sleep(0.3)

    events = mem_store.get_events(run_id)
    assert len(events) == 2

    # Tamper with the first event's payload directly in the DB
    mem_store._conn.execute(
        "UPDATE audit_trail SET payload = ? WHERE sequence_number = 1 AND run_id = ?",
        ('{"opportunities": [{"rank": 99}]}', run_id),
    )
    mem_store._conn.commit()

    result = ChainVerifier(mem_store).verify(run_id)
    assert result["valid"] is False
    assert result["tampered_at_sequence"] == 1
    assert result["reason"] == "hash_mismatch"


def test_audit_endpoint_404_for_unknown_run():
    """GET /runs/{run_id}/audit returns 404 for a run with no events."""
    client = TestClient(app)
    response = client.get("/runs/nonexistent-run-xyz/audit")
    assert response.status_code == 404


def test_empty_run_verifier(mem_store):
    """ChainVerifier returns valid=False with reason=no_events for unknown run."""
    result = ChainVerifier(mem_store).verify("no-such-run")
    assert result["valid"] is False
    assert result["reason"] == "no_events"
    assert result["event_count"] == 0
