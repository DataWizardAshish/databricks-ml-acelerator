"""
AuditStore — backend-agnostic storage interface for audit events.

Current backend: SqliteAuditStore (data/checkpoints.db)
Phase 3 upgrade: DeltaAuditStore (ml_accelerator.audit_trail in Unity Catalog)
"""

import json
import logging
import os
import sqlite3
from typing import Optional
from typing_extensions import Protocol, runtime_checkable

from audit.models import AuditEvent

logger = logging.getLogger(__name__)

_store_instance: Optional["AuditStore"] = None


@runtime_checkable
class AuditStore(Protocol):
    def write(self, event: AuditEvent) -> None: ...
    def get_events(self, run_id: str) -> list[AuditEvent]: ...
    def get_max_sequence(self, run_id: str) -> int: ...
    def ensure_table(self) -> None: ...


class SqliteAuditStore:
    """
    Stores audit events in the existing data/checkpoints.db SQLite database.
    Reuses the same connection pattern as agent/graph.py.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS audit_trail (
            event_id        TEXT    NOT NULL PRIMARY KEY,
            run_id          TEXT    NOT NULL,
            sequence_number INTEGER NOT NULL,
            event_type      TEXT    NOT NULL,
            actor           TEXT    NOT NULL,
            node_name       TEXT    NOT NULL,
            timestamp_utc   TEXT    NOT NULL,
            payload         TEXT    NOT NULL,
            prev_hash       TEXT    NOT NULL,
            event_hash      TEXT    NOT NULL,
            UNIQUE(run_id, sequence_number)
        )
    """
    _CREATE_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_audit_run_seq
        ON audit_trail(run_id, sequence_number)
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.ensure_table()

    def ensure_table(self) -> None:
        self._conn.execute(self._CREATE_TABLE)
        self._conn.execute(self._CREATE_INDEX)
        self._conn.commit()

    def write(self, event: AuditEvent) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO audit_trail
            (event_id, run_id, sequence_number, event_type, actor, node_name,
             timestamp_utc, payload, prev_hash, event_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                event["run_id"],
                event["sequence_number"],
                event["event_type"],
                event["actor"],
                event["node_name"],
                event["timestamp_utc"],
                json.dumps(event["payload"], ensure_ascii=True),
                event["prev_hash"],
                event["event_hash"],
            ),
        )
        self._conn.commit()

    def get_events(self, run_id: str) -> list[AuditEvent]:
        rows = self._conn.execute(
            """
            SELECT event_id, run_id, sequence_number, event_type, actor, node_name,
                   timestamp_utc, payload, prev_hash, event_hash
            FROM audit_trail
            WHERE run_id = ?
            ORDER BY sequence_number ASC
            """,
            (run_id,),
        ).fetchall()
        return [
            AuditEvent(
                event_id=r[0],
                run_id=r[1],
                sequence_number=r[2],
                event_type=r[3],
                actor=r[4],
                node_name=r[5],
                timestamp_utc=r[6],
                payload=json.loads(r[7]),
                prev_hash=r[8],
                event_hash=r[9],
            )
            for r in rows
        ]

    def get_max_sequence(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(sequence_number) FROM audit_trail WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return row[0] if row and row[0] is not None else 0


# Phase 3: DeltaAuditStore goes here
# class DeltaAuditStore:
#     def ensure_table(self):
#         spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.ml_accelerator")
#         spark.sql("CREATE TABLE IF NOT EXISTS ml_accelerator.audit_trail (...) TBLPROPERTIES ('delta.appendOnly'='true')")
#     ...


def get_audit_store() -> AuditStore:
    """
    Return the process-level AuditStore singleton.
    Currently always SqliteAuditStore using data/checkpoints.db.
    Phase 3: add DeltaAuditStore routing when databricks-connect is available.
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    from agent.graph import _get_db_conn
    conn = _get_db_conn()
    _store_instance = SqliteAuditStore(conn)
    logger.info("AuditStore: SQLite at data/checkpoints.db")
    return _store_instance
