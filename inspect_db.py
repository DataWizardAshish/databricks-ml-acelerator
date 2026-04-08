import json
import sqlite3

conn = sqlite3.connect('./data/checkpoints.db')

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("TABLES:", [t[0] for t in tables])

for (tname,) in tables:
    cols = conn.execute(f"PRAGMA table_info({tname})").fetchall()
    count = conn.execute(f"SELECT COUNT(*) FROM [{tname}]").fetchone()[0]
    print(f"\n--- {tname} ({count} rows) ---")
    for col in cols:
        print(f"  {col[1]:30s} {col[2]}")

# Show recent run_history
if any(t[0] == 'run_history' for t in tables):
    print("\n--- run_history (last 5) ---")
    rows = conn.execute(
        "SELECT run_id, catalog, schema_name, use_case, status, updated_at "
        "FROM run_history ORDER BY updated_at DESC LIMIT 5"
    ).fetchall()
    for r in rows:
        print(f"  {r[0][:8]}... | {r[1]}.{r[2]} | {r[3] or '—'} | {r[4]} | {r[5]}")

# Show audit_trail events grouped by run
if any(t[0] == 'audit_trail' for t in tables):
    total = conn.execute("SELECT COUNT(*) FROM audit_trail").fetchone()[0]
    print(f"\n--- audit_trail ({total} total events) ---")

    runs = conn.execute(
        "SELECT DISTINCT run_id FROM audit_trail ORDER BY rowid DESC LIMIT 5"
    ).fetchall()

    for (run_id,) in runs:
        events = conn.execute(
            "SELECT sequence_number, event_type, actor, node_name, timestamp_utc, payload "
            "FROM audit_trail WHERE run_id = ? ORDER BY sequence_number",
            (run_id,)
        ).fetchall()
        print(f"\n  run: {run_id[:8]}... ({len(events)} events)")
        for e in events:
            seq, etype, actor, node, ts, payload_str = e
            ts_short = ts[11:19] if len(ts) > 19 else ts  # HH:MM:SS only
            actor_badge = "👤" if actor == "user" else "🤖"
            print(f"    [{seq:02d}] {actor_badge} {etype:<35s} {node:<30s} {ts_short}")

    # Chain validity check for most recent run
    if runs:
        latest_run_id = runs[0][0]
        events_for_chain = conn.execute(
            "SELECT event_hash, prev_hash, sequence_number FROM audit_trail "
            "WHERE run_id = ? ORDER BY sequence_number",
            (latest_run_id,)
        ).fetchall()

        import hashlib
        chain_ok = True
        broken_at = None
        for i, (ehash, phash, seq) in enumerate(events_for_chain):
            expected_prev = "GENESIS" if i == 0 else events_for_chain[i-1][0]
            if phash != expected_prev:
                chain_ok = False
                broken_at = seq
                break

        status = "✅ VALID" if chain_ok else f"❌ BROKEN at seq {broken_at}"
        print(f"\n  Hash chain (latest run): {status}")
else:
    print("\n  audit_trail table not found — run the app to generate events")

conn.close()
