"""
Connections Score Service
Flask HTTP API wrapping scorer.py with SQLite persistence.
"""

import os
import sqlite3
import tempfile
from datetime import date, datetime, timedelta
from contextlib import contextmanager

from flask import Flask, request, jsonify

from scorer import compute_score_from_bar_image

app = Flask(__name__)

DB_PATH = os.environ.get("DB_PATH", "/data/scores.db")
SPRINT_EPOCH_DEFAULT = "2026-03-07"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db():
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS plays (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                played_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id     TEXT NOT NULL,
                user_name   TEXT NOT NULL,
                score       INTEGER,
                scan_status TEXT NOT NULL,
                sprint_id   INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO config (key, value)
            VALUES ('sprint_epoch', '""" + SPRINT_EPOCH_DEFAULT + """');
        """)


def get_sprint_epoch():
    with db() as conn:
        row = conn.execute("SELECT value FROM config WHERE key='sprint_epoch'").fetchone()
        return date.fromisoformat(row["value"]) if row else date.fromisoformat(SPRINT_EPOCH_DEFAULT)


def current_sprint_id():
    epoch = get_sprint_epoch()
    return (date.today() - epoch).days // 14


def sprint_date_range(sprint_id):
    epoch = get_sprint_epoch()
    start = epoch + timedelta(days=sprint_id * 14)
    end = start + timedelta(days=13)
    return start, end


def resolve_sprint(sprint_param):
    if sprint_param is None or sprint_param == "current":
        return current_sprint_id()
    return int(sprint_param)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/sprint")
def sprint_info():
    sid = current_sprint_id()
    start, end = sprint_date_range(sid)
    return jsonify({
        "sprint_id": sid,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days_remaining": (end - date.today()).days + 1,
    })


@app.post("/score")
def score():
    if "image" not in request.files:
        return jsonify({"error": "missing image field"}), 400

    user_id = request.form.get("user_id", "unknown")
    user_name = request.form.get("user_name", user_id)

    image_file = request.files["image"]
    suffix = os.path.splitext(image_file.filename or "img.jpg")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        image_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        computed_score, status = compute_score_from_bar_image(tmp_path, save_debug=False)
    finally:
        os.unlink(tmp_path)

    sprint_id = current_sprint_id()

    with db() as conn:
        conn.execute(
            """INSERT INTO plays (user_id, user_name, score, scan_status, sprint_id)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, user_name, computed_score, status, sprint_id),
        )

    if status == "success":
        return jsonify({"score": computed_score, "status": "success", "sprint_id": sprint_id})
    else:
        return jsonify({"score": None, "status": status, "sprint_id": sprint_id}), 422


@app.get("/leaderboard")
def leaderboard():
    sid = resolve_sprint(request.args.get("sprint"))
    start, end = sprint_date_range(sid)

    with db() as conn:
        rows = conn.execute("""
            SELECT user_name,
                   COUNT(*) AS plays,
                   SUM(score) AS total_score,
                   MAX(score) AS best_score
            FROM plays
            WHERE sprint_id = ? AND scan_status = 'success'
            GROUP BY user_id, user_name
            ORDER BY total_score DESC
        """, (sid,)).fetchall()

    return jsonify({
        "sprint_id": sid,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "leaderboard": [dict(r) for r in rows],
    })


@app.get("/stats")
def stats():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "missing user_id"}), 400

    sid = resolve_sprint(request.args.get("sprint"))

    with db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS plays,
                   SUM(CASE WHEN scan_status='success' THEN 1 ELSE 0 END) AS scored,
                   SUM(score) AS total_score,
                   MAX(score) AS best_score
            FROM plays
            WHERE user_id = ? AND sprint_id = ?
        """, (user_id, sid)).fetchone()

    return jsonify({
        "user_id": user_id,
        "sprint_id": sid,
        **dict(row),
    })


@app.get("/summary")
def summary():
    sid = resolve_sprint(request.args.get("sprint"))
    start, end = sprint_date_range(sid)

    with db() as conn:
        rows = conn.execute("""
            SELECT user_name,
                   COUNT(*) AS plays,
                   SUM(score) AS total_score,
                   MAX(score) AS best_score
            FROM plays
            WHERE sprint_id = ? AND scan_status = 'success'
            GROUP BY user_id, user_name
            ORDER BY total_score DESC
        """, (sid,)).fetchall()

    lines = [f"Sprint {sid} Summary ({start} - {end}):"]
    for i, r in enumerate(rows, 1):
        lines.append(f"{i}. {r['user_name']} - {r['total_score']} pts ({r['plays']} plays, best: {r['best_score']})")

    return jsonify({
        "sprint_id": sid,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "text": "\n".join(lines),
        "rankings": [dict(r) for r in rows],
    })


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000)
