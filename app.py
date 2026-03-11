"""
Connections Score Service
Flask HTTP API wrapping scorer.py with SQLite persistence.
"""

import os
import sqlite3
import tempfile
from datetime import date, datetime, timedelta, timezone
from contextlib import contextmanager
from zoneinfo import ZoneInfo

from flask import Flask, request, jsonify

from scorer import compute_score_from_bar_image

app = Flask(__name__)

DB_PATH = os.environ.get("DB_PATH", "/data/scores.db")
SPRINT_EPOCH_DEFAULT = "2026-02-06"
IL_TZ = ZoneInfo("Asia/Jerusalem")


def now_il() -> datetime:
    """Current datetime in Israel time."""
    return datetime.now(IL_TZ)


def today_il() -> date:
    """Current date in Israel time."""
    return now_il().date()


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


def today_il_utc_range() -> tuple[datetime, datetime]:
    """Return the UTC start and end of today in Israel time."""
    il_today = today_il()
    il_start = datetime(il_today.year, il_today.month, il_today.day, tzinfo=IL_TZ)
    utc_start = il_start.astimezone(timezone.utc)
    utc_end = utc_start + timedelta(days=1)
    return utc_start, utc_end


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS plays (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                played_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id     TEXT NOT NULL,
                user_name   TEXT NOT NULL,
                chat_id     TEXT,
                score       INTEGER,
                scan_status TEXT NOT NULL,
                sprint_id   INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS members (
                user_id   TEXT PRIMARY KEY,
                user_name TEXT NOT NULL,
                added_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO config (key, value)
            VALUES ('sprint_epoch', '""" + SPRINT_EPOCH_DEFAULT + """');
        """)
        # Migrate: add chat_id column if it doesn't exist yet
        try:
            conn.execute("ALTER TABLE plays ADD COLUMN chat_id TEXT")
        except Exception:
            pass  # Column already exists


def get_sprint_epoch():
    with db() as conn:
        row = conn.execute("SELECT value FROM config WHERE key='sprint_epoch'").fetchone()
        return date.fromisoformat(row["value"]) if row else date.fromisoformat(SPRINT_EPOCH_DEFAULT)


def current_sprint_id():
    epoch = get_sprint_epoch()
    return (today_il() - epoch).days // 14


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
        "days_remaining": (end - today_il()).days + 1,
    })


@app.post("/score")
def score():
    if "image" not in request.files:
        return jsonify({"error": "missing image field"}), 400

    user_id = request.form.get("user_id", "unknown")
    user_name = request.form.get("user_name", user_id)
    chat_id = request.form.get("chat_id")

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
            """INSERT INTO plays (user_id, user_name, chat_id, score, scan_status, sprint_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, user_name, chat_id, computed_score, status, sprint_id),
        )
        # Track member (upsert — update name in case it changed)
        conn.execute(
            """INSERT INTO members (user_id, user_name)
               VALUES (?, ?)
               ON CONFLICT(user_id) DO UPDATE SET user_name=excluded.user_name""",
            (user_id, user_name),
        )

    if status == "success":
        return jsonify({"score": computed_score, "status": "success", "sprint_id": sprint_id})
    else:
        return jsonify({"score": None, "status": status, "sprint_id": sprint_id}), 422


@app.post("/score/correct")
def correct_score():
    data = request.get_json()
    if not data:
        return jsonify({"error": "missing JSON body"}), 400

    user_id = data.get("user_id")
    new_score = data.get("score")
    chat_id = data.get("chat_id")

    if not user_id:
        return jsonify({"error": "missing user_id"}), 400
    if new_score is None:
        return jsonify({"error": "missing score"}), 400
    if not chat_id:
        return jsonify({"error": "missing chat_id"}), 400

    try:
        new_score = int(new_score)
    except (TypeError, ValueError):
        return jsonify({"error": "score must be an integer"}), 400

    if not (1 <= new_score <= 8):
        return jsonify({"error": "score must be between 1 and 8"}), 400

    utc_start, utc_end = today_il_utc_range()

    with db() as conn:
        row = conn.execute(
            """SELECT id, scan_status FROM plays
               WHERE user_id = ?
                 AND chat_id = ?
                 AND played_at >= ?
                 AND played_at <  ?
               ORDER BY played_at DESC
               LIMIT 1""",
            (user_id, chat_id,
             utc_start.strftime("%Y-%m-%d %H:%M:%S"),
             utc_end.strftime("%Y-%m-%d %H:%M:%S")),
        ).fetchone()

        if row is None:
            return jsonify({"error": "no submission found for today from this user"}), 404

        if not row["scan_status"].startswith("failed"):
            return jsonify({"error": "today's submission did not fail — correction not allowed"}), 409

        conn.execute(
            "UPDATE plays SET score = ?, scan_status = 'manual' WHERE id = ?",
            (new_score, row["id"]),
        )

    return jsonify({"score": new_score, "status": "manual"})


@app.get("/leaderboard")
def leaderboard():
    sid = resolve_sprint(request.args.get("sprint"))
    start, end = sprint_date_range(sid)
    chat_id = request.args.get("chat_id")

    query = """
        SELECT user_id,
               (SELECT user_name FROM plays p2
                WHERE p2.user_id = p.user_id
                ORDER BY played_at DESC LIMIT 1) AS user_name,
               COUNT(*) AS plays,
               SUM(score) AS total_score
        FROM plays p
        WHERE sprint_id = ? AND scan_status IN ('success', 'manual')
    """
    params = [sid]
    if chat_id:
        query += " AND chat_id = ?"
        params.append(chat_id)
    query += " GROUP BY user_id ORDER BY total_score DESC"

    with db() as conn:
        rows = conn.execute(query, params).fetchall()

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
    chat_id = request.args.get("chat_id")

    query = """
        SELECT COUNT(*) AS plays,
               SUM(CASE WHEN scan_status IN ('success', 'manual') THEN 1 ELSE 0 END) AS scored,
               SUM(score) AS total_score
        FROM plays
        WHERE user_id = ? AND sprint_id = ?
    """
    params = [user_id, sid]
    if chat_id:
        query += " AND chat_id = ?"
        params.append(chat_id)

    with db() as conn:
        row = conn.execute(query, params).fetchone()

    return jsonify({
        "user_id": user_id,
        "sprint_id": sid,
        **dict(row),
    })


@app.get("/summary")
def summary():
    sid = resolve_sprint(request.args.get("sprint"))
    start, end = sprint_date_range(sid)
    chat_id = request.args.get("chat_id")

    query = """
        SELECT user_id,
               (SELECT user_name FROM plays p2
                WHERE p2.user_id = p.user_id
                ORDER BY played_at DESC LIMIT 1) AS user_name,
               COUNT(*) AS plays,
               SUM(score) AS total_score
        FROM plays p
        WHERE sprint_id = ? AND scan_status IN ('success', 'manual')
    """
    params = [sid]
    if chat_id:
        query += " AND chat_id = ?"
        params.append(chat_id)
    query += " GROUP BY user_id ORDER BY total_score DESC"

    with db() as conn:
        rows = conn.execute(query, params).fetchall()

    lines = [f"Sprint {sid} Summary ({start} - {end}):"]
    for i, r in enumerate(rows, 1):
        lines.append(f"{i}. {r['user_name']} - {r['total_score']} pts ({r['plays']} plays)")

    return jsonify({
        "sprint_id": sid,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "text": "\n".join(lines),
        "rankings": [dict(r) for r in rows],
    })


@app.get("/config/sprint_epoch")
def get_sprint_epoch_endpoint():
    epoch = get_sprint_epoch()
    sid = current_sprint_id()
    start, end = sprint_date_range(sid)
    return jsonify({
        "sprint_epoch": epoch.isoformat(),
        "current_sprint_id": sid,
        "current_sprint_start": start.isoformat(),
        "current_sprint_end": end.isoformat(),
    })


@app.put("/config/sprint_epoch")
def set_sprint_epoch():
    data = request.get_json()
    if not data or "date" not in data:
        return jsonify({"error": "body must be {\"date\": \"YYYY-MM-DD\"}"}), 400
    try:
        new_epoch = date.fromisoformat(data["date"])
    except ValueError:
        return jsonify({"error": "invalid date format, use YYYY-MM-DD"}), 400

    with db() as conn:
        conn.execute(
            "INSERT INTO config (key, value) VALUES ('sprint_epoch', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (new_epoch.isoformat(),),
        )

    sid = (today_il() - new_epoch).days // 14
    start, end = sprint_date_range(sid)
    return jsonify({
        "sprint_epoch": new_epoch.isoformat(),
        "current_sprint_id": sid,
        "current_sprint_start": start.isoformat(),
        "current_sprint_end": end.isoformat(),
    })


@app.get("/missing")
def missing():
    """Return members who haven't submitted a screenshot today (Israel time)."""
    utc_start, utc_end = today_il_utc_range()
    date_il = today_il().isoformat()
    chat_id = request.args.get("chat_id")

    utc_start_s = utc_start.strftime("%Y-%m-%d %H:%M:%S")
    utc_end_s = utc_end.strftime("%Y-%m-%d %H:%M:%S")

    with db() as conn:
        if chat_id:
            # Members of this group = anyone who has ever played in it
            rows = conn.execute("""
                SELECT DISTINCT p.user_id,
                       (SELECT user_name FROM plays p2
                        WHERE p2.user_id = p.user_id AND p2.chat_id = ?
                        ORDER BY played_at DESC LIMIT 1) AS user_name
                FROM plays p
                WHERE p.chat_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM plays p3
                      WHERE p3.user_id = p.user_id
                        AND p3.chat_id = ?
                        AND p3.played_at >= ?
                        AND p3.played_at <  ?
                  )
                ORDER BY user_name
            """, (chat_id, chat_id, chat_id, utc_start_s, utc_end_s)).fetchall()
        else:
            rows = conn.execute("""
                SELECT m.user_id, m.user_name
                FROM members m
                WHERE NOT EXISTS (
                    SELECT 1 FROM plays p
                    WHERE p.user_id = m.user_id
                      AND p.played_at >= ?
                      AND p.played_at <  ?
                )
                ORDER BY m.user_name
            """, (utc_start_s, utc_end_s)).fetchall()

    return jsonify({
        "date": date_il,
        "missing": [dict(r) for r in rows],
        "count": len(rows),
    })


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000)
