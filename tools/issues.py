"""Shared issue tracker (SQLite) for the error-check agent and the main agent.

The error-check agent only ADDS issues. The main agent decides the status.
Duplicates (same file + fingerprint) are not re-added, so the same finding is never reported twice.

CLI:
  python tools/issues.py add <file> <severity> <fingerprint> <description>
  python tools/issues.py list [status]
  python tools/issues.py resolve <id> <fixed|wontfix|later|ignore> [note]
"""
import sqlite3, sys, time
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "issues.db"
STATUSES = ("open", "fixed", "wontfix", "later", "ignore")


def conn():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS issues(
        id INTEGER PRIMARY KEY, file TEXT, severity TEXT, fingerprint TEXT,
        description TEXT, status TEXT DEFAULT 'open', note TEXT,
        created REAL, updated REAL, UNIQUE(file, fingerprint))""")
    return c


def add(file, severity, fingerprint, description):
    c = conn()
    try:
        c.execute("INSERT INTO issues(file,severity,fingerprint,description,created,updated) VALUES(?,?,?,?,?,?)",
                  (file, severity, fingerprint, description, time.time(), time.time()))
        c.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # already known, whatever its status


def resolve(issue_id, status, note=""):
    assert status in STATUSES
    c = conn()
    c.execute("UPDATE issues SET status=?, note=?, updated=? WHERE id=?", (status, note, time.time(), issue_id))
    c.commit()


def rows(status=None):
    c = conn()
    q = "SELECT id,status,severity,file,description,note FROM issues"
    return c.execute(q + (" WHERE status=?" if status else "") + " ORDER BY id", (status,) if status else ()).fetchall()


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["add"]:
        print("added" if add(*a[1:5]) else "duplicate, skipped")
    elif a[:1] == ["list"]:
        for r in rows(a[1] if len(a) > 1 else None):
            print(r)
    elif a[:1] == ["resolve"]:
        resolve(int(a[1]), a[2], " ".join(a[3:]))
    else:
        print(__doc__)
