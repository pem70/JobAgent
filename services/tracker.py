from __future__ import annotations

import datetime as dt
import webbrowser
from typing import Any

from db import get_connection

VALID_APPLICATION_STATUSES = {"saved", "applied", "interviewing", "offer", "rejected"}


def get_job(job_id: int) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM job_posting WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def open_job_url(job_id: int) -> str:
    job = get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found.")
    url = str(job.get("url") or "").strip()
    if not url:
        raise ValueError(f"Job {job_id} has no URL.")
    webbrowser.open(url)
    return url


def set_job_interaction(job_id: int, interaction: str | None = None, status: str | None = None) -> None:
    conn = get_connection()
    try:
        updates: list[str] = []
        params: list[Any] = []
        if interaction is not None:
            updates.append("interaction = ?")
            params.append(interaction)
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if not updates:
            return
        params.append(job_id)
        conn.execute(f"UPDATE job_posting SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    finally:
        conn.close()


def upsert_application(job_id: int, status: str, notes: str | None = None) -> dict[str, Any]:
    normalized = (status or "").strip().lower()
    if normalized not in VALID_APPLICATION_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {', '.join(sorted(VALID_APPLICATION_STATUSES))}")

    conn = get_connection()
    try:
        job = conn.execute("SELECT id FROM job_posting WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        conn.execute(
            """
            INSERT INTO application (job_id, status, notes)
            VALUES (?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status = excluded.status,
                notes = COALESCE(excluded.notes, application.notes),
                updated_at = datetime('now')
            """,
            (job_id, normalized, notes),
        )

        if normalized in {"applied", "interviewing", "offer", "rejected"}:
            conn.execute(
                "UPDATE job_posting SET status = 'tracked', interaction = 'applied' WHERE id = ?",
                (job_id,),
            )
        elif normalized == "saved":
            conn.execute(
                "UPDATE job_posting SET status = 'tracked' WHERE id = ?",
                (job_id,),
            )

        row = conn.execute("SELECT * FROM application WHERE job_id = ?", (job_id,)).fetchone()
        conn.commit()
        return dict(row) if row else {}
    finally:
        conn.close()


def append_note(job_id: int, note_text: str) -> dict[str, Any]:
    note = (note_text or "").strip()
    if not note:
        raise ValueError("Note text cannot be empty.")
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    stamped = f"[{ts}] {note}"

    conn = get_connection()
    try:
        existing = conn.execute("SELECT notes FROM application WHERE job_id = ?", (job_id,)).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO application (job_id, status, notes) VALUES (?, 'saved', ?)",
                (job_id, stamped),
            )
            conn.execute("UPDATE job_posting SET status = 'tracked' WHERE id = ?", (job_id,))
        else:
            prev = str(existing["notes"] or "").strip()
            merged = stamped if not prev else f"{prev}\n{stamped}"
            conn.execute(
                "UPDATE application SET notes = ?, updated_at = datetime('now') WHERE job_id = ?",
                (merged, job_id),
            )
        row = conn.execute("SELECT * FROM application WHERE job_id = ?", (job_id,)).fetchone()
        conn.commit()
        return dict(row) if row else {}
    finally:
        conn.close()


def list_applications(status: str | None = None) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        sql = """
            SELECT a.job_id, j.title, j.company, a.status, a.updated_at, a.notes
            FROM application a
            JOIN job_posting j ON j.id = a.job_id
            WHERE 1=1
        """
        params: list[Any] = []
        if status:
            sql += " AND a.status = ?"
            params.append(status.strip().lower())
        sql += " ORDER BY a.updated_at DESC"
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def dismiss_job(job_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE job_posting SET status = 'dismissed', interaction = 'dismissed' WHERE id = ?",
            (job_id,),
        )
        conn.commit()
    finally:
        conn.close()
