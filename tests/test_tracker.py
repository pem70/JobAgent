"""Unit tests for services/tracker.py"""
from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import init_db
from services.tracker import (
    VALID_APPLICATION_STATUSES,
    append_note,
    dismiss_job,
    get_job,
    list_applications,
    set_job_interaction,
    upsert_application,
)


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    init_db(conn)
    return conn


def _insert_job(conn: sqlite3.Connection, title: str = "Engineer", company: str = "Acme") -> int:
    cursor = conn.execute(
        "INSERT INTO job_posting (title, company, url) VALUES (?, ?, ?)",
        (title, company, "https://example.com/job"),
    )
    conn.commit()
    return cursor.lastrowid


class TestGetJob(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()

    def tearDown(self) -> None:
        self.conn.close()

    def test_returns_job_dict(self) -> None:
        job_id = _insert_job(self.conn)
        with patch("services.tracker.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                job = get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["title"], "Engineer")

    def test_returns_none_for_missing(self) -> None:
        with patch("services.tracker.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                job = get_job(9999)
        self.assertIsNone(job)


class TestUpsertApplication(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()
        self.job_id = _insert_job(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def _call(self, status: str, notes: str | None = None) -> dict:
        with patch("services.tracker.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                return upsert_application(self.job_id, status, notes)

    def test_creates_new_application(self) -> None:
        result = self._call("saved")
        self.assertEqual(result["status"], "saved")
        self.assertEqual(result["job_id"], self.job_id)

    def test_updates_existing_application(self) -> None:
        self._call("saved")
        result = self._call("applied")
        self.assertEqual(result["status"], "applied")

    def test_invalid_status_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self._call("invalid_status")

    def test_nonexistent_job_raises_value_error(self) -> None:
        with patch("services.tracker.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                with self.assertRaises(ValueError):
                    upsert_application(9999, "saved")

    def test_applied_marks_job_as_tracked(self) -> None:
        self._call("applied")
        row = self.conn.execute(
            "SELECT status, interaction FROM job_posting WHERE id = ?", (self.job_id,)
        ).fetchone()
        self.assertEqual(row["status"], "tracked")
        self.assertEqual(row["interaction"], "applied")

    def test_saved_marks_job_as_tracked(self) -> None:
        self._call("saved")
        row = self.conn.execute(
            "SELECT status FROM job_posting WHERE id = ?", (self.job_id,)
        ).fetchone()
        self.assertEqual(row["status"], "tracked")

    def test_all_valid_statuses_accepted(self) -> None:
        for status in VALID_APPLICATION_STATUSES:
            result = self._call(status)
            self.assertEqual(result["status"], status)


class TestAppendNote(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()
        self.job_id = _insert_job(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def _call(self, note: str) -> dict:
        with patch("services.tracker.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                return append_note(self.job_id, note)

    def test_creates_application_with_note(self) -> None:
        result = self._call("First note")
        self.assertIn("First note", result["notes"])

    def test_appends_to_existing_note(self) -> None:
        self._call("Note one")
        result = self._call("Note two")
        self.assertIn("Note one", result["notes"])
        self.assertIn("Note two", result["notes"])

    def test_empty_note_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self._call("")

    def test_note_has_timestamp_prefix(self) -> None:
        result = self._call("Testing")
        import re
        self.assertTrue(re.search(r"\[\d{4}-\d{2}-\d{2}", result["notes"]))


class TestDismissJob(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()
        self.job_id = _insert_job(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def test_sets_status_dismissed(self) -> None:
        with patch("services.tracker.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                dismiss_job(self.job_id)
        row = self.conn.execute(
            "SELECT status, interaction FROM job_posting WHERE id = ?", (self.job_id,)
        ).fetchone()
        self.assertEqual(row["status"], "dismissed")
        self.assertEqual(row["interaction"], "dismissed")


class TestSetJobInteraction(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()
        self.job_id = _insert_job(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def _call(self, **kwargs) -> None:
        with patch("services.tracker.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                set_job_interaction(self.job_id, **kwargs)

    def test_sets_interaction(self) -> None:
        self._call(interaction="viewed")
        row = self.conn.execute(
            "SELECT interaction FROM job_posting WHERE id = ?", (self.job_id,)
        ).fetchone()
        self.assertEqual(row["interaction"], "viewed")

    def test_sets_status(self) -> None:
        self._call(status="tracked")
        row = self.conn.execute(
            "SELECT status FROM job_posting WHERE id = ?", (self.job_id,)
        ).fetchone()
        self.assertEqual(row["status"], "tracked")

    def test_no_op_when_nothing_provided(self) -> None:
        # Should not raise
        self._call()


class TestListApplications(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()

    def tearDown(self) -> None:
        self.conn.close()

    def _make_application(self, title: str, status: str) -> None:
        job_id = _insert_job(self.conn, title=title)
        self.conn.execute(
            "INSERT INTO application (job_id, status) VALUES (?, ?)", (job_id, status)
        )
        self.conn.commit()

    def test_returns_all_applications(self) -> None:
        self._make_application("Job A", "saved")
        self._make_application("Job B", "applied")
        with patch("services.tracker.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                result = list_applications()
        self.assertEqual(len(result), 2)

    def test_filters_by_status(self) -> None:
        self._make_application("Job A", "saved")
        self._make_application("Job B", "applied")
        with patch("services.tracker.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                result = list_applications(status="applied")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "applied")

    def test_includes_title_and_company(self) -> None:
        self._make_application("Engineer", "saved")
        with patch("services.tracker.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                result = list_applications()
        self.assertEqual(result[0]["title"], "Engineer")
        self.assertIn("company", result[0])


if __name__ == "__main__":
    unittest.main()
