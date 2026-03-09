"""Unit tests for db.py"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_connection, init_db, SCHEMA_SQL

EXPECTED_TABLES = {"user_profile", "job_posting", "match_result", "application"}
EXPECTED_INDEXES = {"idx_job_status", "idx_job_company_title", "idx_application_status"}


def _make_conn(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


class TestInitDb(unittest.TestCase):
    def test_creates_all_required_tables(self) -> None:
        conn = _make_conn()
        init_db(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in EXPECTED_TABLES:
            self.assertIn(table, tables)
        conn.close()

    def test_creates_indexes(self) -> None:
        conn = _make_conn()
        init_db(conn)
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        for idx in EXPECTED_INDEXES:
            self.assertIn(idx, indexes)
        conn.close()

    def test_idempotent(self) -> None:
        conn = _make_conn()
        init_db(conn)
        # calling twice should not raise
        init_db(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        self.assertTrue(EXPECTED_TABLES.issubset(tables))
        conn.close()


class TestGetConnection(unittest.TestCase):
    def test_returns_connection_with_row_factory(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = get_connection(db_path)
            self.assertEqual(conn.row_factory, sqlite3.Row)
            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_foreign_keys_enabled(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = get_connection(db_path)
            fk_enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            self.assertEqual(fk_enabled, 1)
            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_tables_initialized_on_connection(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = get_connection(db_path)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertTrue(EXPECTED_TABLES.issubset(tables))
            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestJobPostingSchema(unittest.TestCase):
    """Verify schema columns and constraints."""

    def setUp(self) -> None:
        self.conn = _make_conn()
        init_db(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def test_job_posting_unique_source_external_id(self) -> None:
        self.conn.execute(
            "INSERT INTO job_posting (source, external_id, title, company) VALUES (?,?,?,?)",
            ("jsearch", "abc123", "Engineer", "Acme"),
        )
        self.conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO job_posting (source, external_id, title, company) VALUES (?,?,?,?)",
                ("jsearch", "abc123", "Engineer 2", "Acme"),
            )

    def test_job_posting_default_status_is_new(self) -> None:
        self.conn.execute(
            "INSERT INTO job_posting (title, company) VALUES (?,?)",
            ("Dev", "Corp"),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT status FROM job_posting").fetchone()
        self.assertEqual(row["status"], "new")

    def test_match_result_unique_job_id(self) -> None:
        self.conn.execute(
            "INSERT INTO job_posting (title, company) VALUES (?,?)", ("Dev", "Corp")
        )
        self.conn.commit()
        job_id = self.conn.execute("SELECT id FROM job_posting").fetchone()["id"]
        self.conn.execute(
            "INSERT INTO match_result (job_id, final_score, embedding_similarity, keyword_score) VALUES (?,?,?,?)",
            (job_id, 0.9, 0.8, 0.7),
        )
        self.conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO match_result (job_id, final_score, embedding_similarity, keyword_score) VALUES (?,?,?,?)",
                (job_id, 0.5, 0.4, 0.3),
            )


if __name__ == "__main__":
    unittest.main()
