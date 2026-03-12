"""Unit tests for services/profile.py"""
from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import init_db
from services.profile import (
    _exclude_terms,
    _role_expansions,
    _serialize_profile_for_db,
    _split_csv,
    _unique_preserve_order,
    get_profile,
    update_profile_fields,
    upsert_profile,
)


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    init_db(conn)
    return conn


def _base_profile(**kwargs) -> dict:
    base = {
        "id": 1,
        "name": "Alice",
        "resume_text": "Backend developer with Python experience.",
        "skills": ["python", "fastapi"],
        "target_roles": ["software engineer"],
        "yoe": 3,
        "location_pref": "Austin",
        "remote_ok": True,
        "min_salary": 80000,
        "deal_breakers": [],
    }
    base.update(kwargs)
    return base


class TestSplitCsv(unittest.TestCase):
    def test_splits_comma_separated(self) -> None:
        result = _split_csv("python, fastapi, postgresql")
        self.assertEqual(result, ["python", "fastapi", "postgresql"])

    def test_strips_whitespace(self) -> None:
        result = _split_csv("  python  ,  java  ")
        self.assertEqual(result, ["python", "java"])

    def test_empty_string_returns_empty(self) -> None:
        self.assertEqual(_split_csv(""), [])

    def test_none_returns_empty(self) -> None:
        self.assertEqual(_split_csv(None), [])

    def test_single_item(self) -> None:
        self.assertEqual(_split_csv("python"), ["python"])


class TestUniquePreserveOrder(unittest.TestCase):
    def test_removes_duplicates(self) -> None:
        result = _unique_preserve_order(["python", "fastapi", "python"])
        self.assertEqual(result.count("python"), 1)

    def test_preserves_order(self) -> None:
        result = _unique_preserve_order(["b", "a", "c"])
        self.assertEqual(result, ["b", "a", "c"])

    def test_case_insensitive_dedup(self) -> None:
        result = _unique_preserve_order(["Python", "python"])
        self.assertEqual(len(result), 1)

    def test_removes_empty_strings(self) -> None:
        result = _unique_preserve_order(["python", "", "  "])
        self.assertNotIn("", result)
        self.assertNotIn("  ", result)


class TestRoleExpansions(unittest.TestCase):
    def setUp(self) -> None:
        self.synonyms = {
            "software engineer": ["swe", "software dev"],
            "backend engineer": ["backend dev"],
        }

    def test_includes_original_role(self) -> None:
        result = _role_expansions(["software engineer"], self.synonyms)
        self.assertIn("software engineer", result)

    def test_expands_synonyms(self) -> None:
        result = _role_expansions(["software engineer"], self.synonyms)
        self.assertIn("swe", result)
        self.assertIn("software dev", result)

    def test_multiple_roles_expanded(self) -> None:
        result = _role_expansions(["software engineer", "backend engineer"], self.synonyms)
        self.assertIn("swe", result)
        self.assertIn("backend dev", result)

    def test_deduplicates_results(self) -> None:
        result = _role_expansions(["software engineer"], self.synonyms)
        self.assertEqual(len(result), len(set(r.lower() for r in result)))

    def test_unknown_role_returns_as_is(self) -> None:
        result = _role_expansions(["data engineer"], self.synonyms)
        self.assertIn("data engineer", result)


class TestExcludeTerms(unittest.TestCase):
    def test_junior_excludes_senior_titles(self) -> None:
        terms = _exclude_terms(yoe=1, deal_breakers=[])
        self.assertIn("Staff", terms)
        self.assertIn("Lead", terms)
        self.assertIn("Manager", terms)

    def test_senior_uses_different_set(self) -> None:
        terms = _exclude_terms(yoe=5, deal_breakers=[])
        self.assertIn("Principal", terms)
        self.assertNotIn("Staff", terms)

    def test_deal_breakers_appended(self) -> None:
        terms = _exclude_terms(yoe=1, deal_breakers=["relocation"])
        self.assertIn("relocation", terms)

    def test_no_duplicates_in_result(self) -> None:
        terms = _exclude_terms(yoe=1, deal_breakers=["Staff"])
        self.assertEqual(terms.count("Staff"), 1)


class TestSerializeProfileForDb(unittest.TestCase):
    def test_skills_serialized_as_json(self) -> None:
        profile = _base_profile()
        payload = _serialize_profile_for_db(profile)
        parsed = json.loads(payload["skills"])
        self.assertIsInstance(parsed, list)

    def test_remote_ok_stored_as_int(self) -> None:
        profile_true = _base_profile(remote_ok=True)
        profile_false = _base_profile(remote_ok=False)
        self.assertEqual(_serialize_profile_for_db(profile_true)["remote_ok"], 1)
        self.assertEqual(_serialize_profile_for_db(profile_false)["remote_ok"], 0)


class TestGetProfileAndUpsert(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()

    def tearDown(self) -> None:
        self.conn.close()

    def test_get_profile_returns_none_when_empty(self) -> None:
        with patch("services.profile.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                result = get_profile()
        self.assertIsNone(result)

    def test_upsert_and_get_roundtrip(self) -> None:
        profile = _base_profile()
        with patch("services.profile.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                saved = upsert_profile(profile, conn=self.conn)

        self.assertEqual(saved["name"], "Alice")
        self.assertEqual(saved["yoe"], 3)
        self.assertEqual(saved["min_salary"], 80000)
        self.assertIn("python", saved["skills"])

    def test_upsert_updates_existing_profile(self) -> None:
        profile = _base_profile()
        upsert_profile(profile, conn=self.conn)

        updated = _base_profile(name="Bob", yoe=5)
        saved = upsert_profile(updated, conn=self.conn)
        self.assertEqual(saved["name"], "Bob")
        self.assertEqual(saved["yoe"], 5)

    def test_deal_breakers_preserved(self) -> None:
        profile = _base_profile(deal_breakers=["relocation", "on-call"])
        saved = upsert_profile(profile, conn=self.conn)
        self.assertIn("relocation", saved["deal_breakers"])
        self.assertIn("on-call", saved["deal_breakers"])


class TestUpdateProfileFields(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()
        # Seed a profile
        upsert_profile(_base_profile(), conn=self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def _call(self, **kwargs) -> tuple:
        with patch("services.profile.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                with patch("services.profile.load_config", return_value={}):
                    with patch("services.profile.load_synonyms", return_value={"role_synonyms": {}, "skill_synonyms": {}}):
                        with patch("services.profile.save_config"):
                            return update_profile_fields(**kwargs)

    def test_updates_skills(self) -> None:
        saved, _, changed = self._call(skills="go, rust")
        self.assertIn("go", saved["skills"])
        self.assertIn("rust", saved["skills"])
        self.assertIn("skills", changed)

    def test_updates_salary(self) -> None:
        saved, _, changed = self._call(salary=150000)
        self.assertEqual(saved["min_salary"], 150000)
        self.assertIn("min_salary", changed)

    def test_updates_location(self) -> None:
        saved, _, changed = self._call(location="Seattle")
        self.assertEqual(saved["location_pref"], "Seattle")
        self.assertIn("location_pref", changed)

    def test_updates_remote(self) -> None:
        saved, _, changed = self._call(remote=False)
        self.assertFalse(saved["remote_ok"])
        self.assertIn("remote_ok", changed)

    def test_raises_when_no_profile(self) -> None:
        empty_conn = _make_db()
        with patch("services.profile.get_connection", return_value=empty_conn):
            with patch.object(empty_conn, "close"):
                with self.assertRaises(ValueError):
                    update_profile_fields(salary=100000)
        empty_conn.close()

    def test_unchanged_fields_not_in_changed_set(self) -> None:
        _, _, changed = self._call(salary=90000)
        self.assertNotIn("skills", changed)
        self.assertNotIn("yoe", changed)


if __name__ == "__main__":
    unittest.main()
