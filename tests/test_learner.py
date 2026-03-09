"""Unit tests for services/learner.py"""
from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import init_db
from services.learner import (
    MIN_INTERACTIONS,
    _normalize_weights,
    _parse_json_list,
    _title_keywords,
    apply_learned_weights,
    load_weights_or_empty,
    mark_ignored_jobs,
    update_weights,
)


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    init_db(conn)
    return conn


class TestParseJsonList(unittest.TestCase):
    def test_valid_list(self) -> None:
        result = _parse_json_list('["python", "fastapi", "  "]')
        self.assertIn("python", result)
        self.assertIn("fastapi", result)
        self.assertNotIn("", result)  # empty stripped items removed

    def test_empty_string_returns_empty(self) -> None:
        self.assertEqual(_parse_json_list(""), [])

    def test_none_returns_empty(self) -> None:
        self.assertEqual(_parse_json_list(None), [])

    def test_non_list_json_returns_empty(self) -> None:
        self.assertEqual(_parse_json_list('{"key": "val"}'), [])

    def test_invalid_json_returns_empty(self) -> None:
        self.assertEqual(_parse_json_list("not json"), [])

    def test_lowercases_items(self) -> None:
        result = _parse_json_list('["Python", "JAVA"]')
        self.assertIn("python", result)
        self.assertIn("java", result)


class TestTitleKeywords(unittest.TestCase):
    def test_splits_on_whitespace(self) -> None:
        result = _title_keywords("Software Engineer Backend")
        self.assertIn("software", result)
        self.assertIn("engineer", result)
        self.assertIn("backend", result)

    def test_handles_slashes_and_dashes(self) -> None:
        result = _title_keywords("Full-Stack/Backend Engineer")
        self.assertIn("full", result)
        self.assertIn("stack", result)
        self.assertIn("backend", result)

    def test_empty_title_returns_empty(self) -> None:
        self.assertEqual(_title_keywords(""), [])

    def test_none_returns_empty(self) -> None:
        self.assertEqual(_title_keywords(None), [])

    def test_lowercases_tokens(self) -> None:
        result = _title_keywords("SENIOR Engineer")
        self.assertIn("senior", result)


class TestNormalizeWeights(unittest.TestCase):
    def test_scales_to_minus_one_one(self) -> None:
        weights = {"a": 10.0, "b": -5.0, "c": 2.0}
        normalized = _normalize_weights(weights)
        self.assertAlmostEqual(normalized["a"], 1.0)
        self.assertAlmostEqual(normalized["b"], -0.5)

    def test_empty_dict_returns_empty(self) -> None:
        self.assertEqual(_normalize_weights({}), {})

    def test_all_zeros_returns_zeros(self) -> None:
        weights = {"a": 0.0, "b": 0.0}
        normalized = _normalize_weights(weights)
        self.assertEqual(normalized["a"], 0.0)
        self.assertEqual(normalized["b"], 0.0)

    def test_clamps_to_range(self) -> None:
        weights = {"a": 1.0, "b": -1.0}
        normalized = _normalize_weights(weights)
        for v in normalized.values():
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)


class TestApplyLearnedWeights(unittest.TestCase):
    def test_empty_weights_returns_one(self) -> None:
        boost = apply_learned_weights({"title": "Engineer", "parsed_skills": "[]"}, {})
        self.assertAlmostEqual(boost, 1.0)

    def test_positive_skill_weight_increases_boost(self) -> None:
        weights = {"skills": {"python": 0.8}, "keywords": {}}
        job = {"title": "Engineer", "parsed_skills": json.dumps(["python"])}
        boost = apply_learned_weights(job, weights)
        self.assertGreater(boost, 1.0)

    def test_negative_skill_weight_decreases_boost(self) -> None:
        weights = {"skills": {"java": -1.0}, "keywords": {}}
        job = {"title": "Java Developer", "parsed_skills": json.dumps(["java"])}
        boost = apply_learned_weights(job, weights)
        self.assertLess(boost, 1.0)

    def test_positive_keyword_weight_increases_boost(self) -> None:
        weights = {"skills": {}, "keywords": {"senior": 1.0}}
        job = {"title": "Senior Engineer", "parsed_skills": "[]"}
        boost = apply_learned_weights(job, weights)
        self.assertGreater(boost, 1.0)

    def test_boost_clamped_to_zero_point_five_to_one_point_five(self) -> None:
        # Extreme positive weights
        weights = {"skills": {f"s{i}": 1.0 for i in range(50)}, "keywords": {}}
        skills = [f"s{i}" for i in range(50)]
        job = {"title": "Engineer", "parsed_skills": json.dumps(skills)}
        boost = apply_learned_weights(job, weights)
        self.assertLessEqual(boost, 1.5)
        self.assertGreaterEqual(boost, 0.5)

    def test_list_parsed_skills_also_works(self) -> None:
        weights = {"skills": {"python": 0.5}, "keywords": {}}
        job = {"title": "Engineer", "parsed_skills": ["python"]}
        boost = apply_learned_weights(job, weights)
        self.assertGreater(boost, 1.0)


class TestMarkIgnoredJobs(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()

    def tearDown(self) -> None:
        self.conn.close()

    def _insert_old_seen_job(self, days_ago: int = 10) -> int:
        import datetime as dt
        old_time = (dt.datetime.utcnow() - dt.timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.conn.execute(
            "INSERT INTO job_posting (title, company, status, scraped_at) VALUES (?,?,?,?)",
            ("Engineer", "Acme", "seen", old_time),
        )
        self.conn.commit()
        return cursor.lastrowid

    def test_marks_old_seen_jobs_as_ignored(self) -> None:
        job_id = self._insert_old_seen_job(days_ago=10)
        with patch("services.learner.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                count = mark_ignored_jobs(days=7)
        self.assertEqual(count, 1)
        row = self.conn.execute(
            "SELECT interaction FROM job_posting WHERE id = ?", (job_id,)
        ).fetchone()
        self.assertEqual(row["interaction"], "ignored")

    def test_does_not_mark_recent_seen_jobs(self) -> None:
        job_id = self._insert_old_seen_job(days_ago=2)
        with patch("services.learner.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                count = mark_ignored_jobs(days=7)
        self.assertEqual(count, 0)

    def test_does_not_overwrite_existing_interaction(self) -> None:
        job_id = self._insert_old_seen_job(days_ago=10)
        self.conn.execute(
            "UPDATE job_posting SET interaction = 'applied' WHERE id = ?", (job_id,)
        )
        self.conn.commit()
        with patch("services.learner.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                count = mark_ignored_jobs(days=7)
        self.assertEqual(count, 0)


class TestUpdateWeights(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()

    def tearDown(self) -> None:
        self.conn.close()

    def test_returns_empty_when_insufficient_interactions(self) -> None:
        # Insert fewer than MIN_INTERACTIONS jobs
        for i in range(5):
            self.conn.execute(
                "INSERT INTO job_posting (title, company, interaction, parsed_skills, remote) "
                "VALUES (?,?,?,?,?)",
                ("Engineer", "Acme", "viewed", '["python"]', 0),
            )
        self.conn.commit()
        with patch("services.learner.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                result = update_weights()
        self.assertEqual(result, {})

    def test_returns_empty_when_all_positive_no_negative(self) -> None:
        for i in range(MIN_INTERACTIONS):
            self.conn.execute(
                "INSERT INTO job_posting (title, company, interaction, parsed_skills, remote) "
                "VALUES (?,?,?,?,?)",
                (f"Engineer {i}", "Acme", "viewed", '["python"]', 0),
            )
        self.conn.commit()
        with patch("services.learner.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                with patch("services.learner.save_learned_weights"):
                    result = update_weights()
        # needs both positive and negative interactions
        self.assertEqual(result, {})


class TestLoadWeightsOrEmpty(unittest.TestCase):
    def test_returns_dict_when_valid(self) -> None:
        weights = {"skills": {"python": 0.5}, "keywords": {}}
        with patch("services.learner.load_learned_weights", return_value=weights):
            result = load_weights_or_empty()
        self.assertIsInstance(result, dict)
        self.assertIn("skills", result)

    def test_returns_empty_on_non_dict(self) -> None:
        with patch("services.learner.load_learned_weights", return_value=[1, 2, 3]):
            result = load_weights_or_empty()
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
