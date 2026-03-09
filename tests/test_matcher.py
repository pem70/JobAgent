"""Unit tests for services/matcher.py"""
from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import init_db
from services.matcher import (
    compute_final_scores,
    compute_mechanical_score,
    layer1_hard_filter,
    layer3_llm_rerank,
)


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    init_db(conn)
    return conn


def _insert_job(conn: sqlite3.Connection, **kwargs) -> int:
    defaults: dict = {
        "title": "Software Engineer",
        "company": "Acme",
        "status": "new",
        "salary_max": None,
        "remote": 0,
        "keyword_score": 0.5,
    }
    defaults.update(kwargs)
    cursor = conn.execute(
        """INSERT INTO job_posting (title, company, status, salary_max, remote, keyword_score)
           VALUES (:title, :company, :status, :salary_max, :remote, :keyword_score)""",
        defaults,
    )
    conn.commit()
    return cursor.lastrowid


def _profile(**kwargs) -> dict:
    base = {
        "min_salary": 0,
        "location_pref": "",
        "remote_ok": True,
        "yoe": 3,
        "skills": ["python", "fastapi"],
        "name": "Alice",
        "target_roles": ["software engineer"],
        "resume_text": "experienced backend developer",
    }
    base.update(kwargs)
    return base


class TestComputeMechanicalScore(unittest.TestCase):
    def test_full_match_returns_near_one(self) -> None:
        profile = _profile(min_salary=80000, remote_ok=True, yoe=5)
        job = {
            "salary_max": 120000,
            "remote": 1,
            "min_yoe": 3,
            "parsed_skills": json.dumps(["python", "fastapi"]),
            "keyword_score": 1.0,
        }
        score = compute_mechanical_score(job, profile)
        self.assertGreater(score, 0.7)

    def test_salary_below_min_gives_zero_salary_component(self) -> None:
        profile = _profile(min_salary=150000)
        job = {
            "salary_max": 80000,
            "remote": 0,
            "min_yoe": None,
            "parsed_skills": "[]",
            "keyword_score": 0.0,
        }
        score = compute_mechanical_score(job, profile)
        # salary_score = 0.0, contributes 0/4 to total
        self.assertLess(score, 0.5)

    def test_null_salary_gives_neutral_salary_score(self) -> None:
        profile = _profile(min_salary=100000)
        job_with_null = {
            "salary_max": None, "remote": 0, "min_yoe": None,
            "parsed_skills": "[]", "keyword_score": 0.0,
        }
        job_with_sufficient = {
            "salary_max": 120000, "remote": 0, "min_yoe": None,
            "parsed_skills": "[]", "keyword_score": 0.0,
        }
        score_null = compute_mechanical_score(job_with_null, profile)
        score_sufficient = compute_mechanical_score(job_with_sufficient, profile)
        # null salary = 0.5 (neutral), sufficient = 1.0
        self.assertLess(score_null, score_sufficient)

    def test_remote_mismatch_reduces_score(self) -> None:
        profile = _profile(remote_ok=True)
        job_remote = {"salary_max": None, "remote": 1, "min_yoe": None, "parsed_skills": "[]", "keyword_score": 0.5}
        job_onsite = {"salary_max": None, "remote": 0, "min_yoe": None, "parsed_skills": "[]", "keyword_score": 0.5}
        self.assertGreater(
            compute_mechanical_score(job_remote, profile),
            compute_mechanical_score(job_onsite, profile),
        )

    def test_yoe_sufficient_gives_full_yoe_score(self) -> None:
        profile = _profile(yoe=5)
        job = {"salary_max": None, "remote": 0, "min_yoe": 3, "parsed_skills": "[]", "keyword_score": 0.0}
        # user has 5 yoe, job requires 3 — should pass
        score = compute_mechanical_score(job, profile)
        # yoe_score == 1.0 contributes to total
        self.assertGreater(score, 0.0)

    def test_yoe_insufficient_reduces_score(self) -> None:
        profile = _profile(yoe=1)
        job_ok = {"salary_max": None, "remote": 0, "min_yoe": 1, "parsed_skills": "[]", "keyword_score": 0.0}
        job_too_high = {"salary_max": None, "remote": 0, "min_yoe": 5, "parsed_skills": "[]", "keyword_score": 0.0}
        self.assertGreater(
            compute_mechanical_score(job_ok, profile),
            compute_mechanical_score(job_too_high, profile),
        )

    def test_skill_overlap_raises_score(self) -> None:
        profile = _profile(skills=["python", "fastapi", "postgresql"])
        job_match = {
            "salary_max": None, "remote": 0, "min_yoe": None,
            "parsed_skills": json.dumps(["python", "fastapi"]),
            "keyword_score": 0.0,
        }
        job_no_match = {
            "salary_max": None, "remote": 0, "min_yoe": None,
            "parsed_skills": json.dumps(["rust", "haskell"]),
            "keyword_score": 0.0,
        }
        self.assertGreater(
            compute_mechanical_score(job_match, profile),
            compute_mechanical_score(job_no_match, profile),
        )

    def test_returns_float_in_zero_one_range(self) -> None:
        profile = _profile()
        job = {"salary_max": 200000, "remote": 1, "min_yoe": 0, "parsed_skills": "[]", "keyword_score": 1.0}
        score = compute_mechanical_score(job, profile)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestLayer1HardFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()

    def tearDown(self) -> None:
        self.conn.close()

    def test_returns_new_jobs_by_default(self) -> None:
        _insert_job(self.conn, status="new")
        _insert_job(self.conn, status="seen")
        with patch("services.matcher.get_connection", return_value=self.conn):
            # Prevent connection close in layer1_hard_filter
            with patch.object(self.conn, "close"):
                results = layer1_hard_filter(_profile(), include_seen=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "new")

    def test_include_seen_returns_both(self) -> None:
        _insert_job(self.conn, status="new")
        _insert_job(self.conn, status="seen")
        with patch("services.matcher.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                results = layer1_hard_filter(_profile(), include_seen=True)
        self.assertEqual(len(results), 2)

    def test_salary_filter_excludes_below_min(self) -> None:
        _insert_job(self.conn, salary_max=50000, status="new")
        _insert_job(self.conn, salary_max=150000, status="new")
        _insert_job(self.conn, salary_max=None, status="new")
        profile = _profile(min_salary=100000)
        with patch("services.matcher.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                results = layer1_hard_filter(profile, include_seen=False)
        # Only salary_max=150000 and salary_max=None should pass
        self.assertEqual(len(results), 2)

    def test_dismissed_jobs_excluded(self) -> None:
        _insert_job(self.conn, status="dismissed")
        with patch("services.matcher.get_connection", return_value=self.conn):
            with patch.object(self.conn, "close"):
                results = layer1_hard_filter(_profile())
        self.assertEqual(len(results), 0)


class TestComputeFinalScores(unittest.TestCase):
    def _job(self, job_id: int, mechanical: float, llm: float | None = None, keyword: float = 0.5) -> dict:
        return {
            "id": job_id,
            "mechanical_score": mechanical,
            "llm_score": llm,
            "keyword_score": keyword,
        }

    def test_sorted_descending_by_score(self) -> None:
        jobs = [self._job(1, 0.3), self._job(2, 0.8), self._job(3, 0.5)]
        ranked = compute_final_scores(jobs, llm_enabled=False)
        scores = [j["final_score"] for j in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_llm_enabled_uses_llm_formula(self) -> None:
        job_with_llm = self._job(1, 0.5, llm=80.0)
        job_no_llm = self._job(2, 0.5, llm=None)
        ranked_with = compute_final_scores([job_with_llm], llm_enabled=True)
        ranked_without = compute_final_scores([job_no_llm], llm_enabled=True)
        # job with llm_score=80 → 0.3*0.5 + 0.7*0.8 = 0.71
        self.assertAlmostEqual(ranked_with[0]["final_score"], 0.3 * 0.5 + 0.7 * 0.8, places=5)
        # job without llm falls back to mechanical only
        self.assertAlmostEqual(ranked_without[0]["final_score"], 0.5, places=5)

    def test_llm_disabled_uses_mechanical_only(self) -> None:
        job = self._job(1, 0.6, llm=90.0)
        ranked = compute_final_scores([job], llm_enabled=False)
        self.assertAlmostEqual(ranked[0]["final_score"], 0.6, places=5)

    def test_sets_final_score_on_each_job(self) -> None:
        jobs = [self._job(i, 0.5) for i in range(5)]
        ranked = compute_final_scores(jobs, llm_enabled=False)
        for job in ranked:
            self.assertIn("final_score", job)


class TestLayer3LlmRerank(unittest.TestCase):
    def _job(self, job_id: int, title: str = "Engineer") -> dict:
        return {
            "id": job_id,
            "title": title,
            "company": "Acme",
            "description": "Build software",
            "parsed_skills": '["python"]',
        }

    def test_returns_all_input_jobs(self) -> None:
        jobs = [self._job(i) for i in range(5)]
        with patch("services.matcher.call_kimi_rerank", return_value=[]):
            result = layer3_llm_rerank(jobs, _profile(), top_n=5)
        self.assertEqual(len(result), 5)

    def test_only_top_n_sent_to_llm(self) -> None:
        jobs = [self._job(i) for i in range(10)]
        captured = []

        def fake_rerank(profile_summary, jobs_text):
            captured.append(jobs_text)
            return []

        with patch("services.matcher.call_kimi_rerank", side_effect=fake_rerank):
            layer3_llm_rerank(jobs, _profile(), top_n=3)

        # Only first 3 jobs should be in the LLM prompt
        self.assertEqual(len(captured), 1)
        self.assertIn("job_index=0", captured[0])
        self.assertIn("job_index=1", captured[0])
        self.assertIn("job_index=2", captured[0])
        self.assertNotIn("job_index=3", captured[0])

    def test_llm_scores_applied_to_correct_jobs(self) -> None:
        jobs = [self._job(10), self._job(20)]
        llm_response = [
            {"job_index": 0, "score": 85, "rationale": "Great fit", "skill_gaps": [], "red_flags": []},
            {"job_index": 1, "score": 40, "rationale": "Weak match", "skill_gaps": ["go"], "red_flags": []},
        ]
        with patch("services.matcher.call_kimi_rerank", return_value=llm_response):
            result = layer3_llm_rerank(jobs, _profile(), top_n=5)
        result_by_id = {j["id"]: j for j in result}
        self.assertAlmostEqual(result_by_id[10]["llm_score"], 85.0)
        self.assertAlmostEqual(result_by_id[20]["llm_score"], 40.0)

    def test_empty_jobs_returns_empty(self) -> None:
        with patch("services.matcher.call_kimi_rerank", return_value=[]):
            result = layer3_llm_rerank([], _profile(), top_n=5)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
