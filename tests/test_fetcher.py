"""Unit tests for services/fetcher.py"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import init_db
from services.fetcher import (
    _expand_title_terms,
    build_queries,
    ingest_jobs,
    is_duplicate,
    keyword_filter,
    normalize_jsearch_job,
)


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    init_db(conn)
    return conn


def _insert_job(conn: sqlite3.Connection, **kwargs) -> None:
    defaults = {
        "source": "jsearch",
        "external_id": "ext1",
        "title": "Software Engineer",
        "company": "Acme",
        "location": "Austin, TX",
        "scraped_at": dt.datetime.utcnow().isoformat(),
    }
    defaults.update(kwargs)
    conn.execute(
        """INSERT INTO job_posting (source, external_id, title, company, location, scraped_at)
           VALUES (:source, :external_id, :title, :company, :location, :scraped_at)""",
        defaults,
    )
    conn.commit()


class TestBuildQueries(unittest.TestCase):
    def _cfg(self, roles=None, skills=None, max_q=5) -> dict:
        return {
            "must_have_roles": roles or ["software engineer"],
            "nice_to_have_skills": skills or ["Python", "FastAPI"],
            "max_queries_per_fetch": max_q,
        }

    def test_generates_role_skill_combinations(self) -> None:
        queries = build_queries(self._cfg())
        self.assertGreater(len(queries), 0)
        # Each query should contain a role
        for q in queries:
            self.assertTrue(any(role in q for role in ["software engineer"]))

    def test_respects_max_queries_limit(self) -> None:
        cfg = self._cfg(
            roles=["engineer", "developer", "analyst", "designer", "manager"],
            skills=["Python", "Java"],
            max_q=3,
        )
        queries = build_queries(cfg)
        self.assertLessEqual(len(queries), 3)

    def test_deduplicates_queries(self) -> None:
        # Same role listed twice
        cfg = self._cfg(roles=["software engineer", "Software Engineer"])
        queries = build_queries(cfg)
        lowered = [q.lower() for q in queries]
        self.assertEqual(len(lowered), len(set(lowered)))

    def test_empty_roles_returns_empty(self) -> None:
        cfg = self._cfg(roles=[])
        queries = build_queries(cfg)
        self.assertEqual(queries, [])

    def test_no_skills_uses_role_only(self) -> None:
        cfg = self._cfg(roles=["backend engineer"], skills=[])
        queries = build_queries(cfg)
        self.assertEqual(queries, ["backend engineer"])


class TestKeywordFilter(unittest.TestCase):
    def _cfg(self) -> dict:
        return {
            "must_have_roles": ["software engineer", "backend engineer"],
            "nice_to_have_skills": ["python", "fastapi", "postgresql"],
            "exclude_terms": ["staff", "principal", "director"],
        }

    def _synonyms(self) -> dict:
        return {"role_synonyms": {}, "skill_synonyms": {}}

    def test_matching_title_passes(self) -> None:
        job = {"title": "Software Engineer", "description": "Python fastapi"}
        keep, score = keyword_filter(job, self._cfg(), self._synonyms())
        self.assertTrue(keep)
        self.assertGreater(score, 0)

    def test_excluded_title_rejected(self) -> None:
        job = {"title": "Staff Software Engineer", "description": "Python"}
        keep, score = keyword_filter(job, self._cfg(), self._synonyms())
        self.assertFalse(keep)
        self.assertEqual(score, 0.0)

    def test_empty_title_rejected(self) -> None:
        job = {"title": "", "description": "Python"}
        keep, score = keyword_filter(job, self._cfg(), self._synonyms())
        self.assertFalse(keep)

    def test_score_based_on_nice_to_have_skills(self) -> None:
        # 2 of 3 nice-to-have skills present
        job = {
            "title": "Software Engineer",
            "description": "we use python and fastapi here",
        }
        keep, score = keyword_filter(job, self._cfg(), self._synonyms())
        self.assertTrue(keep)
        self.assertAlmostEqual(score, 2 / 3, places=2)

    def test_no_title_match_still_passes_with_low_score(self) -> None:
        # Bug note: jobs without title match return (True, 0.05) by design
        job = {"title": "Marketing Manager", "description": "Python"}
        keep, score = keyword_filter(job, self._cfg(), self._synonyms())
        self.assertTrue(keep)
        self.assertEqual(score, 0.05)

    def test_synonym_expansion_allows_match(self) -> None:
        synonyms = {"role_synonyms": {"software engineer": ["swe"]}, "skill_synonyms": {}}
        job = {"title": "SWE", "description": "python"}
        keep, score = keyword_filter(job, self._cfg(), synonyms)
        self.assertTrue(keep)


class TestIsDuplicate(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()

    def tearDown(self) -> None:
        self.conn.close()

    def test_exact_match_is_duplicate(self) -> None:
        _insert_job(self.conn, company="Acme", title="Software Engineer", location="Austin")
        job = {"company": "Acme", "title": "Software Engineer", "location": "Austin"}
        self.assertTrue(is_duplicate(job, self.conn))

    def test_completely_different_is_not_duplicate(self) -> None:
        _insert_job(self.conn, company="Acme", title="Software Engineer")
        job = {"company": "BetaCorp", "title": "Data Scientist", "location": "NYC"}
        self.assertFalse(is_duplicate(job, self.conn))

    def test_empty_company_is_duplicate(self) -> None:
        job = {"company": "", "title": "Engineer", "location": ""}
        self.assertTrue(is_duplicate(job, self.conn))

    def test_empty_title_is_duplicate(self) -> None:
        job = {"company": "Acme", "title": "", "location": ""}
        self.assertTrue(is_duplicate(job, self.conn))

    def test_fuzzy_match_is_duplicate(self) -> None:
        _insert_job(self.conn, company="Acme Corp", title="Software Engineer", location="Austin TX")
        job = {"company": "Acme Corp", "title": "Software Engineer", "location": "Austin, TX"}
        self.assertTrue(is_duplicate(job, self.conn))


class TestNormalizeJsearchJob(unittest.TestCase):
    def _raw(self, **kwargs) -> dict:
        base = {
            "job_id": "ext123",
            "job_title": "Engineer",
            "employer_name": "Acme",
            "job_location": "Austin, TX",
            "job_is_remote": False,
            "job_description": "Build stuff",
            "job_apply_link": "https://example.com/apply",
            "job_min_salary": 80000,
            "job_max_salary": 120000,
            "job_salary_period": "YEAR",
            "job_posted_at_datetime_utc": "2024-01-01T00:00:00Z",
        }
        base.update(kwargs)
        return base

    def test_maps_standard_fields(self) -> None:
        job = normalize_jsearch_job(self._raw())
        self.assertEqual(job["title"], "Engineer")
        self.assertEqual(job["company"], "Acme")
        self.assertEqual(job["location"], "Austin, TX")
        self.assertEqual(job["source"], "jsearch")
        self.assertEqual(job["external_id"], "ext123")

    def test_annualizes_hourly_salary(self) -> None:
        job = normalize_jsearch_job(self._raw(job_min_salary=50, job_max_salary=75, job_salary_period="HOUR"))
        self.assertEqual(job["salary_min"], 50 * 2080)
        self.assertEqual(job["salary_max"], 75 * 2080)

    def test_annualizes_monthly_salary(self) -> None:
        job = normalize_jsearch_job(self._raw(job_min_salary=8000, job_max_salary=12000, job_salary_period="MONTH"))
        self.assertEqual(job["salary_min"], 8000 * 12)
        self.assertEqual(job["salary_max"], 12000 * 12)

    def test_remote_flag(self) -> None:
        job = normalize_jsearch_job(self._raw(job_is_remote=True))
        self.assertEqual(job["remote"], 1)

    def test_none_salary_stays_none(self) -> None:
        job = normalize_jsearch_job(self._raw(job_min_salary=None, job_max_salary=None))
        self.assertIsNone(job["salary_min"])
        self.assertIsNone(job["salary_max"])

    def test_fallback_location_from_city_state(self) -> None:
        raw = self._raw(job_location=None)
        raw.pop("job_location", None)
        raw["job_city"] = "Austin"
        raw["job_state"] = "TX"
        raw["job_country"] = "US"
        job = normalize_jsearch_job(raw)
        self.assertIn("Austin", job["location"])


class TestIngestJobs(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _make_db()
        self.config = {
            "must_have_roles": ["software engineer"],
            "nice_to_have_skills": ["python"],
            "exclude_terms": [],
        }
        self.synonyms: dict = {"role_synonyms": {}, "skill_synonyms": {}}

    def tearDown(self) -> None:
        self.conn.close()

    def _raw_job(self, ext_id: str = "j1", title: str = "Software Engineer") -> dict:
        return {
            "job_id": ext_id,
            "job_title": title,
            "employer_name": "Acme",
            "job_location": "Austin, TX",
            "job_is_remote": False,
            "job_description": "Build stuff with python",
            "job_apply_link": "https://example.com",
            "job_min_salary": None,
            "job_max_salary": None,
            "job_salary_period": None,
            "job_posted_at_datetime_utc": "2024-01-01T00:00:00Z",
        }

    def test_inserts_new_job(self) -> None:
        with patch("services.fetcher.llm_extract_job_details", return_value={}):
            inserted, filtered, dupes = ingest_jobs(
                [self._raw_job()], self.config, self.synonyms, self.conn
            )
        self.assertEqual(inserted, 1)
        self.assertEqual(filtered, 0)
        count = self.conn.execute("SELECT COUNT(*) FROM job_posting").fetchone()[0]
        self.assertEqual(count, 1)

    def test_filters_excluded_title(self) -> None:
        cfg = dict(self.config, exclude_terms=["staff"])
        with patch("services.fetcher.llm_extract_job_details", return_value={}):
            inserted, filtered, dupes = ingest_jobs(
                [self._raw_job(title="Staff Software Engineer")], cfg, self.synonyms, self.conn
            )
        self.assertEqual(inserted, 0)
        self.assertEqual(filtered, 1)

    def test_skips_duplicate(self) -> None:
        with patch("services.fetcher.llm_extract_job_details", return_value={}):
            ingest_jobs([self._raw_job()], self.config, self.synonyms, self.conn)
            inserted, filtered, dupes = ingest_jobs(
                [self._raw_job()], self.config, self.synonyms, self.conn
            )
        self.assertEqual(dupes, 1)

    def test_respects_limit(self) -> None:
        jobs = [self._raw_job(ext_id=f"j{i}") for i in range(10)]
        with patch("services.fetcher.llm_extract_job_details", return_value={}):
            inserted, _, _ = ingest_jobs(jobs, self.config, self.synonyms, self.conn, limit=3)
        self.assertEqual(inserted, 3)

    def test_sets_keyword_score(self) -> None:
        with patch("services.fetcher.llm_extract_job_details", return_value={}):
            ingest_jobs([self._raw_job()], self.config, self.synonyms, self.conn)
        row = self.conn.execute("SELECT keyword_score FROM job_posting").fetchone()
        self.assertIsNotNone(row["keyword_score"])


if __name__ == "__main__":
    unittest.main()
