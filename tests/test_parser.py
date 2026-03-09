"""Unit tests for services/parser.py"""
from __future__ import annotations

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.parser import (
    _contains_phrase,
    _to_amount,
    extract_salary,
    extract_skills_from_text,
    extract_yoe,
)


class TestExtractYoe(unittest.TestCase):
    def test_years_of_experience_pattern(self) -> None:
        self.assertEqual(extract_yoe("Requires 5 years of experience."), 5)

    def test_plus_notation(self) -> None:
        self.assertEqual(extract_yoe("3+ years experience in Python."), 3)

    def test_at_least_pattern(self) -> None:
        self.assertEqual(extract_yoe("At least 4 years of exp required."), 4)

    def test_range_pattern_returns_min(self) -> None:
        self.assertEqual(extract_yoe("3-5 years of experience."), 3)

    def test_multiple_matches_returns_minimum(self) -> None:
        text = "2+ years Python. Preferred: 5 years leadership."
        self.assertEqual(extract_yoe(text), 2)

    def test_no_match_returns_none(self) -> None:
        self.assertIsNone(extract_yoe("No experience requirement mentioned."))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(extract_yoe(""))

    def test_case_insensitive(self) -> None:
        self.assertEqual(extract_yoe("10 YEARS of experience"), 10)


class TestToAmount(unittest.TestCase):
    def test_plain_integer(self) -> None:
        self.assertEqual(_to_amount("90000"), 90000)

    def test_k_suffix(self) -> None:
        self.assertEqual(_to_amount("90k"), 90000)

    def test_k_suffix_with_decimal(self) -> None:
        self.assertEqual(_to_amount("87.5k"), 87500)

    def test_with_comma(self) -> None:
        self.assertEqual(_to_amount("120,000"), 120000)

    def test_with_dollar_sign(self) -> None:
        self.assertEqual(_to_amount("$150"), 150)

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_to_amount(""))

    def test_non_numeric_returns_none(self) -> None:
        self.assertIsNone(_to_amount("abc"))


class TestExtractSalary(unittest.TestCase):
    def test_range_with_k_suffix(self) -> None:
        low, high = extract_salary("Salary range: $80k - $120k per year.")
        self.assertEqual(low, 80000)
        self.assertEqual(high, 120000)

    def test_range_with_to_keyword(self) -> None:
        low, high = extract_salary("Compensation: $100,000 to $150,000.")
        self.assertEqual(low, 100000)
        self.assertEqual(high, 150000)

    def test_single_value_returns_both(self) -> None:
        low, high = extract_salary("Pay: $90k annually.")
        self.assertEqual(low, high)
        self.assertIsNotNone(low)

    def test_no_salary_returns_none_none(self) -> None:
        low, high = extract_salary("Competitive compensation.")
        self.assertIsNone(low)
        self.assertIsNone(high)

    def test_empty_string(self) -> None:
        low, high = extract_salary("")
        self.assertIsNone(low)
        self.assertIsNone(high)

    def test_range_preferred_over_single(self) -> None:
        low, high = extract_salary("Pay $80k - $100k, base $90k.")
        self.assertEqual(low, 80000)
        self.assertEqual(high, 100000)


class TestContainsPhrase(unittest.TestCase):
    def test_exact_word_match(self) -> None:
        self.assertTrue(_contains_phrase("we use Python daily", "python"))

    def test_no_partial_word_match(self) -> None:
        # "Go" should not match inside "Google"
        self.assertFalse(_contains_phrase("work at Google", "go"))

    def test_case_insensitive(self) -> None:
        self.assertTrue(_contains_phrase("Knowledge of REST API", "rest api"))

    def test_empty_phrase_returns_false(self) -> None:
        self.assertFalse(_contains_phrase("some text", ""))

    def test_phrase_not_present(self) -> None:
        self.assertFalse(_contains_phrase("Java and Kotlin developer", "python"))


class TestExtractSkillsFromText(unittest.TestCase):
    def setUp(self) -> None:
        self.synonyms: dict = {
            "Python": ["py"],
            "PostgreSQL": ["postgres", "pg"],
            "Docker": [],
        }

    def test_finds_canonical_skill(self) -> None:
        skills = extract_skills_from_text("We use Python and Docker.", self.synonyms)
        self.assertIn("python", skills)
        self.assertIn("docker", skills)

    def test_finds_via_synonym(self) -> None:
        skills = extract_skills_from_text("Must know postgres.", self.synonyms)
        self.assertIn("postgresql", skills)

    def test_returns_lowercase_canonical(self) -> None:
        skills = extract_skills_from_text("Python developer needed.", self.synonyms)
        self.assertIn("python", skills)
        self.assertNotIn("Python", skills)

    def test_empty_description_returns_empty(self) -> None:
        skills = extract_skills_from_text("", self.synonyms)
        self.assertEqual(skills, [])

    def test_empty_synonyms_returns_empty(self) -> None:
        skills = extract_skills_from_text("Python developer", {})
        self.assertEqual(skills, [])

    def test_no_matching_skills(self) -> None:
        skills = extract_skills_from_text("Experience with Fortran.", self.synonyms)
        self.assertEqual(skills, [])


if __name__ == "__main__":
    unittest.main()
