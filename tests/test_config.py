"""Unit tests for config.py"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEFAULT_CONFIG,
    get_env_var,
    load_config,
    load_learned_weights,
    load_synonyms,
    save_config,
    save_learned_weights,
)


class TestLoadConfig(unittest.TestCase):
    def test_returns_defaults_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            config = load_config(path)
            for key in DEFAULT_CONFIG:
                self.assertIn(key, config)

    def test_merges_user_values_over_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            user_data = {"must_have_roles": ["data engineer"], "pages_per_query": 3}
            path.write_text(json.dumps(user_data), encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config["must_have_roles"], ["data engineer"])
            self.assertEqual(config["pages_per_query"], 3)
            # defaults still present
            self.assertIn("nice_to_have_skills", config)

    def test_returns_defaults_on_invalid_json_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config, DEFAULT_CONFIG)

    def test_creates_file_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            self.assertFalse(path.exists())
            load_config(path)
            self.assertTrue(path.exists())


class TestSaveConfig(unittest.TestCase):
    def test_writes_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            data = {"key": "value", "list": [1, 2, 3]}
            save_config(data, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, data)

    def test_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "config.json"
            save_config({"x": 1}, path)
            self.assertTrue(path.exists())


class TestLoadSynonyms(unittest.TestCase):
    def test_returns_empty_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "synonyms.json"
            result = load_synonyms(path)
            self.assertEqual(result, {"role_synonyms": {}, "skill_synonyms": {}})

    def test_loads_valid_synonyms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "synonyms.json"
            data = {"role_synonyms": {"swe": ["software engineer"]}, "skill_synonyms": {}}
            path.write_text(json.dumps(data), encoding="utf-8")
            result = load_synonyms(path)
            self.assertEqual(result["role_synonyms"]["swe"], ["software engineer"])

    def test_returns_empty_on_non_dict_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "synonyms.json"
            path.write_text(json.dumps([1, 2]), encoding="utf-8")
            result = load_synonyms(path)
            self.assertEqual(result, {"role_synonyms": {}, "skill_synonyms": {}})


class TestLoadLearnedWeights(unittest.TestCase):
    def test_returns_empty_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            result = load_learned_weights(path)
            self.assertEqual(result, {"skills": {}, "keywords": {}})

    def test_loads_saved_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            data = {"skills": {"python": 0.8}, "keywords": {"backend": 0.5}}
            path.write_text(json.dumps(data), encoding="utf-8")
            result = load_learned_weights(path)
            self.assertAlmostEqual(result["skills"]["python"], 0.8)

    def test_save_and_reload_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            data = {"skills": {"go": -0.3}, "keywords": {}}
            save_learned_weights(data, path)
            result = load_learned_weights(path)
            self.assertAlmostEqual(result["skills"]["go"], -0.3)


class TestGetEnvVar(unittest.TestCase):
    def test_returns_value_when_set(self) -> None:
        with patch.dict(os.environ, {"MY_TEST_VAR": "hello"}):
            with patch("config.load_env"):
                value = get_env_var("MY_TEST_VAR")
        self.assertEqual(value, "hello")

    def test_raises_when_missing(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "MISSING_VAR_XYZ"}
        with patch.dict(os.environ, env, clear=True):
            with patch("config.load_env"):
                with self.assertRaises(ValueError):
                    get_env_var("MISSING_VAR_XYZ")

    def test_raises_when_empty_string(self) -> None:
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            with patch("config.load_env"):
                with self.assertRaises(ValueError):
                    get_env_var("EMPTY_VAR")


if __name__ == "__main__":
    unittest.main()
