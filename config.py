from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json"
SYNONYMS_PATH = ROOT_DIR / "data" / "synonyms.json"
LEARNED_WEIGHTS_PATH = ROOT_DIR / "data" / "learned_weights.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "must_have_roles": [
        "backend engineer",
        "software engineer",
        "software developer",
    ],
    "nice_to_have_skills": [
        "Python",
        "FastAPI",
        "PostgreSQL",
        "REST API",
    ],
    "exclude_terms": [
        "Staff",
        "Principal",
        "Director",
        "Lead",
        "Manager",
    ],
    "date_range": "3days",
    "max_queries_per_fetch": 5,
    "pages_per_query": 1,
    "llm_rerank_enabled": True,
}


def load_env() -> None:
    load_dotenv(ROOT_DIR / ".env")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        save_config(DEFAULT_CONFIG, config_path)
        return dict(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    merged.update(data)
    return merged


def save_config(config: dict[str, Any], path: Path | str = CONFIG_PATH) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_synonyms(path: Path | str = SYNONYMS_PATH) -> dict[str, Any]:
    synonyms_path = Path(path)
    if not synonyms_path.exists():
        return {"role_synonyms": {}, "skill_synonyms": {}}
    with synonyms_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"role_synonyms": {}, "skill_synonyms": {}}
    return data


def load_learned_weights(path: Path | str = LEARNED_WEIGHTS_PATH) -> dict[str, Any]:
    weights_path = Path(path)
    if not weights_path.exists():
        return {"skills": {}, "keywords": {}}
    with weights_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"skills": {}, "keywords": {}}
    return data


def save_learned_weights(weights: dict[str, Any], path: Path | str = LEARNED_WEIGHTS_PATH) -> None:
    weights_path = Path(path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with weights_path.open("w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=4)


def get_env_var(name: str) -> str:
    load_env()
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value
