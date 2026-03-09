from __future__ import annotations

import re
from typing import Any

from utils.llm import call_kimi_extract

def extract_yoe(description: str) -> int | None:
    """Extract minimum years of experience from text."""
    text = description or ""
    patterns = [
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
        r"at\s+least\s+(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\s*-\s*\d+\s*(?:years?|yrs?)",
    ]
    matches: list[int] = []
    for pattern in patterns:
        for found in re.findall(pattern, text, flags=re.IGNORECASE):
            try:
                matches.append(int(found))
            except ValueError:
                continue
    return min(matches) if matches else None


def _to_amount(token: str) -> int | None:
    raw = token.strip().lower().replace(",", "").replace("$", "")
    if not raw:
        return None
    try:
        if raw.endswith("k"):
            return int(float(raw[:-1]) * 1000)
        return int(float(raw))
    except ValueError:
        return None


def extract_salary(description: str) -> tuple[int | None, int | None]:
    """Extract salary range from text."""
    text = description or ""

    range_pattern = re.search(
        r"\$?\s*([0-9]{2,3}(?:[,\d]{0,3})k?)\s*(?:-|to)\s*\$?\s*([0-9]{2,3}(?:[,\d]{0,3})k?)",
        text,
        flags=re.IGNORECASE,
    )
    if range_pattern:
        low = _to_amount(range_pattern.group(1))
        high = _to_amount(range_pattern.group(2))
        return low, high

    single_pattern = re.search(
        r"\$?\s*([0-9]{2,3}(?:[,\d]{0,3})k?)\s*(?:/yr|per year|annually|year)?",
        text,
        flags=re.IGNORECASE,
    )
    if single_pattern:
        value = _to_amount(single_pattern.group(1))
        return value, value

    return None, None


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower().strip())
    if not escaped:
        return False
    return re.search(rf"\b{escaped}\b", text.lower()) is not None


def extract_skills_from_text(description: str, skill_synonyms: dict[str, list[str]]) -> list[str]:
    """
    Match canonical skills by canonical name or synonyms, whole-word, case-insensitive.
    Returns canonical skill list in lowercase.
    """
    text = description or ""
    found: list[str] = []
    for canonical, synonyms in skill_synonyms.items():
        candidates = [canonical] + list(synonyms or [])
        if any(_contains_phrase(text, candidate) for candidate in candidates):
            found.append(canonical.lower())
    return found


def llm_extract_job_details(description: str) -> dict[str, Any]:
    """
    LLM structured extraction for high-relevance jobs.
    Returns empty dict on failure.
    """
    try:
        return call_kimi_extract(description)
    except Exception:
        return {}
