from __future__ import annotations

import datetime as dt
import json
import math
from collections import Counter
from typing import Any

from config import load_learned_weights, save_learned_weights
from db import get_connection

MIN_INTERACTIONS = 30


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except Exception:
        return []
    return []


def _title_keywords(title: str | None) -> list[str]:
    if not title:
        return []
    cleaned = (
        str(title).lower().replace("/", " ").replace("-", " ").replace(",", " ").replace(".", " ")
    )
    return [token for token in cleaned.split() if token]


def mark_ignored_jobs(days: int = 7) -> int:
    conn = get_connection()
    try:
        cutoff = (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        result = conn.execute(
            """
            UPDATE job_posting
            SET interaction = 'ignored'
            WHERE status = 'seen'
              AND interaction IS NULL
              AND scraped_at <= ?
            """,
            (cutoff,),
        )
        conn.commit()
        return int(result.rowcount or 0)
    finally:
        conn.close()


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        return {}
    max_abs = max(abs(value) for value in weights.values())
    if max_abs == 0:
        return {k: 0.0 for k in weights}
    return {k: max(-1.0, min(1.0, value / max_abs)) for k, value in weights.items()}


def update_weights() -> dict[str, Any]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT title, parsed_skills, remote, interaction
            FROM job_posting
            WHERE interaction IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()

    if len(rows) < MIN_INTERACTIONS:
        return {}

    positive = {"viewed", "applied"}
    negative = {"dismissed", "ignored"}

    pos_skill_counter: Counter[str] = Counter()
    neg_skill_counter: Counter[str] = Counter()
    pos_kw_counter: Counter[str] = Counter()
    neg_kw_counter: Counter[str] = Counter()
    total_pos = 0
    total_neg = 0

    for row in rows:
        interaction = str(row["interaction"]).lower()
        skills = _parse_json_list(row["parsed_skills"])
        keywords = _title_keywords(row["title"])
        if int(row["remote"] or 0) == 1:
            keywords.append("remote")

        if interaction in positive:
            total_pos += 1
            pos_skill_counter.update(set(skills))
            pos_kw_counter.update(set(keywords))
        elif interaction in negative:
            total_neg += 1
            neg_skill_counter.update(set(skills))
            neg_kw_counter.update(set(keywords))

    if total_pos == 0 or total_neg == 0:
        return {}

    all_skills = set(pos_skill_counter.keys()) | set(neg_skill_counter.keys())
    all_keywords = set(pos_kw_counter.keys()) | set(neg_kw_counter.keys())

    raw_skill_weights: dict[str, float] = {}
    raw_keyword_weights: dict[str, float] = {}
    for skill in all_skills:
        freq_pos = pos_skill_counter[skill] / total_pos
        freq_neg = neg_skill_counter[skill] / total_neg
        raw_skill_weights[skill] = math.log2((freq_pos + 0.01) / (freq_neg + 0.01))
    for kw in all_keywords:
        freq_pos = pos_kw_counter[kw] / total_pos
        freq_neg = neg_kw_counter[kw] / total_neg
        raw_keyword_weights[kw] = math.log2((freq_pos + 0.01) / (freq_neg + 0.01))

    weights = {
        "skills": _normalize_weights(raw_skill_weights),
        "keywords": _normalize_weights(raw_keyword_weights),
        "min_interactions": MIN_INTERACTIONS,
        "updated_at": dt.datetime.utcnow().isoformat(),
    }
    save_learned_weights(weights)
    return weights


def apply_learned_weights(job: dict[str, Any], weights: dict[str, Any]) -> float:
    if not weights:
        return 1.0
    boost = 1.0

    parsed_skills: list[str] = []
    raw_skills = job.get("parsed_skills")
    if isinstance(raw_skills, str):
        parsed_skills = _parse_json_list(raw_skills)
    elif isinstance(raw_skills, list):
        parsed_skills = [str(x).strip().lower() for x in raw_skills if str(x).strip()]
    for skill in parsed_skills:
        if skill in weights.get("skills", {}):
            boost += float(weights["skills"][skill]) * 0.1

    for kw in _title_keywords(job.get("title")):
        if kw in weights.get("keywords", {}):
            boost += float(weights["keywords"][kw]) * 0.05

    return max(0.5, min(1.5, boost))


def load_weights_or_empty() -> dict[str, Any]:
    loaded = load_learned_weights()
    if not isinstance(loaded, dict):
        return {}
    return loaded
