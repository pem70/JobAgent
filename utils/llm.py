from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import OpenAI

from config import load_env


def _build_kimi_client() -> OpenAI:
    load_env()
    api_key = os.getenv("KIMI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing KIMI_API_KEY in .env")
    return OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")


def call_kimi(prompt: str, system_prompt: str = "", temperature: float = 0.3, model: str = "moonshot-v1-8k") -> str:
    """
    Call Kimi model via OpenAI-compatible API.
    Retry once on transient failure.
    """
    client = _build_kimi_client()
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            return str(content).strip()
        except Exception:
            if attempt == 0:
                time.sleep(1.5)
                continue
            raise
    return ""


def _extract_json_payload(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "[]"
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def call_kimi_rerank(profile_summary: str, jobs_text: str) -> list[dict[str, Any]]:
    prompt = (
        "You are a job matching expert. Given a candidate profile and a list of job postings, "
        "score each job on how well it matches the candidate.\n\n"
        "## Candidate Profile\n"
        f"{profile_summary}\n\n"
        "## Job Postings\n"
        f"{jobs_text}\n\n"
        "## Instructions\n"
        "For each job, return a JSON array with objects containing:\n"
        "- job_index (int)\n"
        "- score (int 0-100)\n"
        "- rationale (one sentence)\n"
        "- skill_gaps (array)\n"
        "- red_flags (array)\n\n"
        "Return ONLY the JSON array."
    )

    for attempt in range(2):
        raw = call_kimi(prompt=prompt, temperature=0.2, model="kimi-k2-thinking-turbo")
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                parsed = json.loads(_extract_json_payload(raw))
                return parsed if isinstance(parsed, list) else []
            except Exception:
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                return []
    return []


def call_kimi_extract_batch(descriptions: list[str]) -> list[dict[str, Any]]:
    if not descriptions:
        return []
    jobs_text = "\n\n".join(
        f"--- Job {i} ---\n{(desc or '')[:1500]}" for i, desc in enumerate(descriptions)
    )
    prompt = (
        "Extract structured information from each job description below. "
        "Return ONLY a JSON array with one object per job:\n"
        "[\n"
        "  {\n"
        '    "job_index": 0,\n'
        '    "required_skills": [...],\n'
        '    "preferred_skills": [...],\n'
        '    "min_yoe": int or null,\n'
        '    "degree_requirement": "BS" / "MS" / "PhD" / null,\n'
        '    "visa_sponsorship": true / false / null,\n'
        '    "remote_policy": "remote" / "hybrid" / "onsite" / null,\n'
        '    "tech_stack": [...]\n'
        "  },\n"
        "  ...\n"
        "]\n\n"
        f"{jobs_text}"
    )
    for attempt in range(2):
        raw = call_kimi(prompt=prompt, temperature=0.1, model="kimi-k2-turbo-preview")
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                parsed = json.loads(_extract_json_payload(raw))
                return parsed if isinstance(parsed, list) else []
            except Exception:
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                return []
    return []


def call_kimi_extract(description: str) -> dict[str, Any]:
    prompt = (
        "Extract structured information from this job description. Return ONLY a JSON object:\n"
        "{\n"
        '  "required_skills": [...],\n'
        '  "preferred_skills": [...],\n'
        '  "min_yoe": int or null,\n'
        '  "degree_requirement": "BS" / "MS" / "PhD" / null,\n'
        '  "visa_sponsorship": true / false / null,\n'
        '  "remote_policy": "remote" / "hybrid" / "onsite" / null,\n'
        '  "tech_stack": [...]\n'
        "}\n\n"
        "Job Description:\n"
        f"{(description or '')[:2000]}"
    )
    raw = call_kimi(prompt=prompt, temperature=0.1, model="kimi-k2-turbo-preview")
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        try:
            parsed = json.loads(_extract_json_payload(raw))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
