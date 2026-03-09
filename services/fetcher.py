from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import time
from typing import Any

import requests
from rapidfuzz import fuzz

from services.parser import extract_salary, extract_skills_from_text, extract_yoe, llm_extract_job_details

JSEARCH_URL = "https://jsearch.p.rapidapi.com/search"
REQUEST_TIMEOUT = (5, 25)  # (connect timeout, read timeout)


def build_queries(config: dict[str, Any]) -> list[str]:
    must_have = [x.strip().lower() for x in config.get("must_have_roles", []) if str(x).strip()]
    nice_to_have = [x.strip().lower() for x in config.get("nice_to_have_skills", []) if str(x).strip()]
    top_skills = nice_to_have[:2]
    max_queries = int(config.get("max_queries_per_fetch", 5))

    queries: list[str] = []
    for role in must_have:
        if top_skills:
            for skill in top_skills:
                queries.append(f"{role} {skill}".strip())
        else:
            queries.append(role)
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(query)
    return deduped[:max_queries]


def fetch_jobs(query: str, date_posted: str = "3days", page: int = 1) -> list[dict[str, Any]]:
    headers = {
        "x-rapidapi-key": os.getenv("JSEARCH_API_KEY", ""),
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
    }
    base_params = {
        "query": query,
        "page": str(page),
        "num_pages": "1",
        "country": "us",
    }
    if not headers["x-rapidapi-key"]:
        raise ValueError("Missing JSEARCH_API_KEY in .env")

    date_candidates: list[str] = []
    for candidate in [date_posted, "all"]:
        value = (candidate or "").strip()
        if value and value not in date_candidates:
            date_candidates.append(value)

    for date_value in date_candidates:
        params = dict(base_params)
        params["date_posted"] = date_value
        for attempt in range(2):
            try:
                response = requests.get(
                    JSEARCH_URL,
                    headers=headers,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code == 429 and attempt == 0:
                    time.sleep(2)
                    continue
                if response.status_code == 400 and date_value != "all":
                    break
                response.raise_for_status()
                payload = response.json()
                return payload.get("data", []) if isinstance(payload, dict) else []
            except requests.exceptions.ReadTimeout as exc:
                if attempt == 0:
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"JSearch read timeout after {REQUEST_TIMEOUT[1]}s. "
                    f"URL={JSEARCH_URL} params={params}"
                ) from exc
            except requests.exceptions.ConnectTimeout as exc:
                if attempt == 0:
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"JSearch connect timeout after {REQUEST_TIMEOUT[0]}s. "
                    f"URL={JSEARCH_URL}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                if attempt == 0:
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"JSearch connection error. URL={JSEARCH_URL} host=jsearch.p.rapidapi.com"
                ) from exc
            except requests.exceptions.HTTPError:
                raise
            except Exception:
                if attempt == 0:
                    time.sleep(2)
                    continue
                raise
    return []


def _expand_title_terms(config: dict[str, Any], synonyms: dict[str, Any]) -> set[str]:
    must_have = [x.strip().lower() for x in config.get("must_have_roles", []) if str(x).strip()]
    role_synonyms = synonyms.get("role_synonyms", {})
    normalized_map: dict[str, list[str]] = {
        str(k).lower(): [str(v).lower() for v in vals]
        for k, vals in role_synonyms.items()
    }
    reverse_map: dict[str, str] = {}
    for canonical, vals in normalized_map.items():
        for val in vals:
            reverse_map[val] = canonical

    terms: set[str] = set()
    for role in must_have:
        terms.add(role)
        if role in normalized_map:
            terms.update(normalized_map[role])
        elif role in reverse_map:
            canonical = reverse_map[role]
            terms.add(canonical)
            terms.update(normalized_map.get(canonical, []))
    return terms


def keyword_filter(job: dict[str, Any], config: dict[str, Any], synonyms: dict[str, Any]) -> tuple[bool, float]:
    title = str(job.get("title", "")).strip()
    description = str(job.get("description", "")).lower()
    if not title:
        return False, 0.0

    title_lower = title.lower()
    title_tokens = set(title_lower.replace("/", " ").replace("-", " ").split())
    title_terms = _expand_title_terms(config, synonyms)
    has_title_hit = False
    for term in title_terms:
        term_tokens = {token for token in term.lower().split() if token}
        if term_tokens and term_tokens.issubset(title_tokens):
            has_title_hit = True
            break

    exclude_terms = [str(x).lower() for x in config.get("exclude_terms", [])]
    if any(term and term in title_lower for term in exclude_terms):
        return False, 0.0

    if not has_title_hit:
        return True, 0.05

    nice_to_have = [str(x).lower() for x in config.get("nice_to_have_skills", []) if str(x).strip()]
    if not nice_to_have:
        return True, 0.0
    hit_count = sum(1 for skill in nice_to_have if skill in description)
    return True, hit_count / len(nice_to_have)


def is_duplicate(job: dict[str, Any], db_conn: sqlite3.Connection) -> bool:
    company = str(job.get("company", "")).strip().lower()
    title = str(job.get("title", "")).strip().lower()
    location = str(job.get("location", "")).strip().lower()
    if not company or not title:
        return True

    rows = db_conn.execute(
        """
        SELECT company, title, location, scraped_at
        FROM job_posting
        WHERE lower(company) = ? OR lower(title) = ?
        """,
        (company, title),
    ).fetchall()
    now = dt.datetime.utcnow()
    for row in rows:
        row_company = (row["company"] or "").lower()
        row_title = (row["title"] or "").lower()
        row_location = (row["location"] or "").lower()
        ratio = fuzz.ratio(f"{company}|{title}|{location}", f"{row_company}|{row_title}|{row_location}")
        if ratio >= 85:
            return True

        if row_company == company and row_title == title and row["scraped_at"]:
            try:
                scraped = dt.datetime.fromisoformat(str(row["scraped_at"]))
                if now - scraped <= dt.timedelta(days=30):
                    return True
            except ValueError:
                continue
    return False


def normalize_jsearch_job(raw: dict[str, Any]) -> dict[str, Any]:
    def _parse_salary_value(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(",", "").replace("$", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _annualize(value: float | None, period: str | None) -> int | None:
        if value is None:
            return None
        p = (period or "").strip().upper()
        if p == "HOUR":
            annual = value * 2080
        elif p == "MONTH":
            annual = value * 12
        elif p == "WEEK":
            annual = value * 52
        elif p == "YEAR":
            annual = value
        else:
            if value < 500:
                annual = value * 2080
            elif value < 20000:
                annual = value * 12
            else:
                annual = value
        return int(round(annual))

    description = str(raw.get("job_description") or raw.get("description") or "")
    salary_period = raw.get("job_salary_period") or raw.get("salary_period")
    salary_min = _annualize(_parse_salary_value(raw.get("job_min_salary") or raw.get("salary_min")), salary_period)
    salary_max = _annualize(_parse_salary_value(raw.get("job_max_salary") or raw.get("salary_max")), salary_period)
    return {
        "source": "jsearch",
        "external_id": raw.get("job_id") or raw.get("id"),
        "title": raw.get("job_title") or raw.get("title") or "",
        "company": raw.get("employer_name") or raw.get("company_name") or raw.get("company") or "",
        "location": (
            raw.get("job_location")
            or raw.get("location")
            or " ".join(
                x for x in [raw.get("job_city"), raw.get("job_state"), raw.get("job_country")] if x
            )
        ),
        "remote": 1 if raw.get("job_is_remote") in (True, 1, "true", "True") else 0,
        "description": description,
        "url": raw.get("job_apply_link") or raw.get("job_google_link") or raw.get("url"),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "posted_at": raw.get("job_posted_at_datetime_utc") or raw.get("posted_at"),
    }


def ingest_jobs(
    raw_jobs: list[dict[str, Any]],
    config: dict[str, Any],
    synonyms: dict[str, Any],
    db_conn: sqlite3.Connection,
    limit: int = 50,
) -> tuple[int, int, int]:
    inserted = 0
    filtered_out = 0
    duplicates = 0
    skill_synonyms = synonyms.get("skill_synonyms", {})
    candidates: list[tuple[dict[str, Any], float]] = []

    # Phase A: normalize -> keyword_filter -> is_duplicate, collect until limit
    for raw in raw_jobs:
        job = normalize_jsearch_job(raw)
        keep, keyword_score = keyword_filter(job, config, synonyms)
        if not keep:
            filtered_out += 1
            continue
        if is_duplicate(job, db_conn):
            duplicates += 1
            continue
        candidates.append((job, keyword_score))
        if len(candidates) >= max(1, int(limit)):
            break

    if not candidates:
        return inserted, filtered_out, duplicates

    # Phase B: parser extraction + batch INSERT (embedding deferred to match phase)
    insert_sql = """
        INSERT INTO job_posting (
            source, external_id, title, company, location, remote, description, url,
            salary_min, salary_max, parsed_skills, min_yoe, visa_sponsorship, embedding,
            keyword_score, posted_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new')
    """
    insert_rows: list[tuple[Any, ...]] = []
    for job, keyword_score in candidates:
        description = str(job.get("description", "") or "")
        parsed_skills = extract_skills_from_text(description, skill_synonyms)
        min_yoe = extract_yoe(description)
        visa_sponsorship = None

        if float(keyword_score) > 0.3:
            llm_data = llm_extract_job_details(description)
            if isinstance(llm_data, dict) and llm_data:
                llm_skills: list[str] = []
                for key in ("required_skills", "preferred_skills", "tech_stack"):
                    val = llm_data.get(key, [])
                    if isinstance(val, list):
                        llm_skills.extend(str(x).strip().lower() for x in val if str(x).strip())
                if llm_skills:
                    seen: set[str] = set()
                    merged: list[str] = []
                    for s in llm_skills + parsed_skills:
                        if s not in seen:
                            seen.add(s)
                            merged.append(s)
                    parsed_skills = merged
                if llm_data.get("min_yoe") is not None:
                    try:
                        min_yoe = int(llm_data["min_yoe"])
                    except Exception:
                        pass
                if llm_data.get("visa_sponsorship") is True:
                    visa_sponsorship = 1
                elif llm_data.get("visa_sponsorship") is False:
                    visa_sponsorship = 0

        sal_low, sal_high = extract_salary(description)
        if job.get("salary_min") is None:
            job["salary_min"] = sal_low
        if job.get("salary_max") is None:
            job["salary_max"] = sal_high

        insert_rows.append(
            (
                job.get("source"),
                job.get("external_id"),
                job.get("title"),
                job.get("company"),
                job.get("location"),
                job.get("remote", 0),
                description,
                job.get("url"),
                job.get("salary_min"),
                job.get("salary_max"),
                json.dumps(parsed_skills, ensure_ascii=False),
                min_yoe,
                visa_sponsorship,
                None,
                float(keyword_score),
                job.get("posted_at"),
            )
        )

    try:
        db_conn.executemany(insert_sql, insert_rows)
        inserted += len(insert_rows)
    except sqlite3.IntegrityError:
        for row in insert_rows:
            try:
                db_conn.execute(insert_sql, row)
                inserted += 1
            except sqlite3.IntegrityError:
                duplicates += 1
                continue
    db_conn.commit()
    return inserted, filtered_out, duplicates
