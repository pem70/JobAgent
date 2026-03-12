from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from services.learner import apply_learned_weights, load_weights_or_empty, mark_ignored_jobs, update_weights
from db import get_connection
from utils.llm import call_kimi_rerank


def _load_profile(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM user_profile WHERE id = 1").fetchone()
    if not row:
        raise ValueError("Run `agent profile init` first.")
    return {
        "id": row["id"],
        "name": row["name"] or "",
        "resume_text": row["resume_text"] or "",
        "skills": json.loads(row["skills"]) if row["skills"] else [],
        "target_roles": json.loads(row["target_roles"]) if row["target_roles"] else [],
        "yoe": row["yoe"] or 0,
        "location_pref": row["location_pref"] or "",
        "remote_ok": bool(row["remote_ok"]) if row["remote_ok"] is not None else True,
        "min_salary": row["min_salary"] or 0,
    }


def layer1_hard_filter(profile: dict[str, Any], include_seen: bool = False) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        statuses = ("new", "seen") if include_seen else ("new",)
        placeholders = ",".join("?" for _ in statuses)
        sql = f"""
            SELECT * FROM job_posting
            WHERE status IN ({placeholders})
              AND (salary_max IS NULL OR salary_max >= ?)
        """
        params: list[Any] = [*statuses, int(profile.get("min_salary", 0))]
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def compute_mechanical_score(job: dict[str, Any], profile: dict[str, Any]) -> float:
    min_salary = int(profile.get("min_salary", 0) or 0)
    salary_max = job.get("salary_max")
    if salary_max is None:
        salary_score = 0.5
    elif float(salary_max) >= float(min_salary):
        salary_score = 1.0
    else:
        salary_score = 0.0

    user_wants_remote = bool(profile.get("remote_ok", False))
    job_remote = bool(job.get("remote", 0))
    if user_wants_remote and job_remote:
        remote_score = 1.0
    elif not user_wants_remote:
        remote_score = 0.5
    else:
        remote_score = 0.0

    user_yoe = int(profile.get("yoe", 0) or 0)
    job_min_yoe = job.get("min_yoe")
    if job_min_yoe is None:
        yoe_score = 0.5
    else:
        req = int(job_min_yoe)
        if user_yoe >= req:
            yoe_score = 1.0
        elif req - user_yoe == 1:
            yoe_score = 0.5
        else:
            yoe_score = 0.0

    parsed = job.get("parsed_skills")
    parsed_skills: list[str] = []
    if isinstance(parsed, str) and parsed:
        try:
            loaded = json.loads(parsed)
            if isinstance(loaded, list):
                parsed_skills = [str(x).strip().lower() for x in loaded if str(x).strip()]
        except Exception:
            parsed_skills = []
    elif isinstance(parsed, list):
        parsed_skills = [str(x).strip().lower() for x in parsed if str(x).strip()]

    user_skills = {str(x).strip().lower() for x in profile.get("skills", []) if str(x).strip()}
    if parsed_skills:
        overlap = sum(1 for skill in parsed_skills if skill in user_skills)
        skills_score = overlap / max(1, len(parsed_skills))
    else:
        skills_score = float(job.get("keyword_score", 0.0) or 0.0)

    return float((salary_score + remote_score + yoe_score + skills_score) / 4.0)


def layer3_llm_rerank(jobs: list[dict[str, Any]], profile: dict[str, Any], top_n: int = 20) -> list[dict[str, Any]]:
    candidates = jobs[:top_n]
    if not candidates:
        return jobs

    profile_summary = (
        f"- Name: {profile.get('name', '')}\n"
        f"- Target Roles: {', '.join(profile.get('target_roles', []))}\n"
        f"- Skills: {', '.join(profile.get('skills', []))}\n"
        f"- Years of Experience: {profile.get('yoe', 0)}\n"
        f"- Resume Summary: {(profile.get('resume_text', '') or '')[:500]}"
    )

    enriched_by_id: dict[int, dict[str, Any]] = {int(job["id"]): dict(job) for job in candidates}

    batches: list[tuple[int, list[dict[str, Any]], str]] = []
    for batch_start in range(0, len(candidates), 10):
        batch = candidates[batch_start : batch_start + 10]
        lines: list[str] = []
        for idx, job in enumerate(batch):
            lines.append(
                f"job_index={idx}\n"
                f"title={job.get('title', '')}\n"
                f"company={job.get('company', '')}\n"
                f"parsed_skills={job.get('parsed_skills', '[]')}\n"
                f"description={(job.get('description', '') or '')[:500]}\n"
            )
        batches.append((batch_start, batch, "\n".join(lines)))

    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        future_to_batch = {
            executor.submit(call_kimi_rerank, profile_summary=profile_summary, jobs_text=jobs_text): (batch_start, batch)
            for batch_start, batch, jobs_text in batches
        }
        for future in as_completed(future_to_batch):
            batch_start, batch = future_to_batch[future]
            try:
                llm_items = future.result()
            except Exception as exc:
                print(f"[rerank] batch starting at {batch_start} failed: {exc}")
                continue
            for llm_item in llm_items:
                try:
                    local_idx = int(llm_item.get("job_index"))
                except Exception:
                    continue
                if local_idx < 0 or local_idx >= len(batch):
                    continue
                job_id = int(batch[local_idx]["id"])
                target = enriched_by_id[job_id]
                target["llm_score"] = float(llm_item.get("score", 0))
                target["rationale"] = str(llm_item.get("rationale", "") or "")
                target["skill_gaps"] = llm_item.get("skill_gaps", []) or []
                target["red_flags"] = llm_item.get("red_flags", []) or []

    merged: list[dict[str, Any]] = []
    for job in jobs:
        jid = int(job["id"])
        merged.append(enriched_by_id.get(jid, job))
    return merged


def compute_final_scores(jobs: list[dict[str, Any]], llm_enabled: bool, learned_weights: dict | None = None) -> list[dict[str, Any]]:
    weights = learned_weights or {}
    ranked: list[dict[str, Any]] = []
    for job in jobs:
        keyword_score = float(job.get("keyword_score", 0.0) or 0.0)
        base_mechanical = float(job.get("mechanical_score", 0.0) or 0.0)
        mechanical_score = base_mechanical
        if weights:
            boost = apply_learned_weights(job, weights)
            mechanical_score = max(0.0, min(1.0, base_mechanical * boost))
        llm_score = job.get("llm_score")
        if llm_enabled and llm_score is not None:
            final = 0.3 * mechanical_score + 0.7 * (float(llm_score) / 100.0)
        else:
            final = mechanical_score
        enriched = dict(job)
        enriched["keyword_score"] = keyword_score
        enriched["embedding_similarity"] = mechanical_score
        enriched["final_score"] = final
        ranked.append(enriched)
    ranked.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
    return ranked


def _upsert_match_results(conn: sqlite3.Connection, jobs: list[dict[str, Any]]) -> None:
    for job in jobs:
        conn.execute(
            """
            INSERT INTO match_result (
                job_id, final_score, embedding_similarity, llm_score, keyword_score,
                rationale, skill_gaps, red_flags
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                final_score=excluded.final_score,
                embedding_similarity=excluded.embedding_similarity,
                llm_score=excluded.llm_score,
                keyword_score=excluded.keyword_score,
                rationale=excluded.rationale,
                skill_gaps=excluded.skill_gaps,
                red_flags=excluded.red_flags,
                computed_at=datetime('now')
            """,
            (
                int(job["id"]),
                float(job.get("final_score", 0.0)),
                float(job.get("embedding_similarity", 0.0)),
                float(job["llm_score"]) if job.get("llm_score") is not None else None,
                float(job.get("keyword_score", 0.0)),
                str(job.get("rationale", "") or ""),
                json.dumps(job.get("skill_gaps", []) or [], ensure_ascii=False),
                json.dumps(job.get("red_flags", []) or [], ensure_ascii=False),
            ),
        )
    conn.commit()


def run_match_pipeline(
    include_seen: bool = False,
    llm_enabled: bool = True,
    top_n: int = 10,
    learn_first: bool = False,
) -> list[dict[str, Any]]:
    learned_weights: dict[str, Any] | None = None
    if learn_first:
        mark_ignored_jobs(days=7)
        learned_weights = update_weights() or load_weights_or_empty()
    else:
        learned_weights = load_weights_or_empty()

    conn = get_connection()
    try:
        profile = _load_profile(conn)
        jobs = layer1_hard_filter(profile, include_seen=include_seen)

        mechanical_ranked: list[dict[str, Any]] = []
        for job in jobs:
            enriched = dict(job)
            enriched["mechanical_score"] = compute_mechanical_score(enriched, profile)
            mechanical_ranked.append(enriched)
        mechanical_ranked.sort(key=lambda x: float(x.get("mechanical_score", 0.0)), reverse=True)
        top_mechanical = mechanical_ranked[:10]

        scored = layer3_llm_rerank(top_mechanical, profile, top_n=10) if llm_enabled else top_mechanical
        ranked = compute_final_scores(scored, llm_enabled=llm_enabled, learned_weights=learned_weights)

        _upsert_match_results(conn, ranked)
        job_ids = [int(job["id"]) for job in ranked]
        if job_ids:
            placeholders = ",".join("?" for _ in job_ids)
            conn.execute(f"UPDATE job_posting SET status='seen' WHERE id IN ({placeholders})", job_ids)
            conn.commit()
        return ranked[: max(1, int(top_n))]
    finally:
        conn.close()
