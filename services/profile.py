from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pdfplumber

from config import load_config, load_synonyms, save_config
from db import get_connection
from utils.embedding import get_embedding


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return result


def parse_resume_pdf(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    pages_text: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text.strip())
    resume_text = "\n\n".join(pages_text).strip()
    if not resume_text:
        raise ValueError("Resume PDF parsed successfully but text is empty.")
    return resume_text


def _serialize_profile_for_db(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": 1,
        "name": profile.get("name"),
        "resume_text": profile.get("resume_text"),
        "resume_embedding": pickle.dumps(profile.get("resume_embedding")),
        "skills": json.dumps(profile.get("skills", []), ensure_ascii=False),
        "target_roles": json.dumps(profile.get("target_roles", []), ensure_ascii=False),
        "yoe": int(profile.get("yoe", 0)),
        "location_pref": profile.get("location_pref", ""),
        "remote_ok": 1 if bool(profile.get("remote_ok", True)) else 0,
        "min_salary": int(profile.get("min_salary", 0)),
        "deal_breakers": json.dumps(profile.get("deal_breakers", []), ensure_ascii=False),
    }


def _deserialize_profile_row(row: sqlite3.Row | None) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "resume_text": row["resume_text"] or "",
        "resume_embedding": pickle.loads(row["resume_embedding"]) if row["resume_embedding"] else None,
        "skills": json.loads(row["skills"]) if row["skills"] else [],
        "target_roles": json.loads(row["target_roles"]) if row["target_roles"] else [],
        "yoe": row["yoe"] or 0,
        "location_pref": row["location_pref"] or "",
        "remote_ok": bool(row["remote_ok"]) if row["remote_ok"] is not None else True,
        "min_salary": row["min_salary"] or 0,
        "deal_breakers": json.loads(row["deal_breakers"]) if row["deal_breakers"] else [],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_profile(conn: sqlite3.Connection | None = None) -> Optional[dict[str, Any]]:
    owns_conn = conn is None
    active_conn = conn or get_connection()
    try:
        row = active_conn.execute("SELECT * FROM user_profile WHERE id = 1").fetchone()
        return _deserialize_profile_row(row)
    finally:
        if owns_conn:
            active_conn.close()


def upsert_profile(profile: dict[str, Any], conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    owns_conn = conn is None
    active_conn = conn or get_connection()
    payload = _serialize_profile_for_db(profile)
    try:
        active_conn.execute(
            """
            INSERT INTO user_profile (
                id, name, resume_text, resume_embedding, skills, target_roles, yoe,
                location_pref, remote_ok, min_salary, deal_breakers
            )
            VALUES (
                :id, :name, :resume_text, :resume_embedding, :skills, :target_roles, :yoe,
                :location_pref, :remote_ok, :min_salary, :deal_breakers
            )
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                resume_text = excluded.resume_text,
                resume_embedding = excluded.resume_embedding,
                skills = excluded.skills,
                target_roles = excluded.target_roles,
                yoe = excluded.yoe,
                location_pref = excluded.location_pref,
                remote_ok = excluded.remote_ok,
                min_salary = excluded.min_salary,
                deal_breakers = excluded.deal_breakers,
                updated_at = datetime('now')
            """,
            payload,
        )
        active_conn.commit()
        saved = get_profile(active_conn)
        if not saved:
            raise RuntimeError("Failed to save profile.")
        return saved
    finally:
        if owns_conn:
            active_conn.close()


def _role_expansions(target_roles: list[str], role_synonyms: dict[str, list[str]]) -> list[str]:
    expanded: list[str] = []
    normalized_map = {k.lower(): [v.lower() for v in vals] for k, vals in role_synonyms.items()}
    reverse_map: dict[str, str] = {}
    for canonical, values in normalized_map.items():
        for value in values:
            reverse_map[value] = canonical

    for role in target_roles:
        role_lower = role.lower()
        expanded.append(role_lower)
        if role_lower in normalized_map:
            expanded.extend(normalized_map[role_lower])
        elif role_lower in reverse_map:
            canonical = reverse_map[role_lower]
            expanded.append(canonical)
            expanded.extend(normalized_map.get(canonical, []))
    return _unique_preserve_order(expanded)


def _exclude_terms(yoe: int, deal_breakers: list[str]) -> list[str]:
    if yoe < 3:
        base = ["Staff", "Principal", "Director", "Lead", "Manager"]
    else:
        base = ["Principal", "Director", "VP"]
    return _unique_preserve_order(base + deal_breakers)


def regenerate_config_from_profile(
    profile: dict[str, Any],
    changed_fields: set[str] | None = None,
    force_all: bool = False,
) -> dict[str, Any]:
    config = load_config()
    synonyms = load_synonyms()
    changed = changed_fields or set()

    if force_all or "target_roles" in changed:
        role_synonyms = synonyms.get("role_synonyms", {})
        config["must_have_roles"] = _role_expansions(profile.get("target_roles", []), role_synonyms)

    if force_all or "skills" in changed:
        config["nice_to_have_skills"] = _unique_preserve_order(profile.get("skills", []))

    if force_all or "yoe" in changed or "deal_breakers" in changed:
        config["exclude_terms"] = _exclude_terms(
            int(profile.get("yoe", 0)),
            profile.get("deal_breakers", []),
        )

    save_config(config)
    return config


def init_profile_from_resume(
    resume_path: str | Path,
    name: str,
    target_roles: list[str],
    skills: list[str],
    yoe: int,
    location_pref: str,
    remote_ok: bool,
    min_salary: int,
    deal_breakers: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resume_text = parse_resume_pdf(resume_path)
    resume_embedding = get_embedding(resume_text)
    profile = {
        "id": 1,
        "name": name.strip(),
        "resume_text": resume_text,
        "resume_embedding": resume_embedding,
        "skills": _unique_preserve_order(skills),
        "target_roles": _unique_preserve_order(target_roles),
        "yoe": int(yoe),
        "location_pref": location_pref.strip(),
        "remote_ok": bool(remote_ok),
        "min_salary": int(min_salary),
        "deal_breakers": _unique_preserve_order(deal_breakers),
    }
    saved = upsert_profile(profile)
    config = regenerate_config_from_profile(saved, force_all=True)
    return saved, config


def update_profile_fields(
    resume_path: str | Path | None = None,
    skills: str | None = None,
    roles: str | None = None,
    yoe: int | None = None,
    salary: int | None = None,
    location: str | None = None,
    remote: bool | None = None,
    deal_breakers: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    current = get_profile()
    if not current:
        raise ValueError("Run `agent profile init` first.")

    changed_fields: set[str] = set()
    updated = dict(current)

    if resume_path:
        updated["resume_text"] = parse_resume_pdf(resume_path)
        updated["resume_embedding"] = get_embedding(updated["resume_text"])
        changed_fields.update({"resume_text", "resume_embedding"})

    if skills is not None:
        updated["skills"] = _unique_preserve_order(_split_csv(skills))
        changed_fields.add("skills")

    if roles is not None:
        updated["target_roles"] = _unique_preserve_order(_split_csv(roles))
        changed_fields.add("target_roles")

    if yoe is not None:
        updated["yoe"] = int(yoe)
        changed_fields.add("yoe")

    if salary is not None:
        updated["min_salary"] = int(salary)
        changed_fields.add("min_salary")

    if location is not None:
        updated["location_pref"] = location.strip()
        changed_fields.add("location_pref")

    if remote is not None:
        updated["remote_ok"] = bool(remote)
        changed_fields.add("remote_ok")

    if deal_breakers is not None:
        updated["deal_breakers"] = _unique_preserve_order(_split_csv(deal_breakers))
        changed_fields.add("deal_breakers")

    saved = upsert_profile(updated)
    cfg = regenerate_config_from_profile(saved, changed_fields=changed_fields, force_all=False)
    return saved, cfg, changed_fields
