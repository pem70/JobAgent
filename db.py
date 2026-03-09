from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("agent.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_profile (
    id INTEGER PRIMARY KEY DEFAULT 1,
    name TEXT,
    resume_text TEXT,
    resume_embedding BLOB,
    skills TEXT,
    target_roles TEXT,
    yoe INTEGER DEFAULT 0,
    location_pref TEXT,
    remote_ok INTEGER DEFAULT 1,
    min_salary INTEGER DEFAULT 0,
    deal_breakers TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS job_posting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT DEFAULT 'jsearch',
    external_id TEXT,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    remote INTEGER DEFAULT 0,
    description TEXT,
    url TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    parsed_skills TEXT,
    min_yoe INTEGER,
    visa_sponsorship INTEGER,
    embedding BLOB,
    keyword_score REAL DEFAULT 0.0,
    posted_at TEXT,
    scraped_at TEXT DEFAULT (datetime('now')),
    status TEXT DEFAULT 'new',
    interaction TEXT,
    UNIQUE(source, external_id)
);

CREATE TABLE IF NOT EXISTS match_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES job_posting(id),
    final_score REAL,
    embedding_similarity REAL,
    llm_score REAL,
    keyword_score REAL,
    rationale TEXT,
    skill_gaps TEXT,
    red_flags TEXT,
    computed_at TEXT DEFAULT (datetime('now')),
    UNIQUE(job_id)
);

CREATE TABLE IF NOT EXISTS application (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES job_posting(id),
    status TEXT DEFAULT 'saved',
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(job_id)
);

CREATE INDEX IF NOT EXISTS idx_job_status ON job_posting(status);
CREATE INDEX IF NOT EXISTS idx_job_company_title ON job_posting(company, title);
CREATE INDEX IF NOT EXISTS idx_application_status ON application(status);
"""


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()
