from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    id: int = 1
    name: Optional[str] = None
    resume_text: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    yoe: int = 0
    location_pref: Optional[str] = None
    remote_ok: bool = True
    min_salary: int = 0
    deal_breakers: list[str] = Field(default_factory=list)


class JobPosting(BaseModel):
    id: Optional[int] = None
    source: str = "jsearch"
    external_id: Optional[str] = None
    title: str
    company: str
    location: Optional[str] = None
    remote: bool = False
    description: Optional[str] = None
    url: Optional[str] = None
    keyword_score: float = 0.0
    status: str = "new"


class MatchResult(BaseModel):
    job_id: int
    final_score: float
    embedding_similarity: float
    llm_score: Optional[float] = None
    keyword_score: float = 0.0
    rationale: Optional[str] = None
    skill_gaps: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)


class Application(BaseModel):
    job_id: int
    status: str = "saved"
    notes: Optional[str] = None
