"""Pydantic models for the spec the engine produces and the API around it.

Every claim in a spec carries `sources`, the chunk ids it came from, so a reviewer
can click through to the original note instead of trusting the model.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    chunk_id: str
    artifact: str = ""


class Role(BaseModel):
    name: str
    goals: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)


class Feature(BaseModel):
    id: str
    name: str
    description: str
    priority: Literal["must", "should", "could"] = "should"
    sources: list[SourceRef] = Field(default_factory=list)


class Entity(BaseModel):
    name: str
    fields: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)


class Screen(BaseModel):
    name: str
    purpose: str = ""
    sources: list[SourceRef] = Field(default_factory=list)


class Constraint(BaseModel):
    text: str
    kind: Literal["security", "compliance", "technical", "business"] = "technical"
    sources: list[SourceRef] = Field(default_factory=list)


class OpenQuestion(BaseModel):
    id: str
    question: str
    why_it_matters: str = ""
    sources: list[SourceRef] = Field(default_factory=list)


class Spec(BaseModel):
    title: str
    summary: str
    users: list[Role] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    screens: list[Screen] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)


class RunStatus(StrEnum):
    needs_answers = "needs_answers"
    in_review = "in_review"
    approved = "approved"
    failed = "failed"


# ---------- API request bodies ----------


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class TextArtifactIn(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=500_000)


class RunIn(BaseModel):
    provider: str | None = None


class AnswersIn(BaseModel):
    answers: dict[str, str] = Field(min_length=1)


class ReviewIn(BaseModel):
    decision: Literal["approve", "request_changes"]
    note: str = Field(default="", max_length=4000)
