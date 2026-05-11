"""API request/response schemas — strict Pydantic v2 models matching the spec."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Annotated

# Single-letter test type codes derived from the catalog's `keys` field.
# K=Knowledge & Skills, P=Personality & Behavior, A=Ability & Aptitude,
# S=Simulations, B=Biodata & Situational Judgment, C=Competencies,
# D=Development & 360, E=Assessment Exercises.
# A `test_type` may be a comma-joined multi-letter code (e.g. "K,S") for items
# tagged with multiple categories — matches sample-conversation conventions.
TestTypeCode = Annotated[str, StringConstraints(min_length=1, max_length=15, pattern=r"^[KPASBCDE](,[KPASBCDE])*$")]


class Message(BaseModel):
    """A single conversation turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    """Inbound /chat request — stateless, full conversation history."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def latest_turn_must_be_user(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("latest message must be from user")
        return self


class Recommendation(BaseModel):
    """A single shortlisted assessment in the response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    test_type: TestTypeCode


class ChatResponse(BaseModel):
    """Outbound /chat response — strict schema; fixed shape."""

    model_config = ConfigDict(extra="forbid")

    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    """Readiness response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
