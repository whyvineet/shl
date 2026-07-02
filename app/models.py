from typing import List, Literal

from pydantic import BaseModel, HttpUrl


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: HttpUrl
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


Intent = Literal["recommend", "compare", "off_topic", "injection", "chitchat"]


class HiringProfile(BaseModel):
    intent: Intent = "recommend"

    compare_targets: List[str] = []

    role: str | None = None
    seniority: str | None = None

    technical_skills: List[str] = []
    soft_skills: List[str] = []

    assessment_types: List[str] = []

    industry: str | None = None

    language: str | None = None

    remote: bool | None = None

    max_duration: int | None = None

    complete: bool = False