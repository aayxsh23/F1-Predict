"""Request bodies for the POST endpoints: the only user-written input this API
takes, so the only place limits are enforced."""
from typing import Literal

from pydantic import BaseModel, Field

Target = Literal["finish_position", "qualifying", "quali_delta", "race_time"]


class ExplainRequest(BaseModel):
    season: int
    round: int
    driver: str = Field(max_length=3)
    target: Target = "finish_position"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatContext(BaseModel):
    season: int | None = None
    round: int | None = None
    race_name: str | None = Field(default=None, max_length=80)
    driver: str | None = Field(default=None, max_length=3)


class ChatRequest(BaseModel):
    messages: list[ChatTurn] = Field(min_length=1, max_length=40)
    context: ChatContext | None = None
