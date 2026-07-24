from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class IntentAnalysis(BaseModel):
    scene: str = Field(min_length=1, max_length=80)
    goal: str = Field(min_length=1, max_length=160)
    features: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(
        min_length=1,
        max_length=6,
    )
    keywords: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(
        min_length=1,
        max_length=6,
    )
    risk_flags: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(
        default_factory=list,
        max_length=6,
    )


class ChatResult(BaseModel):
    skill: Literal["goutoujunshi"] = "goutoujunshi"
    intent: str = Field(min_length=1, max_length=80)
    judgement: str = Field(min_length=1, max_length=500)
    recommended_reply: str = Field(min_length=1, max_length=1000)
    alternatives: list[Annotated[str, Field(min_length=1, max_length=1000)]] = (
        Field(min_length=2, max_length=2)
    )
    warning: str | None = Field(default=None, max_length=500)


class GeneratedChatResult(BaseModel):
    """Fields the model may generate for a final chat response.

    The server, not the model, owns the response intent and assistant identity.
    Ignoring extra fields keeps an echoed ``<INTENT>`` context from invalidating
    an otherwise usable response.
    """

    model_config = ConfigDict(extra="ignore")

    judgement: str = Field(min_length=1, max_length=500)
    recommended_reply: str = Field(min_length=1, max_length=1000)
    alternatives: list[Annotated[str, Field(min_length=1, max_length=1000)]] = (
        Field(min_length=2, max_length=2)
    )
    warning: str | None = Field(default=None, max_length=500)
