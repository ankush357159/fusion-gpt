from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings
from .model import MistralCore

app = FastAPI(title="VeraGPT - Mistral-Core")
model = MistralCore()


class Message(BaseModel):
    role: str = Field(..., examples=["user", "assistant", "system"])
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


class ChatResponse(BaseModel):
    model: str
    response: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    response = model.generate(
        [m.model_dump() for m in req.messages],
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return ChatResponse(model=settings.model_name, response=response)
