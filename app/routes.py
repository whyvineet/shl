from fastapi import APIRouter, HTTPException

from app.models import ChatRequest, ChatResponse
from app.chat import process_chat

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        return await process_chat(request.messages)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc