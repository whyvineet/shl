import logging

from fastapi import APIRouter, HTTPException

from app.models import ChatRequest, ChatResponse
from app.chat import process_chat

logger = logging.getLogger(__name__)
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
    except Exception as exc:
        logger.exception(f"Unhandled error in /chat: {exc}")
        return ChatResponse(
            reply=(
                "Sorry, I ran into a temporary issue processing that. "
                "Could you try rephrasing or resending your message?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )