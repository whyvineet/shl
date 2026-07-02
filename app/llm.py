import asyncio
import json
import logging
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.models import HiringProfile
from app.prompt_manager import (
    render_profile_extraction,
    render_explanation,
    render_comparison,
)

logger = logging.getLogger(__name__)
model = None

LLM_TIMEOUT_SECONDS = 20
LLM_MAX_RETRIES = 2


class LLMError(Exception):
    """Raised when the LLM call fails after retries, or returns unusable
    output. Callers should catch this and degrade gracefully rather than
    let it bubble into a raw 500."""


def get_model():
    global model

    if model is not None:
        return model

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    model = ChatGoogleGenerativeAI(
        model=os.getenv("MODEL", "gemini-2.0-flash"),
        api_key=api_key,
        temperature=0,
    )
    return model


async def _invoke_with_retry(prompt: str) -> str:
    llm = get_model()
    llm_messages = [HumanMessage(content=prompt)]

    last_error = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(llm_messages),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            return response.content.strip()
        except asyncio.TimeoutError as e:
            last_error = e
            logger.warning(f"LLM call timed out (attempt {attempt}/{LLM_MAX_RETRIES})")
        except Exception as e:
            last_error = e
            logger.warning(
                f"LLM call failed (attempt {attempt}/{LLM_MAX_RETRIES}): {e}"
            )

        if attempt < LLM_MAX_RETRIES:
            await asyncio.sleep(0.5 * attempt)

    raise LLMError(f"LLM call failed after {LLM_MAX_RETRIES} attempts: {last_error}")


def _strip_code_fences(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:].strip()
            if "```" in text:
                text = text.split("```")[0].strip()
    return text


def _parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


async def extract_profile(messages) -> HiringProfile:
    conversation = "\n".join(f"{m.role}: {m.content}" for m in messages)

    prompt = render_profile_extraction(conversation)
    prompt += "\n\nRESPOND WITH ONLY VALID JSON, NO OTHER TEXT:"

    try:
        text = await _invoke_with_retry(prompt)
        text = _strip_code_fences(text)
        data = _parse_json_object(text)
        return HiringProfile(**data)
    except (LLMError, RuntimeError):
        raise
    except Exception as e:
        logger.error(f"Profile extraction failed, falling back to empty profile: {e}")
        return HiringProfile()


async def generate_reply(profile, recommendations) -> str:
    profile_json = profile.model_dump_json(indent=2)
    recommendations_json = json.dumps(recommendations, indent=2)

    prompt = render_explanation(profile_json, recommendations_json)

    try:
        return await _invoke_with_retry(prompt)
    except LLMError as e:
        logger.error(f"Reply generation failed: {e}")
        names = ", ".join(r["name"] for r in recommendations) or "no matching assessments"
        return f"Here are the assessments that best match what you've described: {names}."


async def generate_comparison(profile, items) -> str:
    items_json = json.dumps(
        [
            {
                "name": item["name"],
                "description": item.get("description", ""),
                "keys": item.get("keys", []),
                "job_levels": item.get("job_levels", []),
                "duration": item.get("duration", ""),
                "remote": item.get("remote", ""),
                "url": item.get("link", ""),
            }
            for item in items
        ],
        indent=2,
    )

    prompt = render_comparison(items_json)

    try:
        return await _invoke_with_retry(prompt)
    except LLMError as e:
        logger.error(f"Comparison generation failed: {e}")
        lines = [f"- {item['name']}: {item.get('description', '')[:200]}" for item in items]
        return "Here's what the catalog says about each:\n" + "\n".join(lines)
