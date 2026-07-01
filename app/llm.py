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
)

logger = logging.getLogger(__name__)
model = None


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


def extract_profile(messages):
    model = get_model()

    conversation = "\n".join(
        f"{m.role}: {m.content}"
        for m in messages
    )

    full_prompt = render_profile_extraction(conversation)
    full_prompt += "\n\nRESPOND WITH ONLY VALID JSON, NO OTHER TEXT:"

    llm_messages = [
        HumanMessage(content=full_prompt),
    ]

    response = model.invoke(llm_messages)
    text = response.content.strip()

    logger.debug(f"LLM Response: {text[:300]}")

    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:].strip()
            if "```" in text:
                text = text.split("```")[0].strip()

    try:
        data = json.loads(text)
        return HiringProfile(**data)
    except json.JSONDecodeError as e:
        logger.debug(f"First attempt failed. Raw text: {text[:500]}")
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                logger.debug("Successfully extracted JSON from response")
                return HiringProfile(**data)
            except json.JSONDecodeError:
                pass
        
        raise ValueError(
            f"LLM did not return valid JSON. "
            f"Response started with: {text[:100]}... "
            f"Expected JSON format."
        ) from e


def generate_reply(profile, recommendations):
    model = get_model()

    profile_json = profile.model_dump_json(indent=2)
    recommendations_json = json.dumps(recommendations, indent=2)

    prompt = render_explanation(profile_json, recommendations_json)

    llm_messages = [
        HumanMessage(content=prompt),
    ]

    response = model.invoke(llm_messages)

    return response.content