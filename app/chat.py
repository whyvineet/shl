import logging

from app.catalog import find_matches
from app.models import ChatResponse, Recommendation
from app.llm import extract_profile, generate_reply, generate_comparison
from app.retrieval import retrieve

logger = logging.getLogger(__name__)

MAX_TURNS = 8
FORCE_RECOMMEND_AFTER_MESSAGES = MAX_TURNS - 2  # leave room for one more exchange


def has_enough_info_for_recommendation(profile) -> bool:
    return bool(
        profile.role
        and profile.seniority
        and (
            profile.language
            or profile.technical_skills
            or profile.soft_skills
            or profile.assessment_types
            or profile.industry
        )
    )


def need_clarification(profile, turn_count: int) -> bool:
    if turn_count >= FORCE_RECOMMEND_AFTER_MESSAGES and profile.role:
        return False

    if not profile.role:
        return True

    return not has_enough_info_for_recommendation(profile)


def clarification_question(profile) -> str:
    missing = []

    if not profile.role:
        missing.append("the role you're hiring for")

    if not profile.seniority:
        missing.append("the seniority level")

    if not (profile.technical_skills or profile.soft_skills or profile.assessment_types):
        missing.append("the key skills or competencies you want to assess")

    if not missing:
        return "Could you tell me a little more about the role and what you want to assess?"

    return "Could you tell me " + ", ".join(missing) + "?"


def _build_recommendations(items) -> list[Recommendation]:
    recommendations = []
    for item in items:
        recommendations.append(
            Recommendation(
                name=item["name"],
                url=item["link"],
                test_type=", ".join(item.get("keys", [])) or "General",
            )
        )
    return recommendations


REFUSAL_OFF_TOPIC = (
    "I can only help with selecting or comparing SHL assessments - I'm not "
    "able to give general hiring, legal, or compliance advice. If you'd "
    "like, tell me about the role you're hiring for and I can suggest "
    "relevant assessments."
)

REFUSAL_INJECTION = (
    "I'm only able to help with SHL assessment selection and comparison, "
    "and I can't change my instructions or role. Let me know about the "
    "hiring need you'd like help with instead."
)

CHITCHAT_REPLY = (
    "Hi! I can help you find the right SHL assessments for a role you're "
    "hiring for, or compare specific assessments. What role are you "
    "hiring for?"
)


async def process_chat(messages) -> ChatResponse:
    if not messages:
        return ChatResponse(
            reply=CHITCHAT_REPLY,
            recommendations=[],
            end_of_conversation=False,
        )

    turn_count = len(messages)

    profile = await extract_profile(messages)

    if profile.intent == "off_topic":
        return ChatResponse(reply=REFUSAL_OFF_TOPIC, recommendations=[], end_of_conversation=False)

    if profile.intent == "injection":
        return ChatResponse(reply=REFUSAL_INJECTION, recommendations=[], end_of_conversation=False)

    if profile.intent == "chitchat":
        return ChatResponse(reply=CHITCHAT_REPLY, recommendations=[], end_of_conversation=False)

    if profile.intent == "compare":
        found, unresolved = find_matches(profile.compare_targets)

        if len(found) < 2:
            names = ", ".join(profile.compare_targets) or "those assessments"
            missing_note = f" I couldn't find {', '.join(unresolved)} in the catalog." if unresolved else ""
            return ChatResponse(
                reply=(
                    f"I couldn't confidently match enough of {names} in the SHL catalog "
                    f"to compare.{missing_note} Could you give me the exact assessment name(s)?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        reply = await generate_comparison(profile, found)
        recommendations = _build_recommendations(found)

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=False,
        )

    if need_clarification(profile, turn_count):
        return ChatResponse(
            reply=clarification_question(profile),
            recommendations=[],
            end_of_conversation=False,
        )

    items = retrieve(profile, top_k=10)

    if not items:
        return ChatResponse(
            reply=(
                "I couldn't find assessments in the catalog that clearly match what "
                "you've described so far. Could you share more detail on the skills "
                "or competencies you want to assess?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    recommendations = _build_recommendations(items)

    reply = await generate_reply(
        profile,
        [r.model_dump(mode="json") for r in recommendations],
    )

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=True,
    )