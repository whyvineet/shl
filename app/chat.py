from app.models import ChatResponse, Recommendation
from app.llm import extract_profile, generate_reply
from app.retrieval import retrieve


def has_enough_info_for_recommendation(profile):
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


def need_clarification(profile):
    if not profile.role:
        return True

    if profile.complete:
        return False

    return not has_enough_info_for_recommendation(profile)


def clarification_question(profile):

    missing = []

    if not profile.role:
        missing.append("the role you're hiring for")

    if not profile.seniority:
        missing.append("the seniority level")

    if not profile.technical_skills:
        missing.append("the key technical skills")

    if not missing:
        return (
            "Could you tell me a little more about the role "
            "and what you want to assess?"
        )

    return (
        "Could you tell me "
        + ", ".join(missing)
        + "?"
    )


async def process_chat(messages):

    profile = extract_profile(messages)

    if need_clarification(profile):

        return ChatResponse(
            reply=clarification_question(profile),
            recommendations=[],
            end_of_conversation=False,
        )

    assessments = retrieve(profile, top_k=5)

    recommendations = []

    for item in assessments:

        recommendations.append(
            Recommendation(
                name=item["name"],
                url=item["link"],
                test_type=", ".join(item.get("keys", [])),
            )
        )

    reply = generate_reply(
        profile,
        [a.model_dump(mode="json") for a in recommendations],
    )

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=False,
    )