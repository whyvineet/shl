from app.prompt_manager import (
    render_system_prompt,
    render_profile_extraction,
    render_explanation,
)

SYSTEM_PROMPT = render_system_prompt()
PROFILE_PROMPT = "See prompt_manager.render_profile_extraction()"
EXPLANATION_PROMPT = "See prompt_manager.render_explanation()"