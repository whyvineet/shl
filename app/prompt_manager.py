from pathlib import Path
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parent / "templates"
jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_profile_extraction(conversation: str) -> str:
    template = jinja_env.get_template("profile_extraction.jinja2")
    return template.render(conversation=conversation)


def render_explanation(profile: str, recommendations: str) -> str:
    template = jinja_env.get_template("explanation.jinja2")
    return template.render(profile=profile, recommendations=recommendations)


def render_comparison(items: str) -> str:
    template = jinja_env.get_template("comparison.jinja2")
    return template.render(items=items)
