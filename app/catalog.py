import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "products.json"

_catalog = None
_catalog_by_id = {}
_catalog_by_name = {}


def load_catalog():

    global _catalog, _catalog_by_id, _catalog_by_name

    if _catalog is not None:
        return _catalog

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        _catalog = json.load(f)

    _catalog_by_id = {
        item["entity_id"]: item
        for item in _catalog
    }

    _catalog_by_name = {
        item["name"].lower(): item
        for item in _catalog
    }

    logger.info(f"Loaded {len(_catalog)} assessments")

    return _catalog


def get_catalog():
    if _catalog is None:
        return load_catalog()
    return _catalog


def get_by_id(entity_id: str):
    return _catalog_by_id.get(entity_id)


def get_by_name(name: str):
    return _catalog_by_name.get(name.lower())


def filter_catalog(
    assessments,
    job_level=None,
    remote=None,
    language=None,
    categories=None,
):

    results = []

    for assessment in assessments:

        if job_level:
            if job_level not in assessment.get("job_levels", []):
                continue

        if remote is not None:
            if assessment.get("remote") != ("yes" if remote else "no"):
                continue

        if language:
            if language not in assessment.get("languages", []):
                continue

        if categories:
            keys = assessment.get("keys", [])

            if not any(cat in keys for cat in categories):
                continue

        results.append(assessment)

    return results


def assessment_to_text(assessment):

    return " ".join(
        [
            assessment.get("name", ""),
            assessment.get("description", ""),
            " ".join(assessment.get("keys", [])),
            " ".join(assessment.get("job_levels", [])),
            " ".join(assessment.get("languages", [])),
        ]
    )