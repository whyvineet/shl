from rank_bm25 import BM25Okapi

from app.catalog import (
    load_catalog,
    assessment_to_text,
)

catalog = load_catalog()

documents = [
    assessment_to_text(item)
    for item in catalog
]

tokenized = [
    doc.lower().split()
    for doc in documents
]

bm25 = BM25Okapi(tokenized)


def retrieve(profile, top_k=10):

    query = " ".join(
        filter(
            None,
            [
                profile.role,
                profile.seniority,
                *profile.technical_skills,
                *profile.soft_skills,
                *profile.assessment_types,
            ],
        )
    )

    scores = bm25.get_scores(
        query.lower().split()
    )

    ranked = sorted(
        zip(scores, catalog),
        reverse=True,
        key=lambda x: x[0],
    )

    return [
        assessment
        for _, assessment in ranked[:top_k]
    ]