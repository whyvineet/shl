import logging

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.catalog import (
    load_catalog,
    assessment_to_text,
    filter_catalog,
)

logger = logging.getLogger(__name__)

catalog = load_catalog()

documents = [assessment_to_text(item) for item in catalog]

tokenized = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized)


tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    stop_words="english",
)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)


def _normalize(scores):
    scores = np.array(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def _build_query(profile):
    return " ".join(
        filter(
            None,
            [
                profile.role,
                profile.seniority,
                profile.industry,
                *profile.technical_skills,
                *profile.soft_skills,
                *profile.assessment_types,
            ],
        )
    )


def retrieve(profile, top_k=10, bm25_weight=0.6, tfidf_weight=0.4, min_score=0.05):
    query = _build_query(profile)

    if not query.strip():
        return []

    candidates = catalog
    if profile.remote is not None or profile.language:
        filtered = filter_catalog(
            catalog,
            remote=profile.remote,
            language=profile.language,
        )
        
        if filtered:
            candidates = filtered

    candidate_idx = [catalog.index(item) for item in candidates]

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_scores = np.array([bm25_scores[i] for i in candidate_idx])

    query_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(
        query_vec, tfidf_matrix[candidate_idx]
    ).ravel()

    combined = (
        bm25_weight * _normalize(bm25_scores)
        + tfidf_weight * _normalize(tfidf_scores)
    )

    ranked = sorted(
        zip(combined, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = [item for score, item in ranked if score >= min_score]

    if not results:
        results = [item for _, item in ranked]

    return results[:top_k]
