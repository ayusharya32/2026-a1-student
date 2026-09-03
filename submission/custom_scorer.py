"""
submission/custom_scorer.py — optional combined/custom scorer.

Ensemble of BM25 candidate retrieval, normalized VSM TF-IDF re-ranking,
and Query Term Coverage coordinate matching.
"""
import os
import math
from typing import List, Tuple, Dict

from submission.indexer import InvertedIndex, tokenize
from submission import bm25, boolean_vsm

_loaded_inverted_index = None

BM25_K1 = float(os.getenv("BM25_K1", "1.8"))
BM25_B = float(os.getenv("BM25_B", "0.55"))
CANDIDATE_POOL_SIZE = int(os.getenv("CANDIDATE_POOL_SIZE", "200"))
VSM_ALPHA = float(os.getenv("VSM_ALPHA", "0.22"))
COVERAGE_WEIGHT = float(os.getenv("COVERAGE_WEIGHT", "0.25"))

def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(), not retrieve.build_index()."""
    global _loaded_inverted_index
    _loaded_inverted_index = index

    bm25.build(index)
    boolean_vsm.build(index)


def score(query: str, k: int) -> List[Tuple[int, float]]:
    """Return up to k (doc_int, score) pairs for `query`, ranked by an
    ensemble of BM25 candidate retrieval, normalized VSM TF-IDF re-ranking,
    and Query Term Coverage coordinate matching."""
    candidates = bm25.score(query, k=max(k, CANDIDATE_POOL_SIZE), k1=BM25_K1, b=BM25_B)
    if not candidates:
        return []

    idf_cache = bm25._idf_cache
    q_terms = [t for t in tokenize(query) if t in idf_cache]
    if not q_terms:
        return candidates[:k]

    unique_q = list(set(q_terms))
    num_unique = len(unique_q)
    q_weights = {t: idf_cache[t] for t in q_terms}
    q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
    cand_set = {doc_int for doc_int, _ in candidates}

    # Fetch postings restricted to the candidate pool
    term_postings = {}
    for t in unique_q:
        postings = _loaded_inverted_index.get_postings(t)
        term_postings[t] = {d: tf for d, tf in postings.items() if d in cand_set}

    doc_lens = _loaded_inverted_index.doc_len_by_int
    vsm_scores = {}
    coverage_ratios = {}
    for doc_int, _ in candidates:
        dot_product = 0.0
        matched = 0
        for t in unique_q:
            tf = term_postings[t].get(doc_int, 0)
            if tf > 0:
                dot_product += q_weights.get(t, 1.0) * (tf * idf_cache[t])
                matched += 1

        d_norm = math.sqrt(doc_lens[doc_int] + 1.0)
        vsm_scores[doc_int] = (dot_product / (q_norm * d_norm)) if d_norm > 0 else 0.0
        coverage_ratios[doc_int] = matched / num_unique

    # Min-max scale alignment
    max_b = candidates[0][1]
    min_b = candidates[-1][1]
    range_b = (max_b - min_b) if (max_b - min_b) > 1e-6 else 1.0

    max_v = max(vsm_scores.values()) if vsm_scores else 1.0
    min_v = min(vsm_scores.values()) if vsm_scores else 0.0
    range_v = (max_v - min_v) if (max_v - min_v) > 1e-6 else 1.0

    blended = []
    alpha = VSM_ALPHA
    cw = COVERAGE_WEIGHT
    for doc_int, b_score in candidates:
        norm_b = (b_score - min_b) / range_b
        norm_v = (vsm_scores[doc_int] - min_v) / range_v
        final_score = (norm_b + alpha * norm_v) * (1.0 + cw * coverage_ratios[doc_int])
        blended.append((doc_int, final_score))

    blended.sort(key=lambda x: x[1], reverse=True)
    return blended[:k]