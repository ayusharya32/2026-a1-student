from math import log
from typing import List, Tuple, Dict
import heapq
from operator import itemgetter

from submission.indexer import InvertedIndex, tokenize


_loaded_inverted_index = None
_idf_cache: Dict[str, float] = {}
_doc_len_ratio: List[float] = []
_doc_norm_cache: Dict[float, List[float]] = {}


def build(index: InvertedIndex) -> None:
    global _loaded_inverted_index
    global _idf_cache
    global _doc_len_ratio
    global _doc_norm_cache

    _loaded_inverted_index = index

    # Precompute IDF once during index loading.
    _idf_cache = {}

    for term in index.vocabulary:
        df = index.document_frequency(term)

        _idf_cache[term] = log(
            (index.N - df + 0.5) / (df + 0.5) + 1
        )

    # Precompute doc length ratios for fast document normalization calculation
    avgdl = index.avg_doc_len if index.avg_doc_len > 0 else 1.0

    _doc_len_ratio = [
        doc_len / avgdl
         for doc_len in index.doc_len_by_int
     ]
    _doc_norm_cache = {}


def get_index() -> InvertedIndex:
    return _loaded_inverted_index


def score(
    query: str,
    k: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> List[Tuple[int, float]]:
    query_terms = tokenize(query)
    scores: Dict[int, float] = {}

    # Local references for the hot loop.
    idf_cache = _idf_cache
    doc_norm = _get_doc_norm(b)
    get_postings = _loaded_inverted_index.get_postings
    scores_get = scores.get

    # Same for every posting.
    k1_plus_one = k1 + 1

    for term in query_terms:
        if term not in idf_cache:
            continue

        idf = idf_cache[term]
        postings = get_postings(term)

        for doc_int, tf in postings.items():
            numerator = tf * k1_plus_one

            denominator = (
                tf + k1 * doc_norm[doc_int]
            )

            term_score = idf * (
                numerator / denominator
            )

            scores[doc_int] = (
                scores_get(doc_int, 0.0)
                + term_score
            )

    return heapq.nlargest(
        k,
        scores.items(),
        key=itemgetter(1),
    )


def get_idf(term: str) -> float:
    df = _loaded_inverted_index.document_frequency(term)
    N = _loaded_inverted_index.N

    numerator = N - df + 0.5
    denominator = df + 0.5

    return log(
        numerator / denominator + 1
    )


def get_df(term: str) -> int:
    return _loaded_inverted_index.document_frequency(term)

def _get_doc_norm(b: float) -> List[float]:
    if b not in _doc_norm_cache:
        one_minus_b = 1.0 - b
        _doc_norm_cache[b] = [
            one_minus_b + b * ratio
            for ratio in _doc_len_ratio
        ]
    return _doc_norm_cache[b]