from math import log
from typing import List, Tuple, Dict
import heapq
from operator import itemgetter

from submission.indexer import InvertedIndex, tokenize


_loaded_inverted_index = None
_idf_cache: Dict[str, float] = {}
_doc_norm: List[float] = []


def build(index: InvertedIndex) -> None:
    global _loaded_inverted_index
    global _idf_cache
    global _doc_norm

    _loaded_inverted_index = index

    # Precompute IDF once during index loading.
    _idf_cache = {}

    for term in index.vocabulary:
        df = index.document_frequency(term)

        _idf_cache[term] = log(
            (index.N - df + 0.5) / (df + 0.5) + 1
        )

    # Precompute the document-length normalization part
    # of the BM25 denominator.
    avgdl = index.avg_doc_len
    b = 0.6

    _doc_norm = [
        1 - b + b * doc_len / avgdl
        for doc_len in index.doc_len_by_int
    ]


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
    doc_norm = _doc_norm
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