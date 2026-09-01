"""
submission/bm25.py — Okapi BM25 ranking.

Required component (assignment Section 4.1): "a BM25 implementation with
tunable k1 and b." See the assignment background (Section 3) for the
Robertson & Walker / Robertson & Zaragoza references this is based on.

BM25 score for a query Q = q1...qn against document D:

    score(D, Q) = sum_i  IDF(qi) * ( tf(qi, D) * (k1 + 1) )
                                   / ( tf(qi, D) + k1 * (1 - b + b * |D| / avgdl) )

A standard IDF variant (Robertson-Sparck Jones, +1-smoothed so it stays
non-negative even for terms occurring in more than half the corpus):

    IDF(qi) = ln( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )

where:
    N        = number of documents in the corpus
    df(qi)   = number of documents containing qi
    tf(qi,D) = term frequency of qi in D
    |D|      = length of D in tokens
    avgdl    = average document length across the corpus

k1 (typically 1.2-2.0) controls term-frequency saturation; b (in [0, 1])
controls document-length normalisation strength. Both must be exposed as
parameters, not hard-coded — you need to sweep them for your report
(assignment Section 8, "parameter search procedure for k1, b").
"""
from math import log
from typing import List, Tuple
from submission.indexer import InvertedIndex, tokenize

_loaded_inverted_index = None

def build(index: InvertedIndex) -> None:
    """Optional: precompute anything BM25-specific (e.g. cached IDF values
    per term) from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""


    global _loaded_inverted_index
    _loaded_inverted_index = index


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first."""

    query_terms = tokenize(query)
    scores = {}

    for term in query_terms:
        if term not in _loaded_inverted_index.postings:
            continue

        idf = get_idf(term)

        for doc_id in _loaded_inverted_index.postings[term]:
            tf = _loaded_inverted_index.postings[term][doc_id]
            doc_len = _loaded_inverted_index.doc_len[doc_id]
            avgdl = _loaded_inverted_index.avg_doc_len

            numerator = tf * (k1 +1)
            denominator = tf + k1 * (1 - b + b * doc_len / avgdl)

            term_score = idf * (numerator/denominator)

            if doc_id not in scores:
                scores[doc_id] = 0.0

            scores[doc_id] += term_score

    rankedList = sorted(scores.items(), key= lambda x: x[1], reverse=True)
    return rankedList[:k]



'''----------------------------------- Helper Functions ----------------------------------'''
def get_idf(term: str) -> float:
    df = get_df(term)
    N = _loaded_inverted_index.N

    numerator = N - df + 0.5
    denominator = df + 0.5

    return log(numerator/denominator + 1)

def get_df(term: str) -> int:
    if term not in _loaded_inverted_index.postings:
        return 0
    
    return len(_loaded_inverted_index.postings[term])

    