from math import log
from typing import List, Tuple
import heapq
from submission.indexer import InvertedIndex, tokenize

_loaded_inverted_index = None

def build(index: InvertedIndex) -> None:
    global _loaded_inverted_index
    _loaded_inverted_index = index


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
    query_terms = tokenize(query)
    scores = {}

    avgdl = _loaded_inverted_index.avg_doc_len
    doc_lens = _loaded_inverted_index.doc_len

    for term in query_terms:
        df = _loaded_inverted_index.document_frequency(term)
        if df == 0:
            continue

        idf = get_idf(term)
        postings = _loaded_inverted_index.get_postings(term)

        for doc_id, tf in postings.items():
            doc_len = doc_lens[doc_id]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))

            term_score = idf * (numerator / denominator)

            if doc_id not in scores:
                scores[doc_id] = 0.0

            scores[doc_id] += term_score

    return heapq.nlargest(k, scores.items(), key=lambda x: x[1])


def get_idf(term: str) -> float:
    df = _loaded_inverted_index.document_frequency(term)
    N = _loaded_inverted_index.N

    numerator = N - df + 0.5
    denominator = df + 0.5

    return log(numerator / denominator + 1)

def get_df(term: str) -> int:
    return _loaded_inverted_index.document_frequency(term)

    