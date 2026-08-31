"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces to implement:

1. Boolean retrieval: given a query, treat it as an AND (conjunctive) or
   OR (disjunctive) combination of terms and return the matching document
   set — no ranking, just set membership. Useful as a fast candidate
   filter and as a sanity check ("does my index even find the right
   documents for this query?").

2. Vector-space ranking: represent the query and each candidate document
   as TF-IDF weighted vectors and rank by cosine similarity. A standard
   TF-IDF weight for term t in document d:

       w(t, d) = tf(t, d) * log( N / df(t) )

   (log base is your choice — just be consistent), and cosine similarity
   between query vector q and document vector d:

       sim(q, d) = (q . d) / (||q|| * ||d||)

Both pieces should read from the same InvertedIndex you build in
indexer.py.
"""
from typing import List, Tuple
from math import log, sqrt

from submission.indexer import InvertedIndex, tokenize

_loaded_inverted_index = None

def build(index: InvertedIndex) -> None:
    """Optional: precompute anything VSM-specific (e.g. document vector
    norms) from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""

    global _loaded_inverted_index
    _loaded_inverted_index = index

def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return the (unranked) list of doc_ids matching `query`, treating it
    as a conjunction (`mode="and"`) or disjunction (`mode="or"`) of its
    terms."""

    query_terms = tokenize(query)

    if not query_terms:
        return []
    
    posting_sets = []

    for term in query_terms:
        if term not in _loaded_inverted_index.postings:
            if mode == "and":
                return []
            continue

        docs_ids_containing_term = set(_loaded_inverted_index.postings[term].keys())
        posting_sets.append(docs_ids_containing_term)

    if not posting_sets:
        return []
    
    if mode == "and":
        result = posting_sets[0]

        for docs in posting_sets[1:]:
            result = result & docs

    elif mode == "or":
        result = posting_sets[0]
    
        for docs in posting_sets[1:]:
            result = result | docs

    else:
        raise ValueError("mode must be 'and' or 'or'")

    return list(result)


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    TF-IDF cosine similarity, highest score first."""

    query_terms = tokenize(query)
    if not query_terms:
        return []

    N = _loaded_inverted_index.N

    query_weights = {}

    for term in query_terms:
        if term not in _loaded_inverted_index.postings:
            continue

        df = len(_loaded_inverted_index.postings[term])
        if df == 0:
            continue

        idf = log(N / df)
        query_weights[term] = (
            query_weights.get(term, 0.0) + idf
        )

    if not query_weights:
        return []

    query_norm = sqrt(
        sum(weight * weight for weight in query_weights.values())
    )

    scores = {}

    # For all the docs ids containing any one of the query terms at least once
    candidate_docs = set()

    for term in query_weights:
        candidate_docs.update(_loaded_inverted_index.postings[term].keys())

    for doc_id in candidate_docs:
        dot_product = 0.0
        doc_norm = 0.0

        for term in query_weights:
            if doc_id not in _loaded_inverted_index.postings[term]:
                continue

            tf = _loaded_inverted_index.postings[term][doc_id]
            df = len(_loaded_inverted_index.postings[term])
            idf = log(N / df)

            doc_weight = tf * idf

            dot_product += query_weights[term] * doc_weight
            doc_norm += doc_weight * doc_weight

        doc_norm = sqrt(doc_norm)

        if doc_norm > 0:
            scores[doc_id] = dot_product / (query_norm * doc_norm)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]