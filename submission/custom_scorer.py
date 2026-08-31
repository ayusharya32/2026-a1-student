"""
submission/custom_scorer.py — optional combined/custom scorer.

Not required, but this is explicitly called out in the assignment
(Section 4.1) as "where separation in the leaderboard tends to happen":
any linear or non-linear combination of your Boolean/VSM and BM25
signals, additional features (e.g. proximity/bigram overlap), or your
own heuristic.

If you use this, wire it in from submission/retrieve.py's retrieve()
instead of calling a single scorer directly, and describe what you did
and why in your report (Section 7, "one-paragraph description of your
final competition entry").
"""
from typing import List, Tuple

from submission.indexer import InvertedIndex, tokenize
from submission import bm25, boolean_vsm
import os

_loaded_inverted_index = None

BM25_K1 = float(os.getenv("BM25_K1", "2.25"))
BM25_B = float(os.getenv("BM25_B", "0.5"))

def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(), not retrieve.build_index() — the
    harness runs those two in separate processes. Anything this needs at
    query time either comes from the loaded InvertedIndex or must have
    been written to index_dir by InvertedIndex.save() (which then counts
    toward your index-size score)."""

    global _loaded_inverted_index

    _loaded_inverted_index = index

    bm25.build(index)
    boolean_vsm.build(index)


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by your
    own combined/custom scoring function, highest score first."""


    bm25_results = bm25.score(query, k, k1=BM25_K1, b=BM25_B)
    vsm_results = boolean_vsm.vsm_score(query, k)

    bm25_scores = dict(bm25_results)
    vsm_scores = dict(vsm_results)

    all_docs = set(bm25_scores) | set(vsm_scores)

    scores = {}

    for doc_id in all_docs:
        bm25_score = bm25_scores.get(doc_id, 0.0)
        vsm_score = vsm_scores.get(doc_id, 0.0)

        scores[doc_id] = bm25_score + vsm_score

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:k]
