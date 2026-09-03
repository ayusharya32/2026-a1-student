import json
import os
from typing import List, Optional, Tuple

from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex, tokenize
from submission import bm25, boolean_vsm, custom_scorer

_DOC_ORDER: Optional[List[str]] = None
_DOC_ORDER_FILENAME = "doc_order.json"

BM25_K1 = float(os.getenv("BM25_K1", "1.8"))
BM25_B = float(os.getenv("BM25_B", "0.55"))

def build_index(corpus_path: str, index_dir: str) -> None:
    corpus = load_corpus(corpus_path)
    index = InvertedIndex()
    index.build(corpus)
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    index = InvertedIndex.load(index_dir)
    bm25.build(index)
    boolean_vsm.build(index)
    custom_scorer.build(index)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    results = custom_scorer.score(query, k)
    index = bm25.get_index()

    return [
        (index.int_to_doc[doc_int] if isinstance(doc_int, int) else doc_int, score)
        for doc_int, score in results
    ]


def _baseline_retrieve(query: str, k: int) -> List[Tuple[str, float]]:
    assert _DOC_ORDER is not None
    top = _DOC_ORDER[:k]
    return [(doc_id, float(len(top) - i)) for i, doc_id in enumerate(top)]