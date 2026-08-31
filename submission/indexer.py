"""
submission/indexer.py — build your inverted index here.

This is one of the required components (assignment Section 4.1): you must
build the inverted index yourself, without an existing search/indexing
library (Lucene, Elasticsearch, Pyserini, Whoosh, etc.).

A `tokenize()` helper is provided below purely so that tokenization is
consistent across your Boolean/VSM and BM25 scorers —
feel free to replace it (e.g. add stemming or stopword removal), just make
sure every scorer that reads this index was built with the same tokenizer.

Everything else — the postings representation, what per-document and
collection statistics you track, whether you add positions for
proximity/phrase features — is your design decision. `InvertedIndex`
below sketches a minimal, obviously-sufficient shape; you do not have to
use it, but if you do, filling in `build()` and `document_frequency()` is
enough to support Boolean/VSM and BM25.

Persistence (assignment Section 4.1 / Section 7 "index size" scoring):
`build_index()` in retrieve.py runs in one process and `load_index()` runs
in a separate, later one — so whatever this index needs at query time must
round-trip through `save()`/`load()` below, not just live as Python
attributes. The on-disk byte size of what `save()` writes is graded
directly (smaller, relative to the class median, scores better), so a
compact postings encoding is worth more here than in most course
assignments — see the `save()` docstring for concrete starting points.
"""
import re
import os
import json
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric-only tokenization."""
    tokens = _TOKEN_RE.findall(text.lower())

    stemmed = []

    for token in tokens:
        # plural forms
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]

        # common verb/adjective forms
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 5 and token.endswith("ed"):
            token = token[:-2]

        stemmed.append(token)

    return stemmed


class InvertedIndex:
    """A minimal inverted index skeleton. Extend the data structures here
    however your design needs (e.g. term positions for phrase/proximity
    scoring, a more compact postings representation for the efficiency
    bonus) — this is a starting point, not a fixed schema.
    """

    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: term_freq}
        self.doc_len: Dict[str, int] = {}  # doc_id -> number of tokens
        self.doc_text: Dict[str, str] = {}  # doc_id -> raw text (handy for VSM/debugging)
        self.N: int = 0  # number of documents
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """corpus: list of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.load_corpus()."""

        self.postings = {}
        self.doc_len = {}
        self.doc_text = {}

        for (doc_id, text) in corpus:
            tokens = tokenize(text)

            self.doc_len[doc_id] = len(tokens)
            self.doc_text[doc_id] = text

            for term in tokens:
                if term not in self.postings:
                    self.postings[term] = {}

                if doc_id not in self.postings[term]:
                    self.postings[term][doc_id] = 0

                self.postings[term][doc_id] += 1

        self.N = len(corpus)

        if self.N > 0:
            self.avg_doc_len = sum(self.doc_len.values()) / self.N
        else: 
            self.avg_doc_len = 0.0


    def document_frequency(self, term: str) -> int:

        if term not in self.postings: 
            return 0

        return len(self.postings[term])


    def save(self, index_dir: str) -> None:
        """Persist everything document_frequency() / your scorers need to
        `index_dir`, so `load()` can reconstruct this object in a fresh
        process with no memory of `build()` ever having run. Called from
        retrieve.build_index().

        The on-disk byte size of whatever you write here is graded
        directly (assignment Section 7, "index size", relative to the
        class median) — some starting points, roughly in order of effort:
          - json/pickle-dump self.postings etc. directly (works, but
            verbose: repeats every doc_id string per posting).
          - drop self.doc_text if your scorers don't need raw text at
            query time (BM25/VSM only need term-frequency and length
            statistics, not the original documents).
          - delta-encode each postings list's doc-ids (sorted ascending,
            store gaps instead of absolute ids) and varint/byte-pack them,
            instead of a naive JSON list of integers.

        TODO(you): implement.
        """

        os.makedirs(index_dir, exist_ok=True)

        index_file_data = {}
        index_file_data["postings"] = self.postings
        index_file_data["doc_len"] = self.doc_len
        index_file_data["N"] = self.N
        index_file_data["avg_doc_len"] = self.avg_doc_len

        index_file_path = os.path.join(index_dir, "index.json")
        with open(index_file_path, "w", encoding="utf-8") as file:
            json.dump(index_file_data, file, indent=4)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct an InvertedIndex purely from what save() wrote to
        `index_dir`. Called in a fresh process — do not rely on any state
        other than what's actually on disk in `index_dir`.

        TODO(you): implement, matching whatever format save() wrote.
        """

        index_file_path = os.path.join(index_dir, "index.json")
        with open(index_file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        inverted_index = cls()
        inverted_index.postings = json_data["postings"]
        inverted_index.doc_len = json_data["doc_len"]
        inverted_index.N = json_data["N"]
        inverted_index.avg_doc_len = json_data["avg_doc_len"]

        return inverted_index