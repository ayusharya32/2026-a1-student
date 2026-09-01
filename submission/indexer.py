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
import struct
from typing import Dict, List, Tuple
from nltk.corpus import stopwords

_TOKEN_RE = re.compile(
    r"""
    [a-z]+(?:'[a-z]+)?          # words / contractions
    |
    [a-z]+\d+|\d+[a-z]+         # alphanumeric terms: il6, 10mg
    |
    \d+(?:\.\d+)?               # numbers / decimals
    """,
    re.IGNORECASE | re.VERBOSE
)

STOPWORDS = set(stopwords.words("english"))

def stem_word(token: str) -> str:
    # Don't stem very short words.
    if len(token) <= 3:
        return token

    # ---------------- Plurals ----------------

    # studies -> study
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"

    # Avoid destroying words such as "virus", "analysis", "status".
    if token.endswith(("us", "ss", "is")):
        return token

    # cases -> case
    # causes -> cause
    # diseases -> disease
    if len(token) > 4 and token.endswith("es"):
        if token.endswith(("ses", "zes", "xes", "ches", "shes")):
            return token[:-2]
        return token[:-1]

    # patients -> patient
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]

    # ---------------- -ing forms ----------------

    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]

        # triaging -> triage
        # changing -> change
        # dosing -> dose
        if base.endswith(("ag", "ang", "os", "iz")):
            return base + "e"

        # running -> run
        # stopping -> stop
        if len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]

        return base

    # ---------------- -ed forms ----------------

    if len(token) > 4 and token.endswith("ed"):
        base = token[:-2]

        # changed -> change
        # related -> relate
        # infected -> infect
        if base.endswith(("at", "it", "ic", "iv", "iz")):
            return base + "e"

        # stopped -> stop
        if len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]

        return base

    return token

def tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())

    normalized = []

    for token in tokens:
        if token in STOPWORDS:
            continue

        if "'" in token:
            normalized.append(token)
            continue

        if token[0].isdigit():
            normalized.append(token)
            continue

        normalized.append(stem_word(token))

    return normalized

def write_var_int(file, value: int) -> None:
    """Write a non-negative integer using variable-byte encoding."""
    while value >= 128:
        file.write(bytes([(value & 127) | 128]))
        value >>= 7

    file.write(bytes([value]))

def encode_var_int(buffer: bytearray, value: int) -> None:
    """Append a non-negative integer using variable-byte encoding."""
    while value >= 128:
        buffer.append((value & 127) | 128)
        value >>= 7

    buffer.append(value)

def read_var_int(file) -> int:
    """Read one variable-byte encoded non-negative integer."""
    value = 0
    shift = 0

    while True:
        byte = file.read(1)

        if not byte:
            raise EOFError("Unexpected end of index file")

        byte = byte[0]
        value |= (byte & 127) << shift

        if not (byte & 128):
            return value

        shift += 7

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
        os.makedirs(index_dir, exist_ok=True)

        # Give every original document ID a compact integer ID.
        doc_ids = list(self.doc_len.keys())
        doc_to_int = {doc_id: i for i, doc_id in enumerate(doc_ids)}

        # ---------------------------------------------------------------
        # documents.bin
        #
        # Format:
        #   number of documents
        #   for each document:
        #       ID length
        #       ID bytes
        #       document length
        # ---------------------------------------------------------------
        with open(
            os.path.join(index_dir, "documents.bin"), "wb"
        ) as file:

            write_var_int(file, self.N)

            for doc_id in doc_ids:
                encoded_id = doc_id.encode("utf-8")

                write_var_int(file, len(encoded_id))
                file.write(encoded_id)

                write_var_int(file, self.doc_len[doc_id])

        # ---------------------------------------------------------------
        # postings.bin + vocabulary.bin
        #
        # postings.bin stores:
        #
        #   posting_count
        #   delta_doc_id
        #   term_frequency
        #
        # for every term.
        #
        # vocabulary.bin stores:
        #
        #   term
        #   byte offset
        #   byte length
        #
        # ---------------------------------------------------------------

        vocabulary = []

        with open(
            os.path.join(index_dir, "postings.bin"), "wb"
        ) as postings_file:

            for term in sorted(self.postings):

                start = postings_file.tell()

                posting_list = self.postings[term]

                # Integer document IDs are sorted before delta encoding.
                entries = [
                    (doc_to_int[doc_id], tf)
                    for doc_id, tf in posting_list.items()
                ]

                buffer = bytearray()

                encode_var_int(buffer, len(entries))

                previous_doc = 0

                for doc_int, tf in entries:
                    gap = doc_int - previous_doc

                    encode_var_int(buffer, gap)
                    encode_var_int(buffer, tf)

                    previous_doc = doc_int

                postings_file.write(buffer)

                end = postings_file.tell()

                vocabulary.append(
                    (term, start, end - start)
                )

        # Vocabulary is small enough to keep as a compact binary table.
        #
        # Each entry:
        #   term length
        #   term bytes
        #   postings offset
        #   postings length
        #
        with open(
            os.path.join(index_dir, "vocabulary.bin"), "wb"
        ) as file:

            write_var_int(file, len(vocabulary))

            for term, offset, length in vocabulary:
                encoded_term = term.encode("utf-8")

                write_var_int(file, len(encoded_term))
                file.write(encoded_term)

                write_var_int(file, offset)
                write_var_int(file, length)

        # Small metadata file.
        with open(
            os.path.join(index_dir, "meta.bin"), "wb"
        ) as file:

            file.write(struct.pack("<Id", self.N, self.avg_doc_len))

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":

        inverted_index = cls()

        # ---------------------------------------------------------------
        # Load documents and reconstruct:
        #
        # integer doc ID -> original doc ID
        # original doc ID -> document length
        # ---------------------------------------------------------------

        int_to_doc = []

        with open(
            os.path.join(index_dir, "documents.bin"), "rb"
        ) as file:

            number_of_documents = read_var_int(file)

            for _ in range(number_of_documents):

                id_length = read_var_int(file)
                doc_id = file.read(id_length).decode("utf-8")

                doc_len = read_var_int(file)

                int_to_doc.append(doc_id)
                inverted_index.doc_len[doc_id] = doc_len

        # ---------------------------------------------------------------
        # Load metadata.
        # ---------------------------------------------------------------

        with open(
            os.path.join(index_dir, "meta.bin"), "rb"
        ) as file:

            inverted_index.N, inverted_index.avg_doc_len = struct.unpack(
                "<Id",
                file.read(12),
            )

        # ---------------------------------------------------------------
        # Read vocabulary.
        # ---------------------------------------------------------------

        vocabulary = {}

        with open(
            os.path.join(index_dir, "vocabulary.bin"), "rb"
        ) as file:

            number_of_terms = read_var_int(file)

            for _ in range(number_of_terms):

                term_length = read_var_int(file)
                term = file.read(term_length).decode("utf-8")

                offset = read_var_int(file)
                length = read_var_int(file)

                vocabulary[term] = (offset, length)

        # ---------------------------------------------------------------
        # Reconstruct the same postings structure BM25 already expects:
        #
        # term -> {original_doc_id: tf}
        # ---------------------------------------------------------------

        with open(
            os.path.join(index_dir, "postings.bin"), "rb"
        ) as file:

            for term, (offset, length) in vocabulary.items():

                file.seek(offset)

                posting_count = read_var_int(file)

                postings = {}

                current_doc = 0

                for _ in range(posting_count):

                    gap = read_var_int(file)
                    tf = read_var_int(file)

                    current_doc += gap

                    doc_id = int_to_doc[current_doc]

                    postings[doc_id] = tf

                inverted_index.postings[term] = postings

        return inverted_index