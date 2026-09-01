import os
import struct
from typing import Dict, List, Tuple
import json
from collections import Counter

from submission.utils.encoding import *
from submission.utils.tokenizer import tokenize_string
from submission.utils.lazy_postings_map import LazyPostingsMap


def tokenize(text: str) -> List[str]:
    return tokenize_string(text)


class InvertedIndex:
    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}
        self.doc_len: Dict[str, int] = {}
        self.doc_text: Dict[str, str] = {}
        self.N: int = 0
        self.avg_doc_len: float = 0.0

        self.vocabulary: Dict[str, Tuple[int, int, int]] = {}

        self.int_to_doc: List[str] = []
        self.doc_len_by_int: List[int] = []

        self.postings_file = None
        self._postings_cache: Dict[str, Dict[int, int]] = {}

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """Build index from JSONL filepath or list of (doc_id, text) pairs."""
        self.postings = {}
        self.doc_len = {}

        if isinstance(corpus, str):
            with open(corpus, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    obj = json.loads(line)
                    doc_id = obj["doc_id"]
                    tokens = tokenize(obj["text"])

                    self.doc_len[doc_id] = len(tokens)

                    counts = Counter(tokens)

                    for term, tf in counts.items():
                        if term not in self.postings:
                            self.postings[term] = {}

                        self.postings[term][doc_id] = tf

        else:
            for doc_id, text in corpus:
                tokens = tokenize(text)
                self.doc_len[doc_id] = len(tokens)

                counts = Counter(tokens)

                for term, tf in counts.items():
                    if term not in self.postings:
                        self.postings[term] = {}

                    self.postings[term][doc_id] = tf

        self.N = len(self.doc_len)

        self.avg_doc_len = (
            sum(self.doc_len.values()) / self.N
            if self.N > 0
            else 0.0
        )

    def document_frequency(self, term: str) -> int:
        if term in self.vocabulary:
            return self.vocabulary[term][2]

        if term in self.postings:
            return len(self.postings[term])

        return 0

    def get_postings(self, term: str) -> Dict[int, int]:
        """Fetch and decode postings using integer document IDs."""

        if term in self._postings_cache:
            return self._postings_cache[term]

        if term not in self.vocabulary or self.postings_file is None:
            return {}

        offset, length, _df = self.vocabulary[term]

        self.postings_file.seek(offset)
        raw_bytes = self.postings_file.read(length)

        postings = {}
        pos = 0
        buf_len = len(raw_bytes)

        # Read posting count.
        val = 0
        shift = 0

        while pos < buf_len:
            b = raw_bytes[pos]
            pos += 1

            val |= (b & 127) << shift

            if not (b & 128):
                break

            shift += 7

        count = val

        current_doc = 0

        for _ in range(count):
            # Read gap.
            val = 0
            shift = 0

            while pos < buf_len:
                b = raw_bytes[pos]
                pos += 1

                val |= (b & 127) << shift

                if not (b & 128):
                    break

                shift += 7

            gap = val

            # Read term frequency.
            val = 0
            shift = 0

            while pos < buf_len:
                b = raw_bytes[pos]
                pos += 1

                val |= (b & 127) << shift

                if not (b & 128):
                    break

                shift += 7

            tf = val

            current_doc += gap

            # Keep integer document ID during query-time scoring.
            postings[current_doc] = tf

        self._postings_cache[term] = postings
        return postings

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)

        doc_ids = list(self.doc_len.keys())
        doc_to_int = {
            doc_id: i
            for i, doc_id in enumerate(doc_ids)
        }

        # documents.bin
        doc_buf = bytearray()

        encode_var_int(doc_buf, self.N)

        for doc_id in doc_ids:
            encoded_id = doc_id.encode("utf-8")

            encode_var_int(doc_buf, len(encoded_id))
            doc_buf.extend(encoded_id)
            encode_var_int(doc_buf, self.doc_len[doc_id])

        with open(
            os.path.join(index_dir, "documents.bin"),
            "wb",
        ) as file:
            file.write(doc_buf)

        # postings.bin + vocabulary
        vocabulary = []

        with open(
            os.path.join(index_dir, "postings.bin"),
            "wb",
        ) as postings_file:

            for term in sorted(self.postings):
                start = postings_file.tell()

                posting_list = self.postings[term]

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
                    (
                        term,
                        start,
                        end - start,
                        len(entries),
                    )
                )

        # vocabulary.bin
        vocab_buf = bytearray()

        encode_var_int(
            vocab_buf,
            len(vocabulary),
        )

        for term, offset, length, df in vocabulary:
            encoded_term = term.encode("utf-8")

            encode_var_int(
                vocab_buf,
                len(encoded_term),
            )

            vocab_buf.extend(encoded_term)

            encode_var_int(vocab_buf, offset)
            encode_var_int(vocab_buf, length)
            encode_var_int(vocab_buf, df)

        with open(
            os.path.join(index_dir, "vocabulary.bin"),
            "wb",
        ) as file:
            file.write(vocab_buf)

        # meta.bin
        with open(
            os.path.join(index_dir, "meta.bin"),
            "wb",
        ) as file:
            file.write(
                struct.pack(
                    "<Id",
                    self.N,
                    self.avg_doc_len,
                )
            )

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        inverted_index = cls()

        # Load documents.
        with open(
            os.path.join(index_dir, "documents.bin"),
            "rb",
        ) as file:

            number_of_documents = read_var_int(file)

            for _ in range(number_of_documents):
                id_length = read_var_int(file)

                doc_id = (
                    file.read(id_length)
                    .decode("utf-8")
                )

                doc_len = read_var_int(file)

                inverted_index.int_to_doc.append(doc_id)
                inverted_index.doc_len[doc_id] = doc_len
                inverted_index.doc_len_by_int.append(doc_len)

        # Load metadata.
        with open(
            os.path.join(index_dir, "meta.bin"),
            "rb",
        ) as file:

            inverted_index.N, inverted_index.avg_doc_len = struct.unpack(
                "<Id",
                file.read(12),
            )

        # Load vocabulary.
        with open(
            os.path.join(index_dir, "vocabulary.bin"),
            "rb",
        ) as file:

            number_of_terms = read_var_int(file)

            for _ in range(number_of_terms):
                term_length = read_var_int(file)

                term = (
                    file.read(term_length)
                    .decode("utf-8")
                )

                offset = read_var_int(file)
                length = read_var_int(file)
                df = read_var_int(file)

                inverted_index.vocabulary[term] = (
                    offset,
                    length,
                    df,
                )

        # Open postings file for lazy access.
        postings_path = os.path.join(
            index_dir,
            "postings.bin",
        )

        if os.path.exists(postings_path):
            inverted_index.postings_file = open(
                postings_path,
                "rb",
            )

        inverted_index.postings = LazyPostingsMap(
            inverted_index
        )

        return inverted_index