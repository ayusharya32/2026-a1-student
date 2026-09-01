from typing import TYPE_CHECKING, Dict, List
if TYPE_CHECKING:
    from submission.indexer import InvertedIndex

class LazyPostingsMap:
    """Lazy dictionary wrapper that loads postings on-demand from disk."""
    def __init__(self, index: "InvertedIndex"):
        self._index = index

    def __contains__(self, term: str) -> bool:
        return term in self._index.vocabulary or term in self._index._postings_cache

    def __getitem__(self, term: str) -> Dict[str, int]:
        return self._index.get_postings(term)

    def get(self, term: str, default=None):
        if term in self:
            return self._index.get_postings(term)
        return default

    def __len__(self) -> int:
        return len(self._index.vocabulary)

    def keys(self):
        return self._index.vocabulary.keys()