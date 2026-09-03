import re
from typing import List, Dict
from nltk.stem import SnowballStemmer

from submission.utils.constants import STOPWORDS

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

_STEMMER = SnowballStemmer("english")
_STEM_CACHE: Dict[str, str] = {}

def stem_word(token: str) -> str:
    st = _STEM_CACHE.get(token)
    if st is not None:
        return st
    st = _STEMMER.stem(token)
    _STEM_CACHE[token] = st
    return st

def tokenize_string(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    normalized = []
    _get = _STEM_CACHE.get
    _stem = _STEMMER.stem
    for token in tokens:
        if token in STOPWORDS:
            continue
        if "'" in token or token[0].isdigit():
            normalized.append(token)
            continue
        st = _get(token)
        if st is None:
            st = _stem(token)
            _STEM_CACHE[token] = st
        normalized.append(st)

    return normalized