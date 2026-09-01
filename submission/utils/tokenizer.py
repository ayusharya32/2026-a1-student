import re
from typing import List, Dict

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

_STEM_CACHE: Dict[str, str] = {}

def stem_word_uncached(token: str) -> str:
    if len(token) <= 3:
        return token

    # Plurals
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("us", "ss", "is")):
        return token
    if len(token) > 4 and token.endswith("es"):
        if token.endswith(("ses", "zes", "xes", "ches", "shes")):
            return token[:-2]
        return token[:-1]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]

    # -ing forms
    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]
        if base.endswith(("ag", "ang", "os", "iz")):
            return base + "e"
        if len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base

    # -ed forms
    if len(token) > 4 and token.endswith("ed"):
        base = token[:-2]
        if base.endswith(("at", "it", "ic", "iv", "iz")):
            return base + "e"
        if len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    
    return token

def stem_word(token: str) -> str:
    st = _STEM_CACHE.get(token)
    if st is not None:
        return st
    
    st = stem_word_uncached(token)
    _STEM_CACHE[token] = st

    return st

def tokenize_string(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    normalized = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if "'" in token or token[0].isdigit():
            normalized.append(token)
            continue
        st = _STEM_CACHE.get(token)
        if st is None:
            st = stem_word_uncached(token)
            _STEM_CACHE[token] = st
        normalized.append(st)

    return normalized
