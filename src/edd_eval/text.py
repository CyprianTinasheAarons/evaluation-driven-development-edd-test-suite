"""Conservative text normalization used by the deterministic baseline metrics."""

from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def normalize(text: str) -> str:
    """Normalize casing, accents, whitespace, and punctuation for comparisons."""

    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(TOKEN_RE.findall(without_marks.casefold()))


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def token_f1(left: str, right: str) -> float:
    """Compute set-based F1, making repeated words irrelevant to the score."""

    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def token_recall(candidate: str, reference: str) -> float:
    candidate_tokens = token_set(candidate)
    reference_tokens = token_set(reference)
    if not reference_tokens:
        return 1.0 if not candidate_tokens else 0.0
    return len(candidate_tokens & reference_tokens) / len(reference_tokens)
