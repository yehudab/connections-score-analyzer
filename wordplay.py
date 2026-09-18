"""
wordplay.py

Code-generated wordplay hypotheses for NYT Connections.

Jev cannot "discover" that DIMENSION, PENNYWISE, NICKELODEON and QUARTERBACK
all start with coins: it reads literally and does not do the indirection. But
it is good at judging whether DIME, PENNY, NICKEL and QUARTER belong together.
So code does the letter-level work and hands Jev plain words to judge:

  1. For every board word, derive candidate hidden words by mechanism:
       start  — a dictionary word at the start        (DIMENSION -> DIME)
       end    — a dictionary word at the end          (MARIGOLD  -> GOLD)
       drop1  — remove one letter to get a word       (BORAT     -> BRAT;  "X plus a letter")
       add1   — insert one letter to get a word       (COBBLE    -> COBBLER; "X minus a letter")
     Candidates are filtered by word frequency so junk substrings drop out.
  2. Jev scores which derived tokens belong together (see jev_solver.py).
  3. Code searches, per mechanism, for 4 board words whose chosen tokens have
     the highest mutual affinity, producing ranked quad hypotheses that Jev
     then verifies as plain four-word categories.

Also produces deterministic letter-pattern groups (palindromes, "Y is the only
vowel", no vowels) that need no model at all.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable

from wordfreq import zipf_frequency

MECHANISMS = ("start", "end", "drop1", "add1")
MECHANISM_LABEL = {
    "start": "start of",
    "end": "end of",
    "drop1": "one letter removed from",
    "add1": "one letter added to",
}

MIN_ZIPF = 2.7        # roughly the 60k most common English words
MAX_ZIPF = 6.8        # drop function words like THE / AND
MIN_LEN = 3
MAX_TOKENS_PER_WORD_PER_MECHANISM = 8
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class Token:
    text: str        # the derived hidden word, uppercase
    source: str      # the board word it came from (verbatim)
    mechanism: str   # one of MECHANISMS

    @property
    def label(self) -> str:
        return f"{self.text} ({self.source})"

    def describe(self) -> str:
        return f'"{self.text}" ({MECHANISM_LABEL[self.mechanism]} {self.source})'


def normalize(word: str) -> str:
    """Letters only, uppercase: 'FRANK-N-FURTER' -> 'FRANKNFURTER'."""
    return re.sub(r"[^A-Z]", "", word.upper())


def is_common_word(t: str) -> bool:
    if len(t) < MIN_LEN or not t.isalpha():
        return False
    z = zipf_frequency(t.lower(), "en")
    return MIN_ZIPF <= z <= MAX_ZIPF


def _rank(tokens: Iterable[str]) -> list[str]:
    return sorted(set(tokens), key=lambda t: zipf_frequency(t.lower(), "en"), reverse=True)


def tokens_for(word: str) -> list[Token]:
    norm = normalize(word)
    n = len(norm)
    if n < MIN_LEN + 1:
        cands: dict[str, list[str]] = {"start": [], "end": []}
    else:
        cands = {
            "start": [norm[:k] for k in range(MIN_LEN, n) if is_common_word(norm[:k])],
            "end": [norm[-k:] for k in range(MIN_LEN, n) if is_common_word(norm[-k:])],
        }
    # drop1 / add1: skip trailing-S changes (plural of the same word is not
    # wordplay, and it lets every noun on the board "match" every other noun).
    cands["drop1"] = [
        t
        for i in range(n)
        for t in [norm[:i] + norm[i + 1:]]
        if t != norm and is_common_word(t) and not (i == n - 1 and norm[-1] == "S")
    ]
    cands["add1"] = [
        t
        for i in range(n + 1)
        for c in LETTERS
        for t in [norm[:i] + c + norm[i:]]
        if is_common_word(t) and not (i == n and c == "S")
    ]
    out: list[Token] = []
    for mech, ts in cands.items():
        for t in _rank(ts)[:MAX_TOKENS_PER_WORD_PER_MECHANISM]:
            out.append(Token(t, word, mech))
    return out


def board_tokens(board: list[str]) -> list[Token]:
    return [t for w in board for t in tokens_for(w)]


# ---------------------------------------------------------------------------
# Deterministic letter-pattern groups
# ---------------------------------------------------------------------------


def _is_palindrome(norm: str) -> bool:
    return len(norm) >= 3 and norm == norm[::-1]


def _only_vowel_is_y(norm: str) -> bool:
    return "Y" in norm and not any(c in "AEIOU" for c in norm)


def _no_vowels(norm: str) -> bool:
    return len(norm) >= 3 and not any(c in "AEIOUY" for c in norm)


def _double_letters(norm: str) -> bool:
    return any(a == b for a, b in zip(norm, norm[1:]))


LETTER_PATTERNS = {
    "palindromes": _is_palindrome,
    "Y is the only vowel": _only_vowel_is_y,
    "no vowels": _no_vowels,
}


def letter_pattern_groups(board: list[str]) -> list[tuple[str, frozenset[str]]]:
    """Patterns matched by at least 4 board words: (pattern name, matching words)."""
    out = []
    for name, fn in LETTER_PATTERNS.items():
        matched = frozenset(w for w in board if fn(normalize(w)))
        if len(matched) >= 4:
            out.append((name, matched))
    return out


# ---------------------------------------------------------------------------
# Hypothesis search over token affinities
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    words: frozenset[str]
    tokens: tuple[Token, ...]
    mechanism: str
    affinity: float          # mean pairwise token affinity from Jev (stage A)
    verified: float | None = None   # group-level Noul (stage B)

    def describe(self) -> str:
        toks = "/".join(t.text for t in self.tokens)
        v = "" if self.verified is None else f" ver={self.verified:.2f}"
        return f"{self.mechanism}:{toks} aff={self.affinity:.2f}{v}"


def wordplay_hypotheses(
    tokens: list[Token],
    affinity: dict[frozenset[Token], float],
    *,
    top_n: int = 60,
    subset_shortlist: int = 400,
    min_long_tokens: int = 2,
) -> list[Hypothesis]:
    """Best token assignment per 4-word subset, per mechanism, ranked by mean affinity.

    Per mechanism, `bp[(w1,w2)]` is the best affinity over any token pair from
    the two words. Its sum bounds a subset's score, so we shortlist subsets by
    that bound and then evaluate token combinations exactly on the shortlist.
    """
    hyps: list[Hypothesis] = []
    for mech in MECHANISMS:
        by_word: dict[str, list[Token]] = {}
        for t in tokens:
            if t.mechanism == mech:
                by_word.setdefault(t.source, []).append(t)
        words = sorted(by_word)
        if len(words) < 4:
            continue

        bp: dict[tuple[str, str], float] = {}
        for w1, w2 in itertools.combinations(words, 2):
            best = 0.0
            for t1 in by_word[w1]:
                for t2 in by_word[w2]:
                    a = affinity.get(frozenset((t1, t2)), 0.0)
                    if a > best:
                        best = a
            bp[(w1, w2)] = best

        bounded = []
        for quad in itertools.combinations(words, 4):
            pairs = list(itertools.combinations(quad, 2))
            if any(bp[p] <= 0.0 for p in pairs):
                continue
            bounded.append((sum(bp[p] for p in pairs) / 6, quad))
        bounded.sort(reverse=True)

        for _, quad in bounded[:subset_shortlist]:
            best_score, best_combo = -1.0, None
            for combo in itertools.product(*(by_word[w] for w in quad)):
                # Sets made mostly of 3-letter fragments (BAR/BOR/CAR/MAR) pass
                # frequency filters and even verification; demand some substance.
                if sum(len(t.text) >= 4 for t in combo) < min_long_tokens:
                    continue
                s = sum(
                    affinity.get(frozenset((t1, t2)), 0.0)
                    for t1, t2 in itertools.combinations(combo, 2)
                ) / 6
                if s > best_score:
                    best_score, best_combo = s, combo
            if best_combo is not None:
                hyps.append(Hypothesis(frozenset(quad), best_combo, mech, best_score))

    hyps.sort(key=lambda h: h.affinity, reverse=True)
    return hyps[:top_n]
