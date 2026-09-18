"""
blanks.py

Candidate generation for fill-in-the-blank Connections categories such as
"___ BOWL" (SUPER, FISH, TOILET, DUST) or "SAFETY ___" (PIN, NET, BELT, FIRST).

Jev cannot discover the hidden word: a blank is a moderate collocate of all
four members but the strongest collocate of none, so "which word pairs with
BELT?" returns SEAT or BLACK, never SAFETY. Discovery therefore comes from
data and Jev only verifies:

  1. A phrase dictionary of two-word English Wikipedia article titles
     (data/phrases.txt.gz, built by scripts/build_phrases.py; 2.3M phrases,
     covering 97% of the blank categories in the puzzle archive) is indexed in
     SQLite on first use.
  2. A candidate blank is a common word that forms a dictionary phrase with at
     least three board words, all on the same side ("___ X" or "X ___").
  3. jev_solver.py asks Jev how established each candidate phrase is, and the
     four strongest members per blank become a hypothesis.

Closed compounds (EYE+LASH = EYELASH) are checked against the system word list.
"""

from __future__ import annotations

import gzip
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

VOCAB_SIZE = 10000
MIN_LEN = 3
MAX_ZIPF = 6.5   # drop function words (the, and, of ...)

DATA_DIR = Path(__file__).parent / "data"
PHRASES_GZ = DATA_DIR / "phrases.txt.gz"
PHRASES_DB = DATA_DIR / "phrases.sqlite"
SYSTEM_WORDS = Path("/usr/share/dict/words")

STOPWORDS = set("""
the and for that with this from have not are was but they you all can had her his
one our out has been were their said each which she how will about them than its
who into more some could other these two may then only also over such just being
where after most through before between under while because those very any again
both few many much own same too here there when what why would should does did
done doing having get got make made like well back even still might must shall
let say says tell told see seen saw know knew known think thought take took taken
come came give gave given went gone keep kept put set find found
""".split())


def normalize(word: str) -> str:
    return re.sub(r"[^A-Z]", "", word.upper())


def tokens(word: str) -> list[str]:
    """Alphabetic tokens of a board entry: 'TAKE TO' -> ['TAKE', 'TO']."""
    return [t for t in re.split(r"[^A-Z]+", word.upper()) if t]


def candidate_vocabulary(board: list[str], size: int = VOCAB_SIZE) -> list[str]:
    """Common English words that could be the hidden blank, uppercase, deduplicated."""
    board_norm = {normalize(w) for w in board}
    out: list[str] = []
    seen: set[str] = set()
    for w in top_n_list("en", size):
        if not w.isalpha() or len(w) < MIN_LEN or w in STOPWORDS:
            continue
        if zipf_frequency(w, "en") > MAX_ZIPF:
            continue
        up = w.upper()
        if up in seen or up in board_norm:
            continue
        seen.add(up)
        out.append(up)
    return out


# ---------------------------------------------------------------------------
# Phrase index
# ---------------------------------------------------------------------------


class PhraseIndex:
    """SQLite-backed lookup of two-word phrases: partners of a word on either side."""

    def __init__(self, db_path: Path = PHRASES_DB, gz_path: Path = PHRASES_GZ) -> None:
        if not db_path.exists():
            if not gz_path.exists():
                raise FileNotFoundError(
                    f"{gz_path} missing; run scripts/build_phrases.py to create it"
                )
            self._build(db_path, gz_path)
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self._words: set[str] | None = None

    @staticmethod
    def _build(db_path: Path, gz_path: Path) -> None:
        tmp = db_path.with_suffix(".building")
        if tmp.exists():
            tmp.unlink()
        conn = sqlite3.connect(tmp)
        conn.execute("CREATE TABLE phrases (a TEXT NOT NULL, b TEXT NOT NULL)")
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            rows = (tuple(line.rstrip("\n").split(" ", 1)) for line in f)
            conn.executemany("INSERT INTO phrases VALUES (?, ?)", (r for r in rows if len(r) == 2))
        conn.execute("CREATE INDEX idx_a ON phrases(a)")
        conn.execute("CREATE INDEX idx_b ON phrases(b)")
        conn.commit()
        conn.close()
        tmp.replace(db_path)

    def after(self, word: str) -> set[str]:
        """Words X such that 'word X' is a phrase."""
        return {r[0] for r in self.conn.execute("SELECT b FROM phrases WHERE a = ?", (word,))}

    def before(self, word: str) -> set[str]:
        """Words X such that 'X word' is a phrase."""
        return {r[0] for r in self.conn.execute("SELECT a FROM phrases WHERE b = ?", (word,))}

    def partners(self, entry: str) -> dict[str, str]:
        """Candidate blanks for a board entry -> side of the *member* relative to the blank.

        side "before": the member precedes the blank ("FISH BOWL" for blank BOWL)
        side "after":  the member follows the blank  ("SAFETY PIN" for blank SAFETY)
        Multi-word entries match on the whole entry and on their first/last token.
        """
        out: dict[str, str] = {}
        toks = tokens(entry)
        keys = {"".join(toks), toks[0], toks[-1]} if toks else set()
        for key in keys:
            for x in self.after(key):
                out.setdefault(x, "before")
            for x in self.before(key):
                out.setdefault(x, "after")
        return out

    def system_words(self) -> set[str]:
        if self._words is None:
            self._words = set()
            if SYSTEM_WORDS.exists():
                self._words = {
                    l.strip().upper() for l in SYSTEM_WORDS.read_text().splitlines() if l.strip().isalpha()
                }
        return self._words


_INDEX: PhraseIndex | None = None


def phrase_index() -> PhraseIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = PhraseIndex()
    return _INDEX


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass
class BlankCandidate:
    blank: str
    side: str                         # side of members relative to the blank
    members: dict[str, bool] = field(default_factory=dict)   # board word -> found in dictionary
    generality: int = 0               # how many phrases the blank appears in overall (lower = more specific)

    @property
    def found(self) -> list[str]:
        return [w for w, ok in self.members.items() if ok]


def closed_compounds(entry: str, vocab: set[str], words: set[str]) -> dict[str, str]:
    """Blanks X such that entry+X or X+entry is a dictionary word (EYE+LASH)."""
    out: dict[str, str] = {}
    joined = "".join(tokens(entry))
    if not joined:
        return out
    for x in vocab:
        if len(x) < 3:
            continue
        if (joined + x) in words:
            out.setdefault(x, "before")
        if (x + joined) in words:
            out.setdefault(x, "after")
    return out


def candidate_blanks(
    board: list[str],
    *,
    min_members: int = 3,
    cap: int = 400,
    vocab_size: int = VOCAB_SIZE,
    include_closed: bool = True,
) -> list[BlankCandidate]:
    """Blanks that form a dictionary phrase with >= min_members board words on one side."""
    idx = phrase_index()
    vocab = set(candidate_vocabulary(board, vocab_size))
    words = idx.system_words() if include_closed else set()
    hits: dict[tuple[str, str], set[str]] = {}
    for w in board:
        partners = idx.partners(w)
        if include_closed:
            for x, side in closed_compounds(w, vocab, words).items():
                partners.setdefault(x, side)
        for x, side in partners.items():
            if x in vocab:
                hits.setdefault((x, side), set()).add(w)
    cands: list[BlankCandidate] = []
    for (x, side), ws in hits.items():
        if len(ws) >= min_members:
            c = BlankCandidate(x, side, {w: (w in ws) for w in board})
            c.generality = len(idx.after(x)) + len(idx.before(x))
            cands.append(c)
    # most members first, then the most specific blanks; name/side break ties so
    # the order (and therefore question ids and cache keys) is deterministic
    cands.sort(key=lambda c: (-len(c.found), c.generality, c.blank, c.side))
    return cands[:cap]


def phrase(member: str, blank: str, side: str) -> str:
    return f"{member} {blank}" if side == "before" else f"{blank} {member}"
