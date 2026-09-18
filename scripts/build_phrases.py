#!/usr/bin/env python3
"""
Build data/phrases.txt.gz: two-word English phrases from Wikipedia article titles.

Used by blanks.py to discover fill-in-the-blank candidates ("SAFETY PIN",
"KEY LIME", "SPOILER ALERT"). Only pairs where at least one side is a common
word (the candidate blank vocabulary) are kept, which cuts the file to a size
that loads in a couple of seconds.

Usage:
    python scripts/build_phrases.py [path/to/enwiki-latest-all-titles-in-ns0.gz]

Downloads the ~110 MB titles dump when no path is given.
"""
import gzip
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import blanks  # noqa: E402

DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz"
OUT = Path(__file__).resolve().parent.parent / "data" / "phrases.txt.gz"


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench-data/enwiki-titles.gz")
    if not src.exists():
        print(f"downloading {DUMP_URL} -> {src}", file=sys.stderr)
        src.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DUMP_URL, src)
    vocab = set(blanks.candidate_vocabulary([], blanks.VOCAB_SIZE))
    pairs: set[tuple[str, str]] = set()
    with gzip.open(src, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.rstrip("\n")
            parts = t.split("_")
            if len(parts) != 2:
                continue
            a, b = parts
            if not (a.isalpha() and b.isalpha() and a.isascii() and b.isascii()):
                continue
            a, b = a.upper(), b.upper()
            if a in vocab or b in vocab:
                pairs.add((a, b))
    OUT.parent.mkdir(exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as out:
        for a, b in sorted(pairs):
            out.write(f"{a} {b}\n")
    print(f"wrote {len(pairs):,} phrases to {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
