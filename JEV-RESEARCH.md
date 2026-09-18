# Solving NYT Connections with Jev

A two-day experiment: can [TypeSafe's Jev](https://docs.typesafe.ai), a model that
answers typed questions with probabilities instead of generating text, solve the
daily NYT Connections puzzle? The existing solver in this repo asks an LLM via
OpenRouter to "group these 16 words into 4 categories" and parses the JSON. Jev
cannot do that: it has no text output. Every part of the solver that used to be a
prompt had to become either a typed question or code.

**Result:** 40 of the 50 most recent puzzles solved on an offline benchmark, up
from 21 for the first design, at under one cent per puzzle. Opus 4.6 through
OpenRouter solves 96% of the same puzzles at about 3.4 cents. All grouping logic,
hypothesis generation and feedback handling live in code; Jev supplies judgments.

## How Jev differs

Jev takes a `state` (any JSON) and a map of questions, and returns one typed answer
per question, all evaluated in parallel in one request:

| Question | Returns |
|---|---|
| **Noul** — is this true? | a probability of yes |
| **Choice** — which of these options? | a probability per option, summing to 1 |
| **Score** — which level on this ordered scale? | a probability per level |

No temperature, no seed, no text. Input tokens cost $0.042 per million; output is
free. A request may hold up to 64k tokens of state and questions.

## The strategies, in the order they were built

Benchmark: the 50 most recent puzzles (2026-07-30 to 2026-09-17) from the
[Eyefyre answer archive](https://github.com/Eyefyre/NYT-Connections-Answers),
replayed through a simulated board with the real rules (correct / one away / wrong,
four mistakes). Categories are split by their NYT title into semantic
("PARTS OF A FLIGHT"), wordplay ("STARTING WITH U.S. COINS") and blank ("___ BOWL").

| Strategy | Solved | 0 mistakes | Semantic | Wordplay | Blank |
|---|---|---|---|---|---|
| 1. Pairwise Nouls | 21/50 | 7 | 72% | 31% | 28% |
| 2. Staged Choice beam | 28/50 | 11 | 82% | 43% | 33% |
| 3. + code-generated wordplay hypotheses | 33/50 | 14 | 84% | 77% | 33% |
| 4. + phrase-dictionary blank hypotheses | 39/50 | 20 | 88% | 72% | 67% |
| 5. + one-away follow-up question | **40/50** | **21** | 89% | 83% | 72% |

Each step is cumulative and selectable with `JEV_STRATEGY` (`pairwise`, `beam`,
`wordplay`, `blanks`); `blanks` is the default and includes step 5.

### 1. Pairwise Nouls

One request with 120 questions, one per pair of board words: *"do A and B belong
to the same category of four?"* The answers fill a 16×16 affinity matrix. Code then
runs a branch-and-bound search for the partition into four groups of four that
maximises total within-group affinity.

Game feedback becomes constraints on that search rather than new model calls:
*wrong* means at most two of those four share a group; *one away* means exactly
three do. Solved words drop off the board and the constraints project onto what
remains.

It worked for semantic categories and failed completely on wordplay: true wordplay
groups ranked around 1000th of 1820 possible four-word sets. A pairwise question
cannot see a pattern that only exists across all four words. Nouls were also
under-confident in isolation: the median true same-group pair scored 0.48.

### 2. Staged Choice beam

The user's idea, and the single biggest lesson about the model: **Choice questions
are comparative, Nouls are not.** *"Which other board word belongs with DIME?"*
divides probability among competitors, so the right partner stands out even when
Jev is unsure in absolute terms. Three requests build groups up:

1. 16 Choices: which word belongs with *X*? (15 options)
2. 120 Choices: which word belongs with *X and Y*? (14 options)
3. Top 200 triples: which word completes *X, Y, Z*? (13 options)

Scores accumulate per word set and the partition search runs over four-word sets.
Widening the beam from 200 to 400 triples did not help: the true wordplay triples
were in the beam, but Jev still would not pick the fourth coin word from DIME,
PENNY, NICKEL. The model does not do letter-level indirection, which the
[jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13) says plainly.

### 3. Code-generated wordplay hypotheses

Since Jev cannot *discover* that DIMENSION, PENNYWISE, NICKELODEON and QUARTERBACK
start with coins, code does the letter work and Jev judges plain words
([`wordplay.py`](wordplay.py)):

- Derive hidden-word tokens from each board word: start and end substrings, one
  letter removed ("X plus a letter"), one letter added ("X minus a letter"),
  filtered by word frequency. DIMENSION yields DIME; MARIGOLD yields GOLD.
- Stage A: a Choice per token over its peers, *"which other hidden word belongs in
  the same category as DIME?"*. Normalising each distribution by its maximum
  lifted true groups reaching the shortlist from 8 to 17 of 18.
- Stage B: a Score question per candidate four-token set, *"how well do DIME,
  NICKEL, PENNY, QUARTER form a single specific category?"*, with rules stating that
  fragments, suffixes and plurals are not categories. P(top level) is the score.
- Stage C: duels. Hypotheses sharing three tokens compete for the fourth
  (DIME/PENNY/QUARTER plus NICKEL or FRANK); one Choice settles it.

Choosing the verification wording was the most useful experiment in the project.
Five wordings were tested on 23 true token groups and 160 wrong hypotheses:

| Verification question | AUC | True ≥ 0.65 | Wrong ≥ 0.65 |
|---|---|---|---|
| Noul "do all four belong together?" | 0.92 | 17/23 | 4/160 |
| Noul, strict criteria | 0.94 | 15/23 | 1/160 |
| Choice "which is the odd one out, or none?" | 0.90 | 5/23 | 0/160 |
| Score, P(top level) | 0.93 | 19/23 | 0/160 |
| Noul "is there an odd one out?" (inverted) | 0.77 | 4/23 | 0/160 |

The odd-one-out Choice looked good on paper but Jev almost never answered "none",
so it was useless as a verifier. It became useful later, in step 5, once the
premise "one of these does not belong" was actually true.

Letter-pattern categories (palindromes, "Y is the only vowel") are found
deterministically with no model call.

### 4. Phrase-dictionary blank hypotheses

Blank categories were the largest remaining failure bucket, and the hardest lesson.
Two attempts to have Jev *discover* the blank failed the same way:

- A Choice per vocabulary word over the 16 board words (*"which board word pairs
  with BOWL?"*) put SAFETY at rank 6121 of 9467 candidates for the SAFETY ___ board.
- A Choice per board word over 250-word vocabulary chunks put SAFETY at rank 408
  for BELT and LOVE past rank 1000 for every member.

The reason is structural: a blank is a moderate collocate of all four members and
the strongest collocate of none. Ask what pairs with BELT and you get SEAT or
BLACK, never SAFETY. No best-partner question can find it.

Discovery therefore comes from data ([`blanks.py`](blanks.py),
[`scripts/build_phrases.py`](scripts/build_phrases.py)). Two-word English Wikipedia
article titles ("Safety pin", "Key lime", "Spoiler alert") cover 97% of the blank
categories in the puzzle archive; Norvig's web bigram table covered 11%. The 2.3M
phrases ship as a 9.6 MB file, indexed in SQLite on first use. A candidate blank is
a common word forming a phrase with at least three board words on the same side.
Jev then grades every candidate phrase with a Score question, and a hypothesis is
scored by the *weakest* member's probability of being at least a recognisable
phrase. On the test boards that put every true group at 0.89 or above and every
accidental Wikipedia match at 0.83 or below.

### 5. One-away follow-up

Pre-computing "if this is one away, swap X for Y" from the same cached scores adds
nothing for Jev; the benchmark was 39/50 with or without it. What the feedback gives
is a *new* question that was not askable before: exactly one of these four is an
intruder. After a one-away the solver sends one request with an odd-one-out Choice
over the four members and a completion Choice for each possible trio, and ranks
swaps by Jev's joint probability. On today's live board this is what rescued FLIGHT
(a tasting flight, not a "flight case") on the first retry.

## What Jev is and is not

**Not deterministic.** The docs say System One is "designed to return stable
answers across repeated evaluations" and their own
[self-consistency cookbook](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook)
measures ~0.01 standard deviation per probability with occasional label flips.
Measured here: three identical requests of 16 Choice questions moved probabilities
by up to 0.14 (mean 0.012) and flipped the top choice on 3 of 16; a repeated Score
question ranged 0.55 to 0.65. There is no seed or temperature to set. A Connections
board with a genuine red herring sits exactly at those margins, so the same board
can solve on one run and fail on the next. Averaging three samples per request
(`JEV_REPEATS=3`) made decisions consistent but not more accurate on a six-puzzle
test, so it is off by default.

**Good at:** comparative judgments over a closed set; verifying that four plain
words form a category; grading whether a two-word phrase is established; answering
hundreds of independent questions in one 1–3 second request for a fraction of a cent.

**Bad at:** anything requiring letter-level indirection or discovery of a hidden
word; counting; absolute yes/no judgments in isolation, which come back
under-confident. All of this is consistent with the jaggedness page. The design
pattern that worked every time was: generate candidates in code, let Jev select or
verify, keep the decision logic in code.

## Against a reasoning LLM

Same 45 puzzles (the Opus run stopped at a $1.50 budget):

| | Opus 4.6 via OpenRouter | Jev, `blanks` strategy |
|---|---|---|
| Solved | 43/45 (96%) | 36/45 (80%) |
| Zero-mistake solves | 30 | 18 |
| Semantic / wordplay / blank | 98% / 93% / 94% | 89% / 83% / 69% |
| Model calls per puzzle | 1.4 | ~10 requests |
| Cost per puzzle | ~3.4 cents (90% output tokens) | ~0.8 cents |
| Model time per puzzle | ~25 s | 5–8 s |

Both solved 35, Opus alone 8, Jev alone 1 (a hidden-accessories wordplay category
found by code-generated tokens), neither 1. Opus wins on world knowledge: it placed
FLIGHT with the tasting assortment on its first guess. Jev wins on cost by four
times and on speed by three to five. Both needed feedback-driven re-asking to close
out the hardest board. Neither is deterministic run to run.

## Live board

`./solve.sh --solver jev --no-cdp --headed` plays the real puzzle in a visible
browser. Getting there needed three fixes to the scraper that predate Jev: NYT's
tiles are now hidden checkbox inputs with the word in `aria-label`, a full-page ad
interstitial follows the Play button, and tiles stay selected after a wrong guess.

## Reproduce

```bash
pip install -r requirements.txt            # typesafe-sdk, wordfreq, playwright ...
export TYPESAFE_API_KEY=...                # console.typesafe.ai

./bench_jev.py --strategy blanks --last 50          # Jev, 50 most recent puzzles
./bench_jev.py --strategy blanks --ids 1188 -v      # one puzzle, every question logged
./bench_jev.py --strategy openrouter --last 50 --budget-usd 1.50   # the LLM solver

SOLVER=jev ./solve.sh --no-cdp --headed --debug     # today's board, visible browser
```

Jev answers are cached per board and question wording in `bench-data/`, so re-runs
that only change thresholds or weights are free. The whole research programme,
including every failed discovery experiment, cost under $3 of TypeSafe credit.

## Files

| File | What |
|---|---|
| [`connections_solver.py`](connections_solver.py) | Browser automation and game loop; pluggable solver strategy |
| [`jev_solver.py`](jev_solver.py) | The four Jev strategies, request chunking, caching, one-away follow-up |
| [`wordplay.py`](wordplay.py) | Hidden-word token generation and hypothesis search |
| [`blanks.py`](blanks.py) | Phrase dictionary index and blank-candidate discovery |
| [`bench_jev.py`](bench_jev.py) | Offline benchmark against the puzzle archive, for Jev and OpenRouter |
| [`scripts/build_phrases.py`](scripts/build_phrases.py) | Builds `data/phrases.txt.gz` from the Wikipedia titles dump |

## What is left

The ten remaining failures are homophones ("HOMOPHONES OF FACIAL FEATURES"), which
need a pronunciation dictionary the way blanks needed a phrase dictionary; a few
wordplay mechanisms the generator does not cover ("STARTING WITH BREAD SHAPES",
where BOULE is too rare for the frequency filter); pop-culture sets ("NBC SITCOM
SURNAMES"); and boards where two categories compete for the same words, where the
one-away follow-up could be extended with duels between hypotheses sharing a blank.
