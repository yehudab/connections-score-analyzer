#!/usr/bin/env python3
"""
jev_solver.py

A Connections solving strategy built on TypeSafe's Jev (System One) model.

Jev does not generate text. It answers typed questions with calibrated
probabilities. So instead of asking a model to "partition these 16 words into
4 themed groups", this strategy:

  1. Asks ONE Jev request with a Noul ("is this true?") question per unordered
     word pair: "do A and B belong to the same category?" — 120 questions for a
     16-word board, all evaluated in parallel against the same board state.
  2. Turns the answers into a pairwise affinity matrix.
  3. In code, searches for the partition into groups of 4 that maximises total
     within-group affinity, subject to constraints derived from game feedback:
       - "wrong"    → at most 2 of those 4 words share a group
       - "one away" → exactly 3 of those 4 words share a group
     Feedback never requires another API call; the constraint set just grows
     and the partition search is re-run over the remaining words.

The strategy object is a callable with the same contract `play_game` in
connections_solver.py expects from a solver:

    groups = strategy(remaining_words, failed_guesses)
    # -> [{"theme": str, "members": [w, w, w, w], "alternatives": []}, ...]

Environment:
    TYPESAFE_API_KEY        required
    TYPESAFE_DEFAULT_MODEL  optional (default: jev-latest)
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from typesafe_sdk import Noul, TypeSafeClient

# ---------------------------------------------------------------------------
# Question design
# ---------------------------------------------------------------------------

GAME_RULES = (
    "NYT Connections: the 16 words on the board split into exactly 4 hidden "
    "categories of 4 words each. A category can be a shared meaning, a common "
    "type of thing, words that all precede or follow the same word, homophones, "
    "anagrams, hidden words, or similar wordplay. Some words plausibly fit two "
    "categories as red herrings, but each word belongs to exactly one."
)

PAIR_CRITERIA = {
    "true": (
        "The two words are in the same category of four on this board: the "
        "connection between them also holds for two other board words."
    ),
    "false": (
        "The two words are not in the same category: any link between them is "
        "weaker than the link each has with three other board words, or no "
        "other board words share it."
    ),
}


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def pair_question(a: str, b: str) -> Noul:
    return Noul(
        instructions=(
            f'On this board, do the words "{a}" and "{b}" belong to the same '
            f"category of four?"
        ),
        criteria=PAIR_CRITERIA,
    )


def build_questions(board: list[str]) -> dict[str, Noul]:
    return {
        f"p{i}_{j}": pair_question(board[i], board[j])
        for i, j in itertools.combinations(range(len(board)), 2)
    }


# ---------------------------------------------------------------------------
# Affinity fetching
# ---------------------------------------------------------------------------


@dataclass
class AffinityResult:
    board: list[str]
    pairs: dict[tuple[str, str], float]
    model: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float

    def to_json(self) -> dict:
        return {
            "board": self.board,
            "pairs": {f"{a}||{b}": p for (a, b), p in self.pairs.items()},
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AffinityResult":
        pairs = {}
        for k, p in d["pairs"].items():
            a, b = k.split("||", 1)
            pairs[pair_key(a, b)] = float(p)
        return cls(
            board=list(d["board"]),
            pairs=pairs,
            model=d.get("model", "?"),
            input_tokens=int(d.get("input_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
            elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
        )


def prompt_version() -> str:
    """Short hash of the question wording, so cached answers are keyed to it."""
    import hashlib

    sample = json.dumps(
        {"rules": GAME_RULES, "q": pair_question("A", "B").instructions, "criteria": PAIR_CRITERIA},
        sort_keys=True,
    )
    return hashlib.sha1(sample.encode()).hexdigest()[:8]


def board_key(board: Iterable[str]) -> str:
    return prompt_version() + "#" + "|".join(sorted(board))


def fetch_affinities(board: list[str], client: TypeSafeClient) -> AffinityResult:
    """One Jev request: a Noul per unordered pair of board words."""
    questions = build_questions(board)
    state = {"rules": GAME_RULES, "board": list(board)}
    t0 = time.monotonic()
    response = client.system_one(state=state, questions=questions)
    elapsed = time.monotonic() - t0

    pairs: dict[tuple[str, str], float] = {}
    for i, j in itertools.combinations(range(len(board)), 2):
        pairs[pair_key(board[i], board[j])] = float(response.nouls[f"p{i}_{j}"].noul)

    usage = response.usage
    return AffinityResult(
        board=list(board),
        pairs=pairs,
        model=response.model,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        elapsed_seconds=round(elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Constraints from game feedback
# ---------------------------------------------------------------------------


@dataclass
class Constraints:
    # (words, max_together): no group may contain more than `max_together` of `words`
    max_together: list[tuple[frozenset[str], int]] = field(default_factory=list)
    # every word in the set must land in the same group
    together: list[frozenset[str]] = field(default_factory=list)
    # exactly 3 of these 4 words must share a group (checked on complete partitions)
    needs_three: list[frozenset[str]] = field(default_factory=list)
    # exact groupings that must not be proposed again (fallback safety net)
    tried: set[frozenset[str]] = field(default_factory=set)


def project_constraints(
    remaining: list[str],
    failed_guesses: list[dict] | None,
) -> Constraints:
    """Translate feedback on past guesses into constraints over the remaining words.

    A guess may include words that have since been solved (moved off the board).
    Solved words sit in a group containing none of the remaining words, which
    lets us project each constraint onto the remaining subset:

      wrong (≤2 of 4 together):
        - all 4 remaining        → ≤2 of them together
        - 3 remaining            → ≤2 of them together (still true)
        - ≤2 remaining           → no information
      one away (exactly 3 of 4 together):
        - all 4 remaining        → exactly 3 together, never all 4
        - 3 remaining, 1 solved  → the solved word's group holds none of the
                                   remaining 3, so the 3 must be together
        - ≤2 remaining           → no information
    """
    rem = set(remaining)
    c = Constraints()
    for fg in failed_guesses or []:
        members = frozenset(fg["members"])
        present = members & rem
        c.tried.add(members)
        if fg.get("feedback") == "one_away":
            if len(present) == 4:
                c.max_together.append((present, 3))
                c.needs_three.append(present)
            elif len(present) == 3:
                c.together.append(present)
        else:  # wrong
            if len(present) >= 3:
                c.max_together.append((present, 2))
    return c


# ---------------------------------------------------------------------------
# Partition search
# ---------------------------------------------------------------------------

Objective = Callable[[float], float]

OBJECTIVES: dict[str, Objective] = {
    "linear": lambda p: p,
    "log": lambda p: math.log(max(p, 1e-3)),
    "logit": lambda p: math.log(max(p, 1e-3) / max(1.0 - p, 1e-3)),
}


def group_score(
    group: tuple[str, ...],
    pairs: dict[tuple[str, str], float],
    objective: Objective,
) -> float:
    return sum(objective(pairs[pair_key(a, b)]) for a, b in itertools.combinations(group, 2))


def group_mean_affinity(group: Iterable[str], pairs: dict[tuple[str, str], float]) -> float:
    ps = [pairs[pair_key(a, b)] for a, b in itertools.combinations(list(group), 2)]
    return sum(ps) / len(ps) if ps else 0.0


def _group_allowed(group: frozenset[str], c: Constraints) -> bool:
    if group in c.tried:
        return False
    for words, max_n in c.max_together:
        if len(group & words) > max_n:
            return False
    for words in c.together:
        k = len(group & words)
        if 0 < k < len(words):
            return False
    for words in c.needs_three:
        # exactly 3 of the 4 must share a group; a group holding exactly 2 makes
        # that impossible (the other 2 can no longer form a trio with either).
        if len(group & words) == 2:
            return False
    return True


def _partition_allowed(groups: list[frozenset[str]], c: Constraints) -> bool:
    for words in c.needs_three:
        if not any(len(g & words) == 3 for g in groups):
            return False
    return True


def best_partition(
    words: list[str],
    pairs: dict[tuple[str, str], float] | None,
    constraints: Constraints,
    objective: Objective = OBJECTIVES["linear"],
    scorer: Callable[[frozenset[str]], float] | None = None,
) -> list[frozenset[str]] | None:
    """Exhaustive branch-and-bound over partitions of `words` into groups of 4.

    Group scores come from `scorer(frozenset)` when given, otherwise from the
    sum of `objective(pair probability)` over the group's 6 pairs.
    Returns the partition with the highest total within-group score that
    satisfies `constraints`, or None if no partition satisfies them.
    """
    words = list(words)
    n = len(words)
    if n % 4 != 0 or n == 0:
        raise ValueError(f"cannot partition {n} words into groups of 4")
    if scorer is None:
        if pairs is None:
            raise ValueError("best_partition needs either `pairs` or `scorer`")
        _pairs = pairs
        scorer = lambda g: group_score(tuple(g), _pairs, objective)  # noqa: E731

    # Score cache for every 4-subset (1820 for n=16), plus the best single-group
    # score for the bound.
    subset_scores: dict[frozenset[str], float] = {}
    for combo in itertools.combinations(words, 4):
        g = frozenset(combo)
        subset_scores[g] = scorer(g)
    best_single = max(subset_scores.values())

    best: dict = {"score": -math.inf, "groups": None}

    def recurse(remaining: list[str], chosen: list[frozenset[str]], score: float) -> None:
        if not remaining:
            if score > best["score"] and _partition_allowed(chosen, constraints):
                best["score"] = score
                best["groups"] = list(chosen)
            return
        groups_left = len(remaining) // 4
        if score + best_single * groups_left <= best["score"]:
            return  # cannot beat the incumbent
        first, rest = remaining[0], remaining[1:]
        # Try the most promising groups first so the bound tightens quickly.
        candidates = []
        for combo in itertools.combinations(rest, 3):
            g = frozenset((first, *combo))
            if _group_allowed(g, constraints):
                candidates.append((subset_scores[g], g))
        candidates.sort(key=lambda t: t[0], reverse=True)
        for s, g in candidates:
            if score + s + best_single * (groups_left - 1) <= best["score"]:
                break  # sorted desc: nothing later can beat the incumbent either
            recurse([w for w in rest if w not in g], chosen + [g], score + s)

    recurse(words, [], 0.0)
    return best["groups"]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class JevStrategy:
    """Callable strategy: (remaining_words, failed_guesses) -> ordered groups.

    Holds the affinity matrix for the current board so that feedback-driven
    re-solves cost no API calls. `affinity_cache` (board_key -> AffinityResult
    JSON dict) lets a benchmark replay stored answers without spending credits.
    """

    def __init__(
        self,
        *,
        client: TypeSafeClient | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        objective: str = "linear",
        affinity_cache: dict[str, dict] | None = None,
        debug_dir: Path | None = None,
        verbose: bool = True,
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}")
        self._client = client
        self._model = model
        self._timeout = timeout
        self.objective_name = objective
        self.objective = OBJECTIVES[objective]
        self.affinity_cache = affinity_cache
        self.debug_dir = debug_dir
        self.verbose = verbose

        self.board: list[str] | None = None
        self.affinity: AffinityResult | None = None
        self.model_used: str | None = None
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.search_seconds = 0.0

    # -- API -----------------------------------------------------------------

    @property
    def client(self) -> TypeSafeClient:
        if self._client is None:
            self._client = TypeSafeClient(model=self._model, timeout=self._timeout)
        return self._client

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[jev] {msg}", file=sys.stderr, flush=True)

    def load_board(self, board: list[str]) -> AffinityResult:
        """Fetch (or replay from cache) the affinity matrix for a board."""
        key = board_key(board)
        cached = self.affinity_cache.get(key) if self.affinity_cache is not None else None
        if cached is not None:
            result = AffinityResult.from_json(cached)
            self._log(f"affinities for {len(board)} words replayed from cache (model={result.model})")
        else:
            self._log(f"asking {len(board) * (len(board) - 1) // 2} pairwise questions ...")
            result = fetch_affinities(board, self.client)
            self.requests += 1
            self.input_tokens += result.input_tokens
            self.output_tokens += result.output_tokens
            self._log(
                f"response model={result.model} input_tokens={result.input_tokens}"
                f" output_tokens={result.output_tokens} in {result.elapsed_seconds}s"
            )
            if self.affinity_cache is not None:
                self.affinity_cache[key] = result.to_json()
        self.board = list(board)
        self.affinity = result
        self.model_used = result.model
        if self.debug_dir is not None:
            self._dump_debug(result)
        return result

    def __call__(self, remaining: list[str], failed_guesses: list[dict] | None = None) -> list[dict]:
        remaining = list(remaining)
        if self.board is None or not set(remaining) <= set(self.board):
            self.load_board(remaining)
        assert self.affinity is not None
        pairs = self.affinity.pairs

        constraints = project_constraints(remaining, failed_guesses)
        t0 = time.monotonic()
        partition = best_partition(remaining, pairs, constraints, self.objective)
        if partition is None:
            # Feedback contradicted itself (e.g. misread page). Relax to the
            # minimum: never repeat an exact guess.
            self._log("constraints infeasible — relaxing to exact-repeat exclusion only")
            relaxed = Constraints(tried=constraints.tried)
            partition = best_partition(remaining, pairs, relaxed, self.objective)
        self.search_seconds += time.monotonic() - t0
        if partition is None:
            return []

        groups = []
        for g in partition:
            members = sorted(g, key=remaining.index)
            mean_p = group_mean_affinity(members, pairs)
            groups.append({
                "theme": f"jev affinity {mean_p:.2f}",
                "members": members,
                "alternatives": [],
                "score": round(group_score(tuple(members), pairs, self.objective), 3),
                "mean_affinity": round(mean_p, 3),
            })
        groups.sort(key=lambda g: g["mean_affinity"], reverse=True)

        if self.verbose:
            self._log(
                f"partition of {len(remaining)} words"
                f" (constraints: {len(constraints.max_together)} max-together,"
                f" {len(constraints.together)} together, {len(constraints.needs_three)} needs-three)"
            )
            for g in groups:
                self._log(f"  {g['mean_affinity']:.2f}  {g['members']}")
        return groups

    # -- debugging -------------------------------------------------------------

    def _dump_debug(self, result: AffinityResult) -> None:
        assert self.debug_dir is not None
        self.debug_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        path = self.debug_dir / f"jev_{ts}_affinity_{len(result.board)}.json"
        path.write_text(json.dumps(result.to_json(), indent=2))
        self._log(f"affinity matrix saved to {path}")
        # Human-readable: each word's top-3 partners
        lines = []
        for w in result.board:
            partners = sorted(
                ((result.pairs[pair_key(w, o)], o) for o in result.board if o != w),
                reverse=True,
            )[:3]
            lines.append(f"{w:>18}: " + ", ".join(f"{o} {p:.2f}" for p, o in partners))
        (self.debug_dir / f"jev_{ts}_top3_{len(result.board)}.txt").write_text("\n".join(lines))


def make_jev_strategy(**kwargs) -> JevStrategy:
    return JevStrategy(**kwargs)


# ---------------------------------------------------------------------------
# Staged Choice ("beam") strategy
#
# Instead of 120 independent yes/no pair questions, ask comparative Choice
# questions and build groups up in stages, so that later stages show Jev a
# pair or a triple together (where wordplay patterns become visible):
#
#   stage 1: 16 Choices  "which other word belongs with X?"        (15 options)
#   stage 2: 120 Choices "which word belongs with X and Y?"        (14 options)
#   stage 3: top-K triples → Choice "which word completes X, Y, Z?" (13 options)
#   stage 4 (optional): Noul per top quad "do these four form one category?"
#
# Every stage is one request. Scores are aggregated per word set and the
# partition search runs over 4-subset scores instead of pair sums.
# ---------------------------------------------------------------------------

from typesafe_sdk import Choice, Score  # noqa: E402


def _quote(words: Iterable[str]) -> str:
    ws = [f'"{w}"' for w in words]
    if len(ws) == 1:
        return ws[0]
    return ", ".join(ws[:-1]) + " and " + ws[-1]


def choice_question(anchor: Iterable[str], options: Iterable[str]) -> Choice:
    anchor = list(anchor)
    if len(anchor) == 1:
        text = f"Which other word on the board belongs in the same category of four as {_quote(anchor)}?"
    elif len(anchor) == 2:
        text = f"Which word on the board belongs in the same category of four as both {_quote(anchor)}?"
    else:
        text = f"Which word on the board completes the category of four with {_quote(anchor)}?"
    return Choice(instructions=text, criteria={w: None for w in options})


def quad_noul(quad: Iterable[str]) -> Noul:
    return Noul(
        instructions=(
            f"Do the four words {_quote(quad)} form exactly one complete Connections "
            f"category on this board, with no odd one out?"
        ),
        criteria={
            "true": "All four share one connection that no other board word shares.",
            "false": "At least one of the four does not belong; a different board word fits better.",
        },
    )


@dataclass
class StageResult:
    """Raw answers from one request, cacheable and independent of scoring."""
    answers: dict[str, dict]      # question id -> {"probabilities": {...}} or {"noul": x}
    model: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float

    def to_json(self) -> dict:
        return {
            "answers": self.answers,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_json(cls, d: dict) -> "StageResult":
        return cls(
            answers=d["answers"],
            model=d.get("model", "?"),
            input_tokens=int(d.get("input_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
            elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
        )


def run_stage(client: TypeSafeClient, board: list[str], questions: dict) -> StageResult:
    state = {"rules": GAME_RULES, "board": list(board)}
    t0 = time.monotonic()
    response = client.system_one(state=state, questions=questions)
    elapsed = time.monotonic() - t0
    answers: dict[str, dict] = {}
    for qid, ans in response.answers.items():
        if hasattr(ans, "probabilities"):
            # Choice keys are option strings; Score keys are level ints in the SDK
            answers[qid] = {"probabilities": {str(k): float(v) for k, v in ans.probabilities.items()}}
            if hasattr(ans, "score"):
                answers[qid]["score"] = float(ans.score)
        elif hasattr(ans, "noul"):
            answers[qid] = {"noul": float(ans.noul)}
    usage = response.usage
    return StageResult(
        answers=answers,
        model=response.model,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        elapsed_seconds=round(elapsed, 2),
    )


def beam_prompt_version() -> str:
    import hashlib

    sample = json.dumps(
        {
            "rules": GAME_RULES,
            "q1": choice_question(["A"], ["B"]).instructions,
            "q2": choice_question(["A", "B"], ["C"]).instructions,
            "q3": choice_question(["A", "B", "C"], ["D"]).instructions,
            "noul": quad_noul(["A", "B", "C", "D"]).instructions,
            "noul_criteria": quad_noul(["A", "B", "C", "D"]).criteria,
        },
        sort_keys=True,
    )
    return hashlib.sha1(sample.encode()).hexdigest()[:8]


class JevBeamStrategy:
    """Callable strategy: (remaining_words, failed_guesses) -> ordered groups.

    weights = (pair, triple, quad, noul): how each stage's evidence contributes
    to a 4-subset's score. Each component is in [0, 1]; missing evidence is 0.
    """

    def __init__(
        self,
        *,
        client: TypeSafeClient | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        triple_beam: int = 200,
        verify_quads: int = 0,
        weights: tuple[float, float, float, float] = (1.0, 1.0, 2.0, 2.0),
        affinity_cache: dict[str, dict] | None = None,
        debug_dir: Path | None = None,
        verbose: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout
        self.triple_beam = triple_beam
        self.verify_quads = verify_quads
        self.weights = weights
        self.affinity_cache = affinity_cache
        self.debug_dir = debug_dir
        self.verbose = verbose

        self.board: list[str] | None = None
        self.pair: dict[tuple[str, str], float] = {}
        self.tri: dict[frozenset[str], float] = {}
        self.quad: dict[frozenset[str], float] = {}
        self.quad_noul: dict[frozenset[str], float] = {}
        self.model_used: str | None = None
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.search_seconds = 0.0

    @property
    def client(self) -> TypeSafeClient:
        if self._client is None:
            self._client = TypeSafeClient(model=self._model, timeout=self._timeout)
        return self._client

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[jev-beam] {msg}", file=sys.stderr, flush=True)

    # -- stages ------------------------------------------------------------------

    MAX_QUESTIONS_PER_REQUEST = 200  # ~30k tokens for 13-option Choices; API caps at 64k
    PARALLEL_REQUESTS = 6

    def _stage(self, name: str, board: list[str], questions: dict) -> StageResult:
        """Run one stage, splitting into several requests if it is too large."""
        ids = list(questions)
        per = self.MAX_QUESTIONS_PER_REQUEST
        if len(ids) <= per:
            return self._request(name, board, questions)
        chunks = [
            (f"{name}[{c}]", {q: questions[q] for q in ids[start : start + per]})
            for c, start in enumerate(range(0, len(ids), per))
        ]
        # Chunks are independent: run them concurrently (the API allows 1200 rpm).
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.PARALLEL_REQUESTS) as pool:
            results = list(pool.map(lambda c: self._request(c[0], board, c[1]), chunks))
        merged: dict[str, dict] = {}
        in_tok = out_tok = 0
        elapsed = 0.0
        for r in results:
            merged.update(r.answers)
            in_tok += r.input_tokens
            out_tok += r.output_tokens
            elapsed = max(elapsed, r.elapsed_seconds)  # wall-clock, since parallel
        return StageResult(merged, results[-1].model, in_tok, out_tok, round(elapsed, 2))

    def _request(self, name: str, board: list[str], questions: dict) -> StageResult:
        import hashlib

        qkey = hashlib.sha1(json.dumps(sorted(questions), sort_keys=True).encode()).hexdigest()[:10]
        key = f"beam:{beam_prompt_version()}:{name}:{qkey}#" + "|".join(sorted(board))
        cached = self.affinity_cache.get(key) if self.affinity_cache is not None else None
        if cached is not None:
            result = StageResult.from_json(cached)
            self._log(f"{name}: {len(questions)} questions replayed from cache")
        else:
            self._log(f"{name}: asking {len(questions)} questions ...")
            result = run_stage(self.client, board, questions)
            self.requests += 1
            self.input_tokens += result.input_tokens
            self.output_tokens += result.output_tokens
            self._log(
                f"{name}: model={result.model} input_tokens={result.input_tokens}"
                f" in {result.elapsed_seconds}s"
            )
            if self.affinity_cache is not None:
                self.affinity_cache[key] = result.to_json()
        self.model_used = result.model
        return result

    def load_board(self, board: list[str]) -> None:
        board = list(board)
        others = lambda *anchor: [w for w in board if w not in anchor]  # noqa: E731

        # Stage 1: one Choice per word over the other 15.
        q1 = {f"w{i}": choice_question([w], others(w)) for i, w in enumerate(board)}
        r1 = self._stage("stage1-pairs", board, q1)
        raw: dict[tuple[str, str], list[float]] = {}
        for i, w in enumerate(board):
            for o, p in r1.answers[f"w{i}"]["probabilities"].items():
                raw.setdefault(pair_key(w, o), []).append(p)
        # symmetrise: average of X→Y and Y→X (each row sums to 1)
        self.pair = {k: sum(v) / len(v) for k, v in raw.items()}

        # Stage 2: one Choice per pair over the other 14.
        pairs = list(itertools.combinations(board, 2))
        q2 = {f"p{i}": choice_question(pr, others(*pr)) for i, pr in enumerate(pairs)}
        r2 = self._stage("stage2-triples", board, q2)
        tri_raw: dict[frozenset[str], float] = {}
        for i, pr in enumerate(pairs):
            for z, p in r2.answers[f"p{i}"]["probabilities"].items():
                t = frozenset((*pr, z))
                tri_raw[t] = tri_raw.get(t, 0.0) + p
        # each triple has 3 sub-pairs, all asked → mean
        self.tri = {t: s / 3 for t, s in tri_raw.items()}

        # Stage 3: top-K triples → Choice over the other 13.
        top_tris = sorted(self.tri.items(), key=lambda kv: kv[1], reverse=True)[: self.triple_beam]
        q3 = {
            f"t{i}": choice_question(sorted(t, key=board.index), others(*t))
            for i, (t, _) in enumerate(top_tris)
        }
        r3 = self._stage("stage3-quads", board, q3)
        quad_raw: dict[frozenset[str], float] = {}
        for i, (t, _) in enumerate(top_tris):
            for w, p in r3.answers[f"t{i}"]["probabilities"].items():
                q = t | {w}
                quad_raw[q] = quad_raw.get(q, 0.0) + p
        # 4 sub-triples per quad; unqueried sub-triples count as 0
        self.quad = {q: s / 4 for q, s in quad_raw.items()}

        # Stage 4 (optional): verify the top quads with a group-level Noul.
        self.quad_noul = {}
        if self.verify_quads > 0:
            top_quads = sorted(self.quad.items(), key=lambda kv: kv[1], reverse=True)[: self.verify_quads]
            q4 = {f"q{i}": quad_noul(sorted(q, key=board.index)) for i, (q, _) in enumerate(top_quads)}
            r4 = self._stage("stage4-verify", board, q4)
            for i, (q, _) in enumerate(top_quads):
                self.quad_noul[q] = r4.answers[f"q{i}"]["noul"]

        self.board = board
        if self.debug_dir is not None:
            self._dump_debug()

    # -- scoring ----------------------------------------------------------------

    def subset_score(self, g: frozenset[str]) -> float:
        wp, wt, wq, wn = self.weights
        pair_mean = sum(self.pair[pair_key(a, b)] for a, b in itertools.combinations(g, 2)) / 6
        tri_mean = sum(self.tri.get(frozenset(t), 0.0) for t in itertools.combinations(g, 3)) / 4
        return (
            wp * pair_mean
            + wt * tri_mean
            + wq * self.quad.get(g, 0.0)
            + wn * self.quad_noul.get(g, 0.0)
        )

    def __call__(self, remaining: list[str], failed_guesses: list[dict] | None = None) -> list[dict]:
        remaining = list(remaining)
        if self.board is None or not set(remaining) <= set(self.board):
            self.load_board(remaining)

        constraints = project_constraints(remaining, failed_guesses)
        t0 = time.monotonic()
        partition = best_partition(remaining, None, constraints, scorer=self.subset_score)
        if partition is None:
            self._log("constraints infeasible — relaxing to exact-repeat exclusion only")
            partition = best_partition(
                remaining, None, Constraints(tried=constraints.tried), scorer=self.subset_score
            )
        self.search_seconds += time.monotonic() - t0
        if partition is None:
            return []

        groups = []
        for g in partition:
            s = self.subset_score(g)
            groups.append({
                "theme": f"jev beam {s:.2f}",
                "members": sorted(g, key=remaining.index),
                "alternatives": [],
                "score": round(s, 3),
                "mean_affinity": round(s, 3),
            })
        groups.sort(key=lambda g: g["score"], reverse=True)
        if self.verbose:
            self._log(
                f"partition of {len(remaining)} words"
                f" (constraints: {len(constraints.max_together)} max-together,"
                f" {len(constraints.together)} together, {len(constraints.needs_three)} needs-three)"
            )
            for g in groups:
                self._log(f"  {g['score']:.2f}  {g['members']}")
        return groups

    def _dump_debug(self) -> None:
        assert self.debug_dir is not None and self.board is not None
        self.debug_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        top_quads = sorted(self.quad.items(), key=lambda kv: kv[1], reverse=True)[:25]
        lines = ["top quads (stage-3 score, noul):"]
        for q, s in top_quads:
            n = self.quad_noul.get(q)
            lines.append(f"  {s:.2f} {'' if n is None else f'{n:.2f}'}  {sorted(q, key=self.board.index)}")
        lines.append("\ntop-3 partners per word (stage-1, symmetrised):")
        for w in self.board:
            partners = sorted(((self.pair[pair_key(w, o)], o) for o in self.board if o != w), reverse=True)[:3]
            lines.append(f"{w:>18}: " + ", ".join(f"{o} {p:.2f}" for p, o in partners))
        (self.debug_dir / f"jevbeam_{ts}_{len(self.board)}.txt").write_text("\n".join(lines))
        self._log(f"debug dump saved to {self.debug_dir}")


# ---------------------------------------------------------------------------
# Beam + code-generated wordplay hypotheses
#
# wordplay.py derives hidden-word tokens from each board word (DIMENSION ->
# DIME, MARIGOLD -> GOLD, BORAT -> BRAT ...). Two more Jev requests turn those
# into scored 4-word hypotheses:
#
#   stage A: one Choice per token — "which other hidden word belongs with
#            DIME?" over the other tokens of the same mechanism
#   stage B: one Noul per top hypothesis — "do DIME, PENNY, NICKEL and QUARTER
#            all belong to one category?" — a plain semantic judgment
#
# A verified hypothesis adds wp_weight * noul to that 4-subset's score in the
# partition search. Letter-pattern groups (palindromes etc.) are added directly.
# ---------------------------------------------------------------------------

import wordplay as _wp  # noqa: E402


def token_choice_question(anchor: _wp.Token, options: list[_wp.Token]) -> Choice:
    return Choice(
        instructions=(
            f"Which other hidden word belongs in the same category of four as "
            f"{anchor.describe()}? Categories are things like coins, metals, colors, "
            f"animals, body parts, or synonyms of one idea."
        ),
        criteria={t.label: None for t in options},
    )


def token_group_noul(tokens: Iterable[_wp.Token]) -> Noul:
    words = [t.text for t in tokens]
    return Noul(
        instructions=(
            f"Do the words {_quote(words)} all belong to one common category?"
        ),
        criteria={
            "true": "All four are members of one recognizable category, or all four are synonyms of the same idea.",
            "false": "At least one of the four does not fit a category shared by the other three.",
        },
    )


NONE_OPTION = "none: all four belong together equally"


def token_odd_one_out(tokens: Iterable[_wp.Token]) -> Choice:
    """Comparative verification: which token does not fit, or none?"""
    words = [t.text for t in tokens]
    criteria: dict[str, str | None] = {w: None for w in words}
    criteria[NONE_OPTION] = "All four are members of one recognizable category, or synonyms of one idea."
    return Choice(
        instructions=(
            f"Consider the words {_quote(words)}. If exactly three of them share a clear "
            f"category and one does not fit, which one is the odd one out? "
            f"If all four fit one category, answer none."
        ),
        criteria=criteria,
    )


CATEGORY_LEVELS = [
    "The words are unrelated, or they are mere fragments/suffixes rather than real words",
    "Two or three are related but at least one clearly does not belong",
    "All four belong to one loose or broad category",
    "All four clearly belong to one specific, nameable category such as coins, metals, fruits, or synonyms of one word",
]
CATEGORY_RULES = [
    "Judge them as ordinary standalone English words or proper names.",
    "Being word fragments, suffixes, prefixes, abbreviations, or all the same part of speech is NOT a category.",
    "Plural forms of everyday nouns with nothing else in common is NOT a category.",
]


def token_group_score(tokens: Iterable[_wp.Token]) -> Score:
    """Graded verification; P(top level) separates true groups from near-misses best."""
    words = [t.text for t in tokens]
    return Score(
        instructions={
            "question": f"How well do the words {_quote(words)} form a single specific category?",
            "rules": CATEGORY_RULES,
        },
        criteria=CATEGORY_LEVELS,
    )


def token_duel_question(shared: Iterable[str], candidates: Iterable[str]) -> Choice:
    """Which candidate best completes the category started by the three shared words?"""
    return Choice(
        instructions={
            "question": (
                f"Which word best completes a specific, nameable category together with "
                f"{_quote(sorted(shared))}?"
            ),
            "rules": CATEGORY_RULES,
        },
        criteria={c: None for c in candidates},
    )


class JevWordplayStrategy(JevBeamStrategy):
    """JevBeamStrategy plus code-generated wordplay hypotheses verified by Jev.

    Stage A Choice probabilities are normalised per anchor (p / max p) before
    hypotheses are ranked, so a word's clear favourite partners count the same
    whether the anchor had 12 or 40 options.

    verify_mode: "score" → Score over CATEGORY_LEVELS; score = P(top level)
                 "noul"  → "do all four belong together?" Noul
                 "odd"   → odd-one-out Choice; score = P(none is odd)
    A hypothesis contributes wp_weight * score only when score >= wp_threshold.
    """

    def __init__(
        self,
        *,
        wp_weight: float = 3.0,
        wp_verify: int = 600,
        wp_threshold: float = 0.6,
        wp_verify_mode: str = "score",
        wp_duels: bool = True,
        wp_affinity_norm: str = "relmax",
        wp_pattern_score: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.wp_weight = wp_weight
        self.wp_verify = wp_verify
        self.wp_threshold = wp_threshold
        self.wp_verify_mode = wp_verify_mode
        self.wp_duels = wp_duels
        self.wp_affinity_norm = wp_affinity_norm
        self.wp_pattern_score = wp_pattern_score
        self.wp: dict[frozenset[str], float] = {}
        self.wp_label: dict[frozenset[str], str] = {}
        # kept for analysis / debugging
        self.wp_tokens: list[_wp.Token] = []
        self.wp_affinity: dict[frozenset[_wp.Token], float] = {}
        self.wp_hyps: list[_wp.Hypothesis] = []

    def load_board(self, board: list[str]) -> None:
        super().load_board(board)
        self._load_wordplay(list(board))

    def _load_wordplay(self, board: list[str]) -> None:
        self.wp, self.wp_label = {}, {}
        self.wp_tokens, self.wp_affinity, self.wp_hyps = [], {}, []

        # Deterministic letter patterns need no model.
        for name, matched in _wp.letter_pattern_groups(board):
            score = self.wp_pattern_score if len(matched) == 4 else self.wp_pattern_score * 0.6
            for quad in itertools.combinations(sorted(matched), 4):
                g = frozenset(quad)
                if score > self.wp.get(g, 0.0):
                    self.wp[g], self.wp_label[g] = score, f"pattern:{name}"

        tokens = _wp.board_tokens(board)
        if len(tokens) < 4:
            return
        by_mech: dict[str, list[_wp.Token]] = {}
        for t in tokens:
            by_mech.setdefault(t.mechanism, []).append(t)

        # Stage A: token affinities via Choice, per mechanism.
        qa: dict[str, Choice] = {}
        anchors: dict[str, tuple[_wp.Token, list[_wp.Token]]] = {}
        for mech, ts in by_mech.items():
            for i, t in enumerate(ts):
                opts = [o for o in ts if o.source != t.source]
                if len(opts) < 3:
                    continue
                qid = f"{mech}{i}"
                qa[qid] = token_choice_question(t, opts[:250])
                anchors[qid] = (t, opts[:250])
        if not qa:
            return
        ra = self._stage("stageA-tokens", board, qa)
        raw: dict[frozenset[_wp.Token], list[float]] = {}
        for qid, (t, opts) in anchors.items():
            by_label = {o.label: o for o in opts}
            probs = ra.answers[qid]["probabilities"]
            scale = (max(probs.values()) or 1.0) if self.wp_affinity_norm == "relmax" else 1.0
            for label, p in probs.items():
                o = by_label.get(label)
                if o is not None:
                    raw.setdefault(frozenset((t, o)), []).append(p / scale)
        affinity = {k: sum(v) / len(v) for k, v in raw.items()}

        hyps = _wp.wordplay_hypotheses(
            tokens, affinity, top_n=self.wp_verify, subset_shortlist=max(600, self.wp_verify)
        )
        self.wp_tokens, self.wp_affinity, self.wp_hyps = tokens, affinity, hyps
        if not hyps:
            return

        # Stage B: verify each hypothesis as a plain four-word category.
        if self.wp_verify_mode == "score":
            qb = {f"h{i}": token_group_score(h.tokens) for i, h in enumerate(hyps)}
            rb = self._stage("stageB-score", board, qb)
            top = str(len(CATEGORY_LEVELS) - 1)
            for i, h in enumerate(hyps):
                h.verified = rb.answers[f"h{i}"]["probabilities"].get(top, 0.0)
        elif self.wp_verify_mode == "odd":
            qb = {f"h{i}": token_odd_one_out(h.tokens) for i, h in enumerate(hyps)}
            rb = self._stage("stageB-odd", board, qb)
            for i, h in enumerate(hyps):
                h.verified = rb.answers[f"h{i}"]["probabilities"].get(NONE_OPTION, 0.0)
        else:
            qb = {f"h{i}": token_group_noul(h.tokens) for i, h in enumerate(hyps)}
            rb = self._stage("stageB-verify", board, qb)
            for i, h in enumerate(hyps):
                h.verified = rb.answers[f"h{i}"]["noul"]
        # Stage C: duels. Hypotheses that share three token texts but differ in
        # the fourth compete for the same triple; at most one can be right. Ask
        # which candidate completes the category and scale losers down.
        kept = [h for h in hyps if h.verified is not None and h.verified >= self.wp_threshold]
        if self.wp_duels and kept:
            by_triple: dict[frozenset[str], dict[str, list[_wp.Hypothesis]]] = {}
            for h in kept:
                texts = [t.text for t in h.tokens]
                for t4 in texts:
                    tri = frozenset(texts) - {t4}
                    if len(tri) == 3:
                        by_triple.setdefault(tri, {}).setdefault(t4, []).append(h)
            duels = {tri: c for tri, c in by_triple.items() if len(c) >= 2}
            if duels:
                keys = list(duels)
                qc = {f"d{i}": token_duel_question(tri, duels[tri]) for i, tri in enumerate(keys)}
                rc = self._stage("stageC-duels", board, qc)
                factor: dict[int, float] = {}
                for i, tri in enumerate(keys):
                    probs = rc.answers[f"d{i}"]["probabilities"]
                    mx = max(probs.values()) or 1.0
                    for cand, hs in duels[tri].items():
                        f = probs.get(cand, 0.0) / mx
                        for h in hs:
                            factor[id(h)] = min(factor.get(id(h), 1.0), f)
                for h in kept:
                    h.verified = h.verified * factor.get(id(h), 1.0)
                self._log(f"wordplay: {len(duels)} duels among {len(kept)} verified hypotheses")

        for h in hyps:
            if h.verified > self.wp.get(h.words, 0.0):
                self.wp[h.words] = h.verified
                self.wp_label[h.words] = h.describe()

        if self.verbose:
            top = sorted(self.wp.items(), key=lambda kv: kv[1], reverse=True)[:8]
            self._log(f"wordplay: {len(tokens)} tokens, {len(hyps)} hypotheses verified; top:")
            for g, s in top:
                self._log(f"  {s:.2f}  {sorted(g, key=board.index)}  <- {self.wp_label[g]}")

    def subset_score(self, g: frozenset[str]) -> float:
        # Ramp from 0 at the threshold to wp_weight at a perfect verification,
        # so borderline hypotheses barely nudge the beam score.
        s = self.wp.get(g, 0.0)
        thr = self.wp_threshold
        bonus = self.wp_weight * (s - thr) / (1.0 - thr) if s > thr else 0.0
        return super().subset_score(g) + bonus

    def __call__(self, remaining: list[str], failed_guesses: list[dict] | None = None) -> list[dict]:
        groups = super().__call__(remaining, failed_guesses)
        for grp in groups:
            g = frozenset(grp["members"])
            if g in self.wp_label:
                grp["theme"] = f"jev wp {self.wp[g]:.2f} {self.wp_label[g]}"
        return groups


# ---------------------------------------------------------------------------
# Wordplay + fill-in-the-blank hypotheses
#
# blanks.py finds candidate blanks in a phrase dictionary (Wikipedia titles):
# common words that form a phrase with >= 3 board words on one side. Jev then
# grades every (blank, board word) phrase with a Score question; per blank the
# four strongest members form a hypothesis scored by its weakest member, and
# hypotheses feed the same bonus mechanism as wordplay ones.
# ---------------------------------------------------------------------------

import blanks as _bl  # noqa: E402

PHRASE_LEVELS = [
    "Not a phrase; just two words placed together",
    "Plausible but not a set expression",
    "A recognizable phrase some people use",
    "A fixed, well-known compound, expression, name, or title",
]


def phrase_score_question(member: str, blank: str, side: str) -> Score:
    return Score(
        instructions=f'How established is "{_bl.phrase(member, blank, side)}" as a phrase?',
        criteria=PHRASE_LEVELS,
    )


class JevBlankStrategy(JevWordplayStrategy):
    """JevWordplayStrategy plus fill-in-the-blank hypotheses."""

    def __init__(
        self,
        *,
        bl_cap: int = 400,
        bl_min_members: int = 3,
        bl_weight: float = 3.0,
        bl_threshold: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.bl_cap = bl_cap
        self.bl_min_members = bl_min_members
        self.bl_weight = bl_weight
        self.bl_threshold = bl_threshold
        self.bl: dict[frozenset[str], float] = {}
        self.bl_label: dict[frozenset[str], str] = {}
        self.bl_cands: list[_bl.BlankCandidate] = []
        # (blank, side, word) -> P(level >= 2): "at least a recognizable phrase".
        # The minimum over four members separates real blank groups from
        # accidental Wikipedia-title matches far better than P(top level).
        self.bl_pair: dict[tuple[str, str, str], float] = {}

    def load_board(self, board: list[str]) -> None:
        super().load_board(board)
        self._load_blanks(list(board))

    def _load_blanks(self, board: list[str]) -> None:
        self.bl, self.bl_label, self.bl_pair = {}, {}, {}
        self.bl_cands = _bl.candidate_blanks(board, min_members=self.bl_min_members, cap=self.bl_cap)
        if not self.bl_cands:
            self._log("blanks: no dictionary candidates")
            return

        # Stage V: grade every (blank, board word) phrase on the candidate's side.
        qv: dict[str, Score] = {}
        keys: dict[str, tuple[str, str, str]] = {}
        for i, c in enumerate(self.bl_cands):
            for j, w in enumerate(board):
                qid = f"v{i}_{j}"
                qv[qid] = phrase_score_question(w, c.blank, c.side)
                keys[qid] = (c.blank, c.side, w)
        rv = self._stage("stageV-phrases", board, qv)
        for qid, key in keys.items():
            probs = rv.answers[qid]["probabilities"]
            self.bl_pair[key] = sum(v for lvl, v in probs.items() if int(lvl) >= 2)

        for c in self.bl_cands:
            scored = sorted(((self.bl_pair[(c.blank, c.side, w)], w) for w in board), reverse=True)
            strong = [(p, w) for p, w in scored if p >= self.bl_threshold]
            pool = scored[:4] if len(strong) < 4 else strong[:6]
            for combo in itertools.combinations(pool, 4):
                words = frozenset(w for _, w in combo)
                score = min(p for p, _ in combo)
                if score > self.bl.get(words, 0.0):
                    pattern = f"___ {c.blank}" if c.side == "before" else f"{c.blank} ___"
                    self.bl[words] = score
                    self.bl_label[words] = f"blank:{pattern} score={score:.2f}"

        if self.verbose:
            self._log(
                f"blanks: {len(self.bl_cands)} dictionary candidates ({len(qv)} phrase questions); top:"
            )
            for g, sc in sorted(self.bl.items(), key=lambda kv: kv[1], reverse=True)[:6]:
                self._log(f"  {sc:.2f}  {sorted(g, key=board.index)}  <- {self.bl_label[g]}")

    def subset_score(self, g: frozenset[str]) -> float:
        s = self.bl.get(g, 0.0)
        thr = self.bl_threshold
        bonus = self.bl_weight * (s - thr) / (1.0 - thr) if s > thr else 0.0
        return super().subset_score(g) + bonus

    def __call__(self, remaining: list[str], failed_guesses: list[dict] | None = None) -> list[dict]:
        groups = super().__call__(remaining, failed_guesses)
        for grp in groups:
            g = frozenset(grp["members"])
            if g in self.bl_label and self.bl[g] > self.bl_threshold and not grp["theme"].startswith("jev wp"):
                grp["theme"] = f"jev bl {self.bl[g]:.2f} {self.bl_label[g]}"
        return groups
