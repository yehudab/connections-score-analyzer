#!/usr/bin/env python3
"""
bench_jev.py

Offline benchmark for the Jev Connections strategy. Replays past NYT
Connections puzzles against jev_solver.JevStrategy with a simulated game
(same feedback rules as the real board: correct / one away / wrong, four
mistakes allowed) and reports the solve rate.

Puzzle data: https://github.com/Eyefyre/NYT-Connections-Answers (connections.json),
downloaded on first run into bench-data/. Jev answers are cached per board in
bench-data/jev_cache.json so re-runs (e.g. with a different objective) cost
nothing.

Usage:
    ./bench_jev.py                      # last 50 puzzles, linear objective
    ./bench_jev.py --last 100 --objective log
    ./bench_jev.py --ids 1180,1181 -v   # specific puzzles, verbose per-guess log
    ./bench_jev.py --before 2026-09-17 --last 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from connections_solver import MAX_MISTAKES, _load_env_file
from jev_solver import JevBeamStrategy, JevStrategy, JevWordplayStrategy, OBJECTIVES

_load_env_file()

DATA_URL = (
    "https://raw.githubusercontent.com/Eyefyre/NYT-Connections-Answers/main/connections.json"
)
BENCH_DIR = Path(__file__).parent / "bench-data"
DATA_PATH = BENCH_DIR / "connections.json"
CACHE_PATH = BENCH_DIR / "jev_cache.json"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_puzzles(path: Path = DATA_PATH) -> list[dict]:
    if not path.exists():
        BENCH_DIR.mkdir(exist_ok=True)
        print(f"downloading puzzle data to {path} ...", file=sys.stderr)
        urllib.request.urlretrieve(DATA_URL, path)
    puzzles = json.loads(path.read_text())
    good = [
        p for p in puzzles
        if len(p.get("answers", [])) == 4 and all(len(a["members"]) == 4 for a in p["answers"])
    ]
    return good


def select_puzzles(puzzles: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",")}
        return [p for p in puzzles if p["id"] in wanted]
    if args.before:
        puzzles = [p for p in puzzles if p["date"] < args.before]
    puzzles = sorted(puzzles, key=lambda p: p["date"])
    return puzzles[-args.last:] if args.last else puzzles


def load_cache(path: Path = CACHE_PATH) -> dict[str, dict]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(cache: dict[str, dict], path: Path = CACHE_PATH) -> None:
    BENCH_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Simulated game (mirrors play_game() in connections_solver.py, minus the browser)
# ---------------------------------------------------------------------------


def feedback_for(guess: list[str], solution: list[frozenset[str]]) -> str:
    g = frozenset(guess)
    for group in solution:
        if g == group:
            return "ok"
    if any(len(g & group) == 3 for group in solution):
        return "one_away"
    return "wrong"


def simulate(strategy, board: list[str], solution: list[frozenset[str]], verbose: bool = False) -> dict:
    remaining = list(board)
    mistakes = 0
    solved: list[frozenset[str]] = []
    failed_guesses: list[dict] = []
    tried: set[frozenset[str]] = set()
    guesses: list[dict] = []
    resolves = 0

    groups = strategy(remaining)
    while remaining and mistakes < MAX_MISTAKES:
        if not groups:
            groups = strategy(remaining, failed_guesses)
            resolves += 1
            if not groups:
                break
        group = groups.pop(0)
        members = list(group["members"])
        if any(m not in remaining for m in members):
            continue
        key = frozenset(members)
        if key in tried:
            groups = strategy(remaining, failed_guesses)
            resolves += 1
            continue
        tried.add(key)

        fb = feedback_for(members, solution)
        guesses.append({"members": members, "feedback": fb, "theme": group.get("theme")})
        if verbose:
            mark = {"ok": "✓", "one_away": "~", "wrong": "✗"}[fb]
            print(f"    {mark} {group.get('theme', ''):<20} {members}", file=sys.stderr)

        if fb == "ok":
            for w in members:
                remaining.remove(w)
            solved.append(key)
            groups = [g for g in groups if all(m in remaining for m in g["members"])]
        else:
            mistakes += 1
            failed_guesses.append({"members": members, "feedback": fb})
            if mistakes < MAX_MISTAKES:
                # Jev groups carry no alternatives: always re-solve with constraints.
                groups = strategy(remaining, failed_guesses)
                resolves += 1

    return {
        "success": len(solved) == 4,
        "mistakes": mistakes,
        "groups_solved": len(solved),
        "solved": [sorted(g) for g in solved],
        "guesses": guesses,
        "resolves": resolves,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_puzzle(puzzle: dict, cache: dict, args: argparse.Namespace) -> dict:
    solution = [frozenset(a["members"]) for a in puzzle["answers"]]
    level_of = {frozenset(a["members"]): a["level"] for a in puzzle["answers"]}
    theme_of = {frozenset(a["members"]): a["group"] for a in puzzle["answers"]}

    board = [w for a in puzzle["answers"] for w in a["members"]]
    random.Random(f"{args.seed}:{puzzle['id']}").shuffle(board)

    if args.strategy in ("beam", "wordplay"):
        cls = JevWordplayStrategy if args.strategy == "wordplay" else JevBeamStrategy
        extra = (
            dict(wp_weight=args.wp_weight, wp_verify=args.wp_verify,
                 wp_threshold=args.wp_threshold, wp_verify_mode=args.wp_verify_mode,
                 wp_duels=not args.no_duels)
            if args.strategy == "wordplay" else {}
        )
        strategy = cls(
            **extra,
            triple_beam=args.beam_triples,
            verify_quads=args.verify_quads,
            weights=tuple(float(x) for x in args.weights.split(",")),
            affinity_cache=cache,
            verbose=args.verbose,
            timeout=args.timeout,
        )
    else:
        strategy = JevStrategy(
            objective=args.objective,
            affinity_cache=cache,
            verbose=args.verbose,
            timeout=args.timeout,
        )
    t0 = time.monotonic()
    result = simulate(strategy, board, solution, verbose=args.verbose)
    elapsed = time.monotonic() - t0

    unsolved = [g for g in solution if g not in {frozenset(s) for s in result["solved"]}]
    result.update({
        "id": puzzle["id"],
        "date": puzzle["date"],
        "elapsed_seconds": round(elapsed, 2),
        "search_seconds": round(strategy.search_seconds, 2),
        "requests": strategy.requests,
        "input_tokens": strategy.input_tokens,
        "model": strategy.model_used,
        "unsolved": [
            {"group": theme_of[g], "level": level_of[g], "members": sorted(g)} for g in unsolved
        ],
        "first_guess_level": next(
            (level_of[frozenset(g["members"])] for g in result["guesses"] if g["feedback"] == "ok"),
            None,
        ),
    })
    return result


def print_summary(results: list[dict], price_per_mtok: float) -> None:
    n = len(results)
    solved = sum(r["success"] for r in results)
    mistakes = Counter(r["mistakes"] for r in results)
    groups_hist = Counter(r["groups_solved"] for r in results)
    tokens = sum(r["input_tokens"] for r in results)
    requests = sum(r["requests"] for r in results)
    api_time = sum(r["elapsed_seconds"] - r["search_seconds"] for r in results)
    search_time = sum(r["search_seconds"] for r in results)
    unsolved_levels = Counter(u["level"] for r in results for u in r["unsolved"])

    print()
    print("=" * 64)
    print(f"Puzzles:            {n}")
    print(f"Solved:             {solved}/{n}  ({100.0 * solved / n:.0f}%)")
    print(f"Failed:             {n - solved}")
    print(f"Mistakes histogram: " + ", ".join(f"{k}: {mistakes[k]}" for k in sorted(mistakes)))
    print(f"Groups solved:      " + ", ".join(f"{k}/4: {groups_hist[k]}" for k in sorted(groups_hist)))
    if unsolved_levels:
        print(
            "Unsolved by level:  "
            + ", ".join(f"L{k}: {unsolved_levels[k]}" for k in sorted(unsolved_levels))
            + "   (0 = yellow/easiest … 3 = purple/hardest; -1 = unknown)"
        )
    print(f"Jev requests:       {requests}  ({requests / n:.2f} per puzzle; 0 means cache replay)")
    print(f"Input tokens:       {tokens:,}  (~${tokens / 1e6 * price_per_mtok:.4f} at ${price_per_mtok}/Mtok)")
    print(f"Time:               API {api_time:.1f}s, partition search {search_time:.1f}s")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--last", type=int, default=50, help="use the N most recent puzzles (default 50)")
    ap.add_argument("--before", metavar="YYYY-MM-DD", help="only puzzles dated strictly before this")
    ap.add_argument("--ids", help="comma-separated puzzle ids (overrides --last/--before)")
    ap.add_argument("--strategy", choices=["pairwise", "beam", "wordplay"], default="pairwise",
                    help="pairwise: 120 Nouls; beam: staged Choice questions; wordplay: beam + code-generated hidden-word hypotheses")
    ap.add_argument("--wp-weight", type=float, default=3.0, help="(wordplay) weight of a verified hypothesis")
    ap.add_argument("--wp-verify", type=int, default=600, help="(wordplay) hypotheses sent to verification")
    ap.add_argument("--wp-threshold", type=float, default=0.6, help="(wordplay) min verified score to count")
    ap.add_argument("--no-duels", action="store_true", help="(wordplay) skip the stage-C duel questions")
    ap.add_argument("--wp-verify-mode", choices=["score", "odd", "noul"], default="score", help="(wordplay) verification question")
    ap.add_argument("--objective", choices=sorted(OBJECTIVES), default="linear",
                    help="(pairwise) how pair probabilities combine into a group score")
    ap.add_argument("--beam-triples", type=int, default=200, help="(beam) triples carried into stage 3")
    ap.add_argument("--verify-quads", type=int, default=0, help="(beam) top quads to verify with a Noul (0 = skip)")
    ap.add_argument("--weights", default="1,1,2,2", help="(beam) pair,triple,quad,noul weights")
    ap.add_argument("--seed", default="bench", help="board shuffle seed")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--price", type=float, default=0.042, help="$ per Mtok input, for the cost estimate")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the answer cache")
    ap.add_argument("--cache", default=str(CACHE_PATH), help="answer cache file (default bench-data/jev_cache.json)")
    ap.add_argument("--out", help="write per-puzzle results JSON here (default bench-data/results_<ts>.json)")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every guess and Jev call")
    args = ap.parse_args()

    puzzles = select_puzzles(load_puzzles(), args)
    if not puzzles:
        print("no puzzles selected", file=sys.stderr)
        sys.exit(1)
    cache = {} if args.no_cache else load_cache(Path(args.cache))

    print(
        f"benchmarking {len(puzzles)} puzzles ({puzzles[0]['date']} .. {puzzles[-1]['date']})"
        f" strategy={args.strategy} objective={args.objective} cached_entries={len(cache)}",
        file=sys.stderr,
    )

    results: list[dict] = []
    try:
        for i, puzzle in enumerate(puzzles, 1):
            print(f"[{i}/{len(puzzles)}] #{puzzle['id']} {puzzle['date']}", file=sys.stderr)
            r = run_puzzle(puzzle, cache, args)
            results.append(r)
            status = "SOLVED" if r["success"] else f"FAILED ({r['groups_solved']}/4)"
            miss = "; ".join(f"L{u['level']} {u['group']}" for u in r["unsolved"])
            print(
                f"    {status:<14} mistakes={r['mistakes']} tokens={r['input_tokens']}"
                f" api={r['elapsed_seconds'] - r['search_seconds']:.1f}s search={r['search_seconds']:.1f}s"
                + (f"  missed: {miss}" if miss else ""),
                file=sys.stderr,
            )
            if not args.no_cache:
                save_cache(cache, Path(args.cache))
    except KeyboardInterrupt:
        print("\ninterrupted — summarising what finished", file=sys.stderr)

    if not results:
        return
    print_summary(results, args.price)

    out = Path(args.out) if args.out else BENCH_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
