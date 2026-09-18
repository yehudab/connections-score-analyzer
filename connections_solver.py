#!/usr/bin/env python3
"""
connections_solver.py

Automatically solves the daily NYT Connections puzzle and captures a winning screenshot.

Pipeline:
  1. Connect to a CDP server (Lightpanda or Chrome) at the given WebSocket URL
  2. Open the NYT Connections page via Playwright
  3. Scrape the 16 tile words from the DOM
  4. Ask a solver strategy to group them:
       - openrouter (default): an LLM via OpenRouter returns groups WITH
         pre-computed "one away" alternatives
       - jev: TypeSafe's Jev model answers typed questions about the words;
         grouping, wordplay/blank hypothesis generation and feedback handling
         happen in code (see jev_solver.py, wordplay.py, blanks.py)
  5. Iteratively submit groups:
       - Correct: remove tiles from board, move on
       - "One Away": use pre-computed alternative swaps, else re-ask the strategy
       - Completely wrong: re-ask the strategy with failure history
  6. Screenshot the completed board

Usage:
    python connections_solver.py [--cdp-url ws://127.0.0.1:9222] [--output win.png]

    # Use Playwright's bundled Chromium instead of Lightpanda:
    python connections_solver.py --no-cdp --headed

    # Solve with Jev instead of OpenRouter:
    python connections_solver.py --solver jev --no-cdp

    # Verbose LLM logging + debug screenshots:
    python connections_solver.py --debug

Environment:
    SOLVER               optional — "openrouter" (default) or "jev"
    OPENROUTER_API_KEY   required for the openrouter solver
    OPENROUTER_MODEL     optional — model to use (default: google/gemini-2.5-pro)
    TYPESAFE_API_KEY     required for the jev solver
    JEV_STRATEGY         optional — blanks (default), wordplay, beam, or pairwise

Requirements:
    pip install playwright openai
    playwright install chromium   # only needed for --no-cdp mode
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from playwright.async_api import async_playwright


def _load_env_file() -> None:
    """Load KEY=VALUE pairs from .env in the project dir into os.environ."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:   # env var takes precedence
            os.environ[key] = value


_load_env_file()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NYT_CONNECTIONS_URL = "https://www.nytimes.com/games/connections"
DEFAULT_CDP_URL = "ws://localhost:9222"
LIGHTPANDA_CLOUD_URL = "wss://euwest.cloud.lightpanda.io/ws?browser=chrome"
DEFAULT_MODEL = "google/gemini-2.5-pro"

SOLVER_DEBUG_DIR = Path(__file__).parent / "solver-debug"
SOLVER_IMAGES_DIR = Path(__file__).parent / "solver-images"

TILE_SELECTORS = [
    'label[class*="Card-module_label"]',   # Sept 2026 markup: <label><span aria-hidden>W</span><span>W</span></label>
    '[data-testid="card-label"]',
    '[data-testid="card"] span',
    '.cell-text',
    '[class*="Card"] [class*="label"]',
    '[class*="card"] [class*="label"]',
    '[class*="Cell"] span',
    '[class*="cell"] span',
    '[class*="Tile"] span',
]

# Sept 2026 markup: every tile is a visually-hidden checkbox
#   <input data-testid="card-input" id="inner-card-N" aria-label="WORD" value="WORD">
# with an associated <label> that renders the word (twice: visible + aria-hidden).
# The input is the source of truth: aria-label for the word, .checked for
# selection, .disabled for readiness. Older selectors remain as fallbacks.
TILE_INPUT_SELECTOR = 'input[data-testid="card-input"]'

# Shared JS helpers, prepended to the page snippets below.
_TILE_JS = """
    const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
    const INPUT_SEL = %s;
    const SELECTORS = %s;
    // Word on a legacy tile element: prefer the visible span, else collapse
    // an exactly doubled string ("PILLOWPILLOW" -> "PILLOW").
    const tileText = el => {
        if (el.tagName === 'INPUT') return norm(el.getAttribute('aria-label') || el.value);
        const visible = Array.from(el.querySelectorAll('span')).find(s => !s.hasAttribute('aria-hidden'));
        let t = norm(visible ? visible.textContent : el.textContent);
        const h = t.length / 2;
        if (!visible && t.length %% 2 === 0 && h > 0 && t.slice(0, h) === t.slice(h)) t = t.slice(0, h);
        return t;
    };
    // All tile elements: the inputs when present, else the first legacy selector with 16 hits.
    const tileElements = () => {
        const inputs = Array.from(document.querySelectorAll(INPUT_SEL));
        if (inputs.length) return inputs;
        for (const sel of SELECTORS) {
            const els = Array.from(document.querySelectorAll(sel)).filter(e => tileText(e));
            if (els.length === 16) return els;
        }
        return [];
    };
    const findTile = word => tileElements().find(e => tileText(e) === word) || null;
    // What to click for a tile element: the input's label, or the enclosing label/button.
    const clickable = el => el.tagName === 'INPUT'
        ? ((el.labels && el.labels[0]) || el)
        : (el.closest('label') || el.closest('button') || el);
    const isSelected = el => el.tagName === 'INPUT'
        ? el.checked
        : Array.from(clickable(el).classList).some(c => c.startsWith('Card-module_selected__'));
    const isReady = el => {
        if (el.tagName === 'INPUT') return !el.disabled;
        const t = clickable(el);
        const ctl = t.tagName === 'LABEL' ? (t.control || t.querySelector('input')) : t;
        return !(ctl && ctl.disabled) && t.getAttribute('aria-disabled') !== 'true';
    };
""" % (json.dumps(TILE_INPUT_SELECTOR), json.dumps(TILE_SELECTORS))

# NYT gives 4 mistakes before game over
MAX_MISTAKES = 4

# How long to wait for the 16 tiles to render after clicking Play
TILE_WAIT_SECONDS = 30

# Global debug flag (set in main)
DEBUG = False
_llm_call_index = 0


# ---------------------------------------------------------------------------
# LLM logging
# ---------------------------------------------------------------------------


def log_llm(label: str, text: str) -> None:
    """Print LLM prompt/response to stderr; dump to file when --debug is set."""
    border = "─" * 60
    print(f"\n{border}", file=sys.stderr)
    print(f"  LLM {label}", file=sys.stderr)
    print(border, file=sys.stderr)
    print(text, file=sys.stderr)
    print(border, file=sys.stderr)

    if DEBUG:
        global _llm_call_index
        SOLVER_DEBUG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        fname = SOLVER_DEBUG_DIR / f"llm_{ts}_{_llm_call_index:02d}_{label.replace(' ', '_')}.txt"
        _llm_call_index += 1
        fname.write_text(text)
        print(f"  (saved to {fname})", file=sys.stderr)


# ---------------------------------------------------------------------------
# Overlay / UI helpers
# ---------------------------------------------------------------------------


async def dismiss_overlays(page) -> None:
    candidates = [
        'button[data-testid="GDPR-accept"]',
        '#fides-accept-all-button',
        '#games-fullscreen-modal button[class*="close"]',
        'button[aria-label="Close"]',
        'button:text("Got it")',
        'button:text("Accept")',
        'button:text("Accept All")',
        'button:text("Play")',
        'button:text("Play!")',
        # Sept 2026: an ad/upsell interstitial after Play with a countdown button
        'button:has-text("Continue to Connections")',
        'button:has-text("Continue")',
    ]
    for selector in candidates:
        try:
            btn = page.locator(selector).first
            if await btn.is_visible(timeout=400):
                # Short timeout: a countdown button may still be disabled; the
                # caller polls and we will try again on the next pass.
                await btn.click(timeout=3000)
                await page.wait_for_timeout(600)
        except Exception:
            pass
    # Force-remove any lingering Fides overlay that blocks clicks
    await page.evaluate("document.getElementById('fides-overlay')?.remove()")


# ---------------------------------------------------------------------------
# Tile extraction
# ---------------------------------------------------------------------------


async def extract_tiles(page) -> list[str]:
    tiles = await page.evaluate("() => {" + _TILE_JS + " return tileElements().map(tileText).filter(t => t); }")
    if len(tiles) == 16:
        return tiles

    # Broad fallback: short button texts that look like game tiles
    return await page.evaluate("""() => {
        const skip = /submit|deselect|shuffle|one away|congratulations|got it|accept|close/i;
        return Array.from(document.querySelectorAll('button'))
            .map(b => b.textContent.replace(/\\s+/g, ' ').trim())
            .filter(t => t.length > 0 && t.length < 40 && !skip.test(t));
    }""")


# ---------------------------------------------------------------------------
# LLM solver (OpenRouter)
#
# The LLM returns groups with pre-computed alternatives:
#
#   "alternatives": [
#     {"remove": "WORD_A", "add": "WORD_X"},   // most likely fix if one-away
#     {"remove": "WORD_B", "add": "WORD_X"},   // second most likely
#   ]
#
# This lets us recover from "One Away" without spending an extra blind guess.
# ---------------------------------------------------------------------------

_GROUP_SCHEMA_EXAMPLE = """{
  "theme": "SHORT THEME",
  "members": ["W1", "W2", "W3", "W4"],
  "alternatives": [
    {"remove": "WORD_LEAST_CONFIDENT", "add": "MOST_LIKELY_REPLACEMENT"},
    {"remove": "WORD_SECOND_LEAST", "add": "SECOND_REPLACEMENT"}
  ]
}"""


def _llm_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _parse_groups(raw: str) -> list[dict]:
    """Extract groups from an LLM response that may contain thinking text or markdown.

    Uses a brace-matching scan to find every top-level JSON object in the
    response, then returns the *last* one that contains a 'groups' key.
    This handles models that prepend chain-of-thought or wrap JSON in fences.
    """
    candidates: list[dict] = []
    i = 0
    while i < len(raw):
        if raw[i] == "{":
            depth = 0
            for j in range(i, len(raw)):
                if raw[j] == "{":
                    depth += 1
                elif raw[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(raw[i : j + 1])
                            if "groups" in data:
                                candidates.append(data)
                        except json.JSONDecodeError:
                            pass
                        break
        i += 1

    if not candidates:
        raise ValueError(f"No valid JSON with 'groups' key found in response:\n{raw[:500]}")

    return candidates[-1]["groups"]


def solve_with_llm(
    tiles: list[str],
    model: str,
    failed_guesses: list[dict] | None = None,
) -> list[dict]:
    """Ask the LLM to partition tiles into themed groups with alternatives."""
    client = _llm_client()

    failure_context = ""
    if failed_guesses:
        lines = []
        for fg in failed_guesses:
            note = (
                "one away (3 of 4 correct)"
                if fg["feedback"] == "one_away"
                else "completely wrong"
            )
            lines.append(f"  - {fg['members']} → {note}")
        failure_context = (
            "\n\nPrevious INCORRECT guesses — do NOT repeat these exact groupings:\n"
            + "\n".join(lines)
        )

    n_groups = len(tiles) // 4

    prompt = f"""You are solving the NYT Connections puzzle.

The remaining words/phrases on the board are:
{json.dumps(tiles, indent=2)}
{failure_context}
Group ALL of them into exactly {n_groups} group(s) of 4.
Each group shares a hidden connection. Order groups from easiest to hardest.

For each group also include 1–2 "alternatives": if the guess comes back as "one away"
(exactly 3 of your 4 are correct), which member would you swap out and what would you
replace it with? List your least-confident member first.

Rules:
- Every tile appears in exactly one group.
- Member strings must be copied verbatim from the input list above.
- Return ONLY valid JSON — no prose, no markdown fences.

Required JSON format:
{{
  "groups": [
    {_GROUP_SCHEMA_EXAMPLE},
    ...
  ]
}}"""

    log_llm("PROMPT", prompt)

    print(f"[llm] calling model={model} tiles={len(tiles)} ...", file=sys.stderr, flush=True)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16000,
    )
    usage = response.usage
    print(
        f"[llm] response received id={response.id}"
        f" prompt_tokens={usage.prompt_tokens if usage else '?'}"
        f" completion_tokens={usage.completion_tokens if usage else '?'}",
        file=sys.stderr, flush=True,
    )

    raw = response.choices[0].message.content
    log_llm("RESPONSE", raw)

    groups = _parse_groups(raw)
    for g in groups:
        g.setdefault("alternatives", [])
    return groups


# ---------------------------------------------------------------------------
# Solver strategies
#
# A strategy is a callable:  strategy(remaining_words, failed_guesses) -> groups
# where groups is an ordered list of {"theme", "members", "alternatives"} dicts.
# play_game() only talks to the strategy, so the browser code is shared.
# ---------------------------------------------------------------------------

SOLVERS = ("openrouter", "jev")
DEFAULT_SOLVER = "openrouter"


def make_openrouter_strategy(model: str):
    def strategy(remaining: list[str], failed_guesses: list[dict] | None = None) -> list[dict]:
        return solve_with_llm(remaining, model, failed_guesses)

    strategy.model_used = model  # type: ignore[attr-defined]
    return strategy


def make_strategy(solver: str, *, debug: bool = False):
    """Build the strategy named by `solver` ("openrouter" or "jev")."""
    if solver == "openrouter":
        model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        return make_openrouter_strategy(model)
    if solver == "jev":
        # lazy import: typesafe-sdk is only needed for this path
        from jev_solver import JevBeamStrategy, JevBlankStrategy, JevStrategy, JevWordplayStrategy

        variant = os.environ.get("JEV_STRATEGY", "blanks").strip().lower()
        common = dict(
            model=os.environ.get("TYPESAFE_DEFAULT_MODEL"),
            debug_dir=SOLVER_DEBUG_DIR if debug else None,
        )
        if variant == "pairwise":
            return JevStrategy(**common)
        common["repeats"] = int(os.environ.get("JEV_REPEATS", "1"))
        if variant == "beam":
            return JevBeamStrategy(**common)
        if variant == "wordplay":
            return JevWordplayStrategy(**common)
        if variant == "blanks":
            return JevBlankStrategy(**common)
        raise ValueError(f"unknown JEV_STRATEGY {variant!r}; choose 'blanks', 'wordplay', 'beam' or 'pairwise'")
    raise ValueError(f"unknown solver {solver!r}; choose from {SOLVERS}")


def solver_from_env() -> str:
    return os.environ.get("SOLVER", DEFAULT_SOLVER).strip().lower()


def required_key_for(solver: str) -> str:
    return "TYPESAFE_API_KEY" if solver == "jev" else "OPENROUTER_API_KEY"


# ---------------------------------------------------------------------------
# Game interaction
# ---------------------------------------------------------------------------


def _find_tile_js() -> str:
    """JS snippet: (word) -> the tile element (input or legacy element), or null."""
    return "(word) => {" + _TILE_JS + " return findTile(word); }"


async def click_tile(page, word: str) -> None:
    tile = await page.evaluate_handle(_find_tile_js(), word)
    if await tile.evaluate("el => el === null"):
        raise ValueError(f"Tile not found: {word!r}")

    # NYT keeps tiles selected after a wrong guess; clicking a selected tile
    # would deselect it, so leave it alone.
    if await tile.evaluate("el => {" + _TILE_JS + " return isSelected(el); }"):
        return

    # Click the label (for a hidden checkbox) or the button. force=True bypasses
    # Playwright's actionability checks but still dispatches real pointer
    # events; readiness was already confirmed by wait_for_board_ready.
    target = await tile.evaluate_handle("el => {" + _TILE_JS + " return clickable(el); }")
    await target.as_element().click(force=True)
    await page.wait_for_timeout(300)

    if not await tile.evaluate("el => {" + _TILE_JS + " return isSelected(el); }"):
        raise ValueError(f"Tile clicked but did not become selected: {word!r}")


async def click_submit(page) -> None:
    submitted = await page.evaluate("""() => {
        const btn = Array.from(document.querySelectorAll('button'))
            .find(b => /^submit$/i.test(b.textContent.trim()));
        if (btn && !btn.disabled) { btn.click(); return true; }
        return false;
    }""")
    if not submitted:
        raise ValueError("Submit button not found or is disabled")
    await page.wait_for_timeout(2500)


async def deselect_all(page) -> None:
    """Clear the current selection: the Deselect button first, then any tile still selected."""
    await page.evaluate("""() => {
        const btn = Array.from(document.querySelectorAll('button'))
            .find(b => /deselect/i.test(b.textContent));
        if (btn) btn.click();
    }""")
    await page.wait_for_timeout(400)
    # Fallback for tiles that stayed selected (button disabled mid-animation etc.)
    for _ in range(3):
        still = await page.evaluate(
            "() => {" + _TILE_JS + """
                const sel = tileElements().filter(isSelected);
                sel.forEach(el => clickable(el).click());
                return sel.length;
            }"""
        )
        if not still:
            break
        await page.wait_for_timeout(400)


async def read_feedback(page, submitted: list[str]) -> str:
    """Return 'ok', 'one_away', or 'wrong'.

    Success is confirmed by finding all 4 submitted words inside a
    SolvedCategory-* div (the colored banner that appears after a correct guess).
    """
    solved = await page.evaluate(
        """(members) => {
            const norm = s => s.replace(/\\s+/g, ' ').trim();
            const divs = Array.from(document.querySelectorAll('[class*="SolvedCategory-"]'));
            // A solved banner contains all 4 member words in its text
            return divs.some(div =>
                members.every(m => norm(div.textContent).includes(m))
            );
        }""",
        submitted,
    )
    if solved:
        return "ok"

    text = await page.evaluate("() => document.body.innerText")
    if re.search(r"one away", text, re.I):
        return "one_away"
    return "wrong"


async def wait_for_board_ready(page, members: list[str]) -> None:
    """Wait until the first tile of the next group exists and is interactive."""
    await page.wait_for_function(
        "(word) => {" + _TILE_JS + " const el = findTile(word); return !!el && isReady(el); }",
        arg=members[0],
        timeout=15_000,
    )


async def submit_group(page, members: list[str]) -> str:
    """Wait for board, select tiles, submit, read feedback, and deselect on failure."""
    await wait_for_board_ready(page, members)
    for word in members:
        await click_tile(page, word)
    await click_submit(page)
    feedback = await read_feedback(page, members)
    if feedback != "ok":
        await deselect_all(page)
    return feedback


# ---------------------------------------------------------------------------
# Iterative solver
# ---------------------------------------------------------------------------


async def play_game(page, tiles: list[str], strategy) -> tuple[bool, int, list[dict]]:
    """
    Play the game iteratively using `strategy(remaining, failed_guesses) -> groups`.
    Returns (success, mistakes, solved_groups) where solved_groups is a list of
    {"theme": ..., "members": [...]} dicts for each correctly guessed group.

    Retry strategy:
    - "One Away": apply the group's pre-computed alternative swap (no extra
      blind guesses needed). Falls back to a re-solve if alternatives are
      exhausted or invalid.
    - Completely wrong: re-ask the strategy with full failure history as context.
    """
    remaining: list[str] = list(tiles)
    mistakes = 0
    groups_solved = 0
    solved_groups: list[dict] = []
    failed_guesses: list[dict] = []
    tried_sets: set[frozenset] = set()   # Python-enforced dedup — LLM can't be trusted

    groups = strategy(remaining)

    while remaining and mistakes < MAX_MISTAKES:
        if not groups:
            print("  (Re-solving — queue empty ...)")
            groups = strategy(remaining, failed_guesses)

        group = groups.pop(0)
        members = list(group["members"])
        alternatives = list(group.get("alternatives", []))

        # Skip groups that reference already-solved words
        bad = [m for m in members if m not in remaining]
        if bad:
            print(f"  [skip] already solved: {bad}")
            continue

        # Skip exact groupings we have already tried (LLM sometimes repeats them)
        key = frozenset(members)
        if key in tried_sets:
            print(f"  [skip] already tried: {members}")
            if not groups:
                groups = strategy(remaining, failed_guesses)
            continue
        tried_sets.add(key)

        print(f"  Trying [{group['theme']}]")
        print(f"    → {members}")

        feedback = await submit_group(page, members)

        if feedback == "ok":
            print("    ✓ Correct!")
            for w in members:
                remaining.remove(w)
            groups_solved += 1
            solved_groups.append({"theme": group["theme"], "members": members})
            groups = [
                g for g in groups
                if all(m in remaining for m in g["members"])
            ]

        elif feedback == "one_away":
            mistakes += 1
            print(f"    ~ One Away! ({mistakes}/{MAX_MISTAKES} mistakes used)")
            failed_guesses.append({"members": members, "feedback": "one_away"})

            # A strategy that can use the feedback as new information (Jev asks
            # "which one does not belong?") overrides pre-computed alternatives.
            one_away = getattr(strategy, "one_away", None)
            if one_away is not None and mistakes < MAX_MISTAKES:
                fresh = one_away(members, remaining, failed_guesses)
                if fresh:
                    alternatives = fresh

            if alternatives:
                alt = alternatives.pop(0)
                remove_word = alt.get("remove")
                add_word = alt.get("add")

                if (
                    remove_word in members
                    and add_word in remaining
                    and add_word not in members
                ):
                    new_members = [add_word if w == remove_word else w for w in members]
                    print(f"    → Alt: swap '{remove_word}' → '{add_word}'")
                    groups.insert(0, {
                        "theme": group["theme"],
                        "members": new_members,
                        "alternatives": alternatives,
                    })
                else:
                    if mistakes < MAX_MISTAKES:
                        print(f"    → Alt invalid ({alt}), re-solving ...")
                        groups = strategy(remaining, failed_guesses)
                    else:
                        print(f"    → Alt invalid ({alt}) — no attempts left.")
            else:
                if mistakes < MAX_MISTAKES:
                    print("    → No alternatives, re-solving ...")
                    groups = strategy(remaining, failed_guesses)
                else:
                    print("    → No alternatives — no attempts left.")

        else:  # completely wrong
            mistakes += 1
            failed_guesses.append({"members": members, "feedback": "wrong"})
            if mistakes < MAX_MISTAKES:
                print(f"    ✗ Wrong. ({mistakes}/{MAX_MISTAKES} mistakes) Re-solving ...")
                groups = strategy(remaining, failed_guesses)
            else:
                print(f"    ✗ Wrong. ({mistakes}/{MAX_MISTAKES} mistakes) — no attempts left.")

    success = groups_solved == 4
    if success:
        print(f"\nSolved! ({mistakes} mistake(s))")
    else:
        print(f"\nGame over — {mistakes} mistakes, solved {groups_solved}/4 groups.")
    return success, mistakes, solved_groups


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def solve(
    *,
    cloud: bool = False,
    no_cdp: bool = False,
    headed: bool = False,
    cdp_url: str | None = None,
    debug: bool = False,
    solver: str | None = None,
) -> dict:
    """
    Run the solver and return a result dict with keys:
      success, groups, mistakes, elapsed_seconds, solver, model, image_path

    `solver` is "openrouter" or "jev"; defaults to the SOLVER env var.
    """
    import time
    global DEBUG
    DEBUG = debug

    start_time = time.monotonic()
    solver = (solver or solver_from_env())
    strategy = make_strategy(solver, debug=debug)
    model = getattr(strategy, "model_used", None) or solver
    print(f"Solver: {solver}  Model: {model}")

    async with async_playwright() as pw:
        if no_cdp:
            headless = not headed
            print(f"Launching Playwright Chromium ({'headed' if headed else 'headless'}) ...")
            browser = await pw.chromium.launch(headless=headless)
            context = await browser.new_context(viewport={"width": 1280, "height": 900})
            page = await context.new_page()
        elif cloud:
            token = os.environ.get("LIGHTPANDA_TOKEN")
            if not token:
                raise RuntimeError("LIGHTPANDA_TOKEN is not set.")
            url = f"{LIGHTPANDA_CLOUD_URL}&token={token}"
            print("Connecting to Lightpanda Cloud ...")
            browser = await pw.chromium.connect_over_cdp(url)
            context = await browser.new_context()
            page = await context.new_page()
        else:
            url = cdp_url or DEFAULT_CDP_URL
            print(f"Connecting Playwright directly to {url} ...")
            browser = await pw.chromium.connect_over_cdp(url)
            ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        try:
            print(f"Navigating to {NYT_CONNECTIONS_URL} ...")
            try:
                await page.goto(NYT_CONNECTIONS_URL, wait_until="domcontentloaded", timeout=30_000)
            except Exception as e:
                if "TargetClosedError" in type(e).__name__ or "TargetClosedError" in str(e):
                    raise RuntimeError(
                        "The browser target closed during navigation. "
                        "Try --no-cdp to use Playwright's bundled Chromium instead."
                    ) from e
                raise

            # Dismiss any cookie/GDPR banners that block the Play button
            await dismiss_overlays(page)

            print("Waiting for Play button ...")
            play_btn = page.locator('[data-testid="moment-btn-play"]')
            await play_btn.wait_for(state="visible", timeout=30_000)
            await play_btn.click()
            print("  Clicked Play")
            await page.wait_for_timeout(1000)

            # The board can take a few seconds to render behind NYT's splash
            # screen / upsell interstitials. Poll for the 16 tiles instead of
            # trusting the first scrape, re-dismissing overlays in between.
            print("Extracting tiles ...")
            tiles: list[str] = []
            for attempt in range(TILE_WAIT_SECONDS):
                tiles = await extract_tiles(page)
                if len(tiles) == 16:
                    break
                await dismiss_overlays(page)
                await page.wait_for_timeout(1000)

            if debug:
                SOLVER_DEBUG_DIR.mkdir(exist_ok=True)
                debug_path = str(SOLVER_DEBUG_DIR / "loaded.png")
                await page.screenshot(path=debug_path)
                print(f"  Saved: {debug_path}")

            if len(tiles) != 16:
                SOLVER_DEBUG_DIR.mkdir(exist_ok=True)
                err_path = str(SOLVER_DEBUG_DIR / "no_tiles.png")
                await page.screenshot(path=err_path)
                raise RuntimeError(
                    f"Expected 16 tiles after {TILE_WAIT_SECONDS}s, got {len(tiles)}: {tiles}"
                    f" (screenshot: {err_path})"
                )

            print(f"Tiles: {tiles}\n")

            success, mistakes, solved_groups = await play_game(page, tiles, strategy)

            elapsed = time.monotonic() - start_time
            SOLVER_IMAGES_DIR.mkdir(exist_ok=True)
            image_path = str(SOLVER_IMAGES_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.png")
            await page.screenshot(path=image_path, full_page=False)
            print(f"Done!  {image_path}  (total time: {elapsed:.1f}s)")

            return {
                "success": success,
                "groups": solved_groups,
                "mistakes": mistakes,
                "elapsed_seconds": round(elapsed, 1),
                "solver": solver,
                # Jev reports the versioned model that actually answered
                "model": getattr(strategy, "model_used", None) or model,
                "image_path": image_path,
            }

        finally:
            await browser.close()


async def run(args: argparse.Namespace) -> None:
    result = await solve(
        cloud=args.cloud,
        no_cdp=args.no_cdp,
        headed=args.headed,
        cdp_url=args.cdp_url,
        debug=args.debug,
        solver=args.solver,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    global DEBUG

    parser = argparse.ArgumentParser(
        description="Solve NYT Connections using Lightpanda CDP + an LLM (OpenRouter) or Jev (TypeSafe)"
    )
    parser.add_argument(
        "--solver",
        choices=SOLVERS,
        default=solver_from_env(),
        help="Grouping strategy (default: $SOLVER or 'openrouter')",
    )
    parser.add_argument(
        "--cdp-url",
        metavar="WS_URL",
        help=f"WebSocket CDP URL (default: {DEFAULT_CDP_URL})",
    )
    parser.add_argument(
        "--no-cdp",
        action="store_true",
        help="Use Playwright's bundled Chromium instead of a CDP server",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Use Lightpanda Cloud (Chrome); requires LIGHTPANDA_TOKEN env var",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Chromium in headed (visible) mode; only applies with --no-cdp",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log LLM prompts/responses to stderr and save debug files",
    )
    args = parser.parse_args()
    DEBUG = args.debug

    key = required_key_for(args.solver)
    if not os.environ.get(key):
        print(f"Error: {key} is not set (required by the {args.solver} solver).", file=sys.stderr)
        sys.exit(1)  # intentional: CLI entry point, not called from the web server

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
