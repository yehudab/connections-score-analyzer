#!/usr/bin/env python3
"""
connections_solver.py

Automatically solves the daily NYT Connections puzzle and captures a winning screenshot.

Pipeline:
  1. Connect to a CDP server (Lightpanda or Chrome) at the given WebSocket URL
  2. Open the NYT Connections page via Playwright
  3. Scrape the 16 tile words from the DOM
  4. Ask an LLM (via OpenRouter) to group them, WITH pre-computed "one away" alternatives
  5. Iteratively submit groups:
       - Correct: remove tiles from board, move on
       - "One Away": use pre-computed alternative swaps from the LLM
       - Completely wrong: re-ask LLM with failure history
  6. Screenshot the completed board

Usage:
    python connections_solver.py [--cdp-url ws://127.0.0.1:9222] [--output win.png]

    # Use Playwright's bundled Chromium (required for NYT — Lightpanda crashes on React):
    python connections_solver.py --no-cdp --headed

    # Verbose LLM logging + debug screenshots:
    python connections_solver.py --debug

Environment:
    OPENROUTER_API_KEY   required — your OpenRouter API key
    OPENROUTER_MODEL     optional — model to use (default: google/gemini-2.5-flash)

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

import websockets
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
DEFAULT_MODEL = "google/gemini-2.5-pro"

TILE_SELECTORS = [
    '[data-testid="card-label"]',
    '[data-testid="card"] span',
    '.cell-text',
    '[class*="Card"] [class*="label"]',
    '[class*="card"] [class*="label"]',
    '[class*="Cell"] span',
    '[class*="cell"] span',
    '[class*="Tile"] span',
]

# NYT gives 4 mistakes before game over
MAX_MISTAKES = 4

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
        ts = datetime.now().strftime("%H%M%S")
        fname = f"debug_llm_{ts}_{_llm_call_index:02d}_{label.replace(' ', '_')}.txt"
        _llm_call_index += 1
        with open(fname, "w") as f:
            f.write(text)
        print(f"  (saved to {fname})", file=sys.stderr)


# ---------------------------------------------------------------------------
# Overlay / UI helpers
# ---------------------------------------------------------------------------


async def dismiss_overlays(page) -> None:
    candidates = [
        'button[data-testid="GDPR-accept"]',
        '#games-fullscreen-modal button[class*="close"]',
        'button[aria-label="Close"]',
        'button:text("Got it")',
        'button:text("Accept")',
        'button:text("Play")',
        'button:text("Play!")',
    ]
    for selector in candidates:
        try:
            btn = page.locator(selector).first
            if await btn.is_visible(timeout=400):
                await btn.click()
                await page.wait_for_timeout(600)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tile extraction
# ---------------------------------------------------------------------------


async def extract_tiles(page) -> list[str]:
    for selector in TILE_SELECTORS:
        try:
            tiles = await page.eval_on_selector_all(
                selector,
                "els => els.map(el => el.textContent.replace(/\\s+/g, ' ').trim()).filter(t => t.length > 0)",
            )
            if len(tiles) == 16:
                return tiles
        except Exception:
            pass

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
        print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
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

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16000,
    )

    raw = response.choices[0].message.content
    log_llm("RESPONSE", raw)

    groups = _parse_groups(raw)
    for g in groups:
        g.setdefault("alternatives", [])
    return groups


# ---------------------------------------------------------------------------
# Game interaction
# ---------------------------------------------------------------------------


def _find_tile_js() -> str:
    """JS snippet that returns the button element for a given word, or null."""
    return """(word) => {
        const norm = s => s.replace(/\\s+/g, ' ').trim();
        const labelSelectors = [
            '[data-testid="card-label"]',
            '[data-testid="card"] span',
            '.cell-text',
            '[class*="Card"] [class*="label"]',
            '[class*="card"] [class*="label"]',
            '[class*="Cell"] span',
            '[class*="cell"] span',
            '[class*="Tile"] span',
        ];
        for (const sel of labelSelectors) {
            const el = Array.from(document.querySelectorAll(sel))
                .find(el => norm(el.textContent) === word);
            if (el) return el.closest('button') || el;
        }
        // Fallback: button whose normalised textContent matches exactly
        return Array.from(document.querySelectorAll('button'))
            .find(b => norm(b.textContent) === word) || null;
    }"""


def _is_selected_js() -> str:
    """JS snippet: returns true if the element has a Card-module_selected__ class."""
    return """(el) => Array.from(el.classList).some(c => c.startsWith('Card-module_selected__'))"""


async def click_tile(page, word: str) -> None:
    # Locate the button via JS handle so we can inspect it after clicking
    btn = await page.evaluate_handle(_find_tile_js(), word)
    if await btn.evaluate("el => el === null"):
        raise ValueError(f"Tile not found: {word!r}")

    # force=True bypasses Playwright's actionability checks (enabled/aria-disabled)
    # but still dispatches real pointer events that React's event system handles.
    # Readiness was already confirmed by wait_for_board_ready.
    await btn.as_element().click(force=True)
    await page.wait_for_timeout(300)

    # Verify the tile is now marked as selected
    selected = await btn.evaluate(_is_selected_js())
    if not selected:
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
    await page.evaluate("""() => {
        const btn = Array.from(document.querySelectorAll('button'))
            .find(b => /deselect/i.test(b.textContent));
        if (btn) btn.click();
    }""")
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
    """Wait until the first tile of the next group is visible and fully interactive."""
    word = members[0]
    await page.wait_for_function(
        """(word) => {
            const norm = s => s.replace(/\\s+/g, ' ').trim();
            const labelSelectors = [
                '[data-testid="card-label"]',
                '[class*="Card"] [class*="label"]',
                '[class*="Cell"] span',
            ];
            for (const sel of labelSelectors) {
                const el = Array.from(document.querySelectorAll(sel))
                    .find(e => norm(e.textContent) === word);
                if (el) {
                    const btn = el.closest('button') || el;
                    // Check both the DOM disabled property and aria-disabled attribute
                    const ariaDisabled = btn.getAttribute('aria-disabled');
                    return !btn.disabled && ariaDisabled !== 'true';
                }
            }
            return false;
        }""",
        arg=word,
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


async def play_game(page, tiles: list[str], model: str) -> bool:
    """
    Play the game iteratively. Returns True on a full solve.

    Retry strategy:
    - "One Away": apply the LLM's pre-computed alternative swap (no extra
      blind guesses needed). Falls back to a re-solve if alternatives are
      exhausted or invalid.
    - Completely wrong: re-ask LLM with full failure history as context.
    """
    remaining: list[str] = list(tiles)
    mistakes = 0
    groups_solved = 0
    failed_guesses: list[dict] = []
    tried_sets: set[frozenset] = set()   # Python-enforced dedup — LLM can't be trusted

    groups = solve_with_llm(remaining, model)

    while remaining and mistakes < MAX_MISTAKES:
        if not groups:
            print("  (Re-solving — queue empty ...)")
            groups = solve_with_llm(remaining, model, failed_guesses)

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
                groups = solve_with_llm(remaining, model, failed_guesses)
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
            groups = [
                g for g in groups
                if all(m in remaining for m in g["members"])
            ]

        elif feedback == "one_away":
            mistakes += 1
            print(f"    ~ One Away! ({mistakes}/{MAX_MISTAKES} mistakes used)")
            failed_guesses.append({"members": members, "feedback": "one_away"})

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
                        groups = solve_with_llm(remaining, model, failed_guesses)
                    else:
                        print(f"    → Alt invalid ({alt}) — no attempts left.")
            else:
                if mistakes < MAX_MISTAKES:
                    print("    → No alternatives, re-solving ...")
                    groups = solve_with_llm(remaining, model, failed_guesses)
                else:
                    print("    → No alternatives — no attempts left.")

        else:  # completely wrong
            mistakes += 1
            failed_guesses.append({"members": members, "feedback": "wrong"})
            if mistakes < MAX_MISTAKES:
                print(f"    ✗ Wrong. ({mistakes}/{MAX_MISTAKES} mistakes) Re-solving ...")
                groups = solve_with_llm(remaining, model, failed_guesses)
            else:
                print(f"    ✗ Wrong. ({mistakes}/{MAX_MISTAKES} mistakes) — no attempts left.")

    success = groups_solved == 4
    if success:
        print(f"\nSolved! ({mistakes} mistake(s))")
    else:
        print(f"\nGame over — {mistakes} mistakes, solved {groups_solved}/4 groups.")
    return success


# ---------------------------------------------------------------------------
# Lightpanda CDP proxy
#
# Playwright's connect_over_cdp sends Target.setAutoAttach{flatten:true} during
# init which kills Lightpanda's page target.  This proxy:
#   1. Pre-creates a Lightpanda session (createTarget + attachToTarget{flatten:true})
#      via raw WebSocket before Playwright ever connects.
#   2. Intercepts Playwright's three init commands and responds synthetically.
#   3. Forwards everything else (including flat session messages) directly —
#      Lightpanda supports flatten=True so no wrapping is needed.
# ---------------------------------------------------------------------------


class LightpandaProxy:
    """WebSocket proxy that makes Playwright work with Lightpanda.

    Lightpanda supports flatten=True (flat session mode), so session messages
    are exchanged with sessionId at the top level — no wrapping needed.
    The proxy only needs to intercept Playwright's three init commands so it
    doesn't try to discover/attach targets itself (which would kill the page).
    """

    def __init__(self, lightpanda_url: str = "ws://localhost:9222", proxy_port: int = 9221):
        self.lightpanda_url = lightpanda_url
        self.proxy_port = proxy_port
        self.url = f"ws://localhost:{proxy_port}"
        self._target_id: str | None = None
        self._session_id: str | None = None
        self._server = None
        self._lp_ws = None
        self._next_id = 1

    async def _lp_cmd(self, method: str, params: dict | None = None) -> dict:
        """Send a command to Lightpanda, skip unsolicited events, return response."""
        msg_id = self._next_id
        self._next_id += 1
        await self._lp_ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            raw = await self._lp_ws.recv()
            data = json.loads(raw)
            if data.get("id") == msg_id:
                if "error" in data:
                    raise RuntimeError(f"Lightpanda error for {method}: {data['error']}")
                return data
            # Unsolicited event (e.g. Target.targetCreated) — discard during setup

    async def start(self) -> None:
        print(f"  [proxy] Connecting raw WebSocket to {self.lightpanda_url} ...")
        self._lp_ws = await websockets.connect(
            self.lightpanda_url,
            additional_headers={"Host": "localhost"},
        )

        resp = await self._lp_cmd("Target.createTarget", {"url": "about:blank"})
        self._target_id = resp["result"]["targetId"]

        # flatten=True: Lightpanda supports flat session messages (sessionId at top level)
        resp = await self._lp_cmd("Target.attachToTarget", {"targetId": self._target_id, "flatten": True})
        self._session_id = resp["result"]["sessionId"]

        print(f"  [proxy] target={self._target_id}  session={self._session_id}")

        self._server = await websockets.serve(self._handle_client, "localhost", self.proxy_port)
        print(f"  [proxy] Proxy listening on {self.url}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._lp_ws:
            await self._lp_ws.close()

    def _target_info(self, attached: bool) -> dict:
        return {
            "targetId": self._target_id,
            "type": "page",
            "title": "",
            "url": "about:blank",
            "attached": attached,
            "canAccessOpener": False,
            "browserContextId": "BID-1",
        }

    async def _handle_client(self, pw_ws) -> None:
        """Handle one Playwright WebSocket connection (flat session mode)."""

        async def from_pw() -> None:
            async for raw in pw_ws:
                data = json.loads(raw)
                method = data.get("method", "")
                msg_id = data.get("id")
                sid = data.get("sessionId")  # present when called at session level

                def _reply(result: dict) -> str:
                    msg: dict = {"id": msg_id, "result": result}
                    if sid:
                        msg["sessionId"] = sid
                    return json.dumps(msg)

                def _event(evt_method: str, params: dict) -> str:
                    msg: dict = {"method": evt_method, "params": params}
                    if sid:
                        msg["sessionId"] = sid
                    return json.dumps(msg)

                if method == "Target.setDiscoverTargets":
                    # Only send synthetic targetCreated at the browser level
                    await pw_ws.send(_reply({}))
                    if not sid:
                        await pw_ws.send(_event("Target.targetCreated", {
                            "targetInfo": self._target_info(False),
                        }))

                elif method == "Target.setAutoAttach":
                    await pw_ws.send(_reply({}))
                    # Only attach at the browser level; session-level calls get empty OK
                    if not sid:
                        await pw_ws.send(_event("Target.attachedToTarget", {
                            "sessionId": self._session_id,
                            "targetInfo": self._target_info(True),
                            "waitingForDebugger": False,
                        }))

                elif method == "Target.getTargets":
                    await pw_ws.send(_reply({"targetInfos": [self._target_info(True)]}))

                else:
                    # Everything else (including flat session messages) → Lightpanda
                    await self._lp_ws.send(raw)

        async def from_lp() -> None:
            async for raw in self._lp_ws:
                data = json.loads(raw)
                method = data.get("method", "")
                # Filter Lightpanda's own copies of events we already sent synthetically
                if method in ("Target.targetCreated", "Target.attachedToTarget"):
                    continue
                await pw_ws.send(raw)

        tasks = [asyncio.ensure_future(from_pw()), asyncio.ensure_future(from_lp())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    import time
    start_time = time.monotonic()
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    print(f"Model: {model}")

    async with async_playwright() as pw:
        if args.no_cdp:
            headless = not args.headed
            print(f"Launching Playwright Chromium ({'headed' if args.headed else 'headless'}) ...")
            browser = await pw.chromium.launch(headless=headless)
            context = await browser.new_context(viewport={"width": 1280, "height": 900})
            page = await context.new_page()
        else:
            cdp_url = args.cdp_url or DEFAULT_CDP_URL
            print(f"Starting Lightpanda proxy for {cdp_url} ...")
            proxy = LightpandaProxy(lightpanda_url=cdp_url, proxy_port=9221)
            await proxy.start()

            print(f"Connecting Playwright to proxy at {proxy.url} ...")
            browser = await pw.chromium.connect_over_cdp(proxy.url)

            # The proxy pre-created one page for us — grab it from Playwright's view.
            page = None
            for ctx in browser.contexts:
                if ctx.pages:
                    page = ctx.pages[0]
                    break
            if page is None:
                try:
                    ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
                    page = await ctx.new_page()
                except Exception as e:
                    await proxy.stop()
                    raise RuntimeError(
                        f"Could not get a page from the proxy session: {e}"
                    ) from e

        try:
            print(f"Navigating to {NYT_CONNECTIONS_URL} ...")
            try:
                await page.goto(NYT_CONNECTIONS_URL, wait_until="domcontentloaded", timeout=30_000)
            except Exception as e:
                if "TargetClosedError" in type(e).__name__ or "TargetClosedError" in str(e):
                    raise RuntimeError(
                        "The browser target closed during navigation.\n"
                        "If you are using Lightpanda: it does not yet support React SPAs like NYT Connections "
                        "(exits 139 / SIGSEGV while executing the React bundle).\n"
                        "Run with --no-cdp to use Playwright's bundled Chromium instead."
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

            if args.debug:
                await page.screenshot(path="debug_loaded.png")
                print("  Saved: debug_loaded.png")

            print("Extracting tiles ...")
            tiles = await extract_tiles(page)

            if len(tiles) != 16:
                await page.screenshot(path="debug_no_tiles.png")
                print(
                    f"Expected 16 tiles, got {len(tiles)}: {tiles}",
                    file=sys.stderr,
                )
                if not tiles:
                    sys.exit(1)

            print(f"Tiles: {tiles}\n")

            await play_game(page, tiles, model)

            elapsed = time.monotonic() - start_time
            print(f"\nSaving screenshot → {args.output}")
            await page.screenshot(path=args.output, full_page=False)
            print(f"Done!  {args.output}  (total time: {elapsed:.1f}s)")

        finally:
            await browser.close()
            if not args.no_cdp:
                await proxy.stop()


def main() -> None:
    global DEBUG

    parser = argparse.ArgumentParser(
        description="Solve NYT Connections using Lightpanda CDP + OpenRouter LLM"
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
        "--headed",
        action="store_true",
        help="Run Chromium in headed (visible) mode; only applies with --no-cdp",
    )
    parser.add_argument(
        "--output",
        default="connections_win.png",
        metavar="FILE",
        help="Output screenshot path (default: connections_win.png)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log LLM prompts/responses to stderr and save debug files",
    )
    args = parser.parse_args()
    DEBUG = args.debug

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
