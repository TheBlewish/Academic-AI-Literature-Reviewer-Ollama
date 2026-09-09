# llm_manager.py  (formerly parallel_llm.py)
# LLM Manager — single-model orchestration with DORMANT parallel scaffolding.
#
# In this version the pipeline drives ONE model via run_primary(). The parallel
# search-agent machinery (run_agent / run_parallel / search_agents) is retained
# but unused, so the multi-agent architecture can be wired in later without a
# rewrite. The public class is now LLMManager; ParallelAgentManager remains as a
# backward-compatible alias at the bottom of the file.
#
# CHANGES IN THIS VERSION:
#   - REWORKED DYNAMIC-CONTEXT GRANULARITY into a generated TAPERED GRID that is
#     never coarser than the original ladder (so it can never allocate MORE than
#     before — fixing the earlier regression where dropping 24576/32768 made a
#     few bands round up) and is FINER almost everywhere (more sizes that better
#     match the real need => less wasted context => faster). The grid uses 1K
#     steps up to 16K, then 2K to 32K, 4K to 64K, 8K to 128K, 16K beyond — the
#     step doubles as windows grow so the PERCENT of context wasted by rounding
#     stays roughly constant (1K wasted on a 5K window is ~20%; on a 60K window
#     it's <2%, so fine steps there aren't worth the extra distinct sizes). The
#     grid is generated up to hard_max (self.n_ctx, read live from config) and
#     ALWAYS ends exactly at hard_max, so "the current max" is the last option
#     at any model size, now or after a future n_ctx bump. Tune via
#     DYNAMIC_CTX_TIERS. DYNAMIC_CTX_FLOOR stays at 1024.
#   - DYNAMIC CONTEXT SIZING (ceiling semantics). The task-profile num_ctx is
#     treated as an UPPER CAP / preference, NOT a floor. For every call,
#     generate() measures the actual prompt+system length, estimates tokens
#     conservatively, adds the FULL output budget plus headroom, rounds up to
#     a grid bucket, then:
#         * caps the result at the profile/explicit num_ctx (ceiling), AND
#         * caps at the model's hard max (self.n_ctx), AND
#         * forces the result to be >= the raw computed need so truncation is
#           impossible (if need exceeds the ceiling, the ceiling is overridden
#           up to the model hard max rather than truncating).
#     Net effect: a small call (e.g. an early planning or distillation step)
#     drops to a small bucket (e.g. 5K-8K) instead of sitting at the old fixed
#     profile size (e.g. 16K-64K). Large calls still get large windows.
#       * Token estimate uses 3.0 chars/token (deliberately OVER-estimates).
#       * 1.25x safety multiplier on the input estimate.
#       * Full actual_max_tokens (thinking already factored) reserved as output.
#       * +2048 fixed headroom, then round UP to the next bucket.
#       * Clamp to [DYNAMIC_CTX_FLOOR, ceiling] but always >= need (need may
#         push above ceiling, capped only by model hard max).
#     Set DYNAMIC_CTX_ENABLED = False to fully restore previous fixed behaviour.
#   - _log_call_start reports estimated input tokens and a "CAPPED" note if a
#     prompt is large enough that need exceeds the profile ceiling.
#
# PRIOR FUNCTIONALITY PRESERVED:
#   - Module-level interrupt mechanism: request_interrupt(), clear_interrupt(),
#     is_interrupted(). Closes in-flight streaming responses on SIGINT.
#   - Active responses tracked thread-safely; streaming loop checks interrupt
#     flag each chunk.
#   - Dual-model LLMManager (primary + N search agents), run_primary,
#     run_single, run_agent, run_parallel, JSON extraction, thinking-tag strip.

import json
import time
import re
import logging
import threading
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from colorama import Fore, Style, init
init()

logger = logging.getLogger(__name__)

try:
    from academic_config import (
        get_primary_llm_config, get_agent_llm_config,
        get_thinking_config, get_parallel_config, get_task_profile,
    )
except ImportError:
    def get_primary_llm_config():
        return {"base_url": "http://localhost:11434", "model_name": "qwen3:32b-q8_0",
                "temperature": 0.1, "n_ctx": 65536, "max_tokens": 8192}
    def get_agent_llm_config():
        return {"base_url": "http://localhost:11434", "model_name": "qwen3:32b-q8_0",
                "temperature": 0.2, "n_ctx": 32768, "max_tokens": 4096}
    def get_thinking_config():
        return {"show_model_thinking": False, "thinking_token_multiplier": 2}
    def get_parallel_config():
        return {"num_search_agents": 2, "agent_timeout": 600, "agent_start_delay": 2.0}
    def get_task_profile(name):
        return None

KNOWN_THINKING_MODEL_PATTERNS = [
    'qwen3', 'qwq', 'deepseek-r1', 'deepseek-reasoner',
    'glm-4', 'glm-z1', 'marco', 'openthinker',
    'sky-t1', 'reasoner', 'thinking', 'cot',
]

# Models observed to reject the Ollama `think` field with an HTTP 400. Populated
# at runtime on the first rejection and shared process-wide, so one bad response
# teaches every agent and the field is never sent to that model again. This
# exists because `think` is now sent whenever a caller or task profile states a
# preference, not only for names matching the pattern list above — the pattern
# list cannot know about every thinking-capable model (gemma4:31b emitted a
# thinking block on calls logged as think=off, exhausting their budget).
_THINK_UNSUPPORTED_MODELS = set()


# =============================================================================
# DYNAMIC CONTEXT SIZING  (ceiling semantics)
# =============================================================================
# Goal: allocate the SMALLEST context window that comfortably fits this call's
# real input + output, never larger. The task-profile num_ctx is an UPPER CAP
# (a ceiling preference), not a fixed size and not a floor.
#
# Sizing recipe for each call:
#   need = (input_tokens_estimate * INPUT_SAFETY) + output_budget + HEADROOM
#   bucketed = round need UP to the next allocation bucket
#   chosen = min(bucketed, ceiling)            # ceiling = profile/explicit ctx
#   chosen = max(chosen, DYNAMIC_CTX_FLOOR)    # never absurdly tiny
#   chosen = max(chosen, min(need, hard_max))  # NEVER below real need (no truncation)
#   chosen = min(chosen, hard_max)             # never above model's window
#
# Because the estimate over-counts tokens, the input is multiplied by a safety
# factor, the full output budget is reserved, headroom is added, and we round
# UP — the allocated window is always comfortably larger than what is actually
# consumed. If a genuinely large prompt needs more than the profile ceiling,
# `chosen` is allowed to rise above the ceiling (up to the model hard max)
# rather than truncate.

DYNAMIC_CTX_ENABLED = True

# Estimated characters per token. Real English ~4, code/JSON ~3. We use a SMALL
# divisor so we OVER-estimate token count → bias toward larger (safer) contexts.
DYNAMIC_CTX_CHARS_PER_TOKEN = 3.0

# Multiplier applied to the estimated input tokens as a safety margin.
DYNAMIC_CTX_INPUT_SAFETY = 1.25

# Fixed token headroom added on top of input + output (covers prompt template
# tokens, role markers, BOS/EOS, and estimation error).
DYNAMIC_CTX_HEADROOM = 2048

# The smallest context window we will ever allocate. Lowered to 1K so the new
# fine-grained small buckets (1K/2K/3K) below are reachable instead of being
# clamped up to a larger floor. The no-truncation guarantee in
# _compute_dynamic_ctx still holds regardless of how small this is.
DYNAMIC_CTX_FLOOR = 1024

# Allocation buckets — we round each call's computed need UP to the nearest of
# these. Finer granularity = less wasted context = faster prompt processing and
# generation (every token of context the model doesn't have to allocate/scan is
# saved compute). A call needing ~5K gets a 5K window instead of 6K, ~7K gets 7K
# instead of 8K, ~22K gets 22K instead of 24K, and so on.
#
# WHY A GENERATED TAPERED GRID instead of a flat hand-typed list:
#   1. NO GAPS / NEVER WORSE THAN THE OLD LADDER. The previous hand-list skipped
#      values the old code had (24576, 32768), so needs landing there rounded
#      UP and regressed. A regular grid that includes every old value can never
#      allocate more than the old ladder did — it is finer-or-equal everywhere.
#   2. PROPORTIONAL GRANULARITY. Wasting 1K on a 5K window is ~20%; wasting 1K
#      on a 60K window is <2%. So the step size DOUBLES as windows grow: 1K
#      steps where small calls cluster (most of the pipeline), widening to 2K,
#      4K, 8K... at the top where the proportional waste of a coarse step is
#      tiny. This keeps "% context wasted" roughly constant across the range
#      while keeping the number of distinct sizes sane (changing num_ctx between
#      calls is not free, so we don't want a bucket every 256 tokens).
#   3. AUTO-TRACKS THE CURRENT MAX. The grid is generated up to hard_max
#      (self.n_ctx, read live from academic_config) and ALWAYS ends exactly at
#      hard_max — so "whatever the current max is" is literally the last option,
#      with no big gap before it, at ANY model size (65K today, 96K/128K later).
#
# To retune granularity, edit DYNAMIC_CTX_TIERS below: each entry is
# (tier_ceiling, step). Smaller steps = finer = less waste but more distinct
# context sizes; larger steps = fewer reallocations. Tiers are applied in order;
# the grid for a given tier starts one step above the previous tier's ceiling.
DYNAMIC_CTX_TIERS = [
    (16384,  1024),   # 1K..16K  in 1K steps   (fine — most calls live here)
    (32768,  2048),   # 18K..32K in 2K steps
    (65536,  4096),   # 36K..64K in 4K steps
    (131072, 8192),   # 72K..128K in 8K steps  (only used if n_ctx is this big)
    (262144, 16384),  # 144K..256K in 16K steps (future headroom)
]

# Cache of the generated bucket ladder per hard_max so we build it once per
# distinct model window size, not on every call.
_CTX_BUCKET_CACHE: Dict[int, List[int]] = {}


def _build_ctx_buckets(hard_max: int) -> List[int]:
    """Generate the ascending bucket ladder for a given model window (hard_max).

    Produces a tapered grid (1K steps low, doubling step as size grows) from
    DYNAMIC_CTX_FLOOR up to and INCLUDING hard_max. Guarantees:
      * strictly ascending, no duplicates,
      * every value <= hard_max,
      * the final value is exactly hard_max (the "current max" last option),
      * every classic ladder value (4096, 6144, 8192, 12288, 16384, 24576,
        32768, 40960, 49152, 57344, 65536) is present when within range, so
        this ladder is never coarser than the original — no regressions.
    """
    out: List[int] = []
    prev_ceiling = 0
    for ceiling, step in DYNAMIC_CTX_TIERS:
        start = prev_ceiling + step
        v = start
        cap = min(ceiling, hard_max)
        while v <= cap:
            if v >= DYNAMIC_CTX_FLOOR:
                out.append(v)
            v += step
        prev_ceiling = ceiling
        if hard_max <= ceiling:
            break

    # Make sure the floor is present as the smallest option.
    if not out or out[0] > DYNAMIC_CTX_FLOOR:
        out.insert(0, DYNAMIC_CTX_FLOOR)

    # Make sure hard_max itself is the final option (covers the case where
    # hard_max is not an exact grid point, e.g. an unusual n_ctx).
    out = [b for b in out if b < hard_max]
    out.append(hard_max)

    # De-dup while preserving order (hard_max may already have been a grid point).
    seen = set()
    deduped = []
    for b in out:
        if b not in seen:
            seen.add(b)
            deduped.append(b)
    return deduped


def _ctx_buckets_for(hard_max: int) -> List[int]:
    """Return (and cache) the bucket ladder for this model window size."""
    buckets = _CTX_BUCKET_CACHE.get(hard_max)
    if buckets is None:
        buckets = _build_ctx_buckets(hard_max)
        _CTX_BUCKET_CACHE[hard_max] = buckets
    return buckets


def _estimate_input_tokens(prompt: str, system_prompt: Optional[str]) -> int:
    """Conservatively estimate the number of tokens in the input text."""
    chars = len(prompt or "") + len(system_prompt or "")
    if chars <= 0:
        return 0
    return int(chars / DYNAMIC_CTX_CHARS_PER_TOKEN) + 1


def _round_up_to_bucket(value: int, hard_max: int) -> int:
    """Round value up to the next allocation bucket (capped at hard_max)."""
    for b in _ctx_buckets_for(hard_max):
        if b >= value:
            return min(b, hard_max)
    return hard_max


def _compute_dynamic_ctx(prompt: str, system_prompt: Optional[str],
                         output_tokens: int, hard_max: int,
                         ceiling: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute the context window to use for this call.

    Args:
        ceiling: the profile/explicit num_ctx — treated as an UPPER CAP, not a
                 floor. May be None (then only the model hard_max caps us).

    Returns a dict with the chosen size and diagnostic numbers.

    Guarantees:
      - chosen is sized from the real input + output need, rounded UP to a bucket.
      - chosen <= ceiling, UNLESS the real need exceeds the ceiling — in that
        case chosen rises above the ceiling (up to hard_max) so the prompt is
        never truncated.
      - chosen >= DYNAMIC_CTX_FLOOR.
      - chosen >= min(need, hard_max)  → truncation is impossible within the
        model's window.
      - chosen <= hard_max  → never exceeds what the model supports.
    """
    input_tokens = _estimate_input_tokens(prompt, system_prompt)
    needed = int(input_tokens * DYNAMIC_CTX_INPUT_SAFETY) + int(output_tokens) + DYNAMIC_CTX_HEADROOM

    bucketed = _round_up_to_bucket(needed, hard_max)

    # Effective ceiling = the profile/explicit cap, bounded by the model hard max.
    if ceiling and ceiling > 0:
        effective_ceiling = min(ceiling, hard_max)
    else:
        effective_ceiling = hard_max

    # Start from the bucketed need, then apply the ceiling as an UPPER CAP.
    chosen = min(bucketed, effective_ceiling)

    # Never go below the sensible floor.
    chosen = max(chosen, DYNAMIC_CTX_FLOOR)

    # NEVER below the real need (capped at hard_max). This is the no-truncation
    # guarantee and it deliberately OVERRIDES the ceiling when content demands.
    chosen = max(chosen, min(needed, hard_max))

    # Final clamp to the model's window.
    chosen = min(chosen, hard_max)

    return {
        "chosen": chosen,
        "input_tokens": input_tokens,
        "needed": needed,
        "ceiling": effective_ceiling,
        "hard_max": hard_max,
        "exceeded_ceiling": needed > effective_ceiling,
    }


# =============================================================================
# MODULE-LEVEL INTERRUPT MECHANISM
# =============================================================================
# This allows a SIGINT handler (in academic_researcher.py) to cleanly abort
# any in-flight LLM streaming call. When request_interrupt() is called:
#   1. The interrupt event is set
#   2. All active responses are forcibly closed
#   3. The iter_lines() loop in generate() raises an exception on the next
#      read, which we catch and return as an "Interrupted" AgentResult
#   4. The pipeline can then route to a terminal state

_interrupt_event = threading.Event()
_active_responses_lock = threading.Lock()
_active_responses: List[Any] = []


def request_interrupt():
    """
    Signal all active LLM requests to abort. Called from SIGINT handler.
    Closes all in-flight streaming responses, which unblocks iter_lines().
    """
    _interrupt_event.set()
    with _active_responses_lock:
        for resp in list(_active_responses):
            try:
                resp.close()
            except Exception:
                pass


def clear_interrupt():
    """Clear the interrupt flag. Call before starting a new run."""
    _interrupt_event.clear()
    with _active_responses_lock:
        _active_responses.clear()


def is_interrupted() -> bool:
    """True if request_interrupt() has been called since last clear."""
    return _interrupt_event.is_set()


def _register_response(resp):
    with _active_responses_lock:
        _active_responses.append(resp)


def _unregister_response(resp):
    with _active_responses_lock:
        try:
            _active_responses.remove(resp)
        except ValueError:
            pass


def _log_call_start(agent_id: str, task: Optional[str], num_ctx: int,
                    max_tokens: int, think: bool,
                    input_tokens: Optional[int] = None,
                    exceeded_ceiling: bool = False):
    """Show what's about to run before the LLM call blocks."""
    task_label = task if task else "default"
    think_label = "ON" if think else "off"
    extra = ""
    if input_tokens is not None:
        cap_note = " (need>ceiling — raised)" if exceeded_ceiling else ""
        extra = f" | in~{input_tokens}tok{cap_note}"
    print(f"  {Fore.BLUE}\u25B6 LLM call: task={task_label} | "
          f"ctx={num_ctx//1024}K | max_tokens={max_tokens} | "
          f"think={think_label} | agent={agent_id}{extra}{Style.RESET_ALL}",
          flush=True)


def _locate_json_decode_error(text: str):
    """Best-effort: reproduce the coarse candidate the JSON parser would try and
    return (decode_error, candidate). Strips a ```json fence and slices from the
    first '{' to the last '}', then attempts json.loads so the resulting
    JSONDecodeError carries the exact failure position (.pos/.lineno/.colno).
    Returns (None, candidate) if the candidate actually parses (the failure was
    elsewhere) or (None, text) if no object-like region is found."""
    if not text:
        return None, ""
    cand = text
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cand)
    if m and '{' in m.group(1):
        cand = m.group(1)
    start = cand.find('{')
    end = cand.rfind('}')
    if start != -1 and end != -1 and end > start:
        cand = cand[start:end + 1]
    else:
        # No object-like region — a positioned error would point at arbitrary
        # prose, so let the caller show the full raw answer instead.
        return None, cand
    try:
        json.loads(cand)
        return None, cand
    except json.JSONDecodeError as e:
        return e, cand
    except Exception:
        return None, cand


def _save_raw_failure(text: str, label: str) -> Optional[str]:
    """Persist the COMPLETE raw answer to Logs/json_parse_failures/ so the full
    text is available for adapting the parser even though the terminal view is
    windowed. Fully defensive — returns the path on success, else None."""
    try:
        import os
        from datetime import datetime
        d = os.path.join("Logs", "json_parse_failures")
        os.makedirs(d, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (label or "json")).strip("_")[:60]
        path = os.path.join(d, f"{safe}_{datetime.now():%Y%m%d_%H%M%S_%f}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path
    except Exception:
        return None


def _print_json_failure_window(text: str, context_label: str, radius: int = 400):
    """Print the JSON failure region — the text immediately around the exact
    decode-error position, with a clear marker on the break — plus the full raw
    answer saved to disk. This is what lets the parser be adapted to new edge
    cases: you see precisely what broke and where, not just a truncated head."""
    err, cand = _locate_json_decode_error(text)
    saved = _save_raw_failure(text, context_label)
    if err is None:
        # Couldn't reproduce a positioned error (e.g. empty/think-only answer or
        # the coarse candidate happened to parse). Show the full raw instead.
        print(f"  {Fore.MAGENTA}--- raw answer (full, {len(text)} chars) ---{Style.RESET_ALL}")
        for ln in text.strip().splitlines():
            print(f"  {Fore.MAGENTA}\u2502 {ln}{Style.RESET_ALL}")
        if saved:
            print(f"  {Fore.WHITE}full raw saved: {saved}{Style.RESET_ALL}")
        return
    pos = err.pos
    lo = max(0, pos - radius)
    hi = min(len(cand), pos + radius)
    before = cand[lo:pos]
    after = cand[pos:hi]
    print(f"  {Fore.MAGENTA}--- JSON parse failure ---{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}reason       : {err.msg}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}at           : line {err.lineno}, col {err.colno} "
          f"(char {pos} of {len(cand)}){Style.RESET_ALL}")
    head_ellipsis = "…" if lo > 0 else ""
    tail_ellipsis = "…" if hi < len(cand) else ""
    print(f"  {Fore.WHITE}--- text around the failure (>>>HERE>>> marks the break) ---{Style.RESET_ALL}")
    window = f"{head_ellipsis}{before}>>>HERE>>>{after}{tail_ellipsis}"
    for ln in window.splitlines():
        print(f"  {Fore.MAGENTA}\u2502 {ln}{Style.RESET_ALL}")
    if saved:
        print(f"  {Fore.WHITE}full raw answer saved: {saved}{Style.RESET_ALL}")


def _print_llm_diagnostics(result: "AgentResult", context_label: str = "",
                           raw_response: Optional[str] = None,
                           json_failure: bool = False):
    """Print a clearly-delimited diagnostic block — ONLY when a call fails.

    Shows exactly what the model did so failures (empty/truncated answers,
    unparseable JSON, HTTP/connection errors) are debuggable from the terminal
    instead of collapsing to a bare "failed". Never called on success or on
    user interrupts.

    When json_failure is set, the truncated "answer (head)" is replaced by a
    failure window centred on the exact decode-error position (with the full raw
    answer saved to disk), so the JSON parser can be adapted to the edge case.
    """
    think_text = result.thinking or ""
    ans_text = raw_response if raw_response is not None else (result.response or "")

    def _snip(s: str, n: int = 600) -> str:
        s = s.strip()
        if len(s) <= n:
            return s
        return s[:n] + f" […+{len(s) - n} more chars]"

    print(f"  {Fore.RED}{'─'*60}{Style.RESET_ALL}")
    print(f"  {Fore.RED}{Style.BRIGHT}\u26A0 LLM CALL FAILED — DIAGNOSTICS{Style.RESET_ALL}"
          f"  {Fore.WHITE}({context_label}){Style.RESET_ALL}")
    print(f"  {Fore.WHITE}error        : {Fore.YELLOW}{result.error}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}agent        : {result.agent_id}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}elapsed      : {result.elapsed_time:.1f}s{Style.RESET_ALL}")
    if result.done_reason is not None:
        trunc_note = f" {Fore.RED}(TRUNCATED){Style.RESET_ALL}" if result.truncated else ""
        print(f"  {Fore.WHITE}done_reason  : {result.done_reason}{trunc_note}{Style.RESET_ALL}")
    if result.prompt_eval_count is not None or result.eval_count is not None:
        print(f"  {Fore.WHITE}tokens       : prompt={result.prompt_eval_count} "
              f"output={result.eval_count}{Style.RESET_ALL}")
    if result.was_retried:
        print(f"  {Fore.WHITE}note         : this was already an automatic retry "
              f"with a larger budget{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}thinking     : {len(think_text)} chars | "
          f"answer: {len(ans_text)} chars{Style.RESET_ALL}")
    if think_text:
        print(f"  {Fore.CYAN}--- thinking (tail) ---{Style.RESET_ALL}")
        # The tail is most informative for a truncated think block.
        tail = think_text.strip()
        tail = tail[-600:] if len(tail) > 600 else tail
        for ln in tail.splitlines():
            print(f"  {Fore.CYAN}\u2502 {ln}{Style.RESET_ALL}")
    if json_failure:
        # Full failure region (positioned) + full raw saved to disk.
        src = raw_response if raw_response is not None else (result.response or "")
        if src.strip():
            _print_json_failure_window(src, context_label)
    elif ans_text.strip():
        print(f"  {Fore.MAGENTA}--- answer (head) ---{Style.RESET_ALL}")
        for ln in _snip(ans_text).splitlines():
            print(f"  {Fore.MAGENTA}\u2502 {ln}{Style.RESET_ALL}")
    print(f"  {Fore.RED}{'─'*60}{Style.RESET_ALL}", flush=True)


@dataclass
class AgentResult:
    agent_id: str
    task_name: str
    response: Optional[str] = None
    json_response: Optional[Dict] = None
    thinking: Optional[str] = None
    error: Optional[str] = None
    elapsed_time: float = 0.0
    success: bool = False
    # --- NEW diagnostic fields (added for the empty/truncated-response fix) ---
    # done_reason: Ollama's reason the generation stopped ("stop" = natural end,
    #   "length" = hit the num_predict limit i.e. TRUNCATED).
    # truncated: convenience flag == (done_reason == "length").
    # eval_count / prompt_eval_count: tokens the model emitted / read (from the
    #   final stream chunk), shown in the on-error diagnostics.
    # was_retried: True if this result came from (or after) an automatic
    #   budget-raising retry, so we never loop more than once.
    done_reason: Optional[str] = None
    truncated: bool = False
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    was_retried: bool = False


def _resolve_task_overrides(task, num_ctx, max_tokens, think):
    """If a task name is given, look up its profile and use as baseline."""
    if task is not None:
        profile = get_task_profile(task)
        if profile:
            if num_ctx is None:
                num_ctx = profile.get("num_ctx")
            if max_tokens is None:
                max_tokens = profile.get("max_tokens")
            if think is None:
                think = profile.get("think")
        else:
            logger.warning(f"Unknown task profile: {task} — falling back to defaults")
    return num_ctx, max_tokens, think


class LLMAgent:
    def __init__(self, agent_id: str, llm_config: Dict = None,
                 thinking_config: Dict = None):
        self.agent_id = agent_id
        if llm_config is None:
            llm_config = get_primary_llm_config()
        if thinking_config is None:
            thinking_config = get_thinking_config()

        self.base_url = llm_config.get("base_url", "http://localhost:11434")
        self.model_name = llm_config.get("model_name", "qwen3:32b-q8_0")
        self.temperature = llm_config.get("temperature", 0.1)
        self.n_ctx = llm_config.get("n_ctx", 65536)
        self.max_tokens = llm_config.get("max_tokens", 8192)

        model_lower = self.model_name.lower()
        self.is_thinking_model = any(
            pattern in model_lower for pattern in KNOWN_THINKING_MODEL_PATTERNS
        )
        self.thinking_multiplier = thinking_config.get("thinking_token_multiplier", 2)

    def generate(self, prompt: str, system_prompt: str = None,
                 max_tokens: int = None, temperature: float = None,
                 num_ctx: int = None, think: Optional[bool] = None,
                 task: Optional[str] = None) -> AgentResult:
        start_time = time.time()

        # An explicit num_ctx (from a caller or, after resolution, a task
        # profile) is captured here and used as a CEILING / upper cap — NOT a
        # floor. Dynamic sizing decides the actual window from real content.
        explicit_ctx = num_ctx

        num_ctx, max_tokens, think = _resolve_task_overrides(
            task, num_ctx, max_tokens, think)

        # If the caller didn't pass an explicit ctx, the profile value (now in
        # num_ctx) becomes the ceiling preference.
        ceiling_ctx = explicit_ctx if explicit_ctx is not None else num_ctx

        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        if think is None:
            effective_think = self.is_thinking_model
        else:
            effective_think = bool(think)

        if effective_think:
            actual_max_tokens = max_tokens * self.thinking_multiplier
        else:
            actual_max_tokens = max_tokens

        hard_max = self.n_ctx

        # Estimate the input size once up front so the retry logic below can
        # size a larger output budget that still fits the model's window.
        input_tokens_est_top = _estimate_input_tokens(prompt, system_prompt)

        # ------------------------------------------------------------------ #
        # Inner attempt: performs ONE request+stream with the given output    #
        # budget. Captures done_reason and token counts, strips thinking, and #
        # marks an EMPTY answer as a failure (previously empty answers were   #
        # silently reported as success=True — the root cause of "Curation     #
        # failed" and "No review generated"). All original control flow       #
        # (interrupt handling, response registration, every exception branch) #
        # is preserved exactly; only diagnostics + empty-detection are added. #
        # ------------------------------------------------------------------ #
        def _attempt(out_budget: int, retried: bool) -> AgentResult:
            # ---- Dynamic context sizing (uses THIS attempt's out_budget) ----
            # hard_max is the model's configured window for this agent — never
            # exceeded. The chosen window is sized to the actual input + output,
            # capped at the profile ceiling, but always >= real need.
            a_input_tokens_est = None
            a_exceeded_ceiling = False
            if DYNAMIC_CTX_ENABLED:
                calc = _compute_dynamic_ctx(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    output_tokens=out_budget,
                    hard_max=hard_max,
                    ceiling=ceiling_ctx,
                )
                a_num_ctx = calc["chosen"]
                a_input_tokens_est = calc["input_tokens"]
                a_exceeded_ceiling = calc["exceeded_ceiling"]
            else:
                # Legacy fixed behaviour
                a_num_ctx = self.n_ctx if num_ctx is None else num_ctx
                a_num_ctx = min(a_num_ctx, hard_max)

            # Show what's about to run so user can see what's executing.
            retry_note = f" {Fore.MAGENTA}(retry: bigger budget){Style.RESET_ALL}" if retried else ""
            _log_call_start(self.agent_id, task, a_num_ctx, out_budget,
                            effective_think, input_tokens=a_input_tokens_est,
                            exceeded_ceiling=a_exceeded_ceiling)
            if retried:
                print(retry_note, flush=True)

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": out_budget,
                    "num_ctx": a_num_ctx,
                },
            }
            if system_prompt:
                payload["system"] = system_prompt
            # Send `think` when the model is a KNOWN thinking model OR when the
            # caller (or task profile) explicitly asked for a setting. Before
            # this, `think` was sent ONLY for names matching
            # KNOWN_THINKING_MODEL_PATTERNS, so a task profile requesting
            # think=off on any other model silently sent nothing and the model
            # used its own default — which is how a call logged as "think=off"
            # spent its whole budget inside a thinking block. If a given model
            # rejects the field, that is remembered and it is not sent again.
            send_think = (self.is_thinking_model or think is not None)
            if send_think and self.model_name not in _THINK_UNSUPPORTED_MODELS:
                payload["think"] = effective_think

            # Check interrupt before even starting
            if is_interrupted():
                return AgentResult(
                    agent_id=self.agent_id, task_name="generate",
                    error="Interrupted before request started",
                    elapsed_time=time.time() - start_time,
                )

            response = None
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=(30, 600),
                )

                # Register this response so the signal handler can close it
                _register_response(response)

                if response.status_code != 200:
                    body = response.text[:400]
                    # Older / non-thinking models reject the `think` field with a
                    # 400. Remember that for this model, drop the field, and
                    # re-issue the identical request once so behaviour is exactly
                    # what it was before `think` started being sent explicitly.
                    if (response.status_code == 400
                            and "think" in body.lower()
                            and "think" in payload):
                        _THINK_UNSUPPORTED_MODELS.add(self.model_name)
                        payload.pop("think", None)
                        logger.debug(
                            f"{self.model_name} rejected the 'think' field — "
                            f"retrying without it and not sending it again.")
                        _unregister_response(response)
                        try:
                            response.close()
                        except Exception:
                            pass
                        response = requests.post(
                            f"{self.base_url}/api/generate",
                            json=payload,
                            stream=True,
                            timeout=(30, 600),
                        )
                        _register_response(response)
                    if response.status_code != 200:
                        return AgentResult(
                            agent_id=self.agent_id, task_name="generate",
                            error=f"HTTP {response.status_code}: {response.text[:200]}",
                            elapsed_time=time.time() - start_time,
                            was_retried=retried,
                        )

                full_response = ""
                full_thinking = ""
                was_interrupted = False
                done_reason = None
                eval_count = None
                prompt_eval_count = None

                try:
                    for line in response.iter_lines():
                        # Check interrupt on every chunk
                        if is_interrupted():
                            was_interrupted = True
                            break
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            if chunk.get("thinking"):
                                full_thinking += chunk["thinking"]
                            if chunk.get("response"):
                                full_response += chunk["response"]
                            # Capture Ollama's stop metadata (present on the
                            # final chunk). done_reason == "length" means the
                            # generation was TRUNCATED at num_predict.
                            if chunk.get("done_reason") is not None:
                                done_reason = chunk.get("done_reason")
                            if chunk.get("eval_count") is not None:
                                eval_count = chunk.get("eval_count")
                            if chunk.get("prompt_eval_count") is not None:
                                prompt_eval_count = chunk.get("prompt_eval_count")
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                except (requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.ConnectionError,
                        ConnectionError,
                        OSError) as e:
                    # If we were interrupted, this is expected (signal handler
                    # closed the connection). Otherwise re-raise.
                    if is_interrupted():
                        was_interrupted = True
                    else:
                        raise

                if was_interrupted:
                    return AgentResult(
                        agent_id=self.agent_id, task_name="generate",
                        error="Interrupted by user (Ctrl-C)",
                        elapsed_time=time.time() - start_time,
                        was_retried=retried,
                    )

                truncated = (done_reason == "length")
                clean_response = self._strip_thinking_tags(full_response)

                # NEW: an empty answer is a FAILURE, not a silent success.
                # On thinking models this is almost always the budget being
                # consumed entirely inside the <think> block before any answer
                # text is emitted (done_reason == "length").
                if not clean_response:
                    # Some Ollama builds stream the think block INLINE as
                    # <think>...</think> inside the response field rather than
                    # the separate `thinking` field. If stripping emptied a
                    # non-empty raw response, that raw text WAS the (truncated)
                    # think output — keep it for diagnostics and so the retry
                    # trigger below fires even when done_reason is absent.
                    diag_thinking = full_thinking
                    if not diag_thinking and full_response.strip():
                        diag_thinking = full_response
                    if truncated:
                        reason = ("empty answer — output budget exhausted inside "
                                  "the thinking block (done_reason=length)")
                    elif diag_thinking:
                        reason = ("empty answer — model produced only a thinking "
                                  "block and no answer text")
                    else:
                        reason = "empty answer — model returned no usable text"
                    return AgentResult(
                        agent_id=self.agent_id, task_name="generate",
                        response="",
                        thinking=diag_thinking if diag_thinking else None,
                        error=reason,
                        elapsed_time=time.time() - start_time,
                        success=False,
                        done_reason=done_reason,
                        truncated=truncated,
                        eval_count=eval_count,
                        prompt_eval_count=prompt_eval_count,
                        was_retried=retried,
                    )

                return AgentResult(
                    agent_id=self.agent_id, task_name="generate",
                    response=clean_response,
                    thinking=full_thinking if full_thinking else None,
                    elapsed_time=time.time() - start_time,
                    success=True,
                    done_reason=done_reason,
                    truncated=truncated,
                    eval_count=eval_count,
                    prompt_eval_count=prompt_eval_count,
                    was_retried=retried,
                )
            except requests.exceptions.Timeout:
                return AgentResult(agent_id=self.agent_id, task_name="generate",
                                   error="Request timed out",
                                   elapsed_time=time.time() - start_time,
                                   was_retried=retried)
            except requests.exceptions.ConnectionError:
                return AgentResult(agent_id=self.agent_id, task_name="generate",
                                   error="Connection to Ollama failed — is it running?",
                                   elapsed_time=time.time() - start_time,
                                   was_retried=retried)
            except Exception as e:
                return AgentResult(agent_id=self.agent_id, task_name="generate",
                                   error=f"Error: {str(e)}",
                                   elapsed_time=time.time() - start_time,
                                   was_retried=retried)
            finally:
                if response is not None:
                    _unregister_response(response)
                    try:
                        response.close()
                    except Exception:
                        pass

        # ---- First attempt --------------------------------------------------
        result = _attempt(actual_max_tokens, retried=False)

        # ---- One-shot automatic retry with a larger output budget -----------
        # If a thinking model truncated inside <think> and emitted no answer,
        # the fix is simply MORE output budget so it can finish thinking AND
        # write the answer. Retry once with a bigger budget that still fits the
        # model window. Never retry on interrupts or genuine connection errors.
        # NOTE: this deliberately does NOT require effective_think. A model that
        # Ollama does not report as a thinking model (so `think` is never sent in
        # the payload) can still emit a thinking block from its own template —
        # gemma4:31b did exactly that, burning all 2048 tokens of a task profile
        # marked think=off inside the think block and returning an empty answer.
        # Because effective_think was False the retry was gated off, so
        # refine_planning and title_generation failed outright and the run
        # re-issued identical queries until it stagnated. The clause below
        # already establishes that the answer came back EMPTY (a non-empty
        # answer sets success=True and never reaches here), so a larger budget
        # is the right response regardless of who asked for the thinking.
        should_retry = (
            (not result.success)
            and (result.truncated or (result.thinking and not result.response))
            and not is_interrupted()
            and not result.was_retried
            and (result.error is None or "Interrupted" not in result.error)
        )
        if should_retry:
            # Room left in the window for output after the input + headroom.
            available_out = hard_max - input_tokens_est_top - DYNAMIC_CTX_HEADROOM
            target = actual_max_tokens * 3
            bigger = min(target, available_out)
            if bigger > actual_max_tokens + 256:
                print(f"  {Fore.YELLOW}\u26A0 {result.error}{Style.RESET_ALL}")
                print(f"  {Fore.YELLOW}Retrying once with a larger output budget "
                      f"({actual_max_tokens} \u2192 {bigger} tokens)...{Style.RESET_ALL}",
                      flush=True)
                result = _attempt(bigger, retried=True)

        # ---- On-error diagnostics (printed ONLY when a call fails) ----------
        # Skipped for user interrupts (not an error condition).
        if (not result.success) and result.error and "Interrupted" not in result.error:
            _print_llm_diagnostics(result, context_label=f"task={task or 'default'}")

        return result

    def generate_json(self, prompt: str, system_prompt: str = None,
                      max_tokens: int = None,
                      num_ctx: int = None, think: Optional[bool] = None,
                      task: Optional[str] = None) -> AgentResult:
        result = self.generate(prompt, system_prompt=system_prompt,
                                max_tokens=max_tokens,
                                num_ctx=num_ctx, think=think, task=task)
        if not result.success or not result.response:
            return result
        json_data = self._extract_json(result.response)
        if json_data is not None:
            result.json_response = json_data
        else:
            result.error = (f"Failed to parse JSON from response "
                            f"(answer was {len(result.response)} chars, "
                            f"done_reason={result.done_reason})")
            result.success = False
            logger.debug(f"JSON parse failed. Raw response: {result.response[:500]}")
            # The model returned text but it wasn't valid JSON even after
            # repair — surface what it actually produced so it's debuggable.
            _print_llm_diagnostics(result, context_label=f"task={task or 'default'} (JSON parse)",
                                   raw_response=result.response, json_failure=True)
        return result

    def _extract_json(self, text: str) -> Optional[Dict]:
        if not text:
            return None
        # Strip any thinking-model tags FIRST. A long <think> block followed by
        # truncated JSON is the most common parse failure on think=ON tasks
        # (curation, sufficiency); removing it before extraction makes the
        # brace/bracket slicing below operate on the actual answer.
        text = self._strip_thinking_tags(text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except json.JSONDecodeError:
                pass

        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(text[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass

        # Structure-aware repair of a TRUNCATED object/array runs before the
        # naive lone-bracket slice below, so a truncated top-level object wins
        # over an inner complete array it happens to contain. When the model is
        # cut off (max_tokens reached) the JSON is well-formed up to the cutoff
        # but missing closers; we reconstruct them.
        repaired = self._repair_truncated_json(text)
        if repaired is not None:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        # Embedded-quote repair. A COMPLETE response (done_reason=stop) can
        # still fail to parse when a string VALUE contains a literal, unescaped
        # " — most often deep_analysis quoting a tweet or a nested quotation,
        # e.g.  "quote": "...appreciation," he tweeted...  Previously this lost
        # the entire paper's quotes (3 studies dropped in the Mars run incl. the
        # main economics source). _repair_unescaped_quotes() re-escapes quotes
        # that sit INSIDE a string and are not the real closing quote, leaving
        # structural quotes intact. We try it on the brace-sliced region first,
        # then on the whole text, and — in case the response was ALSO truncated
        # — feed the repaired text back through the truncation closer. Trailing
        # commas are stripped before each parse attempt. This runs AFTER the
        # truncation repair and BEFORE the lone-bracket slice, so neither of
        # those existing paths changes behaviour; this is purely additive.
        fb_eq = text.find('{')
        lb_eq = text.rfind('}')
        eq_candidates = []
        if fb_eq != -1 and lb_eq > fb_eq:
            eq_candidates.append(text[fb_eq:lb_eq + 1])
        eq_candidates.append(text)
        for cand in eq_candidates:
            fixed = self._repair_unescaped_quotes(cand)
            for attempt in (fixed, re.sub(r',\s*([}\]])', r'\1', fixed)):
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    pass
            closed = self._repair_truncated_json(fixed)
            if closed is not None:
                try:
                    return json.loads(closed)
                except json.JSONDecodeError:
                    pass

        first_bracket = text.find('[')
        last_bracket = text.rfind(']')
        if first_bracket != -1 and last_bracket > first_bracket:
            try:
                return json.loads(text[first_bracket:last_bracket + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _repair_truncated_json(self, text: str) -> Optional[str]:
        """Best-effort close of a truncated JSON object/array.

        Tokenises from the first opening bracket, tracking string-literal state
        and bracket depth. Records the index just past the last COMPLETE value
        (a closed string, a fully-formed number, a complete true/false/null
        literal, or a closed brace/bracket) seen while inside a structure.
        Everything after that boundary is a partial token from the cutoff and
        is discarded; a dangling '"key":' with no value is also trimmed. The
        closers still open at the boundary are then appended. Returns None if
        nothing is recoverable.
        """
        start = min([p for p in (text.find('{'), text.find('[')) if p != -1], default=-1)
        if start == -1:
            return None
        s = text[start:]
        n = len(s)

        in_str = False
        escape = False
        depth = 0
        last_valid = -1
        i = 0
        while i < n:
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
                    # A closed string is a complete VALUE only if it is not a
                    # key — i.e. the next non-space character is not a colon.
                    # Keys (e.g. "reasoning") must not mark a boundary, or the
                    # trim would leave a dangling '"key"' with no value.
                    k = i + 1
                    while k < n and s[k] in ' \t\r\n':
                        k += 1
                    if depth > 0 and not (k < n and s[k] == ':'):
                        last_valid = i + 1
                i += 1
                continue

            if ch == '"':
                in_str = True
                i += 1
            elif ch in '{[':
                depth += 1
                i += 1
            elif ch in '}]':
                depth -= 1
                last_valid = i + 1  # closed structure is a complete value
                i += 1
            elif depth > 0 and (ch.isdigit() or ch in '+-.'):
                # Consume a full number; count it only if a delimiter follows
                # (otherwise it may be truncated mid-number).
                j = i + 1
                while j < n and s[j] in '0123456789.eE+-':
                    j += 1
                if j < n:  # something delimits the number => complete
                    last_valid = j
                i = j
            elif depth > 0 and ch in 'tfn':
                # true / false / null — complete only if the whole literal fits
                # AND a delimiter follows.
                matched = False
                for lit in ('true', 'false', 'null'):
                    if s.startswith(lit, i):
                        end = i + len(lit)
                        if end < n:  # delimiter present => complete literal
                            last_valid = end
                        i = end
                        matched = True
                        break
                if not matched:
                    i += 1  # partial literal (e.g. 'tr') — no boundary
            else:
                i += 1

        if last_valid == -1:
            return None

        head = s[:last_valid]
        # Trim a dangling separator. With key-aware boundaries above, head now
        # ends just past a complete value, so only a stray comma can remain.
        head = head.rstrip().rstrip(',').rstrip()

        # Recompute the closers still open over the trimmed head.
        closers = []
        in_str = False
        escape = False
        for ch in head:
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '{':
                closers.append('}')
            elif ch == '[':
                closers.append(']')
            elif ch in '}]':
                if closers:
                    closers.pop()

        if in_str:
            head += '"'
        return head + ''.join(reversed(closers))

    def _repair_unescaped_quotes(self, text: str) -> str:
        """Re-escape double-quote characters that appear INSIDE a JSON string.

        Thinking/instruct models frequently emit a string value that itself
        contains a literal " (most often deep_analysis quoting a tweet or a
        nested quotation, e.g.  ...appreciation," he tweeted...). That inner "
        is not escaped, so json.loads aborts. This walks the text tracking
        string-literal state and, for every " met while inside a string, treats
        it as the REAL closing quote ONLY when the next non-space character is
        structural (one of  , : } ]  or end-of-text); otherwise it is an
        embedded quote and is escaped to \\". Already-escaped quotes (preceded
        by a backslash) are passed through untouched, as is everything OUTSIDE
        strings, so JSON STRUCTURE is preserved exactly.

        Best-effort only — this is a heuristic, not a parser. The one case it
        cannot disambiguate is an embedded " that happens to be followed by a
        comma/brace (it will mis-close there); when that produces invalid JSON
        the caller's remaining strategies run and, failing all, _extract_json
        returns None exactly as before. It never corrupts a payload that was
        already valid, because _extract_json only reaches here after a direct
        json.loads has already failed.
        """
        out = []
        in_string = False
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if not in_string:
                out.append(ch)
                if ch == '"':
                    in_string = True
                i += 1
                continue
            # --- inside a string literal ---
            if ch == '\\':
                # Preserve the escape sequence verbatim (this char + the next).
                out.append(ch)
                if i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            if ch == '"':
                j = i + 1
                while j < n and text[j] in ' \t\r\n':
                    j += 1
                nxt = text[j] if j < n else ''
                if nxt in (',', ':', '}', ']', ''):
                    out.append(ch)          # genuine closing quote
                    in_string = False
                else:
                    out.append('\\"')       # embedded quote -> escape it
                i += 1
                continue
            out.append(ch)
            i += 1
        return ''.join(out)

    def _strip_thinking_tags(self, text: str) -> str:
        if not text:
            return text
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()


class LLMManager:
    def __init__(self, num_agents: int = None, primary_config: Dict = None,
                 agent_config: Dict = None, thinking_config: Dict = None):
        parallel_cfg = get_parallel_config()
        if num_agents is None:
            num_agents = parallel_cfg.get("num_search_agents", 2)

        self.num_agents = num_agents
        self.primary_config = primary_config or get_primary_llm_config()
        self.agent_config = agent_config or get_agent_llm_config()
        self.thinking_config = thinking_config or get_thinking_config()
        self.parallel_config = parallel_cfg

        self.primary_agent = LLMAgent(
            agent_id="primary",
            llm_config=self.primary_config,
            thinking_config=self.thinking_config,
        )
        self.search_agents: Dict[str, LLMAgent] = {}
        for i in range(num_agents):
            agent_id = f"search_agent_{i}"
            self.search_agents[agent_id] = LLMAgent(
                agent_id=agent_id,
                llm_config=self.agent_config,
                thinking_config=self.thinking_config,
            )

        self.executor = ThreadPoolExecutor(max_workers=max(num_agents, 2))

        # Parallel search agents are DORMANT in this version (the pipeline only
        # calls run_primary). They are constructed lazily-cheap (no model load)
        # so the architecture remains available to build on later, but only the
        # main model is announced at startup.
        primary_model = self.primary_config.get("model_name", "unknown")
        logger.info(f"LLMManager: model={primary_model}")
        print(f"  Model: {primary_model}")

    def run_primary(self, prompt: str, system_prompt: str = None,
                    max_tokens: int = None, as_json: bool = False,
                    num_ctx: int = None, think: Optional[bool] = None,
                    task: Optional[str] = None) -> AgentResult:
        if as_json:
            return self.primary_agent.generate_json(
                prompt, system_prompt=system_prompt,
                max_tokens=max_tokens,
                num_ctx=num_ctx, think=think, task=task)
        return self.primary_agent.generate(
            prompt, system_prompt=system_prompt,
            max_tokens=max_tokens,
            num_ctx=num_ctx, think=think, task=task)

    def run_single(self, prompt: str, system_prompt: str = None,
                   max_tokens: int = None, as_json: bool = False,
                   num_ctx: int = None, think: Optional[bool] = None,
                   task: Optional[str] = None) -> AgentResult:
        return self.run_primary(
            prompt, system_prompt=system_prompt,
            max_tokens=max_tokens, as_json=as_json,
            num_ctx=num_ctx, think=think, task=task)

    def run_agent(self, prompt: str, system_prompt: str = None,
                  max_tokens: int = None, as_json: bool = False,
                  num_ctx: int = None, think: Optional[bool] = None,
                  task: Optional[str] = None) -> AgentResult:
        agent = list(self.search_agents.values())[0]
        if as_json:
            return agent.generate_json(
                prompt, system_prompt=system_prompt,
                max_tokens=max_tokens,
                num_ctx=num_ctx, think=think, task=task)
        return agent.generate(
            prompt, system_prompt=system_prompt,
            max_tokens=max_tokens,
            num_ctx=num_ctx, think=think, task=task)

    def run_parallel(self, tasks: List[Dict]) -> List[AgentResult]:
        if not tasks:
            return []
        if len(tasks) == 1:
            task = tasks[0]
            result = self.run_agent(
                task["prompt"], system_prompt=task.get("system_prompt"),
                max_tokens=task.get("max_tokens"), as_json=task.get("as_json", False),
                num_ctx=task.get("num_ctx"), think=task.get("think"),
                task=task.get("task"),
            )
            result.task_name = task.get("task_name", "single_task")
            return [result]

        agent_ids = list(self.search_agents.keys())
        agent_start_delay = self.parallel_config.get("agent_start_delay", 2.0)
        futures = []

        print(f"\n{Fore.CYAN}  Launching {len(tasks)} parallel agent tasks...{Style.RESET_ALL}")

        for i, t in enumerate(tasks):
            agent_id = agent_ids[i % len(agent_ids)]
            agent = self.search_agents[agent_id]
            task_name = t.get("task_name", f"task_{i}")

            def execute_task(a=agent, tt=t, tn=task_name, delay=i * agent_start_delay):
                if delay > 0:
                    time.sleep(delay)
                if tt.get("as_json", False):
                    r = a.generate_json(
                        tt["prompt"], system_prompt=tt.get("system_prompt"),
                        max_tokens=tt.get("max_tokens"),
                        num_ctx=tt.get("num_ctx"), think=tt.get("think"),
                        task=tt.get("task"),
                    )
                else:
                    r = a.generate(
                        tt["prompt"], system_prompt=tt.get("system_prompt"),
                        max_tokens=tt.get("max_tokens"),
                        num_ctx=tt.get("num_ctx"), think=tt.get("think"),
                        task=tt.get("task"),
                    )
                r.task_name = tn
                return r

            future = self.executor.submit(execute_task)
            futures.append(future)
            print(f"  {Fore.YELLOW}  [{agent_id}] -> {task_name}{Style.RESET_ALL}")

        results = [None] * len(tasks)
        for future in as_completed(futures):
            idx = futures.index(future)
            try:
                result = future.result()
                results[idx] = result
                status = f"{Fore.GREEN}\u2713" if result.success else f"{Fore.RED}\u2717"
                print(f"  {status}  {result.task_name} ({result.elapsed_time:.1f}s){Style.RESET_ALL}")
            except Exception as e:
                results[idx] = AgentResult(
                    agent_id="unknown", task_name=f"task_{idx}",
                    error=f"Execution error: {str(e)}",
                )
        return results

    def shutdown(self):
        self.executor.shutdown(wait=False)


# Backward-compatible alias. The class was renamed ParallelAgentManager ->
# LLMManager when parallel agents were made dormant; this keeps any external
# import of the old name working.
ParallelAgentManager = LLMManager
