"""
qa_session.py — Post-review interactive Q&A for the Academic Literature Review System.

WHY THIS FILE EXISTS
--------------------
The Q&A loop that used to live inline inside Academic_Researcher.py's main()
was broken in several independent ways, all of which produced the same visible
symptom: "you type a question, hit Enter, and nothing happens".

The original loop was:

    while True:
        fu = input("Q&A> ").strip()
        if not fu or fu.lower() in ('new', 'quit', 'exit', 'q'):
            break
        r = pipeline.agent_manager.run_primary(..., task="qa_mode")
        if r.success:
            print(r.response)

Bugs, in the order they bite you:

  BUG 1 — STALE INTERRUPT FLAG (the big one).
      llm_manager keeps a module-level threading.Event as the interrupt flag.
      pipeline.run() calls clear_interrupt() at the START of a run, and the
      SIGINT handler calls request_interrupt(). NOTHING ever clears it after
      the run finishes. So if you pressed Ctrl-C even ONCE during the (very
      long) review run, the flag is still set when Q&A starts, and
      AcademicAgent.generate() hits its "Check interrupt before even starting"
      guard and returns AgentResult(error="Interrupted before request started",
      success=False) instantly, forever. Combined with BUG 2 that is completely
      silent. Fixed here by clearing the interrupt state before the session and
      before EVERY question.

  BUG 2 — SILENT FAILURE. `if r.success:` had no `else`. Every failure mode
      (interrupt, HTTP error, connection error, empty answer, JSON/timeout)
      printed absolutely nothing. The loop just re-prompted. This is literally
      "I hit Enter and it goes down a line". Fixed: every failure is reported
      with its reason and diagnostics.

  BUG 3 — BLANK LINE SILENTLY EXITS Q&A. `if not fu ... break`. A single stray
      Enter (or a newline left in the terminal buffer from tapping keys during
      the hours-long run) drops you straight out of Q&A with no message at all.
      Fixed: a blank line just re-prompts.

  BUG 4 — BUFFERED KEYSTROKES. Anything typed during the long run sits in the
      terminal's input buffer and is consumed instantly by the first input()
      call, tripping BUG 3 before you can type anything. Fixed: the stdin
      buffer is flushed before the first prompt.

  BUG 5 — THINKING MODEL EATS THE WHOLE BUDGET. The "qa_mode" task profile in
      academic_config.py is {"num_ctx": 32768, "max_tokens": 4096, "think":
      True}. On a reasoning model the entire budget can be consumed inside the
      <think> block, so generate() returns success=False with "empty answer".
      Combined with BUG 2: silence. Fixed: empty/truncated answers are detected
      and automatically retried with a bigger budget and thinking disabled.

  BUG 6 — NO PROGRESS INDICATION. run_primary() blocks with no output while a
      reasoning model thinks, which can be minutes. It looks hung. Fixed: an
      elapsed-time indicator runs while the model works.

  BUG 7 — Ctrl-C IN Q&A KILLED Q&A PERMANENTLY. The pipeline's SIGINT handler
      set the interrupt flag; the next question then hit BUG 1. Fixed: Ctrl-C
      now cancels just the current answer and returns you to the prompt.

  BUG 8 — EOF / NON-TTY CRASH-OUT. input() raises EOFError on a closed or
      non-interactive stdin; that propagated to main()'s `except EOFError`
      and quit the whole program. Fixed: EOF is handled locally, and a
      non-interactive stdin skips Q&A cleanly with a message.

  BUG 9 — BLIND 10,000-CHARACTER TRUNCATION. The review was cut at 10k chars
      with no system prompt, so the model usually never saw the References, the
      later sections, or any study metadata, and had no instruction to stay
      grounded. Fixed: a budgeted context that keeps the head AND tail of the
      review (so the reference list survives), plus the study list, plus a
      proper grounding system prompt.

  BUG 10 — NO CONVERSATION MEMORY. Every question was independent, so
      follow-ups like "why?" or "expand on that" had no referent. Fixed: a
      rolling short history of recent turns is included.

Nothing about the review pipeline itself is changed by this module. It only
runs AFTER a review has been produced.

Config keys (all optional, read from RESEARCH_CONFIG via pipeline.config, with
safe defaults so no config edit is required):
    qa_mode_enabled       (already exists)  — honoured by the caller
    qa_context_max_chars  default 60000     — grounding budget for the review
    qa_include_study_list default True      — include the analysed-study table
    qa_history_turns      default 4         — how many prior Q&A turns to keep
    qa_save_transcript    default True      — write Logs/<Title>_QA.txt
    qa_retry_without_think default True     — auto-retry empty thinking answers
"""

import os
import re
import sys
import time
import threading
from datetime import datetime

try:
    from colorama import Fore, Style
except Exception:  # pragma: no cover - colorama is a hard dep of the main app
    class _Dummy:
        def __getattr__(self, _):
            return ""
    Fore = Style = _Dummy()

# The interrupt helpers live in llm_manager. They are imported defensively so
# this module can also be imported/tested standalone.
try:
    from llm_manager import clear_interrupt, is_interrupted
except Exception:  # pragma: no cover
    def clear_interrupt():
        pass

    def is_interrupted():
        return False


__all__ = ["run_qa_session"]


_EXIT_WORDS = ("quit", "exit", "q")
_NEW_WORDS = ("new", "next", "another")
_HELP_WORDS = ("help", "?", "commands")


# ============================================================================
# TERMINAL / INPUT PLUMBING
# ============================================================================

def _console():
    """The real console stream.

    During a run SessionLogger replaces sys.stdout with a _TeeStream, and
    colorama replaces it again at import time. Prompts are written to the
    stream we actually have, but flushed explicitly so the prompt can never sit
    in a buffer while we block on a read (a classic "my typing does nothing"
    cause).
    """
    return sys.stdout if sys.stdout is not None else sys.__stdout__


def _stdin_is_interactive() -> bool:
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:
        return False


def _flush_stdin():
    """Discard anything already sitting in the terminal input buffer.

    A review run takes a long time; any keys tapped during it are queued by the
    terminal and would be swallowed by the first Q&A prompt. Best-effort only —
    never raises.
    """
    if not _stdin_is_interactive():
        return
    try:
        import termios  # POSIX
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        return
    except Exception:
        pass
    try:
        import msvcrt  # Windows
        while msvcrt.kbhit():
            msvcrt.getwch()
    except Exception:
        pass


def _read_line(prompt: str):
    """Read one line. Returns the raw string, or None on EOF.

    Deliberately does NOT use input(): input() raises EOFError (which used to
    kill the whole program) and cannot distinguish "user pressed Enter" from
    "stdin closed". sys.stdin.readline() returns '\\n' for the former and ''
    for the latter, which is exactly the distinction we need.
    """
    out = _console()
    try:
        out.write(prompt)
        out.flush()
    except Exception:
        pass
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:
        return None
    if line == "":          # true EOF
        return None
    return line.rstrip("\r\n")


class _Working:
    """Elapsed-time indicator shown while the model is generating.

    Without this the console is completely silent for however long a reasoning
    model takes, which is indistinguishable from a hang.
    """

    def __init__(self, label="thinking"):
        self.label = label
        self._stop = threading.Event()
        self._t = None
        self._enabled = False
        try:
            self._enabled = bool(_console().isatty())
        except Exception:
            self._enabled = False

    def _loop(self):
        start = time.time()
        frames = "|/-\\"
        i = 0
        out = _console()
        while not self._stop.wait(0.25):
            i += 1
            try:
                out.write(f"\r  {Fore.CYAN}{frames[i % 4]} {self.label}… "
                          f"{int(time.time() - start)}s"
                          f"{Style.RESET_ALL}   ")
                out.flush()
            except Exception:
                return

    def __enter__(self):
        if self._enabled:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1.0)
        if self._enabled:
            try:
                out = _console()
                out.write("\r" + " " * 60 + "\r")
                out.flush()
            except Exception:
                pass
        return False


# ============================================================================
# GROUNDING CONTEXT
# ============================================================================

_QA_SYSTEM_PROMPT = (
    "You are answering follow-up questions about a literature review that YOU "
    "have just completed. The full review, and a summary table of every study "
    "analysed for it, are provided below.\n\n"
    "Rules:\n"
    "1. Answer from the review and the study list. Use APA in-text citations "
    "   (Author, Year) exactly as they appear in the review whenever you make "
    "   an evidence claim.\n"
    "2. If the review does not contain the answer, say so plainly — state what "
    "   the review DOES cover that is closest, and what additional evidence "
    "   would be needed. Do not invent studies, findings, numbers or citations.\n"
    "3. Distinguish clearly between what the studies found and your own "
    "   interpretation or inference.\n"
    "4. Be direct and specific. Prose, not headings, unless the question "
    "   genuinely calls for a list."
)


def _build_study_list(state, limit_chars=20000):
    """Compact table of the analysed studies, from the final pipeline state.

    Mirrors the field names used by _build_studies_for_assessment() in
    Academic_Researcher.py so it stays correct if analyses change shape.
    """
    if not isinstance(state, dict):
        return ""
    analyses = state.get("study_analyses") or []
    if not analyses:
        return ""

    lines = []
    used = 0
    for i, a in enumerate(analyses, 1):
        if not isinstance(a, dict):
            continue
        authors = a.get("paper_authors") or a.get("authors") or ""
        if isinstance(authors, (list, tuple)):
            authors = ", ".join(str(x) for x in authors[:3])
        findings = a.get("key_findings") or []
        if isinstance(findings, (list, tuple)):
            findings = "; ".join(str(f) for f in findings[:3])
        entry = (
            f"[{i}] {a.get('paper_title', '?')} "
            f"({a.get('paper_year', '?')})\n"
            f"    Authors: {str(authors)[:160]}\n"
            f"    Design: {a.get('study_type', '?')} | "
            f"Sample: {a.get('sample_size', '?')} | "
            f"Reliability: {a.get('reliability_score', '?')}/10\n"
            f"    Key findings: {str(findings)[:600]}\n"
        )
        if used + len(entry) > limit_chars:
            lines.append(f"[... {len(analyses) - i + 1} further studies omitted "
                         f"for context budget ...]")
            break
        lines.append(entry)
        used += len(entry)
    return "\n".join(lines)


def _budget_review(review: str, max_chars: int) -> str:
    """Fit the review into the character budget WITHOUT losing the tail.

    The old code did review[:10000], which on any real review threw away the
    later sections and the entire APA reference list — so the model could not
    answer anything about its own citations. Here we keep the head and the tail
    and mark the elision.
    """
    review = review or ""
    if len(review) <= max_chars:
        return review
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return (review[:head]
            + "\n\n[... middle of the review omitted to fit the context "
              "window; the sections above and below are complete ...]\n\n"
            + review[-tail:])


def _build_prompt(question, review, study_list, history, title, query,
                  max_chars):
    parts = []
    parts.append(f"REVIEW TITLE: {title or 'Untitled Review'}")
    parts.append(f"ORIGINAL RESEARCH QUESTION: {query or '(not recorded)'}")
    parts.append("\n===== FULL LITERATURE REVIEW =====\n")
    parts.append(_budget_review(review, max_chars))
    if study_list:
        parts.append("\n===== STUDIES ANALYSED FOR THIS REVIEW =====\n")
        parts.append(study_list)
    if history:
        parts.append("\n===== EARLIER IN THIS Q&A SESSION =====\n")
        for q, a in history:
            parts.append(f"Q: {q}\nA: {a}\n")
    parts.append("\n===== QUESTION TO ANSWER NOW =====\n")
    parts.append(question)
    return "\n".join(parts)


# ============================================================================
# TRANSCRIPT
# ============================================================================

class _Transcript:
    """Appends each Q&A turn to Logs/<Review_Name>_QA.txt as it happens, so a
    long answer is never lost to a scrolled-off terminal."""

    def __init__(self, logs_dir, title, enabled=True):
        self.path = None
        if not enabled:
            return
        try:
            os.makedirs(logs_dir or "Logs", exist_ok=True)
            safe = re.sub(r'[^\w\s-]', '', title or '').strip().replace(' ', '_')
            if not safe:
                safe = f"review_{datetime.now():%Y%m%d_%H%M%S}"
            self.path = os.path.join(logs_dir or "Logs", f"{safe}_QA.txt")
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(f"\n{'=' * 78}\nQ&A SESSION — {datetime.now().isoformat()}\n"
                         f"Review: {title}\n{'=' * 78}\n")
        except Exception:
            self.path = None

    def add(self, question, answer):
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(f"\nQ [{datetime.now():%H:%M:%S}]: {question}\n\n"
                         f"A: {answer}\n\n{'-' * 78}\n")
        except Exception:
            pass


# ============================================================================
# THE SESSION
# ============================================================================

def _print_help():
    print(f"""
{Fore.CYAN}Q&A commands{Style.RESET_ALL}
  {Fore.GREEN}<any question>{Style.RESET_ALL}  ask about the review (press Enter to send)
  {Fore.GREEN}review{Style.RESET_ALL}          re-print the full review
  {Fore.GREEN}studies{Style.RESET_ALL}         list the studies the review was built from
  {Fore.GREEN}clear{Style.RESET_ALL}           forget the Q&A conversation history
  {Fore.GREEN}new{Style.RESET_ALL}             leave Q&A and start a new research question
  {Fore.GREEN}quit{Style.RESET_ALL}            leave Q&A and exit the program
  {Fore.GREEN}help{Style.RESET_ALL}            show this list
  {Fore.YELLOW}Ctrl-C{Style.RESET_ALL}          cancel the answer being generated (Q&A stays open)
""")


def _ask(pipeline, prompt, question, retry_without_think=True):
    """One question -> one AgentResult, with the interrupt flag cleared first
    and an automatic no-thinking retry when the reasoning block eats the whole
    output budget."""

    # BUG 1 / BUG 7 FIX: a Ctrl-C from earlier in the run (or from a previous
    # Q&A answer) leaves llm_manager's interrupt Event set, which makes
    # generate() bail out instantly and silently forever. Clear it before every
    # single question.
    clear_interrupt()
    try:
        pipeline._interrupted = False
    except Exception:
        pass

    with _Working("thinking"):
        r = pipeline.agent_manager.run_primary(prompt, system_prompt=_QA_SYSTEM_PROMPT,
                                               task="qa_mode")

    ok = bool(getattr(r, "success", False)) and bool(getattr(r, "response", ""))
    if ok:
        return r

    if is_interrupted():
        return r

    # BUG 5 FIX: qa_mode has think=True. On a reasoning model the whole budget
    # can be spent inside <think> and the answer comes back empty. Retry once
    # with thinking off and a bigger budget rather than showing the user
    # nothing.
    err = str(getattr(r, "error", "") or "")
    empty_ish = ("empty answer" in err.lower()
                 or getattr(r, "truncated", False)
                 or not getattr(r, "response", ""))
    if retry_without_think and empty_ish:
        print(f"  {Fore.YELLOW}First attempt produced no answer text "
              f"({err or 'empty response'}). Retrying with thinking disabled "
              f"and a larger output budget…{Style.RESET_ALL}")
        clear_interrupt()
        with _Working("retrying"):
            r2 = pipeline.agent_manager.run_primary(
                prompt, system_prompt=_QA_SYSTEM_PROMPT,
                task="qa_mode", think=False, max_tokens=6144)
        if getattr(r2, "success", False) and getattr(r2, "response", ""):
            return r2
        return r2
    return r


def _report_failure(r):
    """BUG 2 FIX: never fail silently."""
    err = getattr(r, "error", None) or "no answer returned (no error reported)"
    print(f"\n{Fore.RED}Q&A could not answer that one.{Style.RESET_ALL}")
    print(f"  {Fore.RED}Reason: {err}{Style.RESET_ALL}")
    dr = getattr(r, "done_reason", None)
    if dr:
        print(f"  {Fore.YELLOW}done_reason={dr} | "
              f"tokens_out={getattr(r, 'eval_count', '?')} | "
              f"tokens_in={getattr(r, 'prompt_eval_count', '?')}"
              f"{Style.RESET_ALL}")
    low = str(err).lower()
    if "interrupt" in low:
        print(f"  {Fore.YELLOW}Tip: that was the interrupt flag. It has been "
              f"cleared — just ask again.{Style.RESET_ALL}")
    elif "connection" in low or "http" in low or "timed out" in low:
        print(f"  {Fore.YELLOW}Tip: check Ollama is still running "
              f"(`ollama ps`) — the model may have been unloaded."
              f"{Style.RESET_ALL}")
    else:
        print(f"  {Fore.YELLOW}Tip: try a shorter or more specific question, "
              f"or lower qa_context_max_chars in academic_config.py."
              f"{Style.RESET_ALL}")


def run_qa_session(pipeline, results) -> str:
    """Interactive post-review Q&A.

    Returns:
        "quit" — the user asked to exit the program.
        "new"  — the user wants to ask a new research question (or Q&A ended
                 for any non-exit reason: EOF, non-interactive stdin, etc.).

    Never raises: any unexpected error is reported and treated as "new" so the
    main loop keeps working.
    """
    review = (results or {}).get("review") or ""
    if not review:
        print(f"\n{Fore.YELLOW}No review was produced, so there is nothing for "
              f"Q&A to work from.{Style.RESET_ALL}")
        return "new"

    cfg = getattr(pipeline, "config", {}) or {}
    max_chars = int(cfg.get("qa_context_max_chars", 60000))
    include_studies = bool(cfg.get("qa_include_study_list", True))
    history_turns = int(cfg.get("qa_history_turns", 4))
    save_transcript = bool(cfg.get("qa_save_transcript", True))
    retry_no_think = bool(cfg.get("qa_retry_without_think", True))

    title = results.get("title") or "Untitled Review"
    query = results.get("query") or ""
    state = results.get("state") or {}
    study_list = _build_study_list(state) if include_studies else ""

    logs_dir = "Logs"
    try:
        logs_dir = pipeline.paths.get("logs_directory", "Logs")
    except Exception:
        pass
    transcript = _Transcript(logs_dir, title, enabled=save_transcript)

    # BUG 8 FIX: piped/redirected stdin used to raise EOFError out of the Q&A
    # loop and terminate the whole program with no explanation.
    if not _stdin_is_interactive():
        print(f"\n{Fore.YELLOW}Q&A mode skipped: stdin is not an interactive "
              f"terminal.{Style.RESET_ALL}")
        return "new"

    # BUG 4 FIX: throw away keystrokes queued during the long run, which would
    # otherwise be eaten by the first prompt.
    _flush_stdin()

    # BUG 1 FIX, session level.
    clear_interrupt()
    try:
        pipeline._interrupted = False
    except Exception:
        pass

    n_studies = len(state.get("study_analyses") or []) if isinstance(state, dict) else 0
    print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Q&A MODE — ask anything about this review{Style.RESET_ALL}")
    print(f"{Fore.WHITE}  Review: {title}")
    print(f"  Grounded in: the full review"
          + (f" + {n_studies} analysed studies" if n_studies else "")
          + f"{Style.RESET_ALL}")
    if transcript.path:
        print(f"{Fore.WHITE}  Transcript: {transcript.path}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}  Type a question and press Enter. "
          f"'help' for commands, 'new' for a new topic, 'quit' to exit."
          f"{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")

    history = []

    while True:
        try:
            line = _read_line(f"\n{Fore.GREEN}Q&A> {Style.RESET_ALL}")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}(Ctrl-C at the prompt — type 'quit' to exit "
                  f"or 'new' for a new question.){Style.RESET_ALL}")
            clear_interrupt()
            try:
                pipeline._interrupted = False
            except Exception:
                pass
            continue

        if line is None:  # EOF
            print(f"\n{Fore.CYAN}End of input — leaving Q&A.{Style.RESET_ALL}")
            return "quit"

        fu = line.strip()

        # BUG 3 FIX: a blank line just re-prompts. It used to exit Q&A silently.
        if not fu:
            continue

        low = fu.lower()

        if low in _EXIT_WORDS:
            return "quit"
        if low in _NEW_WORDS:
            return "new"
        if low in _HELP_WORDS:
            _print_help()
            continue
        if low == "clear":
            history = []
            print(f"  {Fore.GREEN}Q&A history cleared.{Style.RESET_ALL}")
            continue
        if low == "review":
            print(f"\n{Fore.WHITE}{review}{Style.RESET_ALL}")
            continue
        if low == "studies":
            if study_list:
                print(f"\n{Fore.WHITE}{study_list}{Style.RESET_ALL}")
            else:
                print(f"  {Fore.YELLOW}No study list available for this run."
                      f"{Style.RESET_ALL}")
            continue

        prompt = _build_prompt(fu, review, study_list, history[-history_turns:],
                               title, query, max_chars)

        try:
            r = _ask(pipeline, prompt, fu, retry_without_think=retry_no_think)
        except KeyboardInterrupt:
            # BUG 7 FIX: cancel just this answer, keep Q&A alive.
            print(f"\n{Fore.YELLOW}Answer cancelled. Q&A is still open."
                  f"{Style.RESET_ALL}")
            clear_interrupt()
            try:
                pipeline._interrupted = False
            except Exception:
                pass
            continue
        except Exception as e:
            print(f"\n{Fore.RED}Q&A error: {type(e).__name__}: {e}"
                  f"{Style.RESET_ALL}")
            continue

        answer = getattr(r, "response", "") or ""
        if getattr(r, "success", False) and answer.strip():
            print(f"\n{Fore.WHITE}{answer}{Style.RESET_ALL}")
            el = getattr(r, "elapsed_time", 0.0) or 0.0
            print(f"  {Fore.CYAN}({len(answer.split())} words in {el:.1f}s)"
                  f"{Style.RESET_ALL}")
            history.append((fu, answer[:1500]))
            transcript.add(fu, answer)
        else:
            if is_interrupted():
                print(f"\n{Fore.YELLOW}Answer cancelled. Q&A is still open."
                      f"{Style.RESET_ALL}")
                clear_interrupt()
                try:
                    pipeline._interrupted = False
                except Exception:
                    pass
            else:
                _report_failure(r)
