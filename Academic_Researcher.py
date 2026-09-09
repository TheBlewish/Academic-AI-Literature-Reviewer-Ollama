#!/usr/bin/env python3
# academic_researcher.py
# Academic Literature Review System — Quote-Driven APA 7th with Attribution Verification
#
# CHANGES IN THIS VERSION (vs. prior full version):
#   - _llm_select_papers_tangential prompt rewritten with THREE HARD GATES
#     (domain anchor, one-hop connection, domain-expert test) plus worked
#     good/bad examples, to stop UNRELATED (vs. genuinely tangential) picks.
#   - node_tangential_curate_evidence prompt tightened with the same three
#     gates so curation is a second line of defence against over-reaching
#     indirect picks.
#   - NOTHING ELSE CHANGED from the previous version: main-mode full-text gate,
#     tangential abstract-quote handling, APA 7th quote-driven synthesis,
#     deterministic attribution verification, dynamic-context support, and
#     removal of overall_reasoning from selection are all preserved.
#
# CHANGES IN THIS VERSION (placeholder quotes + variable usage):
#   1. QUOTE PLACEHOLDERS — the synthesis LLM no longer retypes quotes. Every
#      verified quote in the evidence base is given a STABLE GLOBAL TOKEN such as
#      [[Q7]]. The model places the token where it wants the quote; the exact
#      verified text is pasted in DETERMINISTICALLY at compile time
#      (_expand_quote_placeholders). Tokens are preserved through self-review /
#      self-fix / verification, so no model ever retypes a quote: this removes
#      retype variability, guarantees the verbatim quote, and reduces correction
#      calls. A quote at/above block_quote_word_threshold words, when its token
#      sits alone on a line, is rendered as an APA block quote; otherwise inline
#      in quotation marks.
#   2. VARIABLE QUOTE USAGE + DROP UNUSED STUDIES — extraction is no longer
#      capped at a fixed number (see study_analyser.py). A study with no
#      verified quote is dropped before synthesis (min_verified_quotes_per_study,
#      unchanged); now a study the model QUOTES ZERO TIMES in the final review is
#      also dropped — at compile the REFERENCES section is rebuilt
#      deterministically (_rebuild_references_section) to contain exactly the
#      studies whose quotes were used (plus any still cited in the body, so no
#      orphan citations). Controlled by rebuild_references_from_used_quotes.
#   The quote_registry that maps each token to its verified text + citation +
#   source study lives in ReviewState and is rebuilt in lockstep with the
#   evidence-base cache.
#
# CHANGES IN THIS VERSION (sequential curation + main-mode acquisition gate):
#   3. SEQUENTIAL CURATION — node_curate_evidence now decides INCLUDE/EXCLUDE
#      ONE paper at a time (task profile "curate_study"), with the same UI feel
#      as quick-read: an [i/total] counter with a time-remaining ETA, BLUE while
#      the paper is being judged, GREEN on INCLUDE, YELLOW on EXCLUDE, each with
#      reasoning. Each call sees only the single paper under review plus a
#      compact one-line catalog of the other candidates (so REDUNDANCY can still
#      be judged) — this keeps context small and fixes the "output budget
#      exhausted inside the thinking block" failure that the old whole-batch
#      call hit on large sets. curated_paper_ids / study_summaries /
#      curation_history record structure is unchanged.
#   4. MAIN-MODE ACQUISITION GATE — abstract-only papers (no full text) are now
#      excluded in MAIN mode right after acquisition, so quick-read / filter /
#      curation / deep-analysis time is not spent on papers that cannot yield
#      verified full-text quotes and would be dropped at deep-analysis anyway.
#      They are NOT deleted from the catalog: in TANGENTIAL mode they remain
#      eligible (tangential verifies quotes against abstracts). The deep-analysis
#      main-mode drop is kept as a defensive safety net.

import os
import sys
import json
import time
import re
import signal
import logging
import tempfile
import hashlib
import difflib
import requests
from typing import TypedDict, Optional, List, Dict, Any, Tuple
from datetime import datetime

from colorama import Fore, Style, init
init()

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("LangGraph not installed. Install: pip install langgraph")
    sys.exit(1)

from academic_config import (
    get_primary_llm_config, get_agent_llm_config,
    get_research_config, get_parallel_config, get_paths_config,
    get_low_end_config, is_low_end_enabled,
)
from paper_discovery import PaperDiscoveryEngine, PaperMetadata
from llm_manager import LLMManager, request_interrupt, clear_interrupt
from research_planner import ResearchPlanner, ResearchPlan, FocusArea
from study_analyser import StudyAnalyser
from reference_harvester import ReferenceHarvester

try:
    from document_store import DocumentStore
    HAS_DOC_STORE = True
except ImportError:
    HAS_DOC_STORE = False

try:
    from text_matcher import TextMatcher
    HAS_TEXT_MATCHER = True
except ImportError:
    HAS_TEXT_MATCHER = False

logger = logging.getLogger(__name__)

SYM_CHECK = "\U00002705"
SYM_BOOK = "\U0001F4DA"
SYM_LOOP = "\U0001F504"
SYM_PAPER = "\U0001F4C4"
SYM_SEARCH = "\U0001F50E"
SYM_EYES = "\U0001F440"
SYM_WRENCH = "\U0001F527"
SYM_BRAIN = "\U0001F9E0"
SYM_LAMP = "\U0001F4A1"
SYM_FILTER = "\U0001F3AF"
SYM_INCLUDE = "\u2713"
SYM_EXCLUDE = "\u2717"
SYM_QUOTE = "\U0001F4AC"

SECTION_COLORS = {
    "ABSTRACT": Fore.WHITE,
    "INTRODUCTION": Fore.CYAN,
    "METHOD": Fore.BLUE,
    "METHODOLOGY": Fore.BLUE,
    "EVIDENCE": Fore.GREEN,
    "FINDINGS": Fore.GREEN,
    "DISCUSSION": Fore.YELLOW,
    "LIMITATIONS": Fore.RED,
    "CONCLUSION": Fore.MAGENTA,
    "REFERENCES": Fore.WHITE,
}


# =============================================================================
# PER-RUN SESSION LOGGING
# =============================================================================
# Every run of the pipeline writes a complete, human-readable transcript of
# EVERYTHING that happens (every phase banner, every per-paper line, every
# warning) to its own plain-text file in the Logs/ directory. The file is named
# after the review — exactly like the Papers/<Review_Name> folder — so a
# review's downloaded papers and its log share the same name.
#
# Implementation: the program already prints rich, stage-by-stage output via
# print(). We simply "tee" stdout/stderr so every byte goes to BOTH the console
# (with colour) AND the log file (ANSI colour codes stripped, so the .txt is
# clean). Python `logging` records are routed into the same file too. This
# captures all stages without having to instrument each node individually.

class _TeeStream:
    """A text stream that duplicates writes to the console and a log file.

    ANSI colour escape codes are stripped from the file copy so the saved .txt
    is clean plain text, while the console keeps its colour.
    """
    _ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

    def __init__(self, console_stream, file_handle):
        self._console = console_stream
        self._file = file_handle

    def write(self, data):
        try:
            self._console.write(data)
        except Exception:
            pass
        try:
            self._file.write(self._ANSI_RE.sub('', data))
        except Exception:
            pass
        return len(data) if data else 0

    def flush(self):
        for s in (self._console, self._file):
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return self._console.isatty()
        except Exception:
            return False

    def fileno(self):
        # Some libraries probe fileno(); delegate to the real console stream.
        return self._console.fileno()


class SessionLogger:
    """Per-run verbose logger.

    Usage:
        sl = SessionLogger(logs_dir)
        sl.start(query)            # begin tee-ing stdout/stderr + logging
        ... run the pipeline ...
        sl.stop(review_title)      # restore streams, close + rename the file

    The file starts life as Logs/review_<timestamp>.txt and is renamed to
    Logs/<Review_Name>.txt at the end of the run (same sanitisation as the
    Papers/<Review_Name> folder). If the title is empty (e.g. the run failed
    before a title was generated) the timestamped name is kept so the log is
    still preserved for debugging.
    """

    def __init__(self, logs_dir="Logs"):
        self.logs_dir = logs_dir or "Logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        self._ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.logs_dir, f"review_{self._ts}.txt")
        self._fh = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._log_handler = None
        self._active = False

    def start(self, query=""):
        if self._active:
            return
        try:
            self._fh = open(self.path, "w", encoding="utf-8")
        except Exception as e:
            # If the log file cannot be opened, do not break the run.
            print(f"{Fore.YELLOW}Could not open session log ({e}); continuing "
                  f"without a per-run log.{Style.RESET_ALL}")
            self._fh = None
            return

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, self._fh)
        sys.stderr = _TeeStream(self._orig_stderr, self._fh)

        # Route Python logging records into the same file (file only, so console
        # output is not duplicated by the logging module).
        self._log_handler = logging.StreamHandler(self._fh)
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root = logging.getLogger()
        if root.level > logging.INFO or root.level == 0:
            root.setLevel(logging.INFO)
        root.addHandler(self._log_handler)

        self._active = True

        # Header — written through the tee so it lands in the file (console too).
        try:
            model = get_primary_llm_config().get("model_name", "?")
        except Exception:
            model = "?"
        print("=" * 78)
        print("ACADEMIC LITERATURE REVIEW — SESSION LOG")
        print("=" * 78)
        print(f"Started:  {datetime.now().isoformat()}")
        print(f"Query:    {query}")
        print(f"Model:    {model}")
        print(f"Log file: {self.path}")
        print("=" * 78)

    def stage(self, name):
        """Optional explicit stage marker (the pipeline's own phase banners are
        already captured; this is just an extra, greppable line)."""
        if self._active:
            print(f"\n----- STAGE: {name} @ {datetime.now().strftime('%H:%M:%S')} -----")

    def stop(self, review_title=""):
        if not self._active:
            return
        print("\n" + "=" * 78)
        print(f"Ended:    {datetime.now().isoformat()}")
        print("=" * 78)

        # Restore the real streams BEFORE closing the file so nothing writes to
        # a closed handle afterwards.
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except Exception:
            pass
        try:
            logging.getLogger().removeHandler(self._log_handler)
        except Exception:
            pass
        for closer in (self._log_handler, self._fh):
            try:
                if closer:
                    closer.close()
            except Exception:
                pass
        self._active = False

        self._rename_to_title(review_title)

    def _rename_to_title(self, review_title):
        if not review_title:
            print(f"{Fore.GREEN}Session log saved: {self.path}{Style.RESET_ALL}")
            return
        safe = re.sub(r'[^\w\s-]', '', review_title).strip().replace(' ', '_')
        if not safe:
            print(f"{Fore.GREEN}Session log saved: {self.path}{Style.RESET_ALL}")
            return
        new_path = os.path.join(self.logs_dir, f"{safe}.txt")
        if os.path.abspath(new_path) == os.path.abspath(self.path):
            print(f"{Fore.GREEN}Session log saved: {self.path}{Style.RESET_ALL}")
            return
        try:
            if os.path.exists(new_path):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_path = os.path.join(self.logs_dir, f"{safe}_{ts}.txt")
            os.rename(self.path, new_path)
            self.path = new_path
        except OSError as e:
            logger.warning(f"Could not rename session log: {e}")
        # Streams are already restored here, so this prints to the console only.
        print(f"{Fore.GREEN}Session log saved: {self.path}{Style.RESET_ALL}")


# =============================================================================
# APA 7th HELPERS (module-level — single source of truth)
# =============================================================================
# These are the canonical APA formatters used everywhere: the per-study
# reasoning prompts (relevance filter, curation, tangential curation, evidence
# sufficiency, readiness overview) cite studies with apa_in_text(), and the
# per-paper acquisition / deep-analysis console lines display apa_reference().
# They are pure string functions — no LLM call, no network call — so they add
# no measurable runtime cost. The instance methods _apa_in_text_citation() and
# _apa_reference_entry() delegate here so behaviour is identical project-wide.

_NAME_PARTICLES = {"de", "del", "della", "der", "den", "da", "di", "do", "dos",
                   "du", "la", "le", "van", "von", "ter", "ten", "bin", "al",
                   "st", "st.", "mac", "mc"}


def _is_initials_token(tok: str) -> bool:
    """True if a token is an initials block like 'JM', 'S.', 'F.A.', 'J-M' —
    i.e. 1-3 letters, all uppercase once punctuation is stripped."""
    t = re.sub(r"[^A-Za-z]", "", tok or "")
    return 1 <= len(t) <= 3 and t.isupper()


def _split_author_name(author: str):
    """Return (surname, [initial_letters]) for an author string in ANY common
    order: 'Surname, F. M.' | 'Given M. Surname' | 'Surname JM' (surname-first
    with a trailing initials block, as Europe PMC/PubMed often return) | 'JM
    Surname'. Lowercase nobiliary particles (de, van, der, la, ...) stay attached
    to the surname. This single parser is the source of truth for both the
    in-text surname and the reference-list 'Surname, F. M.' form, so a name like
    'Salotti JM' resolves to surname 'Salotti' (not 'JM')."""
    a = (author or "").strip()
    if not a:
        return "", []
    if "," in a:
        surname, _, rest = a.partition(",")
        inits = [c.upper() for c in re.findall(r"[A-Za-z]", rest)]
        return surname.strip(), inits
    parts = a.split()
    if len(parts) == 1:
        return parts[0], []

    def _initials_of(tok: str):
        s = re.sub(r"[^A-Za-z]", "", tok)
        if not s:
            return []
        # An initials block ('JM','SS','F') contributes every letter; a full given
        # name ('Jack','Suzanne') contributes only its first letter.
        return [c.upper() for c in s] if _is_initials_token(tok) else [s[0].upper()]

    # Case 1 — trailing initials block ("Salotti JM", "Cornet JF", "de la Monte SM").
    # Only when NOT every token looks like initials (else it is ambiguous).
    if _is_initials_token(parts[-1]) and not all(_is_initials_token(p) for p in parts):
        k = len(parts)
        while k > 0 and _is_initials_token(parts[k - 1]):
            k -= 1
        surname = " ".join(parts[:k])
        inits = []
        for p in parts[k:]:
            inits += _initials_of(p)
        return surname, inits
    # Case 2 — given-first ("Jack Kingdon", "J. M. Salotti", "Suzanne M. de la Monte").
    i = len(parts) - 1
    while i > 0 and parts[i - 1].lower().strip(".") in _NAME_PARTICLES:
        i -= 1
    surname = " ".join(parts[i:])
    inits = []
    for p in parts[:i]:
        inits += _initials_of(p)
    return surname, inits


def _apa_last_name(author: str) -> str:
    """Surname for an in-text citation. Handles 'Surname, F. M.', 'Given Surname',
    'Surname JM' (trailing initials), and lowercase nobiliary particles (de, van,
    der, la, von, ...) so e.g. 'Suzanne M. de la Monte' -> 'de la Monte' and
    'Salotti JM' -> 'Salotti'."""
    surname, _ = _split_author_name(author)
    return surname or (author or "").strip()


def apa_in_text(authors, year) -> str:
    """APA 7th in-text citation, e.g. '(Smith et al., 2020)'.

    1 author  -> (Smith, 2020)
    2 authors -> (Smith & Jones, 2020)
    3+ authors-> (Smith et al., 2020)
    Missing year -> 'n.d.'; missing authors -> '(Anonymous, year)'.
    """
    year = year or "n.d."
    authors = authors or []
    last_names = [_apa_last_name(a) for a in authors if a]
    if not last_names:
        return f"(Anonymous, {year})"
    if len(last_names) == 1:
        return f"({last_names[0]}, {year})"
    if len(last_names) == 2:
        return f"({last_names[0]} & {last_names[1]}, {year})"
    return f"({last_names[0]} et al., {year})"


def _apa_format_author_lastfirst(author: str) -> str:
    """One author in reference-list form: 'Surname, F. M.' (spaced initials).
    Uses the shared parser so 'Salotti JM' -> 'Salotti, J. M.' and
    'Alam SS' -> 'Alam, S. S.' rather than the inverted garble."""
    a = (author or "").strip()
    if not a:
        return ""
    if "," in a:
        return a
    surname, inits = _split_author_name(a)
    if not surname:
        return a
    if not inits:
        return surname
    return f"{surname}, " + " ".join(f"{c}." for c in inits)


def apa_reference(authors, year, title, venue=None, doi=None) -> str:
    """Best-effort APA 7th reference-list entry from available metadata.

    Note: PaperMetadata has no volume/issue/page fields, so the entry is APA-
    shaped but omits the 'Volume(Issue), pages' segment when unavailable.
    """
    year = year or "n.d."
    title = title or "Untitled"
    venue = venue or ""
    formatted = [_apa_format_author_lastfirst(a) for a in (authors or []) if a]
    formatted = [f for f in formatted if f]
    if not formatted:
        author_block = "Anonymous"
    elif len(formatted) == 1:
        author_block = formatted[0]
    elif len(formatted) <= 20:
        author_block = ", ".join(formatted[:-1]) + ", & " + formatted[-1]
    else:
        author_block = ", ".join(formatted[:19]) + ", ... " + formatted[-1]
    # Single terminal period (avoid doubling when the block already ends in '.').
    author_str = author_block if author_block.endswith(".") else author_block + "."

    entry = f"{author_str} ({year}). {title}."
    if venue:
        entry += f" {venue}."
    if doi:
        doi_clean = str(doi).strip()
        if doi_clean.lower().startswith("http"):
            entry += f" {doi_clean}"
        else:
            entry += f" https://doi.org/{doi_clean}"
    return entry


class ReviewState(TypedDict):
    original_query: str
    review_title: Optional[str]
    research_plan: Optional[Dict]
    focus_areas_completed: List[str]
    discovery_round: int
    max_discovery_rounds: int
    study_summaries: List[Dict]
    # IDs of every paper ever quick-read this run. Unlike study_summaries (which
    # the relevance filter and curation PRUNE), this set is never reduced, so a
    # paper dropped by the filter is not re-read on the next tangential round.
    read_paper_ids: List[str]
    study_analyses: List[Dict]
    methodology_assessment: Optional[Dict]
    # --- (#2) Methodology gap-fill (once-per-run, weakness-targeted) ---
    # If the comprehensive methodology assessment finds the evidence base too
    # weak to confidently answer the user's question AND identifies fillable
    # gaps, the pipeline runs ONE targeted gap-fill pass: one search per
    # weakness, acquire + quick-read + filter, deep-analyse ONLY the new studies
    # (previous verified quotes are preserved untouched), one more reference
    # harvest, then a fresh methodology assessment for the final review. Done at
    # most once per run, gated by methodology_gap_fill_done.
    methodology_gap_fill_done: bool
    methodology_gaps_significant: bool
    methodology_weaknesses: List[str]
    methodology_gap_queries: List[str]
    evidence_base_cache: Optional[str]
    # Maps each quote token id (e.g. "7" for [[Q7]]) to its verified text,
    # APA citation, source paper id/title and word count. Rebuilt in lockstep
    # with evidence_base_cache. Used to deterministically paste verified quotes
    # into the review at compile time and to validate placeholders.
    quote_registry: Dict[str, Dict]
    literature_review: Optional[str]
    locked_evidence: Optional[str]
    evidence_plan: Optional[Dict]
    self_review_issues: List[Dict]
    self_review_done: bool
    verification_attempts: int
    max_verification_attempts: int
    verification_passed: bool
    last_verification_issues: List[Dict]
    # --- Verification convergence machinery ---
    # _verification_mode: "holistic" for the first N full-rewrite passes, then
    #   "surgical" for one-issue-at-a-time fixes that leave everything else
    #   untouched. Set by node_verify_review, read by _route_verification.
    # verification_refinement_memory: per-attempt record of what was attempted
    #   and which issues resolved / persisted / newly appeared, fed back into the
    #   next fix prompt so the model can see its own progress and stop repeating
    #   mistakes.
    # _surgical_target: the single issue currently being fixed in surgical mode.
    # verification_exhausted: True only if the generous absolute cap is hit while
    #   issues remain (so the run can terminate, reported loudly as FAILED).
    _verification_mode: str
    verification_refinement_memory: List[Dict]
    _surgical_target: Optional[Dict]
    # Per-issue window-fix attempt counter (description -> count) so a window fix
    # that fails repeatedly escalates to a whole-document single-issue fix.
    _surgical_attempts: Dict[str, int]
    verification_exhausted: bool
    ready_to_write: bool
    final_output: Optional[str]
    interrupted: bool
    errors: List[str]
    # Search-strategy learning fields
    search_history: List[Dict]
    strategy_memo: str
    distillations_done: int
    # Tangential mode fields (per-engagement counters; reset on re-engagement)
    tangential_mode_active: bool
    tangential_engagement_count: int
    tangential_round_count: int
    stagnant_round_count: int
    all_study_summaries: List[Dict]
    filter_dropped_total: int
    curation_excluded_total: int
    executed_queries: List[str]
    tangential_papers_added: int
    tangential_distillations_done: int
    in_tangential_round: bool
    last_sufficiency_decision: Optional[str]
    sufficiency_reasoning: str
    # Curation fields
    curated_paper_ids: List[str]
    curation_history: List[Dict]
    rounds_since_last_curation: int
    tangential_rounds_since_last_curation: int
    # --- Run-bookkeeping fields used by the Phase 8b post-deep-analysis gate. ---
    # These MUST be declared here: LangGraph only persists channels that exist in
    # the state schema. Keys set in one node and read in a LATER node are silently
    # dropped if undeclared, which previously made the 35-minute time guard and
    # the retry cap non-functional (run_start_time/post_review_retries always read
    # their fallback). Declaring them makes them persist across the graph.
    run_start_time: float
    post_review_retries: int
    _post_review_action: Optional[str]
    _tangential_curated_count_this_engagement: int
    # Set by node_synthesize_review when zero studies reached deep analysis, so
    # the document in literature_review is a no-evidence REPORT rather than a
    # draft review. Read by node_self_review and node_verify_review to skip the
    # repair loop. Declared here for the same reason as the fields above: an
    # undeclared channel set in one node is dropped before the next one reads it.
    _no_evidence_report: bool
    # Set by node_post_deep_review when the verified evidence base is under
    # min_studies_for_review and the search budget is spent. Read by
    # node_synthesize_review. Declared for the same persistence reason as above.
    _insufficient_evidence_reason: Optional[str]


class AcademicReviewPipeline:

    # Matches a verified-quote placeholder token like [[Q7]] (group 1 = id).
    _PLACEHOLDER_RE = re.compile(r"\[\[Q(\d+)\]\]")
    # Matches a STUDY-SUMMARY placeholder token like [[S3]] (group 1 = study index).
    # Unlike a quote token, this expands to the study's frozen, grounded paraphrase
    # (the reviewer's summary used to INTRODUCE the study) as PLAIN PROSE — never in
    # quotation marks — and is NOT subject to verbatim source verification.
    _SUMMARY_RE = re.compile(r"\[\[S(\d+)\]\]")

    def __init__(self):
        self.config = get_research_config()
        self.paths = get_paths_config()
        # Low-end device mode (chunked execution for a small machine). When off
        # (default) every stage runs exactly as on the big machine.
        self._low_end = is_low_end_enabled()
        self._low_end_cfg = get_low_end_config()
        for d in [self.paths.get("papers_directory", "Papers"),
                  self.paths.get("logs_directory", "Logs"),
                  self.paths.get("output_directory", "Reviews")]:
            os.makedirs(d, exist_ok=True)

        print(f"\n{Fore.CYAN}Initializing Academic Literature Review System...{Style.RESET_ALL}")
        self.agent_manager = LLMManager()
        self.discovery = PaperDiscoveryEngine()
        self.planner = ResearchPlanner(self.agent_manager)
        self.study_analyzer = StudyAnalyser(self.agent_manager)
        self.reference_harvester = ReferenceHarvester(
            self.discovery, self.agent_manager, self.study_analyzer, self.config)

        # Attribution-check helpers — separate doc store so we don't interfere
        # with the StudyAnalyser's doc store during deep_analysis.
        self._attr_doc_store = DocumentStore() if HAS_DOC_STORE else None
        self._attr_text_matcher = TextMatcher() if HAS_TEXT_MATCHER else None

        self._interrupted = False
        self._review_counter = 0
        signal.signal(signal.SIGINT, self._handle_interrupt)
        self.graph = self._build_graph()
        if self._low_end:
            print(f"{Fore.MAGENTA}LOW-END DEVICE MODE: ON — chunked deep-analysis "
                  f"and map-reduce synthesis; LLM context capped at "
                  f"{self.config.get('low_end_n_ctx', 16384)//1024}K.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}System ready.{Style.RESET_ALL}")

    def _handle_interrupt(self, signum, frame):
        if self._interrupted:
            print(f"\n{Fore.RED}Force exit.{Style.RESET_ALL}")
            sys.exit(1)
        self._interrupted = True
        request_interrupt()
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}Interrupt requested — "
              f"aborting active LLM call...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}(Press Ctrl-C again to force-exit immediately){Style.RESET_ALL}")

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================

    def _build_graph(self) -> Any:
        g = StateGraph(ReviewState)
        g.add_node("setup_review", self.node_setup_review)
        g.add_node("plan_research", self.node_plan_research)
        g.add_node("search_and_select", self.node_search_and_select)
        g.add_node("acquire_texts", self.node_acquire_texts)
        g.add_node("quick_read_studies", self.node_quick_read_studies)
        g.add_node("filter_relevance", self.node_filter_relevance)
        g.add_node("gap_analysis", self.node_gap_analysis)
        g.add_node("distill_strategy", self.node_distill_strategy)
        g.add_node("refine_plan", self.node_refine_plan)
        g.add_node("curate_evidence", self.node_curate_evidence)
        g.add_node("evidence_sufficiency", self.node_evidence_sufficiency)
        g.add_node("refine_plan_tangential", self.node_refine_plan_tangential)
        g.add_node("tangential_gap_analysis", self.node_tangential_gap_analysis)
        g.add_node("tangential_curate_evidence", self.node_tangential_curate_evidence)
        g.add_node("deep_analysis", self.node_deep_analysis)
        g.add_node("harvest_references", self.node_harvest_references)
        g.add_node("assess_methodology", self.node_assess_methodology)
        g.add_node("methodology_gap_fill", self.node_methodology_gap_fill)
        g.add_node("post_deep_review", self.node_post_deep_review)
        g.add_node("synthesize_review", self.node_synthesize_review)
        g.add_node("self_review", self.node_self_review)
        g.add_node("self_fix", self.node_self_fix)
        g.add_node("verify_review", self.node_verify_review)
        g.add_node("surgical_fix", self.node_surgical_fix)
        g.add_node("compile_output", self.node_compile_output)

        g.set_entry_point("setup_review")
        g.add_edge("setup_review", "plan_research")
        g.add_edge("plan_research", "search_and_select")
        g.add_edge("search_and_select", "acquire_texts")
        g.add_edge("acquire_texts", "quick_read_studies")
        g.add_edge("quick_read_studies", "filter_relevance")
        g.add_edge("filter_relevance", "gap_analysis")

        g.add_conditional_edges("gap_analysis", self._route_gap,
                                {"curate": "curate_evidence",
                                 "distill": "distill_strategy",
                                 "refine": "refine_plan",
                                 "write": "curate_evidence"})

        g.add_edge("distill_strategy", "refine_plan")
        g.add_edge("refine_plan", "search_and_select")

        g.add_edge("curate_evidence", "evidence_sufficiency")

        g.add_conditional_edges("evidence_sufficiency", self._route_sufficiency,
                                {"sufficient": "deep_analysis",
                                 "bad_picks": "refine_plan",
                                 "sparse": "refine_plan_tangential"})

        g.add_edge("refine_plan_tangential", "search_and_select")
        g.add_conditional_edges("tangential_gap_analysis", self._route_tangential_gap,
                                {"curate": "tangential_curate_evidence",
                                 "distill": "distill_strategy",
                                 "refine": "refine_plan_tangential"})

        g.add_conditional_edges("tangential_curate_evidence", self._route_post_tangential_curate,
                                {"continue": "refine_plan_tangential",
                                 "exit": "deep_analysis"})

        g.add_edge("deep_analysis", "harvest_references")
        g.add_edge("harvest_references", "assess_methodology")
        g.add_conditional_edges("assess_methodology", self._route_methodology,
                                {"gap_fill": "methodology_gap_fill",
                                 "proceed": "post_deep_review"})
        # After the one-time gap-fill, return to the methodology assessment so the
        # final review's methodology section reflects the enlarged evidence base.
        # gap_fill sets methodology_gap_fill_done=True, so the route then proceeds.
        g.add_edge("methodology_gap_fill", "assess_methodology")
        g.add_conditional_edges("post_deep_review", self._route_post_deep_review,
                                {"research": "refine_plan",
                                 "proceed": "synthesize_review"})
        g.add_edge("synthesize_review", "self_review")
        g.add_conditional_edges("self_review", self._route_self_review,
                                {"fix": "self_fix", "skip": "verify_review"})
        g.add_edge("self_fix", "verify_review")
        g.add_conditional_edges("verify_review", self._route_verification,
                                {"holistic": "synthesize_review",
                                 "surgical": "surgical_fix",
                                 "accept": "compile_output"})
        g.add_edge("surgical_fix", "verify_review")
        g.add_edge("compile_output", END)
        return g.compile()

    # =========================================================================
    # ROUTING
    # =========================================================================

    def _route_gap(self, state: ReviewState) -> str:
        if state.get("interrupted"):
            return "write"
        if state.get("discovery_round", 0) >= state.get("max_discovery_rounds", 15):
            return "write"

        if state.get("tangential_mode_active"):
            return self._route_tangential_gap(state)

        # HARD MINIMUM SEARCH ROUNDS — mirror of the guard in node_gap_analysis.
        # Below the floor we never curate (and therefore never reach the
        # sufficiency check or writing); we still allow distillation so search
        # strategy keeps improving between the early rounds.
        min_rounds = int(self.config.get("min_rounds_before_readiness", 3))
        below_min_rounds = state.get("discovery_round", 0) < min_rounds

        curate_every = self.config.get("curate_every_n_rounds", 5)
        rounds_since = state.get("rounds_since_last_curation", 0)
        if not below_min_rounds and (state.get("ready_to_write") or rounds_since >= curate_every):
            return "curate"

        first_distill = self.config.get("first_distill_after_round", 1)
        distill_every = self.config.get("distill_every_n_rounds", 1)
        rounds = state.get("discovery_round", 0)
        distills = state.get("distillations_done", 0)
        if distills == 0:
            if rounds >= first_distill:
                return "distill"
        else:
            rounds_since_first = rounds - first_distill
            if rounds_since_first >= 0 and (rounds_since_first % distill_every == 0):
                expected = 1 + (rounds_since_first // distill_every)
                if distills < expected:
                    return "distill"

        return "refine"

    def _route_tangential_gap(self, state: ReviewState) -> str:
        if state.get("interrupted"):
            return "curate"

        tang_round = state.get("tangential_round_count", 0)
        tang_papers = state.get("tangential_papers_added", 0)
        round_cap = self.config.get("tangential_round_cap", 100)
        paper_target = self.config.get("tangential_paper_target", 75)
        curate_every = self.config.get("tangential_curate_every_n_rounds", 5)
        rounds_since = state.get("tangential_rounds_since_last_curation", 0)

        if tang_round >= round_cap or tang_papers >= paper_target:
            return "curate"

        if state.get("ready_to_write") or rounds_since >= curate_every:
            return "curate"

        first_distill = self.config.get("first_distill_after_round", 1)
        distill_every = self.config.get("distill_every_n_rounds", 1)
        tang_distills = state.get("tangential_distillations_done", 0)
        if tang_distills == 0:
            if tang_round >= first_distill:
                return "distill"
        else:
            rounds_since_first = tang_round - first_distill
            if rounds_since_first >= 0 and (rounds_since_first % distill_every == 0):
                expected = 1 + (rounds_since_first // distill_every)
                if tang_distills < expected:
                    return "distill"

        return "refine"

    def _route_sufficiency(self, state: ReviewState) -> str:
        if state.get("interrupted"):
            return "sufficient"
        if state.get("discovery_round", 0) >= state.get("max_discovery_rounds", 15):
            return "sufficient"

        decision = state.get("last_sufficiency_decision", "sufficient")

        # ---- TERMINATION GUARDS --------------------------------------------
        # This is the only reachable exit from the tangential loop, so every
        # cap has to be checked HERE. Each guard below stops the run and moves
        # to synthesis with whatever evidence exists — a thin review that
        # finishes beats a perfect one that never does.
        if decision in ("sparse_direct_evidence", "bad_picks"):
            stop = self._tangential_stop_reason(state)
            if stop:
                print(f"\n  {Fore.YELLOW}{Style.BRIGHT}STOPPING TANGENTIAL SEARCH — "
                      f"{stop}{Style.RESET_ALL}")
                print(f"  {Fore.WHITE}Proceeding to synthesis with "
                      f"{len(state.get('study_summaries') or [])} curated study(ies) "
                      f"from a catalog of {len(self.discovery.paper_catalog)} papers."
                      f"{Style.RESET_ALL}")
                return "sufficient"

        if decision == "sparse_direct_evidence":
            return "sparse"
        if decision == "bad_picks":
            return "bad_picks"
        return "sufficient"

    def _tangential_stop_reason(self, state: ReviewState) -> Optional[str]:
        """Return a human-readable reason to stop searching, or None to continue.

        Checked on every sufficiency decision because the routers that used to
        own these limits are attached to an unreachable node.
        """
        cfg = self.config
        tang_round = state.get("tangential_round_count", 0)
        round_cap = int(cfg.get("tangential_round_cap", 8))
        if tang_round >= round_cap:
            return (f"reached the tangential round cap ({tang_round}/{round_cap}). "
                    f"Raise 'tangential_round_cap' for a longer sweep.")

        engagements = state.get("tangential_engagement_count", 0)
        max_eng = int(cfg.get("tangential_max_engagements", 2))
        if engagements > max_eng:
            return (f"reached the tangential engagement limit "
                    f"({engagements}/{max_eng}).")

        # Stagnation: rounds that add no new papers cannot change any later
        # decision, so repeating them is pure waste. In the observed failure the
        # last ten rounds each added zero papers and re-ran identical queries.
        stagnant = state.get("stagnant_round_count", 0)
        max_stagnant = int(cfg.get("max_stagnant_rounds", 3))
        if stagnant >= max_stagnant:
            return (f"{stagnant} consecutive round(s) found no new papers — "
                    f"the searches are returning only papers already in the "
                    f"catalog, so further rounds cannot change the outcome.")

        total_rounds = (state.get("discovery_round", 0)
                        + state.get("tangential_round_count", 0))
        hard_cap = int(cfg.get("absolute_round_cap", 40))
        if total_rounds >= hard_cap:
            return f"hit the absolute round cap ({total_rounds}/{hard_cap})."

        started = state.get("run_start_time")
        budget_min = float(cfg.get("search_time_budget_minutes", 0) or 0)
        if started and budget_min > 0:
            elapsed = (time.time() - started) / 60.0
            if elapsed >= budget_min:
                return (f"exceeded the search time budget "
                        f"({elapsed:.0f} of {budget_min:.0f} minutes).")
        return None

    def _route_post_tangential_curate(self, state: ReviewState) -> str:
        if state.get("interrupted"):
            return "exit"

        min_papers = self.config.get("tangential_min_curated_papers", 15)
        round_cap = self.config.get("tangential_round_cap", 100)
        paper_target = self.config.get("tangential_paper_target", 75)

        tang_curated_count = state.get("_tangential_curated_count_this_engagement", 0)
        tang_round = state.get("tangential_round_count", 0)
        tang_papers = state.get("tangential_papers_added", 0)

        if tang_round >= round_cap or tang_papers >= paper_target:
            if tang_curated_count < min_papers:
                print(f"\n  {Fore.YELLOW}{Style.BRIGHT}Curation retained only {tang_curated_count} papers "
                      f"(threshold: {min_papers}).{Style.RESET_ALL}")
                print(f"  {Fore.MAGENTA}{Style.BRIGHT}Re-engaging tangential mode to gather more "
                      f"indirect evidence.{Style.RESET_ALL}")
                print(f"  {Fore.WHITE}Previously curated tangential papers are preserved as your "
                      f"existing selections.{Style.RESET_ALL}\n")
                state["tangential_round_count"] = 0
                state["tangential_papers_added"] = 0
                state["tangential_distillations_done"] = 0
                state["tangential_rounds_since_last_curation"] = 0
                state["_tangential_curated_count_this_engagement"] = 0
                state["tangential_engagement_count"] = state.get("tangential_engagement_count", 1) + 1
                return "continue"
            return "exit"

        if state.get("ready_to_write"):
            if tang_curated_count < min_papers:
                print(f"\n  {Fore.YELLOW}{Style.BRIGHT}Curation retained only {tang_curated_count} papers "
                      f"(threshold: {min_papers}).{Style.RESET_ALL}")
                print(f"  {Fore.MAGENTA}{Style.BRIGHT}Re-engaging tangential mode.{Style.RESET_ALL}\n")
                state["tangential_round_count"] = 0
                state["tangential_papers_added"] = 0
                state["tangential_distillations_done"] = 0
                state["tangential_rounds_since_last_curation"] = 0
                state["_tangential_curated_count_this_engagement"] = 0
                state["tangential_engagement_count"] = state.get("tangential_engagement_count", 1) + 1
                state["ready_to_write"] = False
                return "continue"
            return "exit"

        return "continue"

    def _route_self_review(self, state: ReviewState) -> str:
        if state.get("interrupted"):
            return "skip"
        # Once the verification loop has started (a verify pass has run), the
        # verification node owns all fixing (holistic + surgical). Routing back
        # into self_fix here would act on STALE first-pass self-review issues, so
        # we skip straight to verification on every rewrite.
        if state.get("verification_attempts", 0) > 0:
            return "skip"
        issues = state.get("self_review_issues", [])
        actionable = [i for i in issues
                      if i.get("severity") in ("critical", "moderate")]
        return "fix" if actionable else "skip"

    def _route_verification(self, state: ReviewState) -> str:
        if state.get("interrupted") or state.get("verification_passed", True):
            return "accept"
        return "surgical" if state.get("_verification_mode") == "surgical" else "holistic"

    # =========================================================================
    # UI HELPERS
    # =========================================================================

    def _print_phase_banner(self, title: str, color=Fore.CYAN):
        print(f"\n{color}{'='*60}\n {title}\n{'='*60}{Style.RESET_ALL}")

    def _print_stage_banner(self, title: str, color=Fore.MAGENTA):
        print(f"\n{color}{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}{Style.RESET_ALL}")

    # =========================================================================
    # NODE: SETUP
    # =========================================================================

    def _route_post_deep_review(self, state: ReviewState) -> str:
        return "research" if state.get("_post_review_action") == "research" else "proceed"

    def _route_methodology(self, state: ReviewState) -> str:
        """After the methodology assessment, decide whether to run the ONE-TIME
        targeted gap-fill pass (#2). Routes to 'gap_fill' only if: gap-fill is
        enabled, it has not already run this run, the assessment flagged
        significant fillable gaps, there is at least one gap query, an interrupt
        is not pending, and the run-time budget is not already spent. Otherwise
        proceeds to the existing post-deep-analysis gate."""
        if state.get("interrupted"):
            return "proceed"
        if not self.config.get("methodology_gap_fill_enabled", True):
            return "proceed"
        if state.get("methodology_gap_fill_done"):
            return "proceed"
        if not state.get("methodology_gaps_significant"):
            return "proceed"
        if not state.get("methodology_gap_queries"):
            return "proceed"
        max_minutes = float(self.config.get("post_review_max_minutes", 35))
        elapsed_min = (time.time() - state.get("run_start_time", time.time())) / 60.0
        if elapsed_min >= max_minutes:
            print(f"  {Fore.YELLOW}Methodology gaps found, but the {max_minutes:.0f}m time "
                  f"budget is reached — skipping gap-fill and proceeding.{Style.RESET_ALL}")
            state["methodology_gap_fill_done"] = True
            return "proceed"
        return "gap_fill"

    def node_setup_review(self, state: ReviewState) -> ReviewState:
        self._review_counter += 1
        placeholder = f"Review_{self._review_counter}"
        state["review_title"] = placeholder
        print(f"  {Fore.WHITE}Starting: {Fore.CYAN}{placeholder}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}(Proper title will be generated after the review is written){Style.RESET_ALL}")
        papers_subdir = os.path.join(self.paths.get("papers_directory", "Papers"), placeholder)
        os.makedirs(papers_subdir, exist_ok=True)
        self.discovery.papers_dir = papers_subdir
        state["run_start_time"] = time.time()
        state["post_review_retries"] = 0
        return state

    # =========================================================================
    # NODE: PLAN
    # =========================================================================

    def node_plan_research(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 1: RESEARCH PLANNING")
        plan = self.planner.create_initial_plan(state["original_query"])
        if plan:
            state["research_plan"] = plan.to_dict()
        else:
            state["errors"].append("Planning failed")
        return state

    # =========================================================================
    # NODE: SEARCH + SELECT (strict selection with scope + per-paper reasoning)
    # =========================================================================

    def node_search_and_select(self, state: ReviewState) -> ReviewState:
        rnd = state.get("discovery_round", 0)
        is_tang = state.get("in_tangential_round", False) or state.get("tangential_mode_active", False)
        mode = "TANGENTIAL" if is_tang else "STANDARD"
        color = Fore.MAGENTA if is_tang else Fore.CYAN

        if is_tang:
            tang_round = state.get("tangential_round_count", 0) + 1
            engagement = state.get("tangential_engagement_count", 1)
            self._print_phase_banner(
                f"PHASE 2: SEARCH & SELECT — Tangential Engagement {engagement}, Round {tang_round}",
                color=color)
        else:
            self._print_phase_banner(f"PHASE 2: SEARCH & SELECT — Round {rnd+1} [{mode}]", color=color)

        plan_data = state.get("research_plan", {})
        focus_areas = plan_data.get("focus_areas", [])

        if "search_history" not in state or state["search_history"] is None:
            state["search_history"] = []

        identified_scope = (plan_data or {}).get("identified_scope", "")

        for fa_data in focus_areas:
            if self._interrupted:
                break

            area = fa_data.get("area", "?")
            query = fa_data.get("search_query", "")
            priority = fa_data.get("priority", 3)

            if not query:
                continue

            # ---- DUPLICATE QUERY SUPPRESSION -------------------------------
            # The tangential planner regenerates the same phrasing round after
            # round ("stimulant induced hypertension left ventricular
            # hypertrophy mechanism" ran eight times in one observed run). The
            # databases are deterministic, so a repeated query returns papers
            # already in the catalog and burns a full search + selection call
            # for nothing. Skip verbatim repeats and let the planner know.
            qkey = re.sub(r'[^a-z0-9]+', ' ', query.lower()).strip()
            executed = state.get("executed_queries") or []
            if qkey and qkey in executed:
                print(f"\n  {color}P{priority}: {area}{Style.RESET_ALL}")
                print(f"  {Fore.YELLOW}  {SYM_SEARCH} {query}{Style.RESET_ALL}")
                print(f"  {Fore.YELLOW}    Already searched this exact query earlier — "
                      f"skipping (it would return the same papers).{Style.RESET_ALL}")
                state["search_history"].append({
                    "round": (f"T{state.get('tangential_round_count', 0) + 1}"
                              if is_tang else f"R{rnd + 1}"),
                    "mode": "tangential" if is_tang else "standard",
                    "query": query, "focus_area": area,
                    "total_candidates": 0, "selected_count": 0,
                    "outcome": "skipped_duplicate_query",
                    "candidate_titles": [], "selected_titles": [],
                    "selection_reasoning": "Skipped: identical query already run.",
                })
                continue
            state["executed_queries"] = executed + [qkey]

            print(f"\n  {color}P{priority}: {area}{Style.RESET_ALL}")
            print(f"  {Fore.BLUE}  {SYM_SEARCH} {query}{Style.RESET_ALL}")

            search_results = self.discovery.search_all_apis(query, limit_per_api=20)
            self.planner.record_query_used(query)
            self.planner.record_focus_area_used(area)

            current_round_label = (f"T{state.get('tangential_round_count', 0) + 1}"
                                    if is_tang else f"R{rnd + 1}")

            history_entry = {
                "round": current_round_label,
                "mode": mode.lower(),
                "engagement": state.get("tangential_engagement_count", 1) if is_tang else None,
                "focus_area": area,
                "priority": priority,
                "query": query,
                "total_candidates": len(search_results),
                "candidate_titles": [
                    {"title": p.title, "year": p.year,
                     "venue": p.venue or "", "citations": p.citation_count}
                    for p in search_results
                ],
                "selected_indices": [],
                "selected_count": 0,
                "selection_reasoning": "",
                "per_paper_reasoning": [],
            }

            if not search_results:
                print(f"  {Fore.YELLOW}  No results found.{Style.RESET_ALL}")
                history_entry["outcome"] = "no_results"
                state["search_history"].append(history_entry)
                continue

            print(f"  {Fore.WHITE}  {len(search_results)} unique papers found. LLM selecting...{Style.RESET_ALL}")

            if is_tang:
                selected, per_paper_reasoning = self._llm_select_papers_tangential(
                    state["original_query"], area, search_results,
                    identified_scope=identified_scope)
            else:
                selected, per_paper_reasoning = self._llm_select_papers(
                    state["original_query"], area, search_results,
                    identified_scope=identified_scope)

            if per_paper_reasoning:
                history_entry["selection_reasoning"] = "Per-paper reasoning: " + \
                    "; ".join(per_paper_reasoning[:5])
            else:
                history_entry["selection_reasoning"] = ""
            history_entry["per_paper_reasoning"] = per_paper_reasoning
            history_entry["selected_count"] = len(selected)
            history_entry["selected_titles"] = [
                {"title": p.title, "year": p.year} for p in selected]
            history_entry["outcome"] = "selected" if selected else "none_relevant"

            if selected:
                added = self.discovery.add_selected_papers(selected)
                print(f"  {Fore.GREEN}  {SYM_CHECK} LLM selected {len(selected)} papers "
                      f"({added} new to catalog){Style.RESET_ALL}")
                for sp_i, sp in enumerate(selected):
                    print(f"  {Fore.WHITE}    -> [{sp.year or '?'}] {sp.title}{Style.RESET_ALL}")
                    if sp_i < len(per_paper_reasoning):
                        reasoning_text = per_paper_reasoning[sp_i]
                        if reasoning_text:
                            label = "Intended use" if is_tang else "Reason"
                            line_color = Fore.MAGENTA if is_tang else Fore.CYAN
                            print(f"  {line_color}        {label}: {reasoning_text}{Style.RESET_ALL}")

                if is_tang:
                    state["tangential_papers_added"] = state.get("tangential_papers_added", 0) + added
                state["_papers_added_this_round"] = \
                    state.get("_papers_added_this_round", 0) + added
            else:
                print(f"  {Fore.YELLOW}  LLM found no relevant papers in results.{Style.RESET_ALL}")

            state["search_history"].append(history_entry)

        state["focus_areas_completed"] = state.get("focus_areas_completed", []) + \
            [fa.get("area", "?") for fa in focus_areas]

        # ---- STAGNATION TRACKING -------------------------------------------
        # A round that adds nothing to the catalog cannot change any downstream
        # decision. Counting these is what lets the run stop instead of
        # re-running the same searches for hours.
        added_this_round = state.get("_papers_added_this_round", 0)
        if added_this_round > 0:
            state["stagnant_round_count"] = 0
        else:
            state["stagnant_round_count"] = state.get("stagnant_round_count", 0) + 1
            limit = int(self.config.get("max_stagnant_rounds", 3))
            print(f"  {Fore.YELLOW}No new papers this round "
                  f"({state['stagnant_round_count']}/{limit} stagnant)."
                  f"{Style.RESET_ALL}")
        state["_papers_added_this_round"] = 0

        if is_tang:
            state["tangential_round_count"] = state.get("tangential_round_count", 0) + 1
            state["tangential_rounds_since_last_curation"] = \
                state.get("tangential_rounds_since_last_curation", 0) + 1
            state["in_tangential_round"] = False
        else:
            state["discovery_round"] = rnd + 1
            state["rounds_since_last_curation"] = state.get("rounds_since_last_curation", 0) + 1

        state["interrupted"] = self._interrupted
        total = len(self.discovery.paper_catalog)
        print(f"\n  {Fore.GREEN}Catalog total: {total} papers{Style.RESET_ALL}")
        if is_tang:
            print(f"  {Fore.MAGENTA}Tangential engagement {state.get('tangential_engagement_count',1)}: "
                  f"{state.get('tangential_papers_added',0)} papers added, "
                  f"round {state.get('tangential_round_count',0)}/{self.config.get('tangential_round_cap',100)}"
                  f"{Style.RESET_ALL}")
        return state

    # =========================================================================
    # SELECTION INTEGRITY — deterministic index/title cross-check
    # =========================================================================
    # The model is asked to echo each selected paper's title verbatim alongside
    # its index. The echoed title is then checked against the paper sitting at
    # that index. This catches the failure mode where a model emits 0-based
    # indices (or otherwise mis-numbers), which silently substituted the
    # NEIGHBOURING paper for the one it actually reasoned about and flooded the
    # catalog with unrelated studies. Nothing here changes WHICH papers the
    # model may choose — it only guarantees the paper added is the paper meant.

    SELECTION_TITLE_MATCH_MIN = 0.75

    @staticmethod
    def _norm_title_for_match(t: str) -> str:
        # Papers are listed to the model as "12. [2019] Title...", so it often
        # copies the bracketed year into its echo. Strip a leading year marker
        # before comparing, otherwise short titles can fall under the match
        # threshold and get dropped for no good reason.
        t = re.sub(r'^\s*\[?\s*(19|20)\d{2}\s*\]?\s*', '', (t or ""))
        return re.sub(r'[^a-z0-9]+', ' ', t.lower()).strip()

    @classmethod
    def _title_similarity(cls, a: str, b: str) -> float:
        na, nb = cls._norm_title_for_match(a), cls._norm_title_for_match(b)
        if not na or not nb:
            return 0.0
        if na == nb:
            return 1.0
        return difflib.SequenceMatcher(None, na, nb).ratio()

    def _resolve_selections(self, selections, papers, reason_key, label):
        """Map raw LLM selections onto papers, verifying index against title.

        Returns (selected_papers, reasons). Emits a terminal notice for every
        correction or drop so mis-numbering is visible instead of silent.
        """
        selected_papers, reasons = [], []
        seen_ids = set()
        corrected = dropped = unverified = 0

        for sel in selections or []:
            if not isinstance(sel, dict):
                continue
            idx = sel.get("paper_index")
            if isinstance(idx, str) and idx.strip().isdigit():
                idx = int(idx.strip())
            reason = sel.get(reason_key, "") or sel.get("reasoning", "") or ""
            echoed = (sel.get("title") or "").strip()

            by_index = None
            if isinstance(idx, int) and 1 <= idx <= len(papers):
                by_index = papers[idx - 1]

            chosen = None
            if not echoed:
                # No echo to check against — fall back to the index alone.
                chosen = by_index
                if chosen is not None:
                    unverified += 1
            else:
                sim_idx = self._title_similarity(echoed, by_index.title) if by_index else 0.0
                if sim_idx >= self.SELECTION_TITLE_MATCH_MIN:
                    chosen = by_index
                else:
                    best, best_sim = None, 0.0
                    for p in papers:
                        sim = self._title_similarity(echoed, p.title)
                        if sim > best_sim:
                            best, best_sim = p, sim
                    if best is not None and best_sim >= self.SELECTION_TITLE_MATCH_MIN:
                        chosen = best
                        corrected += 1
                        print(f"  {Fore.YELLOW}    ! index {idx} pointed at "
                              f"\"{(by_index.title if by_index else 'nothing')[:60]}\" but the "
                              f"model named \"{echoed[:60]}\" — using the named paper."
                              f"{Style.RESET_ALL}")
                    else:
                        dropped += 1
                        print(f"  {Fore.YELLOW}    ! dropped a selection: index {idx} and "
                              f"title \"{echoed[:60]}\" do not match any candidate."
                              f"{Style.RESET_ALL}")
                        continue

            if chosen is None:
                dropped += 1
                continue
            if chosen.paper_id in seen_ids:
                continue
            seen_ids.add(chosen.paper_id)
            selected_papers.append(chosen)
            reasons.append(reason)

        if corrected or dropped:
            print(f"  {Fore.YELLOW}    Selection integrity ({label}): "
                  f"{corrected} corrected, {dropped} dropped.{Style.RESET_ALL}")
        if unverified:
            print(f"  {Fore.YELLOW}    {unverified} selection(s) had no title echo — "
                  f"accepted on index alone (unverified).{Style.RESET_ALL}")
        return selected_papers, reasons

    def _llm_select_papers(self, query: str, focus_area: str,
                           papers: List[PaperMetadata],
                           identified_scope: str = ""):
        """
        Strict question-anchored selection for standard mode.
        Returns: (selected_papers, per_paper_reasoning_list)
        """
        paper_lines = []
        for i, p in enumerate(papers):
            abstract_snippet = ""
            if p.abstract:
                abstract_snippet = p.abstract[:500]
                if len(p.abstract) > 500:
                    abstract_snippet += "..."
            paper_lines.append(
                f"{i+1}. [{p.year or '?'}] {p.title}\n"
                f"   Venue: {p.venue or '?'} | Citations: {p.citation_count or '?'}\n"
                f"   Abstract: {abstract_snippet or 'No abstract'}")
        papers_text = "\n".join(paper_lines)

        scope_block = ""
        if identified_scope:
            scope_block = f"""

IDENTIFIED SCOPE OF THE QUESTION (from the initial plan — selections must
stay within this scope):
{identified_scope}
"""

        prompt = f"""You are selecting academic papers for a literature review. Your job is to
pick papers that genuinely contribute to answering the USER'S QUESTION below.

USER'S RESEARCH QUESTION: "{query}"
{scope_block}
CURRENT FOCUS AREA (this is a sub-aspect of the user's question): "{focus_area}"

================ THE SELECTION TEST ================

For EACH paper, ask yourself:

  Will this paper help answer the user's question above — or provide
  essential context that genuinely informs an answer to it?

This question — not the focus area — is your anchor. A later curation step
will pick the best ones for the final review. But do NOT include adjacent-
topic papers just because they share a mechanism, neighbour the topic, or
appear in similar literature. The question is what defines relevance — not
topical proximity.

================ THE PAPERS ================

{papers_text}

================ OUTPUT FORMAT ================

REMINDER — the question every selection must serve is:
  "{query}"

For EACH paper you INCLUDE, provide a one-or-two-sentence justification of
how it informs the user's question.

NUMBERING RULE (critical): the papers above are numbered starting at 1. Use
that exact printed number as "paper_index". Do NOT count from zero. You must
also copy the paper's title EXACTLY as printed above into "title" — the title
is cross-checked against the index, and any selection whose title and index
disagree is discarded.

Respond with ONLY JSON (NO overall_reasoning field — only per-paper reasons):
{{
    "selections": [
        {{"paper_index": 1, "title": "exact title of paper 1, copied verbatim", "reasoning": "specific reason naming what this paper contributes to answering the user's question"}},
        {{"paper_index": 3, "title": "exact title of paper 3, copied verbatim", "reasoning": "specific reason naming what this paper contributes to answering the user's question"}}
    ]
}}

If genuinely NONE of these papers directly inform the user's question, respond:
{{"selections": []}}"""

        result = self.agent_manager.run_primary(
            prompt, as_json=True, task="paper_selection")

        selected_papers = []
        per_paper_reasoning = []
        if result.success and result.json_response:
            selections = result.json_response.get("selections", [])
            selected_papers, per_paper_reasoning = self._resolve_selections(
                selections, papers, reason_key="reasoning", label="standard")
        else:
            error_info = result.error or "Unknown error"
            print(f"  {Fore.RED}  Selection failed: {error_info}{Style.RESET_ALL}")
            if result.response:
                print(f"  {Fore.RED}  Raw response: {result.response[:300]}{Style.RESET_ALL}")

        return selected_papers, per_paper_reasoning

    def _llm_select_papers_tangential(self, query: str, focus_area: str,
                                       papers: List[PaperMetadata],
                                       identified_scope: str = ""):
        """
        Tangential-mode selection — each pick requires intended-use reasoning.
        Returns: (selected_papers, per_paper_intended_uses)

        CHANGED: now enforces THREE HARD GATES (domain anchor, one-hop
        connection, domain-expert test) with worked good/bad examples so the
        model selects genuinely tangential — not unrelated — papers.
        """
        paper_lines = []
        for i, p in enumerate(papers):
            abstract_snippet = ""
            if p.abstract:
                abstract_snippet = p.abstract[:500]
                if len(p.abstract) > 500:
                    abstract_snippet += "..."
            paper_lines.append(
                f"{i+1}. [{p.year or '?'}] {p.title}\n"
                f"   Venue: {p.venue or '?'} | Citations: {p.citation_count or '?'}\n"
                f"   Abstract: {abstract_snippet or 'No abstract'}")
        papers_text = "\n".join(paper_lines)

        scope_block = ""
        if identified_scope:
            scope_block = f"""

ORIGINAL QUESTION'S IDENTIFIED SCOPE:
{identified_scope}

In tangential mode you may step outside the narrowest reading of this scope to
gather indirect evidence, but every selection must remain in the SAME
real-world subject domain as the question and inform an answer within a single
inferential step.
"""

        prompt = f"""You are in TANGENTIAL MODE selecting papers for a literature review.

USER'S RESEARCH QUESTION: "{query}"
{scope_block}
CURRENT TANGENTIAL FOCUS AREA: "{focus_area}"

You are searching INDIRECT routes because direct evidence on the user's
specific question is sparse. The papers below were returned from an
indirect-route search.

================ WHAT "TANGENTIAL" MEANS HERE ================

Tangential means a CLOSE NEIGHBOUR of the question that a researcher IN THE
QUESTION'S OWN FIELD would still recognise as relevant evidence. It does NOT
mean "any paper I can connect to the topic through a clever chain of
reasoning." The most common error is taking a generic physical, chemical,
statistical, or engineering concept that merely shares a WORD with the
question (e.g. "pressure", "contact", "transfer", "area", "exposure") and
building an analogy chain to the topic. That is UNRELATED, not tangential.
Do NOT do this.

================ THREE HARD GATES — A PAPER MUST PASS ALL THREE ============

GATE 1 — DOMAIN ANCHOR:
The paper must belong to the SAME real-world subject domain as the question.
If the question is about human personal-hygiene behaviour, the paper must be
about human hygiene behaviour, bathroom/toileting practices, or directly
related human health behaviour — NOT industrial surfaces, machine tribology,
chemical/occupational exposure modelling, or abstract physics, even if those
share vocabulary like "contact pressure" or "surface area".

GATE 2 — ONE-HOP CONNECTION:
The link from the paper's ACTUAL findings to the original question must be a
SINGLE, DIRECT inferential step. If explaining the relevance needs a chain
("this measures X, which could model Y, which lets us infer Z, which relates
to the question"), the paper FAILS. If your justification stacks the words
"model", "infer", "estimate", or "extrapolate", reject the paper.

GATE 3 — DOMAIN-EXPERT TEST:
Would a researcher who actually studies the question's topic say "yes, that's
relevant adjacent evidence" — or "that has nothing to do with my field"? If
the latter, reject it.

================ WORKED EXAMPLES ================

GOOD tangential (passes all gates) — question on a medication's effect on
sleep in adults:
  - a study on the drug's CLASS and sleep architecture (class-level, same domain)
  - a validated insomnia self-report measure in adults (proxy outcome, same field)

BAD over-reaching (REJECT) — question on female toilet-paper folding behaviour:
  - "contact pressure between rough surfaces" (machine tribology — fails Gate 1)
  - "oil transfer to surface sample media" (industrial contamination — fails 1)
  - "dermal exposure during spraying/wiping activities" (occupational chemistry —
    fails Gate 1; the path to 'paper folds' needs a multi-step chain — fails 2)
A domain expert in hygiene behaviour would not call any of those evidence.

================ THE SELECTION TEST ================

For EACH paper, ask: does it pass ALL THREE gates? Only then can you articulate
a SINGLE-HOP role it plays in answering the original question.

It is BETTER to select FEWER papers (even zero) that genuinely pass than to
include over-reaching ones. If none pass, return an empty selections list.

================ THE PAPERS ================

{papers_text}

================ OUTPUT FORMAT ================

For EACH paper you INCLUDE, provide an intended use that (a) names the SINGLE
hop connecting it to the question and (b) states the subject domain it belongs
to, so the gate is auditable.

REMINDER — the original question every selection must serve is:
  "{query}"

NUMBERING RULE (critical): the papers above are numbered starting at 1. Use
that exact printed number as "paper_index". Do NOT count from zero. You must
also copy the paper's title EXACTLY as printed above into "title" — the title
is cross-checked against the index, and any selection whose title and index
disagree is discarded.

Respond with ONLY JSON (NO overall_reasoning field — only per-paper intended uses):
{{
    "selections": [
        {{"paper_index": 1, "title": "exact title of paper 1, copied verbatim", "intended_use": "SINGLE-hop connection + the domain this paper belongs to"}},
        {{"paper_index": 3, "title": "exact title of paper 3, copied verbatim", "intended_use": "SINGLE-hop connection + the domain this paper belongs to"}}
    ]
}}

If no papers pass all three gates:
{{"selections": []}}"""

        result = self.agent_manager.run_primary(
            prompt, as_json=True, task="paper_selection")

        selected_papers = []
        per_paper_intended_uses = []
        if result.success and result.json_response:
            selections = result.json_response.get("selections", [])
            selected_papers, per_paper_intended_uses = self._resolve_selections(
                selections, papers, reason_key="intended_use", label="tangential")
        else:
            error_info = result.error or "Unknown error"
            print(f"  {Fore.RED}  Tangential selection failed: {error_info}{Style.RESET_ALL}")

        return selected_papers, per_paper_intended_uses

    # =========================================================================
    # NODE: ACQUIRE / QUICK READ / FILTER
    # =========================================================================

    def _in_tangential_round(self, state: ReviewState) -> bool:
        """Mode signal during the search -> acquire -> quick-read loop, matching
        node_search_and_select exactly. True during tangential rounds (where
        abstract-only papers are allowed and quotes are verified against
        abstracts); False in main mode (full text required)."""
        return bool(state.get("in_tangential_round", False)
                    or state.get("tangential_mode_active", False))

    def node_acquire_texts(self, state: ReviewState) -> ReviewState:
        print(f"\n{Fore.CYAN}{SYM_PAPER} Acquiring full texts for selected papers...{Style.RESET_ALL}")
        s, f = self.discovery.acquire_full_texts_batch()
        summary = self.discovery.get_catalog_summary()
        print(f"  {Fore.GREEN}Full text: {s}/{s+f} | Total: {summary['total_papers']}{Style.RESET_ALL}")

        # MAIN-MODE ACQUISITION GATE: abstract-only papers cannot yield verified
        # full-text quotes and would be dropped at deep-analysis anyway. Announce
        # here (right after acquisition) that they will be skipped, so we don't
        # spend quick-read / filter / curation / deep-analysis time on them. They
        # are NOT removed from the catalog — if tangential mode is engaged later,
        # node_quick_read_studies will make them eligible again.
        if not self._in_tangential_round(state):
            read_ids = set(state.get("read_paper_ids", []))
            unread_abstract_only = [
                p for p in self.discovery.get_all_papers()
                if p.paper_id not in read_ids and not p.full_text_available and p.abstract]
            if unread_abstract_only:
                print(f"  {Fore.YELLOW}Main mode: {len(unread_abstract_only)} abstract-only "
                      f"paper(s) (no full text) will be skipped to save time. "
                      f"They remain available if tangential mode is engaged.{Style.RESET_ALL}")
        return state

    def node_quick_read_studies(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 3: QUICK READ — Understanding Studies")
        # Dedup against EVERY paper ever read this run, not just the ones still
        # in study_summaries — the relevance filter / curation prune that list,
        # and reading is the expensive step we must never repeat (#5).
        read_ids = set(state.get("read_paper_ids", []))
        # Back-compat: also treat anything already summarised as read.
        read_ids |= {s.get("paper_id") for s in state.get("study_summaries", [])}
        all_papers = self.discovery.get_all_papers()
        if self._in_tangential_round(state):
            # TANGENTIAL: abstract-only papers are allowed (quotes verified
            # against the abstract). Behaviour unchanged.
            to_read = [p for p in all_papers
                       if p.paper_id not in read_ids and (p.full_text_available or p.abstract)]
        else:
            # MAIN MODE: only read papers WITH full text. Abstract-only papers
            # are skipped here (announced at acquisition) so no quick-read /
            # filter / curation / deep-analysis time is spent on papers that
            # cannot yield verified full-text quotes and would be dropped later.
            eligible = [p for p in all_papers if p.paper_id not in read_ids]
            to_read = [p for p in eligible if p.full_text_available]
            skipped = [p for p in eligible if not p.full_text_available and p.abstract]
            if skipped:
                print(f"  {Fore.YELLOW}Main mode: skipping {len(skipped)} abstract-only "
                      f"paper(s) (no full text).{Style.RESET_ALL}")

        total = len(to_read)
        if total == 0:
            print(f"  {Fore.YELLOW}No new papers to read.{Style.RESET_ALL}")
            return state

        print(f"  {Fore.WHITE}{total} new papers to read{Style.RESET_ALL}")
        summaries = list(state.get("study_summaries", []))
        start_time = time.time()

        for i, paper in enumerate(to_read):
            if self._interrupted:
                break
            elapsed = time.time() - start_time
            if i > 0:
                per_study = elapsed / i
                remaining = per_study * (total - i)
                eta_m, eta_s = int(remaining // 60), int(remaining % 60)
                progress = f"(~{eta_m}m{eta_s}s remaining)"
            else:
                progress = ""
            print(f"\n  {Fore.CYAN}[{i+1}/{total}] {progress}{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}{apa_reference(paper.authors, paper.year, paper.title, paper.venue, paper.doi)}{Style.RESET_ALL}")
            # Record the read regardless of whether a summary comes back, so a
            # paper that yields no summary is still not retried next round.
            read_ids.add(paper.paper_id)
            summary = self.study_analyzer.quick_read(paper, state["original_query"])
            if summary:
                summaries.append(summary)

        total_time = time.time() - start_time
        print(f"\n  {Fore.GREEN}Read {len(summaries)} studies in "
              f"{int(total_time//60)}m {int(total_time%60)}s{Style.RESET_ALL}")
        state["study_summaries"] = summaries
        state["read_paper_ids"] = [pid for pid in read_ids if pid]

        # ---- APPEND-ONLY EVIDENCE POOL --------------------------------------
        # study_summaries is the working set and gets replaced downstream by the
        # relevance filter and curation. This pool is never pruned, so a study
        # dropped in an early round can still be recovered for synthesis instead
        # of being lost for the rest of the run.
        pool = list(state.get("all_study_summaries") or [])
        seen = {s.get("paper_id") for s in pool}
        for s in summaries:
            pid = s.get("paper_id")
            if pid and pid not in seen:
                pool.append(s)
                seen.add(pid)
        state["all_study_summaries"] = pool
        print(f"  {Fore.WHITE}Evidence pool: {len(pool)} study summary(ies) "
              f"retained across all rounds.{Style.RESET_ALL}")
        return state

    def node_filter_relevance(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 3b: RELEVANCE FILTER")

        summaries = state.get("study_summaries", [])
        if len(summaries) <= 5:
            print(f"  {Fore.YELLOW}Only {len(summaries)} studies — skipping filter.{Style.RESET_ALL}")
            return state

        lines = []
        for i, s in enumerate(summaries, 1):
            relevance = s.get("relevance_to_question", "?")
            findings = "; ".join(s.get("key_findings", [])[:2])
            apa = apa_in_text(s.get("paper_authors") or [], s.get("paper_year"))
            lines.append(
                f"{i}. {s.get('paper_title', '?')} [{s.get('paper_year', '?')}]\n"
                f"   APA in-text: {apa}\n"
                f"   Type: {s.get('study_type', '?')}\n"
                f"   Relevance per quick-read: {relevance}\n"
                f"   Findings: {findings}")
        studies_text = "\n\n".join(lines)

        prompt = f"""You are filtering a list of studies for an academic literature review.

RESEARCH QUESTION: "{state['original_query']}"

Below are {len(summaries)} studies that were summarised from search results.
Identify which should be DROPPED because they do NOT address the question.

DROP a study if:
- Its actual topic is clearly different from the research question
- Its relevance summary indicates it is only tangentially related and provides
  no useful evidence or context
- It is a methodological/measurement paper that does not actually report on
  the topic of the research question

KEEP a study if it directly studies the topic, studies a relevant mechanism/
population/outcome, is a review/meta-analysis on the topic, or provides
essential context the synthesis will need.

STUDIES:
{studies_text}

Respond with ONLY JSON:
{{
    "drops": [<list of indices to drop>],
    "reasoning": "brief overall justification"
}}

In the "reasoning" string, refer to any study using its APA 7th in-text
citation (the "APA in-text" value shown for each study, e.g. (Smith et al.,
2020), or narratively as Smith et al. (2020)) — do NOT refer to studies by
their number. The "drops" array must still use the numeric indices."""

        result = self.agent_manager.run_primary(
            prompt, as_json=True, task="relevance_filter")
        if not result.success or not result.json_response:
            print(f"  {Fore.YELLOW}Filter failed — keeping all.{Style.RESET_ALL}")
            return state

        drops = result.json_response.get("drops", [])
        if not isinstance(drops, list):
            drops = []
        drop_set = set()
        for d in drops:
            try:
                drop_set.add(int(d))
            except (ValueError, TypeError):
                pass

        reasoning = result.json_response.get("reasoning", "")
        print(f"  {Fore.WHITE}Reasoning: {reasoning}{Style.RESET_ALL}")

        # ---- GUARD 1: never drop a study that already survived curation -----
        # study_summaries accumulates across rounds, so an over-eager filter
        # could otherwise discard the hard-won curated evidence base and leave
        # curation with nothing to do.
        protected_ids = set(state.get("curated_paper_ids") or [])
        rescued = []
        if protected_ids:
            for i, s in enumerate(summaries, 1):
                if i in drop_set and s.get("paper_id") in protected_ids:
                    drop_set.discard(i)
                    rescued.append(s.get("paper_title", "?"))
        if rescued:
            print(f"  {Fore.YELLOW}Protected {len(rescued)} already-curated "
                  f"study(ies) from being dropped:{Style.RESET_ALL}")
            for t in rescued:
                print(f"  {Fore.YELLOW}    ✓ {t[:90]}{Style.RESET_ALL}")

        kept = [s for i, s in enumerate(summaries, 1) if i not in drop_set]

        # ---- GUARD 2: a 100% wipe is treated as a filter malfunction --------
        # Dropping every single study is almost always a numbering/parsing
        # error rather than a real judgement, and it strands curation with an
        # empty set. Keep everything and say so, rather than silently emptying
        # the evidence base.
        if summaries and not kept:
            print(f"  {Fore.RED}Filter marked ALL {len(summaries)} studies as "
                  f"irrelevant — treating as a filter malfunction and keeping "
                  f"all of them. Curation will make the real call."
                  f"{Style.RESET_ALL}")
            state["study_summaries"] = summaries
            return state

        dropped_titles = [s.get("paper_title", "?")
                          for i, s in enumerate(summaries, 1) if i in drop_set]
        if dropped_titles:
            print(f"  {Fore.YELLOW}Dropped {len(dropped_titles)}:{Style.RESET_ALL}")
            for t in dropped_titles:
                print(f"  {Fore.YELLOW}    ✗ {t[:90]}{Style.RESET_ALL}")

        print(f"  {Fore.GREEN}{SYM_CHECK} Kept {len(kept)} relevant studies.{Style.RESET_ALL}")
        state["filter_dropped_total"] = (state.get("filter_dropped_total", 0)
                                         + len(dropped_titles))
        state["study_summaries"] = kept
        return state

    # =========================================================================
    # NODE: GAP ANALYSIS
    # =========================================================================

    def node_gap_analysis(self, state: ReviewState) -> ReviewState:
        if state.get("tangential_mode_active"):
            return self.node_tangential_gap_analysis(state)

        self._print_phase_banner("PHASE 4: GAP ANALYSIS")
        summaries = state.get("study_summaries", [])
        if not summaries:
            state["ready_to_write"] = False
            return state

        # HARD MINIMUM SEARCH ROUNDS — do NOT run a readiness/sufficiency check
        # until at least `min_rounds_before_readiness` full search rounds have
        # completed. A "round" runs the full set of focus-area searches, so with
        # the standard 5-focus-area plan 3 rounds == 15 individual searches. Until
        # the floor is reached we never declare ready and keep searching, so the
        # pipeline cannot curate / check sufficiency / write early. The only
        # override is hitting max discovery rounds (handled below), which can never
        # trigger before the floor when min_rounds <= max_discovery_rounds.
        min_rounds = int(self.config.get("min_rounds_before_readiness", 3))
        rnd_now = state.get("discovery_round", 0)
        if rnd_now < min_rounds and rnd_now < state.get("max_discovery_rounds", 15):
            print(f"  {Fore.YELLOW}Readiness check held until {min_rounds} search "
                  f"rounds complete (currently {rnd_now}); searching more before "
                  f"any sufficiency check.{Style.RESET_ALL}")
            state["ready_to_write"] = False
            return state

        studies_text = self._build_studies_overview(summaries)
        result = self.planner.check_readiness(state["original_query"], studies_text)
        ready = result.get("ready_to_write", False)
        confidence = result.get("confidence", 0)

        if len(summaries) >= 10 and confidence >= 0.4:
            ready = True
        # Stricter floor: do not declare ready with too few studies given the
        # heavy attrition still ahead (relevance filter, curation, and the
        # full-text verified-quote gate). Below the floor, keep searching.
        min_before_ready = int(self.config.get("min_studies_before_ready", 10))
        if ready and len(summaries) < min_before_ready:
            ready = False
            print(f"  {Fore.YELLOW}Holding off: only {len(summaries)} studies "
                  f"(< {min_before_ready}); searching more before writing.{Style.RESET_ALL}")
        if state.get("discovery_round", 0) >= state.get("max_discovery_rounds", 15):
            ready = True
            print(f"  {Fore.YELLOW}Max discovery rounds reached.{Style.RESET_ALL}")

        state["ready_to_write"] = ready
        return state

    def node_tangential_gap_analysis(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 4: TANGENTIAL GAP ANALYSIS", color=Fore.MAGENTA)
        summaries = state.get("study_summaries", [])
        if not summaries:
            state["ready_to_write"] = False
            return state

        studies_text = self._build_studies_overview(summaries)
        tang_papers = state.get("tangential_papers_added", 0)
        result = self.planner.tangential_readiness_check(
            state["original_query"], studies_text, tang_papers)
        ready = result.get("ready_to_write", False)
        state["ready_to_write"] = ready
        return state

    def _build_studies_overview(self, summaries: List[Dict]) -> str:
        lines = [f"Total studies: {len(summaries)}\n"]
        for i, s in enumerate(summaries, 1):
            findings = s.get("key_findings", [])
            findings_text = "; ".join(findings[:3]) if findings else "No findings extracted"
            apa = apa_in_text(s.get("paper_authors") or [], s.get("paper_year"))
            lines.append(
                f"Study {i}: {s.get('paper_title', '?')}\n"
                f"  APA in-text: {apa}\n"
                f"  Type: {s.get('study_type', '?')} | Year: {s.get('paper_year', '?')} | "
                f"Reliability: {s.get('reliability_score', '?')}/10\n"
                f"  Findings: {findings_text}\n"
                f"  Relevance: {s.get('relevance_to_question', '?')}")
        return "\n".join(lines)

    # =========================================================================
    # NODE: DISTILL STRATEGY
    # =========================================================================

    def node_distill_strategy(self, state: ReviewState) -> ReviewState:
        is_tang = state.get("tangential_mode_active", False)
        self._print_stage_banner(f"{SYM_BRAIN}  Examining Search Strategy"
                                  f"{' (Tangential Mode)' if is_tang else ''}...",
                                  color=Fore.MAGENTA)

        previous_memo = state.get("strategy_memo", "")
        search_history = state.get("search_history", [])
        if not search_history:
            print(f"  {Fore.YELLOW}No search history to distill from.{Style.RESET_ALL}")
            return state

        history_text = self._format_search_history_for_distill(search_history)
        result = self.planner.distill_search_strategy(
            query=state["original_query"],
            full_search_history=history_text,
            previous_memo=previous_memo,
        )

        new_memo = result.get("new_memo", "").strip()
        reasoning = result.get("reasoning", "").strip()

        if not new_memo:
            print(f"  {Fore.RED}Distillation produced empty memo. Keeping previous.{Style.RESET_ALL}")
            return state

        if previous_memo:
            print(f"\n  {Fore.WHITE}{Style.BRIGHT}Previous search strategy:{Style.RESET_ALL}")
            for line in previous_memo.split("\n"):
                print(f"    {Fore.WHITE}{line}{Style.RESET_ALL}")
        else:
            print(f"\n  {Fore.WHITE}{Style.BRIGHT}Previous search strategy:{Style.RESET_ALL} "
                  f"{Fore.YELLOW}(none — first distillation){Style.RESET_ALL}")

        print(f"\n  {Fore.GREEN}{Style.BRIGHT}New search strategy:{Style.RESET_ALL}")
        for line in new_memo.split("\n"):
            print(f"    {Fore.GREEN}{line}{Style.RESET_ALL}")

        if reasoning:
            print(f"\n  {Fore.CYAN}{Style.BRIGHT}Reasoning:{Style.RESET_ALL}")
            for line in reasoning.split("\n"):
                print(f"    {Fore.CYAN}{line}{Style.RESET_ALL}")

        print(f"\n  {Fore.MAGENTA}{'─'*60}{Style.RESET_ALL}")

        state["strategy_memo"] = new_memo
        if is_tang:
            state["tangential_distillations_done"] = state.get("tangential_distillations_done", 0) + 1
        else:
            state["distillations_done"] = state.get("distillations_done", 0) + 1
        return state

    def _format_search_history_for_distill(self, search_history: List[Dict]) -> str:
        """Render the search history under a hard size budget.

        The unbounded version emitted EVERY candidate title from EVERY search
        for the whole run. By round 10 that was ~167,000 tokens fed into a
        65,536-token window, so Ollama silently discarded the front of the
        prompt (including the research question) and the sufficiency check was
        deciding half-blind — which is what kept the loop running.

        Detail is preserved where it is actually used: the most recent rounds
        keep their full candidate lists (that is what "bad_picks" detection
        reads), while older rounds collapse to query + counts + what was
        selected. A final character ceiling drops the oldest entries entirely
        if the result would still be oversized.
        """
        if not search_history:
            return ""

        full_n = int(self.config.get("search_history_full_detail_entries", 8))
        max_chars = int(self.config.get("search_history_max_chars", 40000))
        max_cand = int(self.config.get("search_history_max_candidates_per_entry", 25))

        total_entries = len(search_history)
        split = max(0, total_entries - full_n)
        older, recent = search_history[:split], search_history[split:]

        def render_full(entry) -> List[str]:
            rnd = entry.get("round", "?")
            mode = entry.get("mode", "?").upper()
            query = entry.get("query", "?")
            area = entry.get("focus_area", "?")
            total = entry.get("total_candidates", 0)
            selected = entry.get("selected_count", 0)
            outcome = entry.get("outcome", "?")
            reasoning = entry.get("selection_reasoning", "")

            out = [f"\n--- {rnd} [{mode}] | Focus: {area} ---",
                   f"Query: \"{query}\"",
                   f"Outcome: {outcome} | Candidates: {total} | Selected: {selected}"]
            cands = entry.get("candidate_titles", []) or []
            if total == 0 or not cands:
                out.append("CANDIDATE TITLES: (none returned)")
            else:
                out.append("CANDIDATE TITLES:")
                for c in cands[:max_cand]:
                    out.append(f"  - [{c.get('year') or '?'}] {c.get('title') or '?'}")
                if len(cands) > max_cand:
                    out.append(f"  - ...and {len(cands) - max_cand} further candidates")
            if entry.get("selected_titles"):
                out.append("YOU SELECTED:")
                for s in entry["selected_titles"]:
                    out.append(f"  ✓ [{s.get('year') or '?'}] {s.get('title') or '?'}")
            if reasoning:
                out.append(f"SELECTION REASONING: {reasoning}")
            return out

        def render_compact(entry) -> List[str]:
            rnd = entry.get("round", "?")
            mode = entry.get("mode", "?").upper()
            query = entry.get("query", "?")
            total = entry.get("total_candidates", 0)
            selected = entry.get("selected_count", 0)
            outcome = entry.get("outcome", "?")
            out = [f"\n--- {rnd} [{mode}] | \"{query}\" → "
                   f"{total} candidates, {selected} selected ({outcome})"]
            for s in (entry.get("selected_titles") or []):
                out.append(f"  ✓ [{s.get('year') or '?'}] {s.get('title') or '?'}")
            return out

        older_blocks = [("\n".join(render_compact(e))) for e in older]
        recent_blocks = [("\n".join(render_full(e))) for e in recent]

        # Recent detail is the priority; shed the oldest compact blocks until
        # the whole thing fits the ceiling.
        omitted = 0
        while older_blocks and (
                sum(len(b) for b in older_blocks) + sum(len(b) for b in recent_blocks)
                > max_chars):
            older_blocks.pop(0)
            omitted += 1

        header = []
        if omitted:
            header.append(f"({omitted} earlier search(es) omitted for length; "
                          f"{len(older_blocks)} summarised, {len(recent_blocks)} "
                          f"shown in full)")
        elif older_blocks:
            header.append(f"({len(older_blocks)} earlier search(es) summarised; "
                          f"{len(recent_blocks)} most recent shown in full)")

        text = "\n".join(header + older_blocks + recent_blocks)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    # =========================================================================
    # NODE: REFINE PLAN (standard + tangential variants)
    # =========================================================================

    def _format_recent_search_history_for_refine(self, search_history: List[Dict],
                                                  last_n: int = 10) -> str:
        recent = search_history[-last_n:] if search_history else []
        if not recent:
            return ""
        lines = []
        for entry in recent:
            rnd = entry.get("round", "?")
            query = entry.get("query", "?")
            total = entry.get("total_candidates", 0)
            selected = entry.get("selected_count", 0)
            outcome = entry.get("outcome", "?")
            lines.append(f"{rnd} \"{query}\" → {total} candidates, {selected} selected ({outcome})")
        return "\n".join(lines)

    def node_refine_plan(self, state: ReviewState) -> ReviewState:
        studies_text = self._build_studies_overview(state.get("study_summaries", []))
        strategy_memo = state.get("strategy_memo", "")
        recent_history = self._format_recent_search_history_for_refine(
            state.get("search_history", []), last_n=10)

        new_plan = self.planner.refine_plan(
            state["original_query"], studies_text,
            state.get("focus_areas_completed", []),
            strategy_memo=strategy_memo,
            recent_search_history=recent_history,
        )

        if new_plan and new_plan.focus_areas:
            state["research_plan"] = new_plan.to_dict()
            state["ready_to_write"] = False
        else:
            state["ready_to_write"] = True
        return state

    def node_refine_plan_tangential(self, state: ReviewState) -> ReviewState:
        if not state.get("tangential_mode_active"):
            self._print_stage_banner(f"{SYM_LAMP}  ENTERING TANGENTIAL MODE", color=Fore.MAGENTA)
            state["tangential_mode_active"] = True
            state["tangential_engagement_count"] = state.get("tangential_engagement_count", 0) + 1
            state["tangential_round_count"] = 0
            state["tangential_papers_added"] = 0
            state["tangential_distillations_done"] = 0
            state["tangential_rounds_since_last_curation"] = 0
            state["_tangential_curated_count_this_engagement"] = 0

        studies_text = self._build_studies_overview(state.get("study_summaries", []))
        strategy_memo = state.get("strategy_memo", "")
        recent_history = self._format_recent_search_history_for_refine(
            state.get("search_history", []), last_n=15)
        sufficiency_reasoning = state.get("sufficiency_reasoning", "")
        tang_round = state.get("tangential_round_count", 0) + 1

        new_plan = self.planner.refine_plan_tangential(
            state["original_query"], studies_text,
            state.get("focus_areas_completed", []),
            strategy_memo=strategy_memo,
            recent_search_history=recent_history,
            sufficiency_reasoning=sufficiency_reasoning,
            tangential_round=tang_round,
        )

        if new_plan and new_plan.focus_areas:
            state["research_plan"] = new_plan.to_dict()
            state["in_tangential_round"] = True
            state["ready_to_write"] = False
        else:
            print(f"  {Fore.YELLOW}Tangential refinement failed — proceeding to synthesis.{Style.RESET_ALL}")
            state["ready_to_write"] = True
        return state

    # =========================================================================
    # NODE: CURATE EVIDENCE
    # =========================================================================

    def _rescue_deadlocked_curation(self, summaries, included_summaries,
                                    decision_records, label):
        """Reinstate the strongest candidates when curation excluded ALL of them.

        Excluding every paper is virtually always a mutual-redundancy deadlock
        (A dropped as "redundant with B" while B is dropped as "redundant with
        A"), not a real judgement that no candidate is usable. Handing the
        synthesis an empty evidence base is never the right outcome, so the top
        candidates are reinstated using the same criteria the curator was asked
        to apply: methodological quality (reliability score + study-type weight)
        first, then recency. Mutates included_summaries and decision_records in
        place; returns the reinstated list.
        """
        floor = int(self.config.get("curation_rescue_min_papers", 5))
        weights = self.config.get("study_type_weights", {}) or {}

        def _strength(s):
            try:
                rel = float(s.get("reliability_score") or 0)
            except (TypeError, ValueError):
                rel = 0.0
            stype = str(s.get("study_type") or "").strip().lower()
            w = float(weights.get(stype, 3))
            try:
                yr = int(s.get("paper_year") or 0)
            except (TypeError, ValueError):
                yr = 0
            return (rel + w, yr)

        ranked = sorted(summaries, key=_strength, reverse=True)
        rescued = ranked[:max(1, min(floor, len(ranked)))]
        print(f"\n  {Fore.RED}{Style.BRIGHT}{label} excluded ALL "
              f"{len(summaries)} papers — this is a mutual-redundancy "
              f"deadlock, not a real verdict.{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Reinstating the {len(rescued)} strongest "
              f"candidate(s) so the review has an evidence base:{Style.RESET_ALL}")
        for s in rescued:
            print(f"  {Fore.YELLOW}    {SYM_INCLUDE} "
                  f"{(s.get('paper_title') or '?')[:85]} "
                  f"[{s.get('paper_year','?')}, {s.get('study_type','?')}, "
                  f"reliability {s.get('reliability_score','?')}/10]{Style.RESET_ALL}")
            included_summaries.append(s)
            for rec in decision_records:
                if rec.get("paper_id") == s.get("paper_id"):
                    rec["include"] = True
                    rec["reasoning"] = ("(reinstated — curation deadlocked and "
                                        "excluded every candidate) " + rec.get("reasoning", ""))
                    break
        return rescued

    def node_curate_evidence(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner(f"{SYM_FILTER} PHASE 5: CURATION (Selecting Papers for the Review)",
                                  color=Fore.CYAN)

        summaries = state.get("study_summaries", [])
        if not summaries:
            print(f"  {Fore.YELLOW}No studies to curate.{Style.RESET_ALL}")
            return state

        # Compact one-line catalog of ALL candidate papers. Each per-study call
        # gets this so it can still judge REDUNDANCY across the set without
        # receiving every paper's full detail (sending all of them in one call
        # is what exhausted the output budget inside the thinking block).
        def _render_catalog(decided_by_index):
            """Catalog of all candidates, annotated with the decision so far.

            Without the annotations the model treats every candidate as still
            in play, which lets it exclude A as redundant with B and then
            exclude B as redundant with A — a mutual-redundancy deadlock that
            can empty the entire evidence base.
            """
            lines = []
            for idx, s in enumerate(summaries, 1):
                apa = apa_in_text(s.get("paper_authors") or [], s.get("paper_year"))
                status = decided_by_index.get(idx)
                if status is True:
                    tag = "[ALREADY INCLUDED]"
                elif status is False:
                    tag = "[ALREADY EXCLUDED — cannot make anything redundant]"
                else:
                    tag = "[not yet decided]"
                lines.append(
                    f"{idx}. {tag} {apa} — {s.get('paper_title','?')} "
                    f"[{s.get('paper_year','?')}, {s.get('study_type','?')}]")
            return "\n".join(lines)

        decided_by_index = {}

        # Prior decisions (by paper_id) from the last curation pass, for
        # consistency across passes (the model may still revise).
        prior_decision_by_id = {}
        prior = state.get("curation_history", [])
        if prior:
            for d in prior[-1].get("decisions", []):
                if d.get("paper_id") is not None:
                    prior_decision_by_id[d.get("paper_id")] = d.get("include")

        total = len(summaries)
        print(f"  {Fore.WHITE}Curating {total} papers one at a time...{Style.RESET_ALL}")

        included_summaries, excluded_summaries = [], []
        decision_records = []
        included_count, excluded_count = 0, 0
        start_time = time.time()

        for i, s in enumerate(summaries, 1):
            title = s.get("paper_title", "?")
            year = s.get("paper_year", "?")
            apa = apa_in_text(s.get("paper_authors") or [], s.get("paper_year"))

            # If interrupted, keep the remaining papers by default (no LLM call)
            # so a partial curation never silently discards studies.
            if self._interrupted:
                included_count += 1
                included_summaries.append(s)
                decision_records.append({
                    "paper_id": s.get("paper_id"), "title": title, "year": year,
                    "include": True, "reasoning": "(interrupted — kept by default)",
                })
                continue

            # ETA — identical pattern to quick_read.
            elapsed = time.time() - start_time
            if i > 1:
                per = elapsed / (i - 1)
                remaining = per * (total - (i - 1))
                eta_m, eta_s = int(remaining // 60), int(remaining % 60)
                progress = f"(~{eta_m}m{eta_s}s remaining)"
            else:
                progress = ""

            # BLUE while the paper is being judged (mirrors quick-read UI).
            print(f"\n  {Fore.CYAN}[{i}/{total}] {progress}{Style.RESET_ALL}")
            print(f"  {Fore.BLUE}{apa_reference(s.get('paper_authors') or [], year, title, s.get('paper_venue'), s.get('paper_doi'))}{Style.RESET_ALL}")
            print(f"  {Fore.BLUE}Curating...{Style.RESET_ALL}", end=" ", flush=True)

            findings = s.get("key_findings", [])
            findings_text = "; ".join(findings[:4]) if findings else "(no findings extracted)"

            prior_hint = ""
            if s.get("paper_id") in prior_decision_by_id:
                pv = prior_decision_by_id[s.get("paper_id")]
                prior_hint = (f"\nPRIOR PASS DECISION for this paper: "
                              f"{'INCLUDE' if pv else 'EXCLUDE'} (you may revise it).")

            prompt = f"""You are curating the evidence base for an academic literature review,
deciding ONE paper at a time whether to INCLUDE it in the review or EXCLUDE it.

USER'S RESEARCH QUESTION: "{state['original_query']}"

================ CURATION CRITERIA ================

Decide INCLUDE or EXCLUDE for THE PAPER UNDER REVIEW based on:

1. RELEVANCE — does it directly inform an answer to the user's question, or
   provide essential context the review needs to cite?
2. RECENCY — is it recent enough given the topic? Foundational classics are
   fine for stable topics; prefer recent work in fast-moving fields.
3. METHODOLOGICAL QUALITY — prefer meta-analyses and systematic reviews,
   well-designed RCTs and large cohort studies, and direct outcome
   measurements over case reports, opinion pieces, or methodologically-limited
   work.
4. REDUNDANCY — you may EXCLUDE this paper as redundant ONLY IF a paper marked
   [ALREADY INCLUDED] below covers the same ground as well or better.
   You must NOT exclude it as redundant against a paper marked
   [not yet decided] or [ALREADY EXCLUDED]. If nothing has been included yet
   that covers this ground, then THIS paper is the strongest representative so
   far and redundancy is not a valid reason to drop it — judge it on relevance,
   recency and quality alone. Excluding every candidate as "redundant with each
   other" leaves the review with no evidence at all, which is always wrong.
5. OVERALL FIT — would a strong literature review actually cite THIS paper? If
   you would skip it as a reviewer, EXCLUDE it.

================ THE PAPER UNDER REVIEW (this is paper #{i}) ================

APA in-text: {apa}
Title: {title}
Year: {year} | Type: {s.get('study_type','?')} | Reliability: {s.get('reliability_score','?')}/10
Sample: {s.get('sample_size','?')}
Methodology: {s.get('methodology_summary','(not extracted)')}
Key findings: {findings_text}
Relevance (per quick-read): {s.get('relevance_to_question','?')}{prior_hint}

================ OTHER CANDIDATE PAPERS (context for redundancy only) ================
(You are NOT deciding on these now. They are listed so you can judge whether the
paper above is redundant. The paper above appears as #{i} in this list. Each
entry shows the decision made so far — only [ALREADY INCLUDED] papers can make
this one redundant.)
{_render_catalog(decided_by_index)}

================ OUTPUT FORMAT ================

Respond with ONLY JSON:
{{
    "include": true,
    "reasoning": "specific reason — name the role this paper plays in the review OR why it is excluded; refer to papers by their APA 7th in-text citation, e.g. (Smith et al., 2020)"
}}"""

            decision_start = time.time()
            result = self.agent_manager.run_primary(
                prompt, as_json=True, task="curate_study")
            decision_elapsed = time.time() - decision_start

            if result.success and result.json_response:
                include = bool(result.json_response.get("include", True))
                reasoning = result.json_response.get("reasoning", "")
            else:
                # Fail-safe: keep the paper (never silently drop on an API error).
                include = True
                reasoning = "(curation call failed — kept by default)"

            decided_by_index[i] = include

            if include:
                included_count += 1
                included_summaries.append(s)
                print(f"{Fore.GREEN}{SYM_INCLUDE} INCLUDE ({decision_elapsed:.0f}s){Style.RESET_ALL}")
                print(f"    {Fore.GREEN}{reasoning}{Style.RESET_ALL}")
            else:
                excluded_count += 1
                excluded_summaries.append(s)
                print(f"{Fore.YELLOW}{SYM_EXCLUDE} EXCLUDE ({decision_elapsed:.0f}s){Style.RESET_ALL}")
                print(f"    {Fore.YELLOW}{reasoning}{Style.RESET_ALL}")

            decision_records.append({
                "paper_id": s.get("paper_id"),
                "title": title,
                "year": year,
                "include": include,
                "reasoning": reasoning,
            })

        # ---- DEADLOCK RESCUE ------------------------------------------------
        # Curating every paper out leaves nothing to write a review from, and it
        # is virtually always a mutual-redundancy deadlock rather than a real
        # judgement that no candidate is usable. Rather than proceed with an
        # empty evidence base, reinstate the strongest candidates by the same
        # criteria the curator was asked to apply: methodological quality
        # (reliability score and study type) first, then recency.
        if summaries and not included_summaries:
            self._rescue_deadlocked_curation(summaries, included_summaries,
                                             decision_records, "Curation")
            included_count = len(included_summaries)
            excluded_count = len(summaries) - included_count

        total_time = time.time() - start_time
        print(f"\n  {Fore.GREEN}{Style.BRIGHT}Curation complete: "
              f"{included_count} included, {excluded_count} excluded "
              f"({int(total_time//60)}m {int(total_time%60)}s){Style.RESET_ALL}")

        summary_assessment = (f"{included_count} included, {excluded_count} excluded "
                              f"(sequential per-study curation).")
        state["curation_excluded_total"] = (state.get("curation_excluded_total", 0)
                                            + excluded_count)

        state["curated_paper_ids"] = [s.get("paper_id") for s in included_summaries]
        state["study_summaries"] = included_summaries
        state["curation_history"] = state.get("curation_history", []) + [{
            "round": state.get("discovery_round", 0),
            "mode": "standard",
            "decisions": decision_records,
            "summary": summary_assessment,
        }]
        state["rounds_since_last_curation"] = 0
        return state

    def node_tangential_curate_evidence(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner(f"{SYM_FILTER} TANGENTIAL CURATION", color=Fore.MAGENTA)

        summaries = state.get("study_summaries", [])
        if not summaries:
            print(f"  {Fore.YELLOW}No studies to curate.{Style.RESET_ALL}")
            return state

        paper_blocks = []
        for i, s in enumerate(summaries, 1):
            findings = s.get("key_findings", [])
            findings_text = "; ".join(findings[:4]) if findings else "(no findings)"
            apa = apa_in_text(s.get("paper_authors") or [], s.get("paper_year"))
            paper_blocks.append(
                f"PAPER {i}:\n"
                f"  APA in-text: {apa}\n"
                f"  Title: {s.get('paper_title','?')}\n"
                f"  Year: {s.get('paper_year','?')} | Type: {s.get('study_type','?')} | "
                f"Reliability: {s.get('reliability_score','?')}/10\n"
                f"  Sample: {s.get('sample_size','?')}\n"
                f"  Methodology: {s.get('methodology_summary','(not extracted)')}\n"
                f"  Key findings: {findings_text}\n"
                f"  Relevance: {s.get('relevance_to_question','?')}")
        papers_text = "\n\n".join(paper_blocks)

        prior_curation_block = ""
        prior = state.get("curation_history", [])
        if prior:
            last = prior[-1]
            kept_titles = [d.get("title", "?") for d in last.get("decisions", [])
                           if d.get("include")]
            prior_curation_block = f"""

PRIOR CURATION DECISIONS (you may revise):
PREVIOUSLY INCLUDED ({len(kept_titles)}):
{chr(10).join('  - ' + t for t in kept_titles[:20])}
"""

        prompt = f"""You are in TANGENTIAL MODE curating the combined evidence base for a
literature review. The evidence base contains BOTH direct-relevance studies
(from the standard search rounds) AND indirect-route studies (from tangential
searches because direct evidence was sparse).

USER'S RESEARCH QUESTION: "{state['original_query']}"
{prior_curation_block}

================ WHAT "INDIRECT EVIDENCE" MEANS HERE ================

Indirect evidence is a CLOSE NEIGHBOUR of the question that a researcher IN THE
QUESTION'S OWN FIELD would still recognise as relevant. It is NOT "any paper
connectable to the topic through a clever chain of reasoning." The most common
error is keeping a generic physical, chemical, statistical, or engineering
paper that merely shares a WORD with the question ("pressure", "contact",
"transfer", "area", "exposure") and building an analogy chain. That is
UNRELATED, not indirect. EXCLUDE such papers.

================ THREE HARD GATES — AN INDIRECT PAPER MUST PASS ALL THREE =====

GATE 1 — DOMAIN ANCHOR: the paper belongs to the SAME real-world subject
domain as the question (not a different field that merely shares vocabulary).

GATE 2 — ONE-HOP CONNECTION: the link from the paper's findings to the
question is a SINGLE, DIRECT inferential step — not a chain of "model… infer…
estimate… extrapolate".

GATE 3 — DOMAIN-EXPERT TEST: a researcher who actually studies the question's
topic would call it relevant adjacent evidence, not "nothing to do with my
field".

EXCLUDE any indirect paper that fails ANY gate, EVEN IF it was selected during
search. This curation step is the second line of defence against over-reaching
picks — be stricter than the search stage was.

================ CURATION CRITERIA ================

Decide INCLUDE/EXCLUDE for each paper based on:

1. RELEVANCE — for direct-relevance studies: directly inform the question.
   For indirect studies: must PASS ALL THREE GATES above and have a clear,
   single-hop route from finding to answering the user's question.

2. RECENCY — prefer recent papers where the field is moving.

3. METHODOLOGICAL QUALITY — meta-analyses > systematic reviews > RCTs >
   cohort > cross-sectional > case reports.

4. REDUNDANCY — drop redundant weaker entries.

5. INDIRECT-EVIDENCE BAR — an indirect study is INCLUDED only if it passes the
   three gates AND you can name the single role it plays. If you cannot, or if
   it is from a different subject domain, EXCLUDE it.

================ THE PAPERS ================

{papers_text}

================ OUTPUT ================

For EVERY paper, decide include/exclude with reasoning. For indirect papers,
the reasoning MUST name the single-hop connection and the subject domain so the
gates are auditable. The decisions array must have EXACTLY {len(summaries)} entries.

Respond with ONLY JSON:
{{
    "decisions": [
        {{"paper_index": 1, "include": true/false, "reasoning": "for indirect papers: single-hop connection + domain; for direct papers: how it informs the question; for exclusions: which gate it fails or why"}},
        ...
    ],
    "summary": "1-2 sentence overall assessment of the curated set"
}}

In every "reasoning" string and in the "summary", refer to papers using their
APA 7th in-text citation (the "APA in-text" value shown for each paper, e.g.
(Smith et al., 2020), or narratively as Smith et al. (2020)) — not by their
number. The "paper_index" field must still carry the numeric index."""

        print(f"  {Fore.WHITE}Reviewing {len(summaries)} papers (direct + indirect)...{Style.RESET_ALL}")
        result = self.agent_manager.run_primary(
            prompt, as_json=True, task="tangential_curate_evidence")

        if not result.success or not result.json_response:
            print(f"  {Fore.RED}Tangential curation failed — keeping all studies.{Style.RESET_ALL}")
            state["curated_paper_ids"] = [s.get("paper_id") for s in summaries]
            state["_tangential_curated_count_this_engagement"] = len(summaries)
            return state

        decisions = result.json_response.get("decisions", [])
        summary_assessment = result.json_response.get("summary", "")

        included_summaries, excluded_summaries = [], []
        decision_records = []
        included_count, excluded_count = 0, 0

        for i, s in enumerate(summaries, 1):
            decision = None
            for d in decisions:
                if d.get("paper_index") == i:
                    decision = d
                    break
            if decision is None:
                decision = {"include": True, "reasoning": "(no decision returned — kept by default)"}

            include = bool(decision.get("include", True))
            reasoning = decision.get("reasoning", "")
            title = s.get("paper_title", "?")
            year = s.get("paper_year", "?")

            if include:
                included_count += 1
                included_summaries.append(s)
                print(f"\n  {Fore.GREEN}[{i}/{len(summaries)}] {SYM_INCLUDE} INCLUDE: "
                      f"[{year}] {title}{Style.RESET_ALL}")
                print(f"    {Fore.WHITE}{reasoning}{Style.RESET_ALL}")
            else:
                excluded_count += 1
                excluded_summaries.append(s)
                print(f"\n  {Fore.YELLOW}[{i}/{len(summaries)}] {SYM_EXCLUDE} EXCLUDE: "
                      f"[{year}] {title}{Style.RESET_ALL}")
                print(f"    {Fore.YELLOW}{reasoning}{Style.RESET_ALL}")

            decision_records.append({
                "paper_id": s.get("paper_id"),
                "title": title,
                "year": year,
                "include": include,
                "reasoning": reasoning,
            })

        # Same deadlock guard as standard curation — tangential mode is where a
        # long run spends most of its time, and an empty result there strands
        # the pipeline just as badly.
        if summaries and not included_summaries:
            self._rescue_deadlocked_curation(summaries, included_summaries,
                                             decision_records, "Tangential curation")
            included_count = len(included_summaries)
            excluded_count = len(summaries) - included_count

        print(f"\n  {Fore.MAGENTA}{Style.BRIGHT}Tangential curation complete: "
              f"{included_count} included, {excluded_count} excluded{Style.RESET_ALL}")
        if summary_assessment:
            print(f"  {Fore.WHITE}{summary_assessment}{Style.RESET_ALL}")

        state["curated_paper_ids"] = [s.get("paper_id") for s in included_summaries]
        state["study_summaries"] = included_summaries
        state["curation_history"] = state.get("curation_history", []) + [{
            "round": state.get("tangential_round_count", 0),
            "mode": "tangential",
            "engagement": state.get("tangential_engagement_count", 1),
            "decisions": decision_records,
            "summary": summary_assessment,
        }]
        state["tangential_rounds_since_last_curation"] = 0
        state["_tangential_curated_count_this_engagement"] = included_count
        return state

    # =========================================================================
    # NODE: EVIDENCE SUFFICIENCY
    # =========================================================================

    def _direct_evidence_inventory(self, state: ReviewState) -> Dict[str, int]:
        """Count the direct evidence the pipeline is holding but not using.

        The sufficiency check only ever saw the post-curation working set, which
        the relevance filter and curation shrink every round. A curated set of
        one looks exactly like a sparse literature even when the catalog holds
        a hundred usable full texts — which is how a question as heavily
        researched as creatine and kidney function ended up in tangential mode.
        """
        try:
            papers = self.discovery.get_all_papers()
        except Exception:
            papers = []
        read_ids = set(state.get("read_paper_ids") or [])
        curated_ids = set(state.get("curated_paper_ids") or [])
        pool = state.get("all_study_summaries") or []

        full_text = [p for p in papers if getattr(p, "full_text_available", False)]
        unread_full_text = [p for p in full_text if p.paper_id not in read_ids]
        pool_unused = [s for s in pool if s.get("paper_id") not in curated_ids]

        return {
            "catalog": len(papers),
            "full_text": len(full_text),
            "unread_full_text": len(unread_full_text),
            "pool": len(pool),
            "pool_unused": len(pool_unused),
            "curated": len(state.get("study_summaries") or []),
            "filter_dropped": int(state.get("filter_dropped_total", 0)),
            "curation_excluded": int(state.get("curation_excluded_total", 0)),
            "standard_rounds": int(state.get("discovery_round", 0)),
        }

    def _tangential_entry_block(self, state: ReviewState):
        """Return a reason to REFUSE tangential mode, or None to allow it.

        Tangential mode exists for genuinely under-studied questions. It must
        not be reachable while the pipeline is still sitting on unexamined
        direct evidence — that is a curation problem, not a literature problem,
        and the fix is another standard round with better queries.
        """
        cfg = self.config
        inv = self._direct_evidence_inventory(state)

        # Already holding a healthy set of DIRECT studies: the literature is
        # plainly not sparse, so write the review rather than chasing proxies.
        healthy = int(cfg.get("tangential_block_min_curated", 8))
        if inv["curated"] >= healthy:
            return ("sufficient",
                    f"{inv['curated']} direct study(ies) are already curated "
                    f"(>= {healthy}) — that is not a sparse literature", inv)

        min_rounds = int(cfg.get("min_standard_rounds_before_tangential", 3))
        if inv["standard_rounds"] < min_rounds:
            return ("bad_picks",
                    f"only {inv['standard_rounds']} standard search round(s) "
                    f"completed (minimum {min_rounds} before indirect evidence "
                    f"is considered)", inv)

        unread_cap = int(cfg.get("tangential_block_unread_full_text", 10))
        if inv["unread_full_text"] >= unread_cap:
            return ("bad_picks",
                    f"{inv['unread_full_text']} paper(s) with full text have not "
                    f"been read yet — direct evidence is still available", inv)

        pool_cap = int(cfg.get("tangential_block_unused_pool", 10))
        if inv["pool_unused"] >= pool_cap:
            return ("bad_picks",
                    f"{inv['pool_unused']} already-read study(ies) were discarded "
                    f"by the filter or curation and never used — the literature "
                    f"is not sparse, the selection was", inv)

        return None, None, inv

    def _abstract_only_paper_count(self) -> int:
        """Papers in the catalog that have an abstract but no full text.

        These are invisible to main mode by design (node_quick_read_studies
        skips them) and readable ONLY in tangential mode, where quotes are
        verified against the abstract.
        """
        try:
            papers = self.discovery.get_all_papers()
        except Exception:
            return 0
        return sum(1 for p in papers
                   if not getattr(p, "full_text_available", False)
                   and getattr(p, "abstract", None))

    def _tangential_entry_force(self, state: ReviewState):
        """Return (reason, inv) when tangential mode MUST be engaged, else (None, inv).

        Counterpart to _tangential_entry_block. That gate only ever refuses
        tangential mode; nothing ever forced it on. The consequence, observed in
        a run that acquired 0 full texts from 8 papers: main mode skips every
        abstract-only paper, so nothing is read, nothing is curated, nothing is
        deep-analysed, and the review comes out blank — while the sufficiency
        model keeps answering "bad_picks" and ordering more standard rounds that
        cannot possibly help, because another standard round still cannot read an
        abstract-only paper.

        The condition is deliberately narrow: it fires ONLY when the pipeline is
        holding no readable direct evidence whatsoever (no full text retrieved
        all run AND nothing read into the evidence pool), while the catalog does
        hold abstract-bearing papers that tangential mode could read. Any run
        with even one full text or one read study is untouched by this.
        """
        cfg = self.config
        inv = self._direct_evidence_inventory(state)

        if not cfg.get("tangential_escalate_on_no_readable_evidence", True):
            return None, inv
        # Already in tangential mode — nothing to escalate.
        if state.get("tangential_mode_active"):
            return None, inv
        # Any readable direct evidence at all means main mode still has a path.
        if inv["full_text"] > 0 or inv["pool"] > 0 or inv["curated"] > 0:
            return None, inv

        min_rounds = int(cfg.get("escalate_min_standard_rounds", 3))
        if inv["standard_rounds"] < min_rounds:
            return None, inv

        n_abstract = self._abstract_only_paper_count()
        min_abstract = int(cfg.get("escalate_min_abstract_papers", 3))
        if n_abstract < min_abstract:
            return None, inv

        return (f"{inv['catalog']} paper(s) in the catalog but 0 full texts "
                f"retrieved and 0 studies read in {inv['standard_rounds']} "
                f"standard round(s); {n_abstract} abstract-only paper(s) are "
                f"readable only in tangential mode"), inv

    def node_evidence_sufficiency(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 6: EVIDENCE SUFFICIENCY CHECK", color=Fore.MAGENTA)

        analyses = state.get("study_summaries", [])
        evidence_summary_lines = [
            f"Curated studies: {len(analyses)}",
            "",
            "Curated study summaries:"
        ]
        for i, a in enumerate(analyses, 1):
            findings = a.get("key_findings", [])
            findings_text = "; ".join(findings[:3]) if findings else "(no findings)"
            relevance = a.get("relevance_to_question", "?")
            apa = apa_in_text(a.get("paper_authors") or [], a.get("paper_year"))
            evidence_summary_lines.append(
                f"  {i}. {a.get('paper_title','?')} [{a.get('paper_year','?')}]\n"
                f"     APA in-text: {apa}\n"
                f"     Type: {a.get('study_type','?')}\n"
                f"     Findings: {findings_text}\n"
                f"     Relevance: {relevance}")
        evidence_text = "\n".join(evidence_summary_lines)

        inv = self._direct_evidence_inventory(state)
        inventory_text = (
            f"Papers in catalog: {inv['catalog']}\n"
            f"Papers with full text retrieved: {inv['full_text']}\n"
            f"Full-text papers NOT yet read: {inv['unread_full_text']}\n"
            f"Studies read across the whole run: {inv['pool']}\n"
            f"Read studies discarded by the relevance filter: {inv['filter_dropped']}\n"
            f"Read studies excluded during curation: {inv['curation_excluded']}\n"
            f"Standard search rounds completed: {inv['standard_rounds']}")

        history_text = self._format_search_history_for_distill(state.get("search_history", []))
        memo = state.get("strategy_memo", "")
        memo_block = ""
        if memo.strip():
            memo_block = f"\n\nSTRATEGY MEMO:\n{memo}"

        prompt = f"""You have completed multiple rounds of searching, reading, filtering, and
curating studies for an academic literature review. Decide: do you have enough
direct evidence to write the review, or do you need more searching?

RESEARCH QUESTION: "{state['original_query']}"

COMPLETE SEARCH HISTORY:
{history_text}
{memo_block}

CURATED EVIDENCE BASE (post-curation):
{evidence_text}

WHAT THE PIPELINE IS ACTUALLY HOLDING:
{inventory_text}

Read those numbers before you decide. The curated set above is what survived
filtering and curation — it is NOT a measure of how much literature exists. If
the catalog holds many papers and many were read then discarded, the shortfall
is in this system's selection, not in the field.

================ DECISION ================

Choose ONE:

A — "sufficient": curated evidence is enough to write a credible review.
   Proceed to synthesis.

B — "bad_picks": topic is well-studied (you see promising candidate titles
   in past searches that weren't selected, or queries clearly used wrong
   terminology). Do another standard search round with refined queries.

C — "sparse_direct_evidence": you have searched comprehensively. The topic
   itself is under-studied at the direct level. Unlock TANGENTIAL MODE to
   gather indirect evidence (class-level, mechanistic, adjacent populations,
   proxy outcomes).

Pick "sparse_direct_evidence" ONLY if the literature itself is thin. Before
choosing it, confirm all of the following:
  - the search history shows you tried the obvious direct terminology for this
    exact question, not just adjacent phrasings;
  - the candidate titles returned across rounds genuinely do not contain
    directly relevant studies;
  - few papers were read and discarded (a large "discarded" count means the
    studies existed and were thrown away — that is "bad_picks", not sparse).

A small curated set on a well-studied topic is almost always "bad_picks".
Well-researched questions — common supplements, licensed medications, standard
clinical interventions — have direct literature; if you found none, your
queries or your selection were wrong, not the field.

================ OUTPUT ================

Respond with ONLY JSON:
{{
    "decision": "sufficient | bad_picks | sparse_direct_evidence",
    "reasoning": "specific justification — name what's covered, what isn't, and what the search history shows",
    "evidence_gaps": ["specific gap 1", "specific gap 2"],
    "search_strategy_assessment": "1-2 sentences"
}}

In the "reasoning", "evidence_gaps", and "search_strategy_assessment" strings,
refer to any study using its APA 7th in-text citation (the "APA in-text" value
shown for each study, e.g. (Smith et al., 2020), or narratively as Smith et al.
(2020)) — do NOT refer to studies by their number."""

        result = self.agent_manager.run_primary(
            prompt, as_json=True, task="evidence_sufficiency")

        # ---- TANGENTIAL ENTRY GATE (deterministic) --------------------------
        # Whatever the model decides, tangential mode stays locked while direct
        # evidence is demonstrably unexploited. Downgrading to "bad_picks" sends
        # the run back for another STANDARD round with refined queries, which is
        # the correct response to a selection failure.
        if result.success and result.json_response:
            if result.json_response.get("decision") == "sparse_direct_evidence":
                action, block, inv = self._tangential_entry_block(state)
                if action:
                    print(f"\n  {Fore.YELLOW}{Style.BRIGHT}TANGENTIAL MODE REFUSED"
                          f"{Style.RESET_ALL}")
                    print(f"  {Fore.YELLOW}The model judged the literature sparse, "
                          f"but {block}.{Style.RESET_ALL}")
                    print(f"  {Fore.WHITE}Catalog {inv['catalog']} | full text "
                          f"{inv['full_text']} | read {inv['pool']} | "
                          f"filter dropped {inv['filter_dropped']} | curation "
                          f"excluded {inv['curation_excluded']}{Style.RESET_ALL}")
                    if action == "sufficient":
                        print(f"  {Fore.WHITE}Enough direct evidence is in hand — "
                              f"proceeding to synthesis.{Style.RESET_ALL}")
                    else:
                        print(f"  {Fore.WHITE}Treating this as a selection problem "
                              f"and running another standard round.{Style.RESET_ALL}")
                    result.json_response["decision"] = action
                    result.json_response["reasoning"] = (
                        f"[Overridden by the tangential entry gate: {block}.] "
                        + str(result.json_response.get("reasoning", "")))

        # ---- TANGENTIAL ESCALATION GATE (deterministic) ---------------------
        # The refusal gate above can only ever lock tangential mode. This one
        # unlocks it when main mode has provably run out of road: no full text
        # retrieved, nothing read, but abstract-bearing papers sitting in the
        # catalog that only tangential mode is allowed to read. Applied whatever
        # the model decided (including a failed call), because "run another
        # standard round" cannot change an outcome that main mode's own
        # full-text requirement has already determined.
        force_reason, force_inv = self._tangential_entry_force(state)
        if force_reason:
            print(f"\n  {Fore.MAGENTA}{Style.BRIGHT}TANGENTIAL MODE FORCED"
                  f"{Style.RESET_ALL}")
            print(f"  {Fore.MAGENTA}{force_reason}.{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}Another standard round cannot read those papers "
                  f"(main mode requires full text), so indirect/abstract evidence "
                  f"is unlocked instead.{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}Catalog {force_inv['catalog']} | full text "
                  f"{force_inv['full_text']} | read {force_inv['pool']} | "
                  f"curated {force_inv['curated']}{Style.RESET_ALL}")
            # Stagnation counted while main mode was structurally unable to use
            # anything it found. Tangential mode uses a different planner AND
            # different eligibility rules, so past stagnation no longer predicts
            # the next round; leaving it set would trip the stop guard in
            # _route_sufficiency and end the run at the moment of escalation.
            state["stagnant_round_count"] = 0
            prior = ""
            if result.success and result.json_response:
                prior = str(result.json_response.get("reasoning", ""))
            state["last_sufficiency_decision"] = "sparse_direct_evidence"
            state["sufficiency_reasoning"] = (
                f"[Escalated by the tangential escalation gate: {force_reason}.] "
                + prior).strip()
            print(f"  {Fore.MAGENTA}{Style.BRIGHT}Decision: sparse_direct_evidence "
                  f"(forced){Style.RESET_ALL}")
            return state

        if not result.success or not result.json_response:
            print(f"  {Fore.RED}Sufficiency check failed — defaulting to 'sufficient'.{Style.RESET_ALL}")
            state["last_sufficiency_decision"] = "sufficient"
            state["sufficiency_reasoning"] = "Decision call failed; proceeding."
            return state

        data = result.json_response
        decision = data.get("decision", "sufficient").strip().lower().replace(" ", "_")
        reasoning = data.get("reasoning", "")
        gaps = data.get("evidence_gaps", []) or []
        strategy_assess = data.get("search_strategy_assessment", "")

        if decision not in ("sufficient", "bad_picks", "sparse_direct_evidence"):
            decision = "sufficient"

        state["last_sufficiency_decision"] = decision
        state["sufficiency_reasoning"] = (reasoning + ("\n" + strategy_assess if strategy_assess else "")).strip()

        color = {"sufficient": Fore.GREEN, "bad_picks": Fore.YELLOW,
                 "sparse_direct_evidence": Fore.MAGENTA}.get(decision, Fore.WHITE)
        print(f"  {color}{Style.BRIGHT}Decision: {decision}{Style.RESET_ALL}")
        if reasoning:
            print(f"  {Fore.WHITE}Reasoning: {reasoning}{Style.RESET_ALL}")
        if strategy_assess:
            print(f"  {Fore.WHITE}Strategy assessment: {strategy_assess}{Style.RESET_ALL}")
        if gaps:
            print(f"  {Fore.WHITE}Remaining gaps:{Style.RESET_ALL}")
            for g in gaps[:5]:
                print(f"    {Fore.WHITE}- {g}{Style.RESET_ALL}")
        return state

    # =========================================================================
    # NODE: DEEP ANALYSIS
    # =========================================================================

    def node_deep_analysis(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 7: DEEP ANALYSIS — Verified Quote Extraction")

        in_tangential = state.get("tangential_mode_active", False) or \
            state.get("tangential_engagement_count", 0) > 0

        if state.get("tangential_mode_active"):
            state["tangential_mode_active"] = False
            state["in_tangential_round"] = False

        mode_label = "tangential" if in_tangential else "main"
        mode_color = Fore.MAGENTA if in_tangential else Fore.CYAN
        print(f"  {mode_color}Deep-analysis mode: {mode_label.upper()}{Style.RESET_ALL}")
        if not in_tangential:
            print(f"  {Fore.YELLOW}Main mode — papers without full text WILL BE DROPPED.{Style.RESET_ALL}")
        else:
            print(f"  {Fore.MAGENTA}Tangential mode — abstract-only papers allowed "
                  f"(quotes verified against abstract).{Style.RESET_ALL}")

        summaries = state.get("study_summaries", [])
        all_papers_map = {p.paper_id: p for p in self.discovery.get_all_papers()}
        min_verified = self.config.get("min_verified_quotes_per_study", 1)

        # ---- EVIDENCE TOP-UP -------------------------------------------------
        # Quotes can only be verified against text the system actually holds, so
        # a curated set of abstract-only papers yields an empty review. Refill
        # from the append-only pool, full-text papers first.
        target = int(self.config.get("min_studies_for_deep_analysis", 8))
        pool = state.get("all_study_summaries") or []
        if pool and len(summaries) < target:
            have = {s.get("paper_id") for s in summaries}
            weights = self.config.get("study_type_weights", {}) or {}

            def _pool_rank(s):
                paper = all_papers_map.get(s.get("paper_id"))
                has_text = 1 if (paper is not None and
                                 getattr(paper, "full_text_available", False)) else 0
                try:
                    rel = float(s.get("reliability_score") or 0)
                except (TypeError, ValueError):
                    rel = 0.0
                w = float(weights.get(str(s.get("study_type") or "").strip().lower(), 3))
                try:
                    yr = int(s.get("paper_year") or 0)
                except (TypeError, ValueError):
                    yr = 0
                return (has_text, rel + w, yr)

            extras = sorted((s for s in pool if s.get("paper_id") not in have),
                            key=_pool_rank, reverse=True)[:target - len(summaries)]
            if extras:
                with_text = sum(1 for s in extras
                                if getattr(all_papers_map.get(s.get("paper_id")),
                                           "full_text_available", False))
                print(f"  {Fore.YELLOW}Only {len(summaries)} curated study(ies) — "
                      f"topping up from the {len(pool)}-study evidence pool."
                      f"{Style.RESET_ALL}")
                print(f"  {Fore.YELLOW}Added {len(extras)} study(ies) "
                      f"({with_text} with full text):{Style.RESET_ALL}")
                for s in extras:
                    ft = ("full text" if getattr(
                        all_papers_map.get(s.get("paper_id")),
                        "full_text_available", False) else "abstract only")
                    print(f"  {Fore.YELLOW}    + {(s.get('paper_title') or '?')[:78]} "
                          f"[{s.get('paper_year','?')}, {ft}]{Style.RESET_ALL}")
                summaries = list(summaries) + extras
                state["study_summaries"] = summaries

        print(f"  {Fore.WHITE}Deep-analyzing {len(summaries)} curated studies.{Style.RESET_ALL}")

        analyses = []
        dropped = []
        total = len(summaries)
        start_time = time.time()

        for i, summary in enumerate(summaries):
            if self._interrupted:
                analyses.append(summary)
                continue
            paper_id = summary.get("paper_id", "")
            paper = all_papers_map.get(paper_id)
            if not paper:
                dropped.append((summary.get("paper_title", "?"), "paper not in catalog"))
                continue

            elapsed = time.time() - start_time
            if i > 0:
                per = elapsed / i
                remaining = per * (total - i)
                progress = f"(~{int(remaining//60)}m{int(remaining%60)}s remaining)"
            else:
                progress = ""

            print(f"\n  {Fore.CYAN}[{i+1}/{total}] {progress}{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}{apa_reference(paper.authors, paper.year, paper.title, paper.venue, paper.doi)}{Style.RESET_ALL}")

            analysis = self.study_analyzer.deep_analysis(
                paper, state["original_query"],
                existing_summary=summary, mode=mode_label)

            if analysis is None:
                dropped.append((paper.title, "no usable full text (main mode)"))
                continue

            verified_count = sum(
                1 for q in analysis.get("key_quotes", [])
                if isinstance(q, dict) and q.get("verified") and q.get("quote"))
            if verified_count < min_verified:
                dropped.append((paper.title, f"only {verified_count} verified quotes "
                                              f"(threshold: {min_verified})"))
                continue

            analyses.append(analysis)

        total_time = time.time() - start_time
        v_count = sum(1 for a in analyses if a.get("quotes_verified"))
        print(f"\n  {Fore.GREEN}Deep analysis: {len(analyses)} studies retained, "
              f"{v_count} with verified quotes ({int(total_time//60)}m {int(total_time%60)}s)"
              f"{Style.RESET_ALL}")

        if dropped:
            print(f"  {Fore.YELLOW}Dropped {len(dropped)} studies:{Style.RESET_ALL}")
            for title, reason in dropped[:20]:
                print(f"    {Fore.YELLOW}- {title}  ({reason}){Style.RESET_ALL}")
            if len(dropped) > 20:
                print(f"    {Fore.YELLOW}... and {len(dropped) - 20} more.{Style.RESET_ALL}")

        state["study_analyses"] = analyses
        state["evidence_base_cache"] = None
        return state

    # =========================================================================
    # NODE: METHODOLOGY / SYNTHESIS / SELF-REVIEW / SELF-FIX / VERIFY / COMPILE
    # =========================================================================

    def node_harvest_references(self, state: ReviewState) -> ReviewState:
        """
        PHASE 7b: BACKWARD CITATION CHASING (snowballing).

        For each retained study we mine its References section, ask the LLM which
        cited works are directly relevant PRIMARY studies, acquire them through
        the existing discovery engine, and run them through deep_analysis so their
        findings can be quoted DIRECTLY as primary evidence. This complements the
        primary-evidence-only rule: sentences that merely cite other studies are
        rejected as quotes, but the studies they point to are pursued here.

        Fully guarded: disabled-by-config, interrupt, or any error simply leaves
        the existing corpus untouched.
        """
        if not self.config.get("enable_reference_harvesting", True):
            return state
        self._print_phase_banner(
            "PHASE 7b: REFERENCE HARVEST — Chasing Cited Primary Studies",
            color=Fore.MAGENTA)
        if self._interrupted:
            print(f"  {Fore.YELLOW}Interrupted — skipping reference harvest.{Style.RESET_ALL}")
            return state

        analyses = state.get("study_analyses", [])
        if not analyses:
            print(f"  {Fore.YELLOW}No retained studies to harvest from.{Style.RESET_ALL}")
            return state

        try:
            new_analyses = self.reference_harvester.harvest(
                analyses, state["original_query"],
                log_dir=self.paths.get("logs_directory", "Logs"))
        except Exception as e:
            logger.warning(f"Reference harvest failed, continuing without it: {e}")
            new_analyses = []

        if new_analyses:
            state["study_analyses"] = analyses + new_analyses
            state["evidence_base_cache"] = None
            print(f"  {Fore.GREEN}Corpus grew by {len(new_analyses)} harvested "
                  f"primary studies (total {len(state['study_analyses'])}).{Style.RESET_ALL}")
        return state

    def node_post_deep_review(self, state: ReviewState) -> ReviewState:
        """
        PHASE 8b: POST-DEEP-ANALYSIS EVIDENCE & QUALITY GATE (authoritative).

        Runs AFTER deep analysis + reference harvest + methodology assessment, so
        it judges the studies that ACTUALLY produced verified primary quotes — not
        the pre-quote curated set (that earlier check could greenlight studies the
        quote gate then dropped). If the verified evidence base is too thin OR the
        methodology was assessed as weak, AND budget remains (discovery rounds
        left, under the time limit, retries left), it requests another search
        round with an adjusted strategy. Otherwise it proceeds to synthesis.
        Decision-only: mutates bookkeeping flags, never the evidence itself.
        """
        analyses = state.get("study_analyses", [])
        n_verified = sum(
            1 for a in analyses
            if any(isinstance(q, dict) and q.get("verified") and q.get("quote")
                   for q in a.get("key_quotes", [])))
        quality = str((state.get("methodology_assessment", {}) or {}).get(
            "overall_quality", "?")).lower()

        min_studies = int(self.config.get("min_studies_for_review", 3))
        max_minutes = float(self.config.get("post_review_max_minutes", 35))
        max_retries = int(self.config.get("post_review_max_retries", 2))
        rnd = state.get("discovery_round", 0)
        max_rnd = state.get("max_discovery_rounds", 15)
        retries = state.get("post_review_retries", 0)
        elapsed_min = (time.time() - state.get("run_start_time", time.time())) / 60.0

        thin = n_verified < min_studies
        weak = (quality == "weak")

        self._print_phase_banner("PHASE 8b: EVIDENCE & QUALITY GATE")
        print(f"  Studies with verified primary quotes: {n_verified} "
              f"(min {min_studies}) | methodology: {quality}")
        print(f"  Round {rnd}/{max_rnd} | retries {retries}/{max_retries} | "
              f"elapsed {elapsed_min:.0f}m (limit {max_minutes:.0f}m)")

        budget_ok = (rnd < max_rnd and retries < max_retries
                     and elapsed_min < max_minutes and not state.get("interrupted"))

        if (thin or weak) and budget_ok:
            reasons = []
            if thin:
                reasons.append(f"only {n_verified} verified-quote study(ies) "
                               f"(< {min_studies})")
            if weak:
                reasons.append("methodology assessed as weak")
            print(f"  {Fore.YELLOW}{Style.BRIGHT}Insufficient: {'; '.join(reasons)}. "
                  f"Re-searching with an adjusted strategy.{Style.RESET_ALL}")
            state["post_review_retries"] = retries + 1
            state["ready_to_write"] = False
            # force a fresh curation pass on the next loop so the newly found
            # papers are re-curated and re-deep-analysed promptly
            state["rounds_since_last_curation"] = self.config.get("curate_every_n_rounds", 5)
            state["_post_review_action"] = "research"
            return state

        if thin or weak:
            # ---- HARD EVIDENCE FLOOR ---------------------------------------
            # Previously this branch ALWAYS proceeded, which is how
            # min_studies_for_review became a preference that was discarded the
            # moment the clock ran out: the gate printed "only 0 verified-quote
            # study(ies) (< 3)" and then synthesised anyway. Falling below the
            # floor now ends the run with an explicit report instead of a review.
            #
            # Only the COUNT triggers the abort. A weak methodology across a
            # sufficient number of studies is a real finding that belongs in the
            # Limitations section, not a reason to withhold the review.
            floor_enabled = bool(self.config.get("abort_below_min_studies", True))
            if thin and floor_enabled:
                print(f"  {Fore.RED}{Style.BRIGHT}BELOW THE EVIDENCE FLOOR — "
                      f"no review will be written.{Style.RESET_ALL}")
                print(f"  {Fore.WHITE}{n_verified} study(ies) carry verified "
                      f"quotes; the configured minimum is {min_studies} "
                      f"('min_studies_for_review'). Search budget is spent, so "
                      f"there is no way to reach the floor this run.{Style.RESET_ALL}")
                print(f"  {Fore.WHITE}An explicit evidence report will be produced "
                      f"instead, including every study and verified quote found. "
                      f"Set 'abort_below_min_studies': False to synthesise a "
                      f"below-floor review anyway.{Style.RESET_ALL}")
                state["_insufficient_evidence_reason"] = (
                    f"{n_verified} study(ies) produced verified quotes, below the "
                    f"configured minimum of {min_studies} "
                    f"(min_studies_for_review), and the search budget "
                    f"(rounds/retries/time) was exhausted before the floor could "
                    f"be reached")
                state["_post_review_action"] = "proceed"
                return state
            print(f"  {Fore.YELLOW}Budget exhausted (round/retry/time limit) — "
                  f"proceeding to synthesis with the evidence on hand.{Style.RESET_ALL}")
        else:
            print(f"  {Fore.GREEN}{SYM_CHECK} Evidence base sufficient — "
                  f"proceeding to synthesis.{Style.RESET_ALL}")
        state["_post_review_action"] = "proceed"
        return state

    def node_assess_methodology(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 8: METHODOLOGY ASSESSMENT")
        analyses = state.get("study_analyses", [])
        if not analyses:
            return state

        already_filled = bool(state.get("methodology_gap_fill_done"))
        gap_enabled = bool(self.config.get("methodology_gap_fill_enabled", True))
        max_wk = int(self.config.get("methodology_gap_max_weaknesses", 6))

        studies = self._build_studies_for_assessment(analyses)
        # The model is asked to (a) grade each study, (b) reason about the body of
        # evidence as a whole, (c) state STRENGTHS and WEAKNESSES *relative to
        # answering the user's question*, (d) rate it, and (e) decide whether the
        # gaps are significant enough that NEW studies should be sought — and if so
        # propose ONE targeted academic search query per weakness/gap.
        gap_instruction = ""
        if gap_enabled and not already_filled:
            gap_instruction = f"""
6. GAP-FILLING DECISION. Decide whether the methodology is too WEAK to confidently
   answer the user's question AND whether new studies could plausibly be found to
   fill the gaps. If yes, set "significant_gaps": true and provide, for EACH
   distinct weakness/gap, ONE specific academic database search query that would
   find studies to address THAT gap (consider the kinds of searches a researcher
   would run: missing study designs, populations, mechanisms, outcomes, time
   periods, or direct-vs-indirect evidence). Provide at most {max_wk} weaknesses
   and the SAME number of matching queries (weakness i ↔ query i)."""
        else:
            gap_instruction = """
6. GAP-FILLING DECISION. A gap-fill pass has already been completed this run (or
   is disabled), so set "significant_gaps": false and leave "gap_search_queries"
   empty. Still report strengths and weaknesses honestly for the final write-up."""

        prompt = f"""You are critically assessing the METHODOLOGICAL QUALITY of the body of
studies collected to answer a specific research question. Be comprehensive,
structured, and give your REASONING — not one-word verdicts.

RESEARCH QUESTION (assess everything in relation to ANSWERING THIS): "{state['original_query']}"

STUDIES ({len(analyses)}):
{studies}

Work through the following:

IMPORTANT — HOW TO REFER TO STUDIES: In every prose field below ("reasoning",
"strengths", "weaknesses"), refer to each study by its APA 7th in-text citation
(e.g. (Smith et al., 2020), or narratively as Smith et al. (2020)) or by its
title — NEVER as "Study 5", "Studies 1-4", or any bare number. The numeric
"study_index" is used ONLY inside the "assessments" array to map back to the
list; it must not appear in any human-readable sentence.

1. PER-STUDY GRADING. For each study, grade methodology using the standard
   evidence hierarchy (systematic reviews / meta-analyses of RCTs highest; then
   RCTs; then cohort; then case-control / cross-sectional; then case reports /
   expert opinion lowest), also weighing sample size, blinding, control quality,
   and bias risk.

2. EVIDENCE-BASE REASONING. Reason about the studies AS A WHOLE: what designs
   dominate, how directly they bear on the question, how consistent they are, and
   how much confidence they support.

3. STRENGTHS. List the concrete methodological strengths of this evidence base
   for answering the research question.

4. WEAKNESSES / GAPS. List the concrete methodological weaknesses and COVERAGE
   GAPS *specifically relative to answering the user's question* (e.g. missing
   study designs, populations, mechanisms, outcomes, reliance on indirect or
   low-tier evidence, thin sample of direct studies).

5. RATING. Give an overall quality label and a 1-10 rating, with a short
   justification that references the strengths and weaknesses above.
{gap_instruction}

Respond with ONLY JSON:
{{
    "assessments": [
        {{"study_index": 1, "final_reliability": 1-10, "evidence_weight": "high|moderate|low", "bias_risk": "low|moderate|high", "bias_reasoning": "1 sentence"}}
    ],
    "overall_evidence_quality": "strong|moderate|weak|mixed",
    "rating": 1-10,
    "reasoning": "2-4 sentences reasoning about the body of evidence as a whole",
    "strengths": ["...", "..."],
    "weaknesses": ["...", "..."],
    "significant_gaps": true/false,
    "gap_search_queries": ["one query per weakness, same order as weaknesses"]
}}"""

        result = self.agent_manager.run_primary(
            prompt, as_json=True, task="methodology_assessment")
        if result.success and result.json_response:
            data = result.json_response
            for a in data.get("assessments", []):
                idx = a.get("study_index", 0) - 1
                if 0 <= idx < len(analyses):
                    for k in ("final_reliability", "evidence_weight", "bias_risk", "bias_reasoning"):
                        analyses[idx][k] = a.get(k)

            strengths = [s for s in (data.get("strengths") or []) if str(s).strip()]
            weaknesses = [w for w in (data.get("weaknesses") or []) if str(w).strip()]
            # Deterministically turn any 'Study N' references into APA citations
            # so the reader can always tell which study is meant.
            summary_txt = self._dereference_study_numbers(
                data.get("reasoning", "") or data.get("evidence_summary", ""), analyses)
            strengths = [self._dereference_study_numbers(s, analyses) for s in strengths]
            weaknesses = [self._dereference_study_numbers(w, analyses) for w in weaknesses]
            queries = [q for q in (data.get("gap_search_queries") or []) if str(q).strip()]
            significant = bool(data.get("significant_gaps")) and not already_filled and gap_enabled
            # Pair each query with a weakness; cap both at max_wk.
            queries = queries[:max_wk]
            weaknesses_capped = weaknesses[:max_wk]

            state["methodology_assessment"] = {
                "overall_quality": data.get("overall_evidence_quality", "?"),
                "rating": data.get("rating"),
                "summary": summary_txt,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "significant_gaps": bool(data.get("significant_gaps")),
            }
            state["methodology_weaknesses"] = weaknesses_capped
            state["methodology_gap_queries"] = queries if significant else []
            state["methodology_gaps_significant"] = significant and bool(queries)

            self._print_methodology_assessment(state["methodology_assessment"],
                                                significant and bool(queries),
                                                weaknesses_capped, queries,
                                                already_filled)
        else:
            print(f"  {Fore.YELLOW}Methodology assessment returned no usable JSON — "
                  f"keeping prior assessment.{Style.RESET_ALL}")
        state["study_analyses"] = analyses
        state["evidence_base_cache"] = None
        return state

    def _print_methodology_assessment(self, ma, will_gap_fill, weaknesses,
                                      queries, already_filled):
        """Comprehensive, structured print of the methodology assessment (#2)."""
        rating = ma.get("rating")
        rating_str = f" | rating: {rating}/10" if rating not in (None, "") else ""
        print(f"  {Fore.GREEN}{SYM_CHECK} Overall quality: "
              f"{ma.get('overall_quality','?')}{rating_str}{Style.RESET_ALL}")
        if ma.get("summary"):
            print(f"  {Fore.WHITE}Assessment: {ma['summary']}{Style.RESET_ALL}")
        if ma.get("strengths"):
            print(f"  {Fore.GREEN}Strengths:{Style.RESET_ALL}")
            for s in ma["strengths"]:
                print(f"    {Fore.GREEN}+ {s}{Style.RESET_ALL}")
        if ma.get("weaknesses"):
            print(f"  {Fore.YELLOW}Weaknesses / gaps (re: answering the question):{Style.RESET_ALL}")
            for w in ma["weaknesses"]:
                print(f"    {Fore.YELLOW}- {w}{Style.RESET_ALL}")
        if already_filled:
            print(f"  {Fore.CYAN}(Gap-fill already completed this run — this is the final "
                  f"methodology assessment for the review.){Style.RESET_ALL}")
        elif will_gap_fill:
            print(f"  {Fore.MAGENTA}{Style.BRIGHT}Significant gaps detected — running a ONE-TIME "
                  f"targeted gap-fill ({len(queries)} weakness search(es)):{Style.RESET_ALL}")
            for i, q in enumerate(queries):
                wk = weaknesses[i] if i < len(weaknesses) else "(weakness)"
                print(f"    {Fore.MAGENTA}{i+1}. weakness: {wk}{Style.RESET_ALL}")
                print(f"       {Fore.BLUE}{SYM_SEARCH} {q}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.GREEN}No significant fillable gaps — proceeding to synthesis."
                  f"{Style.RESET_ALL}")

    # ---------- (#2) Methodology gap-fill (one-time, weakness-targeted) ----------

    def _deep_analyse_main(self, summaries: List[Dict], query: str) -> List[Dict]:
        """Deep-analyse a set of study summaries in MAIN mode (full text required,
        quotes verified against full text). Mirrors node_deep_analysis's core but
        is forced to main mode and operates on a caller-supplied subset, so the
        gap-fill can analyse ONLY the new studies while leaving previously
        analysed studies (and their verified quotes) untouched."""
        if not summaries:
            return []
        all_papers_map = {p.paper_id: p for p in self.discovery.get_all_papers()}
        min_verified = self.config.get("min_verified_quotes_per_study", 1)
        analyses: List[Dict] = []
        dropped: List[Tuple[str, str]] = []
        total = len(summaries)
        start_time = time.time()
        for i, summary in enumerate(summaries):
            if self._interrupted:
                break
            paper = all_papers_map.get(summary.get("paper_id", ""))
            if not paper:
                dropped.append((summary.get("paper_title", "?"), "paper not in catalog"))
                continue
            print(f"\n  {Fore.CYAN}[{i+1}/{total}] (gap-fill){Style.RESET_ALL}")
            print(f"  {Fore.CYAN}{apa_reference(paper.authors, paper.year, paper.title, paper.venue, paper.doi)}{Style.RESET_ALL}")
            analysis = self.study_analyzer.deep_analysis(
                paper, query, existing_summary=summary, mode="main")
            if analysis is None:
                dropped.append((paper.title, "no usable full text (main mode)"))
                continue
            verified_count = sum(
                1 for q in analysis.get("key_quotes", [])
                if isinstance(q, dict) and q.get("verified") and q.get("quote"))
            if verified_count < min_verified:
                dropped.append((paper.title, f"only {verified_count} verified quotes "
                                              f"(threshold: {min_verified})"))
                continue
            analyses.append(analysis)
        print(f"\n  {Fore.GREEN}Deep analysis (gap-fill): {len(analyses)} new study(ies) "
              f"retained.{Style.RESET_ALL}")
        if dropped:
            print(f"  {Fore.YELLOW}Dropped {len(dropped)}:{Style.RESET_ALL}")
            for title, reason in dropped[:20]:
                print(f"    {Fore.YELLOW}- {title}  ({reason}){Style.RESET_ALL}")
            if len(dropped) > 20:
                print(f"    {Fore.YELLOW}... and {len(dropped) - 20} more.{Style.RESET_ALL}")
        return analyses

    def node_methodology_gap_fill(self, state: ReviewState) -> ReviewState:
        """ONE-TIME, weakness-targeted gap-fill (#2). Sequence (matching the spec):
        one search per identified weakness (seeded with the user's question AND
        that weakness) -> acquire -> quick-read (reuses the main-mode node) ->
        relevance filter (reuses the node) -> deep-analyse ONLY the new studies
        (previous verified quotes preserved) -> ONE more reference harvest on the
        new studies -> merge. Control then returns to the methodology assessment
        for a fresh, up-to-date judgement used by the final review. Runs at most
        once per run (guarded by methodology_gap_fill_done)."""
        self._print_phase_banner(
            "PHASE 8c: METHODOLOGY GAP-FILL (one-time, weakness-targeted)",
            color=Fore.MAGENTA)

        queries = list(state.get("methodology_gap_queries", []) or [])
        weaknesses = list(state.get("methodology_weaknesses", []) or [])
        cap = int(self.config.get("methodology_gap_max_weaknesses", 6))
        queries = queries[:cap]
        weaknesses = weaknesses[:cap]

        if not queries or self._interrupted:
            print(f"  {Fore.YELLOW}No gap queries (or interrupted) — skipping gap-fill."
                  f"{Style.RESET_ALL}")
            state["methodology_gap_fill_done"] = True
            return state

        preserved = list(state.get("study_analyses", []) or [])
        preserved_ids = {(a.get("paper_id") or a.get("source_paper_id")) for a in preserved}
        preserved_ids.discard(None)
        # Everything seen BEFORE this gap-fill (read, summarised, or analysed), so
        # "new studies" below means ONLY papers the gap searches newly discover —
        # not previously-seen studies that were dropped earlier in the run.
        pre_seen_ids = set(state.get("read_paper_ids", []) or [])
        pre_seen_ids |= {(s.get("paper_id") or s.get("source_paper_id"))
                         for s in (state.get("study_summaries", []) or [])}
        pre_seen_ids |= preserved_ids
        pre_seen_ids.discard(None)
        print(f"  {Fore.GREEN}Preserving {len(preserved)} already-analysed study(ies) WITH "
              f"their verified quotes.{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}Targeting {len(queries)} methodology weakness(es), "
              f"1 search each.{Style.RESET_ALL}")

        identified_scope = (state.get("research_plan") or {}).get("identified_scope", "")
        oq = state["original_query"]

        # 1) ONE targeted search per weakness, seeded with the user's question AND
        #    the specific weakness (selection still anchored to the question).
        for i, q in enumerate(queries):
            if self._interrupted:
                break
            wk = weaknesses[i] if i < len(weaknesses) else q
            print(f"\n  {Fore.MAGENTA}Gap {i+1}/{len(queries)} — weakness: {wk}{Style.RESET_ALL}")
            print(f"  {Fore.BLUE}  {SYM_SEARCH} {q}{Style.RESET_ALL}")
            try:
                results = self.discovery.search_all_apis(q, limit_per_api=20)
                self.planner.record_query_used(q)
            except Exception as e:
                logger.warning(f"Gap-fill search failed for '{q}': {e}")
                results = []
            state.setdefault("search_history", [])
            if not results:
                print(f"  {Fore.YELLOW}  No results.{Style.RESET_ALL}")
                state["search_history"].append({
                    "round": f"GAP{i+1}", "mode": "gap_fill", "focus_area": wk,
                    "query": q, "total_candidates": 0, "selected_count": 0,
                    "outcome": "no_results", "candidate_titles": []})
                continue
            focus = f"Methodology gap to fill: {wk}"
            selected, per_paper_reasoning = self._llm_select_papers(
                oq, focus, results, identified_scope=identified_scope)
            added = 0
            if selected:
                added = self.discovery.add_selected_papers(selected)
                print(f"  {Fore.GREEN}  {SYM_CHECK} Selected {len(selected)} paper(s) "
                      f"({added} new to catalog).{Style.RESET_ALL}")
                for sp in selected:
                    print(f"  {Fore.WHITE}    -> [{sp.year or '?'}] {sp.title}{Style.RESET_ALL}")
            else:
                print(f"  {Fore.YELLOW}  No relevant papers selected for this gap.{Style.RESET_ALL}")
            state["search_history"].append({
                "round": f"GAP{i+1}", "mode": "gap_fill", "focus_area": wk, "query": q,
                "total_candidates": len(results), "selected_count": len(selected),
                "outcome": "selected" if selected else "none_relevant",
                "candidate_titles": [{"title": p.title, "year": p.year} for p in results],
                "selected_titles": [{"title": p.title, "year": p.year} for p in selected]})

        # 2) Acquire full texts for the newly selected papers.
        print(f"\n  {Fore.CYAN}{SYM_PAPER} Acquiring full texts for gap-fill papers..."
              f"{Style.RESET_ALL}")
        try:
            s, f = self.discovery.acquire_full_texts_batch()
            print(f"  {Fore.GREEN}Full text: {s}/{s+f}{Style.RESET_ALL}")
        except Exception as e:
            logger.warning(f"Gap-fill acquisition failed: {e}")

        # 3) Quick-read new papers + 4) relevance filter — reuse the existing
        #    main-mode nodes (in_tangential_round is False here, so they run in
        #    main mode exactly as the normal first pass does).
        state = self.node_quick_read_studies(state)
        state = self.node_filter_relevance(state)

        # 5) Identify which (surviving) summaries are NEW vs already seen.
        all_summaries = list(state.get("study_summaries", []) or [])
        new_summaries = [s for s in all_summaries
                         if (s.get("paper_id") or s.get("source_paper_id")) not in pre_seen_ids]
        print(f"\n  {Fore.WHITE}{len(new_summaries)} new study summary(ies) to deep-analyse; "
              f"{len(preserved)} previous study(ies) kept with their quotes.{Style.RESET_ALL}")

        # 6) Deep-analyse ONLY the new studies (main mode). Previous analyses and
        #    their verified quotes are never touched.
        new_analyses = self._deep_analyse_main(new_summaries, oq)

        # 7) ONE more reference harvest, on the NEW studies only.
        if (new_analyses and not self._interrupted
                and self.config.get("enable_reference_harvesting", True)):
            state["study_analyses"] = new_analyses
            state = self.node_harvest_references(state)   # -> new_analyses + harvested
            new_analyses = state.get("study_analyses", new_analyses)

        # 8) Merge preserved (kept verbatim, quotes intact) + new (dedup by id).
        merged = list(preserved)
        seen = set(preserved_ids)
        for a in new_analyses:
            pid = a.get("paper_id") or a.get("source_paper_id")
            if pid in seen:
                continue
            seen.add(pid)
            merged.append(a)
        state["study_analyses"] = merged
        state["study_summaries"] = all_summaries
        state["evidence_base_cache"] = None
        state["methodology_gap_fill_done"] = True

        n_new = len(merged) - len(preserved)
        print(f"\n  {Fore.GREEN}{Style.BRIGHT}Gap-fill complete: added {n_new} new study(ies); "
              f"corpus now {len(merged)} (previous quotes preserved).{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}Re-running methodology assessment for the final review so the "
              f"methodology section is up to date...{Style.RESET_ALL}")
        return state

    def _build_studies_for_assessment(self, analyses):
        lines = []
        for i, a in enumerate(analyses, 1):
            cite = self._apa_in_text_citation(a)
            lines.append(f"\n--- Study {i} — refer to this study as {cite} ---\n"
                         f"Title: {a.get('paper_title','?')}\n"
                         f"APA in-text citation (USE THIS, not 'Study {i}', in any prose): {cite}\n"
                         f"Type: {a.get('study_type','?')} | Sample: {a.get('sample_size','?')}\n"
                         f"Year: {a.get('paper_year','?')} | Citations: {a.get('paper_citation_count','?')}\n"
                         f"Findings: {json.dumps(a.get('key_findings',[]))}\n"
                         f"Limitations: {json.dumps(a.get('limitations',[]))}\n"
                         f"Reliability: {a.get('reliability_score','?')}/10")
        return "\n".join(lines)

    # ---------- APA helpers ----------

    def _apa_in_text_citation(self, analysis: Dict) -> str:
        """Return an APA 7th in-text citation string like '(Smith et al., 2020)'."""
        return apa_in_text(analysis.get("paper_authors") or [], analysis.get("paper_year"))

    def _dereference_study_numbers(self, text: str, analyses: List[Dict]) -> str:
        """Deterministically replace bare 'Study N' / 'Studies 1-4' / 'Studies 5
        and 6' references with the studies' APA in-text citations, so the reader
        can always tell WHICH study is meant (the index has no meaning outside the
        assessment input). Unmappable numbers are left as-is."""
        if not text or not analyses:
            return text

        def cite_for(num: int):
            idx = num - 1
            if 0 <= idx < len(analyses):
                return self._apa_in_text_citation(analyses[idx]) or None
            return None

        def join_cites(cites: List[str]) -> str:
            cites = [c for c in cites if c]
            if not cites:
                return ""
            if len(cites) == 1:
                return cites[0]
            return ", ".join(cites[:-1]) + " and " + cites[-1]

        # 1) Ranges, e.g. "Studies 1-4".
        def repl_range(m):
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b and (b - a) <= 50:
                joined = join_cites([cite_for(k) for k in range(a, b + 1)])
                if joined:
                    return joined
            return m.group(0)
        text = re.sub(r'\bStudies?\s+(\d+)\s*[-\u2013\u2014]\s*(\d+)',
                      repl_range, text, flags=re.IGNORECASE)

        # 2) Lists, e.g. "Studies 5 and 6", "Studies 5 & 6", "Studies 5, 6".
        def repl_list(m):
            nums = [int(x) for x in re.findall(r'\d+', m.group(0))]
            joined = join_cites([cite_for(k) for k in nums])
            return joined if joined else m.group(0)
        text = re.sub(r'\bStudies\s+\d+(?:\s*(?:,|&|and)\s*\d+)+',
                      repl_list, text, flags=re.IGNORECASE)

        # 3) Singles, e.g. "Study 5".
        def repl_single(m):
            c = cite_for(int(m.group(1)))
            return c if c else m.group(0)
        text = re.sub(r'\bStudy\s+(\d+)', repl_single, text, flags=re.IGNORECASE)
        return text

    def _apa_reference_entry(self, analysis: Dict) -> str:
        """Best-effort APA 7th reference entry from available metadata."""
        return apa_reference(
            analysis.get("paper_authors") or [],
            analysis.get("paper_year"),
            analysis.get("paper_title"),
            analysis.get("paper_venue"),
            analysis.get("paper_doi"),
        )

    # ---------- Evidence block (used by synthesis & verification) ----------

    def _assign_registry(self, analyses):
        """Assign STABLE GLOBAL quote tokens [[Qn]] across all studies in order
        and return the registry. This is the single source of token numbering;
        _render_evidence_full reuses these exact ids so the rendered evidence
        text is identical whether produced whole (full mode) or per-batch
        (low-end mode). registry[str(n)] = {text, citation, source_paper_id,
        source_paper_title, word_count, study_index}.
        """
        registry: Dict[str, Dict] = {}
        qid = 0
        for i, a in enumerate(analyses, 1):
            apa_in_text = self._apa_in_text_citation(a)
            source_pid = a.get("paper_id") or a.get("source_paper_id")
            # STUDY-SUMMARY token [[Si]] — the frozen, grounded paraphrase used to
            # introduce this study. Keyed "S{i}" so it never collides with numeric
            # quote ids. is_summary=True marks it as a paraphrase (pasted as plain
            # prose, not a verbatim quote, and not verbatim-verified).
            summ = (a.get("study_summary") or "").strip()
            if summ:
                registry[f"S{i}"] = {
                    "text": summ,
                    "citation": apa_in_text,
                    "source_paper_id": source_pid,
                    "source_paper_title": a.get("paper_title"),
                    "word_count": len(summ.split()),
                    "study_index": i,
                    "is_summary": True,
                }
            for q in a.get("key_quotes", []):
                if isinstance(q, dict) and q.get("verified") and q.get("quote"):
                    qid += 1
                    qtext = q.get("quote", "")
                    registry[str(qid)] = {
                        "text": qtext,
                        "citation": apa_in_text,
                        "source_paper_id": source_pid,
                        "source_paper_title": a.get("paper_title"),
                        "word_count": len(qtext.split()),
                        "study_index": i,
                    }
        return registry

    def _render_evidence_full(self, analyses, registry, only_studies=None):
        """Render the full evidence text (with quote text + global tokens). When
        only_studies (a set of 1-based study indices) is given, only those
        studies are rendered, but their STUDY numbers and token ids stay global
        — used to build per-batch evidence in low-end synthesis. With
        only_studies=None this reproduces the original evidence base exactly.
        """
        by_study: Dict[int, List[int]] = {}
        for qid_str, e in registry.items():
            if e.get("is_summary") or not str(qid_str).isdigit():
                continue
            by_study.setdefault(e["study_index"], []).append(int(qid_str))
        for k in by_study:
            by_study[k].sort()

        lines = []
        for i, a in enumerate(analyses, 1):
            if only_studies is not None and i not in only_studies:
                continue
            rel = a.get("final_reliability") or a.get("reliability_score", "?")
            apa_in_text = self._apa_in_text_citation(a)
            text_type = a.get("deep_analysis_text_type") or (
                "FULL TEXT" if a.get("has_full_text") else "ABSTRACT ONLY")
            lines.append(f"\n{'='*60}\nSTUDY {i}: {a.get('paper_title','?')}\n{'='*60}")
            lines.append(f"APA in-text: {apa_in_text}")
            lines.append(f"Authors: {', '.join(a.get('paper_authors',[])[:7])}")
            lines.append(f"Year: {a.get('paper_year','?')} | Venue: {a.get('paper_venue','?')} | "
                         f"DOI: {a.get('paper_doi','N/A')}")
            lines.append(f"Type: {a.get('study_type','?')} | Sample: {a.get('sample_size','?')} | "
                         f"Reliability: {rel}/10 | Evidence weight: {a.get('evidence_weight','?')}")
            lines.append(f"Source text used for quotes: {text_type}")
            lines.append(f"Methodology: {a.get('methodology_summary','N/A')}")

            summ = (a.get("study_summary") or "").strip()
            if summ:
                lines.append(
                    f"\n[STUDY-INTRO TOKEN [[S{i}]] — a frozen, grounded paraphrase of THIS "
                    f"study (its aim, design/methods, sample/scope, and the nature of its "
                    f"findings). To INTRODUCE this study, write [[S{i}]] at the point the "
                    f"introduction belongs; it is pasted in as PLAIN PROSE (NOT a quote, no "
                    f"quotation marks). Place it BEFORE any of this study's quotes on first "
                    f"mention, add the citation {apa_in_text} after it, and write your own "
                    f"connecting sentences around it. Do NOT wrap it in quotation marks.]")
                lines.append(f'   [[S{i}]] expands to: "{summ}"')

            findings = a.get("key_findings", [])
            if findings:
                lines.append("Reported findings:")
                for f in findings:
                    lines.append(f"  - {f}")

            verified = [q for q in a.get("key_quotes", [])
                        if isinstance(q, dict) and q.get("verified") and q.get("quote")]

            if verified:
                qids = by_study.get(i, [])
                lines.append("\n[VERIFIED QUOTES — to use one, write its TOKEN exactly (e.g. "
                             "[[Q1]]) at the spot you want it. The exact verified text is pasted "
                             "in automatically; do NOT retype the quote text yourself]:")
                for pos, q in enumerate(verified):
                    token = f"[[Q{qids[pos]}]]" if pos < len(qids) else "[[Q?]]"
                    qtext = q.get("quote", "")
                    wc = len(qtext.split())
                    lines.append(f'  {token}  ({wc} words) — "{qtext}"')
                    lines.append(f'      CITE AS: {apa_in_text}')
                    ctx = q.get("context", "")
                    if ctx:
                        lines.append(f"      (context: {ctx})")
            else:
                lines.append("\n[NO VERIFIED QUOTES AVAILABLE — do NOT use direct quotation "
                             "from this study; you may only describe it in your own words "
                             "where helpful.]")

            lims = a.get("limitations", [])
            if lims:
                lines.append(f"Limitations: {', '.join(lims)}")
            if a.get("bias_reasoning"):
                lines.append(f"Bias: {a.get('bias_reasoning')}")
        return "\n".join(lines)

    def _render_evidence_compact(self, analyses, registry, only_studies=None):
        """A compact study index: per study, its citation/type and the list of
        available quote TOKENS (with word counts) but NO quote text. Used for
        the review-checking stages and the low-end reduce step. Safe because
        quote integrity is guaranteed by the token system, not by the LLM
        re-reading quote text."""
        by_study: Dict[int, List[int]] = {}
        for qid_str, e in registry.items():
            if e.get("is_summary") or not str(qid_str).isdigit():
                continue
            by_study.setdefault(e["study_index"], []).append(int(qid_str))
        for k in by_study:
            by_study[k].sort()

        lines = []
        for i, a in enumerate(analyses, 1):
            if only_studies is not None and i not in only_studies:
                continue
            rel = a.get("final_reliability") or a.get("reliability_score", "?")
            apa_in_text = self._apa_in_text_citation(a)
            lines.append(f"STUDY {i}: {a.get('paper_title','?')}")
            lines.append(f"  Cite as: {apa_in_text} | Type: {a.get('study_type','?')} | "
                         f"Sample: {a.get('sample_size','?')} | Reliability: {rel}/10")
            qids = by_study.get(i, [])
            if qids:
                toks = " ".join(f"[[Q{q}]]({registry[str(q)]['word_count']}w)" for q in qids)
                lines.append(f"  Quote tokens available: {toks}")
            else:
                lines.append(f"  Quote tokens available: none")
        return "\n".join(lines)

    def _build_evidence(self, analyses):
        """Build the evidence-base string AND a quote registry (full mode).

        Returns (evidence_text, registry), identical to the previous behaviour;
        internally delegates to _assign_registry + _render_evidence_full.
        """
        registry = self._assign_registry(analyses)
        text = self._render_evidence_full(analyses, registry)
        return text, registry

    def _get_evidence(self, state):
        cached = state.get("evidence_base_cache")
        if cached and state.get("quote_registry"):
            return cached
        ev, registry = self._build_evidence(state.get("study_analyses", []))
        state["evidence_base_cache"] = ev
        state["quote_registry"] = registry
        return ev

    def _get_evidence_for_checks(self, state):
        """Evidence shown to the review-checking stages (self-review, self-fix,
        verification). In full mode this is the complete evidence base (current
        behaviour). In low-end mode it is the COMPACT token index, so these
        stages fit a small context window; quote integrity is still enforced
        deterministically by the token system + attribution check."""
        if self._low_end:
            if not state.get("quote_registry"):
                state["quote_registry"] = self._assign_registry(
                    state.get("study_analyses", []))
                if not state.get("evidence_base_cache"):
                    state["evidence_base_cache"] = "(low-end: compact evidence index)"
            return self._render_evidence_compact(
                state.get("study_analyses", []), state.get("quote_registry") or {})
        return self._get_evidence(state)

    # ---------- Synthesis (APA 7th, quote-driven) ----------

    # ==================================================================
    #  TWO-PHASE EVIDENCE SYNTHESIS
    #  The model emits only a PLAN (numbers); code assembles the evidence.
    # ==================================================================

    def _narrative_cite(self, apa_in_text: str) -> str:
        """'(Hoffman et al., 2022)' -> 'Hoffman et al. (2022)'. Falls back to the
        original string if it does not look like a standard parenthetical."""
        if not apa_in_text:
            return apa_in_text
        m = re.match(r"^\((.*),\s*(\d{4}[a-z]?)\)$", apa_in_text.strip())
        if m:
            return f"{m.group(1)} ({m.group(2)})"
        return apa_in_text

    def _valid_study_quote_map(self, registry: Dict, analyses: List[Dict]):
        """Return (valid_studies, study_quotes, study_cite):
        - valid_studies: 1-based indices that have >=1 verified quote token
        - study_quotes[i]: ordered list of that study's global quote ids (ints)
        - study_cite[i]: the APA in-text citation for that study
        """
        study_quotes: Dict[int, List[int]] = {}
        study_cite: Dict[int, str] = {}
        for key, e in registry.items():
            if str(key).isdigit():
                si = e.get("study_index")
                if si:
                    study_quotes.setdefault(si, []).append(int(key))
                    study_cite[si] = e.get("citation", "")
        for i in study_quotes:
            study_quotes[i].sort()
        # citation fallback from summary entries / analyses
        for i in list(study_quotes.keys()):
            if not study_cite.get(i):
                se = registry.get(f"S{i}")
                if se and se.get("citation"):
                    study_cite[i] = se["citation"]
                elif 1 <= i <= len(analyses):
                    study_cite[i] = self._apa_in_text_citation(analyses[i - 1])
        valid_studies = sorted(study_quotes.keys())
        return valid_studies, study_quotes, study_cite

    def _default_evidence_plan(self, registry: Dict, analyses: List[Dict]) -> Dict:
        """Fallback plan: every quoted study in one section, registry order, all
        of each study's quotes."""
        valid, sq, _ = self._valid_study_quote_map(registry, analyses)
        return {"sections": [{
            "heading": "Evidence",
            "studies": [{"study": i, "quotes": list(sq[i])} for i in valid],
        }]}

    def _request_evidence_plan(self, state: ReviewState, evidence: str) -> Optional[Dict]:
        """Ask the model to GROUP studies thematically and choose quote order.
        It outputs ONLY numbers (study indices + quote token ids), so nothing it
        says can leak into the evidence text."""
        prompt = f"""You are PLANNING the EVIDENCE section of a literature review that must
answer the research question below. You will NOT write any prose here — you only
decide the STRUCTURE, as numbers.

RESEARCH QUESTION: "{state['original_query']}"

EVIDENCE BASE (each STUDY has an index, a citation, and verified quote tokens
[[Qn]] with their text shown for your reference):
{evidence}

YOUR TASK — produce a thematic plan that best answers the question:
1. Group the studies into a small number of THEMATIC sections (typically 2-5),
   each with a short descriptive heading derived from the question/evidence.
2. Order the studies within each section so the evidence reads logically.
3. For each study, choose WHICH of its quote tokens to include and in WHAT order
   (pick the quotes that best support the point; you may include several).
4. A study may appear in MORE THAN ONE section if its evidence spans themes — it
   will be introduced on first appearance and back-referenced afterwards.
5. Omit studies that do not help answer the question.

Output ONLY this JSON (numbers only — NO quote text, NO prose):
{{
  "sections": [
    {{
      "heading": "Short thematic heading",
      "studies": [
        {{"study": <study index number>, "quotes": [<quote token number>, ...]}}
      ]
    }}
  ]
}}

Use the integer STUDY index (e.g. 3 for STUDY 3) and the integer quote token id
(e.g. 12 for [[Q12]]). Every quote id you list MUST belong to that study. Do not
invent ids."""
        result = self.agent_manager.run_primary(prompt, task="synthesis")
        if result.success and result.json_response:
            return result.json_response
        if result.success and result.response:
            try:
                from json_parser import parse_json_response
                return parse_json_response(result.response)
            except Exception:
                return None
        return None

    def _validate_evidence_plan(self, plan: Dict, registry: Dict,
                                analyses: List[Dict]):
        """Normalise + hard-validate the plan against the registry. Returns
        (clean_plan, dropped_studies). Guarantees: every study/quote id exists,
        each quote used at most once globally, each listed study has >=1 valid
        quote, headings are non-empty strings. Falls back to a default plan if
        nothing valid remains."""
        valid, study_quotes, _ = self._valid_study_quote_map(registry, analyses)
        valid_set = set(valid)
        used_quotes: set = set()
        clean_sections = []
        placed_studies: set = set()
        if isinstance(plan, dict) and isinstance(plan.get("sections"), list):
            for sec in plan["sections"]:
                if not isinstance(sec, dict):
                    continue
                heading = str(sec.get("heading", "") or "").strip()
                items_in = sec.get("studies") or sec.get("items") or []
                if not isinstance(items_in, list):
                    continue
                clean_items = []
                for it in items_in:
                    if not isinstance(it, dict):
                        continue
                    try:
                        si = int(it.get("study"))
                    except (TypeError, ValueError):
                        continue
                    if si not in valid_set:
                        continue
                    allowed = study_quotes.get(si, [])
                    qs_in = it.get("quotes") or it.get("quote_ids") or []
                    if not isinstance(qs_in, list):
                        qs_in = []
                    chosen = []
                    for q in qs_in:
                        try:
                            qi = int(q)
                        except (TypeError, ValueError):
                            continue
                        if qi in allowed and qi not in used_quotes and qi not in chosen:
                            chosen.append(qi)
                    if not chosen:
                        continue
                    used_quotes.update(chosen)
                    clean_items.append({"study": si, "quotes": chosen})
                    placed_studies.add(si)
                if clean_items:
                    clean_sections.append({
                        "heading": heading or "Evidence",
                        "studies": clean_items,
                    })
        if not clean_sections:
            return self._default_evidence_plan(registry, analyses), []
        # Append any quoted-but-unplaced studies (with their remaining quotes) so
        # genuinely useful evidence is never silently lost — into a final section.
        leftover_items = []
        for i in valid:
            if i in placed_studies:
                continue
            remaining = [q for q in study_quotes[i] if q not in used_quotes]
            if remaining:
                used_quotes.update(remaining)
                leftover_items.append({"study": i, "quotes": remaining})
        if leftover_items:
            clean_sections.append({
                "heading": "Further Evidence",
                "studies": leftover_items,
            })
        dropped = [i for i in valid if i not in placed_studies and
                   not any(it["study"] == i for it in leftover_items)]
        return {"sections": clean_sections}, dropped

    def _assemble_evidence(self, plan: Dict, registry: Dict, analyses: List[Dict]):
        """Deterministically build the EVIDENCE section text (with [[Sn]]/[[Qn]]
        tokens + connector slot markers) from a validated plan. Each study is
        introduced (summary token) on FIRST appearance and back-referenced on
        later appearances. Returns (evidence_text, n_slots)."""
        _, _, study_cite = self._valid_study_quote_map(registry, analyses)
        introduced: set = set()
        slot = 0
        out: List[str] = []
        for sec in plan["sections"]:
            out.append(f"### {sec['heading']}")
            for item in sec["studies"]:
                si = item["study"]
                cite = study_cite.get(si, "")
                slot += 1
                out.append(f"<<CONNECTOR:{slot}>>")
                if si not in introduced:
                    introduced.add(si)
                    out.append(f"[[S{si}]] {cite}".rstrip())
                else:
                    out.append(f"{self._narrative_cite(cite)} further reported:".strip())
                for q in item["quotes"]:
                    out.append(f"[[Q{q}]] {cite}".rstrip())
        # Blank lines between blocks so quote tokens sit alone on their own line.
        return "\n\n".join(out), slot

    def _request_connectors(self, state: ReviewState, preview: str,
                            n_slots: int) -> Dict[str, str]:
        """Ask the model for ONE short navigational transition per slot. It sees
        the assembled evidence (expanded for readability) but returns ONLY short
        sentences keyed by slot number — it never re-emits the evidence."""
        if n_slots <= 0:
            return {}
        prompt = f"""Below is the EVIDENCE section of a literature review, already assembled
and LOCKED. It answers: "{state['original_query']}"

At each marker <<CONNECTOR:k>> insert ONE short NAVIGATIONAL transition sentence
that leads the reader into what follows it. This is the ONLY thing you write.

STRICT RULES:
- Navigational/transitional ONLY (e.g. "Turning from transit to surface
  operations," or "A further study examined the same constraint experimentally,"
  or "Beyond the engineering picture,").
- NEVER state a finding, number, statistic, result, or any empirical claim — all
  findings live in the quotes themselves. If you find yourself describing what a
  study found, STOP and write a purely navigational lead-in instead.
- Keep each under ~20 words. For a marker where no transition is needed (e.g. the
  very first one, or a study reappearing), use an empty string "".
- Do NOT reproduce, paraphrase, or summarise any quote or study description.
- Do NOT output any of the evidence text or any [[tokens]].

EVIDENCE (for your reading — do NOT reproduce it):
{preview}

Output ONLY this JSON, with a key for every marker number 1..{n_slots}:
{{ "1": "transition or empty", "2": "...", ... }}"""
        result = self.agent_manager.run_primary(prompt, task="synthesis")
        data = None
        if result.success and result.json_response:
            data = result.json_response
        elif result.success and result.response:
            try:
                from json_parser import parse_json_response
                data = parse_json_response(result.response)
            except Exception:
                data = None
        out: Dict[str, str] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                kk = re.sub(r"\D", "", str(k))
                if kk and isinstance(v, str):
                    out[kk] = v
        return out

    def _apply_connectors(self, evidence_text: str, conn: Dict[str, str],
                          registry: Dict) -> str:
        """Substitute connector sentences into the slot markers, then clean. Any
        connector that parrots a quote/summary is removed by the transcription
        strip in _sanitize_review."""
        def clean_sentence(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"\[\[[SQ]?\d*\]\]", "", s)          # no tokens
            s = s.replace("<<", "").replace(">>", "")
            s = re.sub(r'["\u201c\u201d\u2018\u2019]', "", s)  # no quote marks
            s = re.sub(r"\s+", " ", s).strip()
            words = s.split()
            if len(words) > 40:                              # hard length cap
                s = " ".join(words[:40])
            return s

        def repl(m):
            k = m.group(1)
            sent = clean_sentence(conn.get(k, ""))
            return sent

        text = re.sub(r"<<CONNECTOR:(\d+)>>", repl, evidence_text)
        # Tidy the blank lines left where empty connectors were removed.
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Transcription-strip protects against a connector that copies evidence.
        text = self._sanitize_review(text, registry)
        return text.strip()

    def _request_analysis(self, state: ReviewState, evidence_expanded: str) -> Dict[str, str]:
        """SEPARATE call with a FRESH context: given only the question and the
        finished EVIDENCE (expanded, read-only), write the analytical sections.
        No tokens, no new direct quotes — reasoning in the model's own words."""
        summary = self.discovery.get_catalog_summary()
        ma = state.get("methodology_assessment") or {}
        meth_ctx = ""
        if ma:
            bits = [f"Overall methodological quality: {ma.get('overall_quality','?')}"]
            if ma.get("rating") not in (None, ""):
                bits.append(f"Rating: {ma.get('rating')}/10")
            if ma.get("summary"):
                bits.append(f"Assessment: {ma.get('summary')}")
            if ma.get("weaknesses"):
                bits.append("Weaknesses/gaps: " + "; ".join(ma["weaknesses"]))
            meth_ctx = ("\n\nMETHODOLOGICAL ASSESSMENT (inform METHOD/LIMITATIONS in your "
                        "OWN words; do NOT quote or cite it):\n- " + "\n- ".join(bits))
        tang = ""
        if state.get("tangential_engagement_count", 0) > 0:
            tang = ("\n\nNOTE: some evidence came from tangential/indirect searches and some "
                    "quotes were verified against abstracts; say so explicitly in LIMITATIONS.")
        prompt = f"""You are writing the ANALYTICAL sections of an APA 7th literature review
that answers the research question. The EVIDENCE section is already written
(shown below, read-only) — do NOT rewrite it, and do NOT add an EVIDENCE or a
REFERENCES section.

RESEARCH QUESTION: "{state['original_query']}"

SEARCH METHODOLOGY:
Databases: Semantic Scholar, OpenAlex, CORE, Europe PMC, Crossref
Papers in catalog: {summary['total_papers']} | Full text reviewed: {summary['with_full_text']}{meth_ctx}{tang}

THE EVIDENCE SECTION (already finalised — reason ABOUT it; quote nothing):
{evidence_expanded}

Write these sections in your OWN words. Reference studies by their APA in-text
citation exactly as they appear above. Use NO direct quotations and NO [[tokens]].

- INTRODUCTION (1-2 paragraphs): the question and why it matters; scope/purpose.
- METHOD (1 paragraph): databases, screening counts, inclusion criteria (only
  studies with verified quotes were retained), and search limitations.
- DISCUSSION (3-5 paragraphs): the real analysis — what the quoted findings
  collectively mean for the question, how studies reinforce or contradict each
  other, the weight of evidence, and what remains unresolved.
- LIMITATIONS: limitations of the studies and of this review.
- CONCLUSION (1 paragraph): the direct, evidence-based answer to the question,
  stating what is established vs uncertain.

Output ONLY this JSON (prose strings, no markdown headers inside them):
{{
  "introduction": "...",
  "method": "...",
  "discussion": "...",
  "limitations": "...",
  "conclusion": "..."
}}"""
        result = self.agent_manager.run_primary(prompt, task="synthesis")
        data = None
        if result.success and result.json_response:
            data = result.json_response
        elif result.success and result.response:
            try:
                from json_parser import parse_json_response
                data = parse_json_response(result.response)
            except Exception:
                data = None
        out = {"introduction": "", "method": "", "discussion": "",
               "limitations": "", "conclusion": ""}
        if isinstance(data, dict):
            for k in out:
                v = data.get(k)
                if isinstance(v, str):
                    out[k] = v.strip()
        return out

    def _assemble_full_review(self, sections: Dict[str, str], locked_evidence: str) -> str:
        """Stitch the final token'd review: analysis sections + locked EVIDENCE +
        a REFERENCES placeholder (rebuilt deterministically at compile)."""
        parts = [
            "INTRODUCTION", "", sections.get("introduction", "").strip(), "",
            "METHOD", "", sections.get("method", "").strip(), "",
            "EVIDENCE", "", locked_evidence.strip(), "",
            "DISCUSSION", "", sections.get("discussion", "").strip(), "",
            "LIMITATIONS", "", sections.get("limitations", "").strip(), "",
            "CONCLUSION", "", sections.get("conclusion", "").strip(), "",
            "REFERENCES", "",
        ]
        return "\n".join(parts).strip() + "\n"

    # ----- evidence locking helpers (used by the fix nodes) -----

    _EVIDENCE_HEADING_RE = re.compile(r"(?im)^[ \t]*(?:\d+\.\s*)?EVIDENCE[ \t]*$")
    _POST_EVIDENCE_HEADING_RE = re.compile(
        r"(?im)^[ \t]*(?:\d+\.\s*)?(?:DISCUSSION|LIMITATIONS|CONCLUSION|REFERENCES)[ \t]*$")
    _EVIDENCE_MASK = "\u27e6LOCKED_EVIDENCE\u27e7"

    def _evidence_body_span(self, review: str):
        """Return (start, end) character offsets of the EVIDENCE section BODY
        (between the EVIDENCE heading and the next analysis heading), or None."""
        if not review:
            return None
        m = self._EVIDENCE_HEADING_RE.search(review)
        if not m:
            return None
        start = m.end()
        m2 = self._POST_EVIDENCE_HEADING_RE.search(review, start)
        end = m2.start() if m2 else len(review)
        return (start, end)

    def _mask_evidence(self, review: str):
        """Replace the EVIDENCE body with a single mask marker so a fixer LLM
        cannot touch it. Returns the masked review (or the original if no span)."""
        span = self._evidence_body_span(review)
        if not span:
            return review
        s, e = span
        return review[:s] + "\n\n" + self._EVIDENCE_MASK + "\n\n" + review[e:]

    def _unmask_evidence(self, review: str, state: ReviewState) -> str:
        """Swap the mask marker back for the locked evidence. If the marker was
        lost, re-impose by heading span as a fallback."""
        locked = state.get("locked_evidence")
        if not locked:
            return review
        if self._EVIDENCE_MASK in review:
            return review.replace(self._EVIDENCE_MASK, locked.strip())
        return self._reimpose_evidence_by_span(review, state)

    def _reimpose_evidence_by_span(self, review: str, state: ReviewState) -> str:
        """Overwrite whatever sits in the EVIDENCE body with the locked evidence,
        guaranteeing a fix can never corrupt the verified evidence section."""
        locked = state.get("locked_evidence")
        if not locked:
            return review
        span = self._evidence_body_span(review)
        if not span:
            return review
        s, e = span
        return review[:s] + "\n\n" + locked.strip() + "\n\n" + review[e:]

    def _synthesize_two_phase(self, state: ReviewState) -> ReviewState:
        """PHASE 9 (two-phase): plan -> deterministic assemble -> locked
        connectors -> separate (fresh-context) analysis -> stitched review."""
        analyses = state.get("study_analyses", [])
        if not analyses:
            state["literature_review"] = "Insufficient studies."
            return state
        evidence = self._get_evidence(state)
        registry = state.get("quote_registry") or {}

        # 9a — PLAN (numbers only).
        self._print_phase_banner("PHASE 9a: EVIDENCE PLAN (thematic grouping + quote order)")
        print("  Asking the model to group studies and choose quote order...", flush=True)
        raw_plan = self._request_evidence_plan(state, evidence)
        plan, dropped = self._validate_evidence_plan(raw_plan or {}, registry, analyses)
        n_sections = len(plan["sections"])
        n_studies = len({it["study"] for sec in plan["sections"] for it in sec["studies"]})
        n_quotes = sum(len(it["quotes"]) for sec in plan["sections"] for it in sec["studies"])
        print(f"  {Fore.GREEN}{SYM_CHECK} Plan: {n_sections} section(s), {n_studies} study(ies), "
              f"{n_quotes} quote(s).{Style.RESET_ALL}")
        if dropped:
            print(f"  {Fore.YELLOW}{len(dropped)} study(ies) not placed by the plan.{Style.RESET_ALL}")
        state["evidence_plan"] = plan

        # 9b — ASSEMBLE (deterministic; tokens only).
        self._print_phase_banner("PHASE 9b: ASSEMBLE EVIDENCE (deterministic)")
        evidence_text, n_slots = self._assemble_evidence(plan, registry, analyses)

        # 9c — CONNECTORS (locked pass; short navigational sentences only).
        if self.config.get("evidence_connectors_enabled", True) and n_slots > 0:
            self._print_phase_banner("PHASE 9c: CONNECTORS (locked — navigational sentences only)")
            preview, _, _ = self._expand_quote_placeholders(evidence_text, registry)
            conn = self._request_connectors(state, preview, n_slots)
            locked_evidence = self._apply_connectors(evidence_text, conn, registry)
            filled = sum(1 for k in conn if conn.get(k, "").strip())
            print(f"  {Fore.GREEN}{SYM_CHECK} {filled}/{n_slots} connector slot(s) filled."
                  f"{Style.RESET_ALL}")
        else:
            locked_evidence = self._apply_connectors(evidence_text, {}, registry)
        state["locked_evidence"] = locked_evidence.strip()

        # 9d — ANALYSIS (separate call, fresh context, reads finished evidence).
        self._print_phase_banner("PHASE 9d: ANALYSIS (separate context — intro/method/discussion/etc.)")
        evidence_expanded, _, _ = self._expand_quote_placeholders(
            state["locked_evidence"], registry)
        print("  Writing the analytical sections from the finished evidence...", flush=True)
        sections = self._request_analysis(state, evidence_expanded)
        if not any(sections.values()):
            state["errors"].append("Two-phase analysis returned empty sections.")

        review = self._assemble_full_review(sections, state["locked_evidence"])
        state["literature_review"] = self._sanitize_review(review, registry)
        wc = len(self._expand_quote_placeholders(
            state["literature_review"], registry)[0].split())
        print(f"  {Fore.GREEN}{SYM_CHECK} Assembled review (~{wc} words, evidence locked)."
              f"{Style.RESET_ALL}")
        return state

    def _build_no_evidence_report(self, state: ReviewState) -> str:
        """Write an explicit account of WHY no review could be produced.

        Reaching synthesis with zero analysed studies used to emit the two-word
        string "Insufficient studies.", which the self-fix and verification
        passes then spent ~20 minutes inflating into an empty seven-heading
        skeleton that PASSED verification — a document that reported nothing and
        looked like a successful run. A run that finds nothing should say so, in
        full, with the numbers that explain it.
        """
        summary = self.discovery.get_catalog_summary()
        papers = []
        try:
            papers = self.discovery.get_all_papers()
        except Exception:
            pass
        abstract_only = [p for p in papers
                         if not getattr(p, "full_text_available", False)
                         and getattr(p, "abstract", None)]
        no_text_at_all = [p for p in papers
                          if not getattr(p, "full_text_available", False)
                          and not getattr(p, "abstract", None)]

        lines = [
            "NO REVIEW PRODUCED — INSUFFICIENT RETRIEVABLE EVIDENCE",
            "",
            f'RESEARCH QUESTION: "{state.get("original_query", "")}"',
            "",
            "OUTCOME",
            "No literature review was written because no study reached the "
            "deep-analysis stage with verifiable text. This is a retrieval "
            "outcome, not a finding about the research question: it says "
            "nothing about whether the literature exists or what it concludes.",
            "",
        ]
        floor_reason = state.get("_insufficient_evidence_reason")
        if floor_reason:
            # Below-floor variant: some evidence exists, it just did not reach
            # min_studies_for_review. Replace the zero-evidence wording.
            lines[0] = "NO REVIEW PRODUCED — EVIDENCE BELOW THE REQUIRED MINIMUM"
            lines[5] = (
                f"No literature review was written because {floor_reason}. "
                f"Everything the run did find is reproduced in full below, so "
                f"nothing is lost — it is simply not enough to support a review. "
                f"This is a statement about what this run retrieved, not about "
                f"what the literature contains.")
        lines += [
            "WHAT THE SEARCH ACTUALLY FOUND",
            f"  Papers identified and catalogued: {summary.get('total_papers', 0)}",
            f"  Full texts successfully retrieved: {summary.get('with_full_text', 0)}",
            f"  Abstract only (no full text): {len(abstract_only)}",
            f"  Neither full text nor abstract: {len(no_text_at_all)}",
            f"  Standard search rounds completed: {state.get('discovery_round', 0)}",
            f"  Tangential engagements: {state.get('tangential_engagement_count', 0)}",
            "",
        ]

        if summary.get("total_papers", 0) and not summary.get("with_full_text", 0):
            lines += [
                "MOST LIKELY CAUSE",
                "Papers were found but no full text could be downloaded for any "
                "of them. Quotes are verified against retrieved text, so with no "
                "text there is nothing to verify and nothing to quote. Check the "
                "acquisition lines in the log above: repeated 'failed' entries "
                "against a valid open-access DOI point at the download step "
                "(publisher bot-blocking, network, or a missing UNPAYWALL_EMAIL) "
                "rather than at the literature being unavailable.",
                "",
            ]
        elif not summary.get("total_papers", 0):
            lines += [
                "MOST LIKELY CAUSE",
                "No papers entered the catalog at all. Either the search APIs "
                "returned nothing usable (check for rate-limit and missing-API-key "
                "messages in the log) or the selection step rejected every "
                "candidate.",
                "",
            ]
        else:
            n_kept = len(state.get("study_analyses") or [])
            if floor_reason and n_kept:
                lines += [
                    "MOST LIKELY CAUSE",
                    f"Retrieval and analysis worked — {n_kept} study(ies) came "
                    f"through with usable text — but too few cleared quote "
                    f"verification to meet the configured floor. Either the "
                    f"question is genuinely thinly studied, or the search stopped "
                    f"before it had covered the literature. The relevance-filter "
                    f"and curation reasoning in the log above shows which.",
                    "",
                ]
            else:
                lines += [
                    "MOST LIKELY CAUSE",
                    "Papers with text were retrieved, but none survived reading, "
                    "filtering, curation, and quote verification. Review the "
                    "relevance-filter and curation reasoning in the log above.",
                    "",
                ]

        if papers:
            lines.append("PAPERS IDENTIFIED (catalogued but not usable as evidence)")
            for p in papers:
                status = ("full text" if getattr(p, "full_text_available", False)
                          else ("abstract only" if getattr(p, "abstract", None)
                                else "metadata only"))
                lines.append(
                    f"  - [{status}] "
                    f"{apa_reference(p.authors, p.year, p.title, p.venue, p.doi)}")
            lines.append("")

        # ---- RETAINED EVIDENCE ----------------------------------------------
        # An abort must never silently bin work the run actually completed. Any
        # study that reached deep analysis is reproduced here with its verified
        # quotes and APA reference, so a below-floor run still hands back
        # everything it found in citable form.
        analyses = state.get("study_analyses") or []
        if analyses:
            lines.append("EVIDENCE RETAINED (analysed, with verified quotes)")
            lines.append(
                "These studies were read and analysed successfully. They are "
                "reproduced verbatim rather than discarded; the quotes below "
                "passed the same verification a finished review would apply.")
            lines.append("")
            for i, a in enumerate(analyses, 1):
                lines.append(
                    f"  {i}. {apa_reference(a.get('paper_authors') or [], a.get('paper_year'), a.get('paper_title', '?'), a.get('paper_venue'), a.get('paper_doi'))}")
                if a.get("study_type"):
                    lines.append(f"     Design: {a.get('study_type')}")
                verified = [q for q in (a.get("key_quotes") or [])
                            if isinstance(q, dict) and q.get("verified") and q.get("quote")]
                if verified:
                    for q in verified:
                        loc = q.get("source_section") or q.get("section_heading") or ""
                        loc = f" [{loc}]" if loc else ""
                        lines.append(f'     Verified quote{loc}: "{q.get("quote")}"')
                        if q.get("context"):
                            lines.append(f"       Context: {q.get('context')}")
                else:
                    lines.append("     No quote passed verification for this study.")
                lines.append("")

        lines.append("SUGGESTED NEXT STEPS")
        if floor_reason:
            lines += [
                f"  1. Lower 'min_studies_for_review' (currently "
                f"{int(self.config.get('min_studies_for_review', 3))}) if a "
                f"smaller evidence base is acceptable for this question, or set "
                f"'abort_below_min_studies': False to synthesise a below-floor "
                f"review with the studies listed above.",
                "  2. Raise the search budget — 'post_review_max_retries', "
                "'post_review_max_minutes' or 'max_discovery_rounds' — so the "
                "run has room to reach the floor before it stops.",
                "  3. Re-run with broader search terms; the evidence above shows "
                "the topic is not empty, only thinly retrieved.",
            ]
        else:
            lines += [
                "  1. Confirm UNPAYWALL_EMAIL is set to a real address.",
                "  2. Add the free API keys the log reported as missing "
                "(Semantic Scholar, CORE) — both raise the share of retrievable "
                "open-access text considerably.",
                "  3. Re-run the question; if papers are again found but never "
                "downloaded, the failure is in acquisition and the search terms are "
                "not the problem.",
            ]
        return "\n".join(lines)

    def node_synthesize_review(self, state: ReviewState) -> ReviewState:
        # ---- ZERO-EVIDENCE SHORT CIRCUIT ------------------------------------
        # Every synthesis path already bailed to "Insufficient studies." on an
        # empty analyses list, but that stub was then handed to self-review,
        # self-fix, and up to three verification passes, which turned it into an
        # empty seven-section skeleton and declared it PASSED. Catch it once,
        # here, write a truthful report instead, and flag it so the repair loop
        # is skipped — there is nothing in a retrieval failure for those passes
        # to fix.
        if not state.get("study_analyses"):
            self._print_phase_banner("PHASE 9: SYNTHESIS — NO EVIDENCE TO SYNTHESISE")
            print(f"  {Fore.YELLOW}No study reached deep analysis with verifiable "
                  f"text — writing an explicit no-evidence report instead of a "
                  f"review.{Style.RESET_ALL}")
            state["literature_review"] = self._build_no_evidence_report(state)
            state["_no_evidence_report"] = True
            return state
        # ---- BELOW-FLOOR SHORT CIRCUIT --------------------------------------
        # Phase 8b decided the verified evidence base is under
        # min_studies_for_review and the search budget is spent. Same treatment:
        # an honest report (which reproduces every study and verified quote
        # found), and no repair loop — the shortfall is in the evidence, and no
        # amount of rewriting can fix that.
        if state.get("_insufficient_evidence_reason"):
            self._print_phase_banner("PHASE 9: SYNTHESIS — EVIDENCE BELOW THE REQUIRED MINIMUM")
            print(f"  {Fore.YELLOW}{state['_insufficient_evidence_reason']}."
                  f"{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}Writing an evidence report that retains every "
                  f"study and verified quote found.{Style.RESET_ALL}")
            state["literature_review"] = self._build_no_evidence_report(state)
            state["_no_evidence_report"] = True
            return state
        if self._low_end:
            return self._node_synthesize_review_low_end(state)
        if self.config.get("two_phase_synthesis", True):
            return self._synthesize_two_phase(state)
        return self._synthesize_one_shot(state)

    def _synthesize_one_shot(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 9: SYNTHESIS — APA 7th Quote-Driven Literature Review")
        analyses = state.get("study_analyses", [])
        if not analyses:
            state["literature_review"] = "Insufficient studies."
            return state

        evidence = self._get_evidence(state)
        summary = self.discovery.get_catalog_summary()

        prior_issues = state.get("last_verification_issues", [])
        issues_section = ""
        if prior_issues:
            review_for_anno = state.get("literature_review", "") or ""
            annotated = self._annotate_issues_in_review(review_for_anno, prior_issues)
            memory = self._render_refinement_memory(state)
            mem_block = f"\n{memory}\n" if memory else ""
            issues_section = (
                f"\n\nTHIS IS A REWRITE. Your PREVIOUS draft FAILED verification with the "
                f"issues below. You MUST resolve EVERY one of them while keeping everything "
                f"that was already correct. The review will NOT pass until all critical (and "
                f"ideally all moderate) issues are gone.{mem_block}\n"
                f"ISSUES TO RESOLVE (with exactly where each occurs and how to fix it):\n"
                f"{annotated}\n")

        rules_block = self._super_critical_quote_rules()

        tangential_note = ""
        if state.get("tangential_engagement_count", 0) > 0:
            tangential_note = """

NOTE: Some evidence below comes from TANGENTIAL searches (indirect routes such
as class-level, mechanistic, or adjacent-population studies) because direct
evidence on the exact research question is sparse. Some quotes were verified
against abstracts rather than full texts (see "Source text used for quotes"
on each study). In LIMITATIONS, explicitly note this and that some inference
relies on indirect evidence."""

        ma = state.get("methodology_assessment") or {}
        methodology_context = ""
        if ma:
            mc = [f"Overall methodological quality: {ma.get('overall_quality','?')}"]
            if ma.get("rating") not in (None, ""):
                mc.append(f"Rating: {ma.get('rating')}/10")
            if ma.get("summary"):
                mc.append(f"Assessment: {ma.get('summary')}")
            if ma.get("strengths"):
                mc.append("Strengths: " + "; ".join(ma["strengths"]))
            if ma.get("weaknesses"):
                mc.append("Weaknesses/gaps relative to the question: " + "; ".join(ma["weaknesses"]))
            methodology_context = (
                "\n\nMETHODOLOGICAL ASSESSMENT (use this to inform the METHOD and "
                "LIMITATIONS sections in your OWN words; do NOT quote it and do NOT add a "
                "citation to it):\n- " + "\n- ".join(mc) + "\n")

        prompt = f"""You are writing a high-distinction, university-grade academic literature
review in APA 7th style that DIRECTLY ANSWERS the research question below.

RESEARCH QUESTION: "{state['original_query']}"

SEARCH METHODOLOGY:
Databases: Semantic Scholar, OpenAlex, CORE, Europe PMC, Crossref
Total papers in catalog: {summary['total_papers']} | Full text reviewed: {summary['with_full_text']}
Studies analysed: {len(analyses)}
{methodology_context}
EVIDENCE BASE (each study includes APA in-text citation and VERIFIED QUOTES):
{evidence}
{issues_section}{tangential_note}
================ APA 7th STRUCTURE ================

Use these EXACT section headings, in this order, in APA 7th format:

1. INTRODUCTION
2. METHOD
3. EVIDENCE
4. DISCUSSION
5. LIMITATIONS
6. CONCLUSION
7. REFERENCES

================ SECTION REQUIREMENTS ================

INTRODUCTION (1-2 paragraphs)
- State the research question and why it matters.
- Establish the scope and purpose of the review.

METHOD (1 paragraph)
- Databases searched.
- Number of papers screened/included.
- Inclusion criteria (note: only papers with verified full-text quotes were
  retained in main mode; in tangential mode, verified abstract quotes were
  also accepted).
- Limitations of the search strategy.

EVIDENCE (the BULK of the review — it ORDERS the verified evidence with light
connective tissue ONLY; it does NOT analyse. ALL interpretation/analysis goes in
the DISCUSSION section.)

In this section you are an ORGANISER of evidence, not a commentator. You build
the section almost entirely from TOKENS; your own writing here is limited to
short sentences that make it flow.

For each study, the FIRST (and only) time it appears:
  1. Write its STUDY-INTRO token [[Sn]] exactly once, immediately followed by its
     "CITE AS" citation. This token expands to a grounded, plain-prose
     introduction of the study (what it examined, its design, scope, and the
     nature of its findings). THE TOKEN IS THE INTRODUCTION — you do NOT write
     your own summary or description of the study.
  2. Then place that study's QUOTE token(s) [[Qn]], EACH ALONE ON ITS OWN LINE,
     in the order you want the evidence to read. Each renders as a clearly
     set-apart evidence block. Put the "CITE AS" citation right after each token.
  3. You MAY write ONE short connecting sentence (occasionally two) to move to the
     next study or theme, ONLY for flow — e.g. "A second study approached the same
     constraint experimentally." or "Turning from transit to surface operations,".

ABSOLUTELY FORBIDDEN IN THE EVIDENCE SECTION — these make the review unreadable
and are CRITICAL errors:
  - Do NOT state, restate, paraphrase, summarise, or "preview" what a quote says,
    before OR after it. The [[Qn]] token ALREADY contains the exact finding. The
    quote speaks for itself.
  - Do NOT paraphrase, restate, or expand the [[Sn]] study summary in your own
    words. The token is the whole introduction.
  - Do NOT write interpretation, significance, comparison, implications, or
    analysis here. That is the DISCUSSION section's job, not this one.
  - Do NOT type any quote text yourself, and do NOT wrap a token in quotation
    marks.
  Your prose between tokens is ONLY brief connective tissue — never a description
  of what the evidence contains.

The pattern is simply: [[Sn]] (+citation) -> [[Qn]] block (+citation) -> [[Qn]]
block (+citation) -> one short connector -> next study's [[Sn]] ... Order the
evidence and let the tokens speak.

HOW TO INSERT A QUOTE: write the quote's TOKEN (e.g. [[Q7]]) ALONE on its own line
where you want that evidence block, with its "CITE AS" citation immediately after.
The verified text is pasted in automatically — NEVER retype, paraphrase, or edit
quote text. Use ONLY tokens shown in the evidence base; never invent a token.
Quotes may be long; that is fine — they are rendered as set-apart blocks.

CITATIONS: copy each token's "CITE AS" string CHARACTER-FOR-CHARACTER (e.g. never
turn "de la Monte" into "SM"; never write "M & K"). Do NOT add provenance notes
such as "(verified quote from full text)" — the in-text citation is the only
parenthetical after a quote.

Use each [[Sn]] and each [[Qn]] token AT MOST ONCE. Use as many or as few quote
tokens per study as the evidence warrants. If a study offers nothing worth
quoting, do not use it at all (do not introduce or cite it) — unquoted studies
are removed from the final review (including from REFERENCES).

When naming authors narratively, use the surnames from the "CITE AS" string —
never initials.

DISCUSSION (3-5 paragraphs — this is where ALL of your analysis lives)
- The EVIDENCE section deliberately contains NO analysis — it only orders the
  quoted evidence. THIS section is where you do all the reasoning, so make it
  substantial and genuinely analytical.
- In your OWN words (no quotation marks, no tokens here), reason ABOUT the quoted
  evidence: what the findings collectively mean for the research question, how
  they connect, where studies reinforce or contradict each other, what the
  weight of evidence implies, and what remains unresolved. Compare across study
  types and across the themes from EVIDENCE.
- Every claim here must refer to a study you already quoted in EVIDENCE, via its
  APA in-text citation. Do NOT introduce new direct quotes or tokens here; you
  may briefly paraphrase a previously-quoted study when reasoning about it, but
  precise factual claims must trace to a study you quoted (cite it).

LIMITATIONS
- Limitations of the studies in the evidence base.
- Limitations of this review (search scope, language, etc).
- If tangential mode was engaged, explicitly state that direct evidence is
  limited and some inference relies on indirect evidence verified against
  abstracts rather than full texts.

CONCLUSION (1 paragraph)
- The evidence-based answer to the research question.
- What is established vs uncertain.

REFERENCES (APA 7th)
- One reference entry per cited study, alphabetised by first author surname.
- Use the metadata (authors, year, title, venue, DOI) from the evidence
  base. Include the DOI as https://doi.org/<doi> if available.

{rules_block}

================ ABSOLUTE RULES ================

- Every direct quotation MUST be inserted as a [[Qn]] TOKEN from the evidence
  base. NEVER type quotation-marked text yourself; only tokens become quotes.
- Use each token's "CITE AS" string immediately after it. Attribution is
  checked deterministically — a token already carries its correct source.
- Every in-text citation MUST correspond to a study you are quoting via a
  token. Do NOT cite or discuss a study you are not quoting; unquoted studies
  are removed from the review (body and REFERENCES) automatically.
- Every in-text citation MUST exist in the evidence base.
- Aim for 2000+ words. EVIDENCE should be the longest section.
- Use APA 7th sentence case for titles in REFERENCES; use APA author-date
  format for in-text citations.
- Do NOT include content that is not anchored in the evidence base."""

        print(f"  Generating quote-driven APA 7th review...", flush=True)
        result = self.agent_manager.run_primary(prompt, task="synthesis")
        if result.success and result.response:
            state["literature_review"] = self._sanitize_review(result.response, state.get("quote_registry"))
            print(f"  {Fore.GREEN}{SYM_CHECK} {len(result.response.split())} words "
                  f"({result.elapsed_time:.0f}s){Style.RESET_ALL}")
        else:
            state["literature_review"] = None
            state["errors"].append(f"Synthesis failed: {result.error}")
        return state

    # ---------- Low-end synthesis (map-reduce, quote-token preserving) ----------

    def _node_synthesize_review_low_end(self, state: ReviewState) -> ReviewState:
        """Chunked synthesis for a small-context device.

        MAP   — studies are written up in batches (reading order); each batch
                produces a thematic EVIDENCE fragment using only that batch's
                quote tokens.
        REDUCE— fragments are merged (hierarchically if large) and assembled
                into the full APA review with the surrounding sections.
        Quotes remain [[Qn]] tokens throughout, so nothing is retyped and the
        deterministic compile-time paste + reference rebuild are identical to
        full mode.
        """
        self._print_phase_banner("PHASE 9: SYNTHESIS — LOW-END (map-reduce, quote-driven)")
        analyses = state.get("study_analyses", [])
        if not analyses:
            state["literature_review"] = "Insufficient studies."
            return state

        registry = self._assign_registry(analyses)
        state["quote_registry"] = registry
        # Sentinel so _get_evidence won't try to rebuild a giant full string;
        # the checking stages use the compact index instead.
        state["evidence_base_cache"] = "(low-end: evidence rendered per batch)"

        summary = self.discovery.get_catalog_summary()
        tangential = state.get("tangential_engagement_count", 0) > 0
        batch_chars = int(self.config.get("low_end_synthesis_map_batch_chars", 9000) or 9000)

        # Per-study evidence rendered with global tokens, then greedily grouped
        # into batches under the character budget (studies kept in order).
        per_study = [(i, self._render_evidence_full(analyses, registry, only_studies={i}))
                     for i in range(1, len(analyses) + 1)]
        batches, cur, cur_len = [], [], 0
        for i, txt in per_study:
            if cur and cur_len + len(txt) > batch_chars:
                batches.append(cur)
                cur, cur_len = [], 0
            cur.append((i, txt))
            cur_len += len(txt)
        if cur:
            batches.append(cur)

        print(f"  {Fore.MAGENTA}Map step: {len(batches)} batch(es) across "
              f"{len(analyses)} studies (≈{batch_chars} chars/batch).{Style.RESET_ALL}")

        fragments = []
        for bi, batch in enumerate(batches, 1):
            if self._interrupted:
                break
            ev = "\n".join(t for _, t in batch)
            print(f"  {Fore.CYAN}  Map batch {bi}/{len(batches)} "
                  f"(studies {batch[0][0]}-{batch[-1][0]})...{Style.RESET_ALL}", flush=True)
            frag = self._lowend_map_fragment(state, ev, bi, len(batches), tangential)
            if frag:
                fragments.append(frag)
                print(f"  {Fore.GREEN}    done ({len(frag.split())} words).{Style.RESET_ALL}")
            else:
                print(f"  {Fore.YELLOW}    batch produced no usable fragment.{Style.RESET_ALL}")

        if not fragments:
            state["literature_review"] = None
            state["errors"].append("Low-end synthesis: no evidence fragments produced.")
            return state

        merged_evidence = self._lowend_reduce_fragments(state, fragments)
        compact_index = self._render_evidence_compact(analyses, registry)
        print(f"  {Fore.MAGENTA}Reduce step: assembling final APA review...{Style.RESET_ALL}",
              flush=True)
        review = self._lowend_assemble_review(
            state, merged_evidence, compact_index, summary, tangential)

        if review:
            state["literature_review"] = self._sanitize_review(review, state.get("quote_registry"))
            print(f"  {Fore.GREEN}{SYM_CHECK} Low-end review assembled "
                  f"({len(review.split())} words){Style.RESET_ALL}")
        else:
            state["literature_review"] = None
            state["errors"].append("Low-end synthesis: final assembly failed.")
        return state

    _LOWEND_TOKEN_RULES = (
        "QUOTE TOKENS (CRITICAL):\n"
        "- Direct quotes are inserted ONLY as their token, e.g. [[Q7]], placed where\n"
        "  you want the quote. NEVER type quotation-marked quote text yourself.\n"
        "- For a quote of 40+ words, put its token ALONE on its own line (it becomes\n"
        "  an APA block quote). For shorter quotes, place the token inline.\n"
        "- Put the token's \"CITE AS\" citation immediately after it, copied exactly.\n"
        "- Use ONLY tokens shown here; do not invent tokens. Use as many or as few as\n"
        "  the evidence warrants. Do not discuss or cite a study you are not quoting.\n"
    )

    def _lowend_map_fragment(self, state, evidence, batch_idx, n_batches, tangential):
        tang = ("\nSome studies are tangential/abstract-verified; you may note this where "
                "relevant.\n") if tangential else ""
        prompt = f"""You are drafting PART {batch_idx} of {n_batches} of the EVIDENCE section of an
APA 7th, quote-driven academic literature review. Write ONLY the thematic
evidence prose for the studies in THIS batch — no INTRODUCTION, METHOD,
DISCUSSION, or REFERENCES (those are added later).

RESEARCH QUESTION: "{state['original_query']}"
{tang}
{self._LOWEND_TOKEN_RULES}
EVIDENCE FOR THIS BATCH (each study's verified quotes are shown as tokens):
{evidence}

Write tight thematic paragraphs that present these studies' evidence using their
quote tokens. Introduce each study on first mention with its design/sample in a
short clause, then its quote token(s) and CITE AS citation. Group related
findings. Output ONLY the evidence prose for this batch."""
        result = self.agent_manager.run_primary(prompt, task="synthesis")
        if result.success and result.response:
            return self._sanitize_review(result.response, state.get("quote_registry"))
        return None

    def _lowend_reduce_fragments(self, state, fragments):
        """Combine map fragments into one EVIDENCE body. Concatenates when the
        combined size fits the reduce budget (zero token-loss risk); otherwise
        merges fragments in groups via the LLM (preserving every token), up to a
        few rounds, then concatenates whatever remains."""
        group_chars = int(self.config.get("low_end_synthesis_reduce_group_chars", 11000) or 11000)

        def total(frs):
            return sum(len(f) for f in frs) + 2 * len(frs)

        rounds = 0
        while total(fragments) > group_chars and len(fragments) > 1 and rounds < 3:
            rounds += 1
            print(f"  {Fore.MAGENTA}  Reduce round {rounds}: merging "
                  f"{len(fragments)} fragments...{Style.RESET_ALL}", flush=True)
            merged, cur, cur_len = [], [], 0
            for fr in fragments:
                if cur and cur_len + len(fr) > group_chars:
                    merged.append(self._lowend_merge_group(state, cur))
                    cur, cur_len = [], 0
                cur.append(fr)
                cur_len += len(fr)
            if cur:
                merged.append(self._lowend_merge_group(state, cur))
            fragments = [m for m in merged if m]
            if not fragments:
                break

        return "\n\n".join(fragments)

    def _lowend_merge_group(self, state, group):
        if len(group) == 1:
            return group[0]
        joined = "\n\n".join(group)
        prompt = f"""Merge these EVIDENCE-section drafts into one tighter, well-organised
thematic evidence narrative for an APA 7th literature review. Remove repetition
and group related findings.

RESEARCH QUESTION: "{state['original_query']}"

{self._LOWEND_TOKEN_RULES}
PRESERVE EVERY [[Qn]] TOKEN AND EVERY [[Sn]] STUDY-INTRO TOKEN EXACTLY as it
appears (do not drop, renumber, or alter any token) and keep each token's CITE AS
citation beside it. Output ONLY the merged evidence prose.

DRAFTS TO MERGE:
{joined}"""
        result = self.agent_manager.run_primary(prompt, task="synthesis")
        if result.success and result.response:
            return self._sanitize_review(result.response, state.get("quote_registry"))
        # Fall back to concatenation so no tokens are lost on a failed merge.
        return joined

    def _lowend_assemble_review(self, state, merged_evidence, compact_index,
                                summary, tangential):
        tang = ""
        if tangential:
            tang = ("\nNOTE: Some evidence is tangential / verified against abstracts; state "
                    "this in LIMITATIONS.\n")
        prior_issues = state.get("last_verification_issues", [])
        issues_section = ""
        if prior_issues:
            issues_text = "\n".join(
                f"- [{i.get('severity', '?')}] {i.get('description', '')}"
                for i in prior_issues[:10])
            issues_section = f"\nPREVIOUS REVIEW HAD THESE ISSUES — ADDRESS EACH:\n{issues_text}\n"

        prompt = f"""Assemble a complete high-distinction APA 7th, quote-driven literature
review from the EVIDENCE body already drafted below. Add the surrounding
sections; keep the EVIDENCE body as the EVIDENCE section.

RESEARCH QUESTION: "{state['original_query']}"

SEARCH METHODOLOGY:
Databases: Semantic Scholar, OpenAlex, CORE, Europe PMC, Crossref
Total papers in catalog: {summary['total_papers']} | Full text reviewed: {summary['with_full_text']}
{tang}{issues_section}
{self._LOWEND_TOKEN_RULES}
Use these EXACT section headings in order: INTRODUCTION, METHOD, EVIDENCE,
DISCUSSION, LIMITATIONS, CONCLUSION, REFERENCES.
- INTRODUCTION: state the question and why it matters.
- METHOD: databases, screening, inclusion criteria, search limitations.
- EVIDENCE: use the drafted evidence body below; you may reorganise it
  thematically but PRESERVE every [[Qn]] quote token and every [[Sn]] study-intro
  token exactly and keep its CITE AS citation. Keep 40+ word quote tokens alone on
  their own line. Keep each study's [[Sn]] intro before that study's first quote.
- DISCUSSION: reason about the quoted evidence; do NOT introduce new tokens.
- LIMITATIONS: study + review limitations (and tangential note if relevant).
- CONCLUSION: the evidence-based answer.
- REFERENCES: one APA 7th entry per QUOTED study. (This list is also rebuilt
  automatically, so focus on the body.)

STUDY INDEX (citation + available tokens; no quote text):
{compact_index}

DRAFTED EVIDENCE BODY (already contains quote tokens):
{merged_evidence}

Output ONLY the complete review."""
        result = self.agent_manager.run_primary(prompt, task="synthesis")
        if result.success and result.response:
            return result.response
        return None

    # ---------- Self-review ----------

    def node_self_review(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 10: SELF-REVIEW")
        review = state.get("literature_review")
        if not review:
            state["self_review_issues"] = []
            state["self_review_done"] = True
            return state
        # A no-evidence report is a factual account of a retrieval failure, not a
        # draft review. Critiquing it against literature-review structure only
        # produces "missing Introduction/Method/References" issues that the fix
        # passes cannot resolve, which is exactly the loop that turned a failed
        # run into an empty skeleton.
        if state.get("_no_evidence_report"):
            print(f"  {Fore.YELLOW}No-evidence report — nothing to self-review."
                  f"{Style.RESET_ALL}")
            state["self_review_issues"] = []
            state["self_review_done"] = True
            return state
        if state.get("self_review_done"):
            print(f"  {Fore.YELLOW}Self-review already done.{Style.RESET_ALL}")
            return state

        evidence = self._get_evidence_for_checks(state)
        current_date = datetime.now().strftime("%B %Y")

        prompt = f"""You are critically reviewing a draft APA 7th, quote-driven academic
literature review.

CURRENT DATE: {current_date}
(Papers dated this year or before are NOT future-dated fabrications.)

RESEARCH QUESTION: "{state['original_query']}"

EVIDENCE BASE (with verified quotes per study):
{evidence}

DRAFT REVIEW:
{review}

NOTE ON QUOTES: direct quotes are inserted as TOKENS like [[Q7]] that are
replaced with the exact verified text automatically. A token is ALWAYS a
correct, verbatim quote — do NOT flag tokens as missing, inexact, or
unverifiable. Each study is also introduced with a STUDY-INTRO TOKEN like [[S3]],
which expands to a grounded paraphrase of that study as PLAIN PROSE (it is NOT a
quotation) — do NOT flag [[Sn]] tokens as hand-typed quotes or as unverifiable,
and do NOT expect quotation marks around them. Your job re: quotes is to check
(a) that no quotation-marked text was hand-typed instead of using a token, and
(b) that every study that is cited or discussed is actually used via at least one
token — a [[Qn]] quote and/or its [[Sn]] intro (studies used by no token must be
removed).

CHECKS:
1. DIRECTNESS: does the review answer the question?
2. APA 7th STRUCTURE: are all 7 sections present (INTRODUCTION, METHOD,
   EVIDENCE, DISCUSSION, LIMITATIONS, CONCLUSION, REFERENCES)?
3. STUDY INTRODUCTION: on a study's first mention in EVIDENCE, is it introduced
   with its [[Sn]] token (aim/design/method/scope) BEFORE that study's first
   quote? Flag any study whose quote appears with no preceding [[Sn]] intro.
4. QUOTE TOKENS: are all direct quotes inserted as [[Qn]] tokens (not
   hand-typed in quotation marks)? Flag any hand-typed quotation. Do NOT flag
   the tokens themselves (neither [[Qn]] nor [[Sn]]).
5. CITATION VALIDITY: every APA in-text citation must correspond to a study
   present in the evidence base AND quoted via at least one token. Flag any
   study that is cited or discussed but never quoted (it must be removed).
6. EVIDENCE-DRIVEN: is the EVIDENCE section primarily organised around
   quote tokens (token alone on a line for 40+ word block quotes, inline
   otherwise)?
7. DISCUSSION DEPTH: does the DISCUSSION reason about the quoted evidence
   without introducing new quote tokens?
8. REFERENCES: are all quoted studies present in REFERENCES, in APA 7th
   format, alphabetised, with DOIs where available?
9. SCOPE DRIFT: does the review stay within the user's question scope?
10. QUANTITATIVE DETAIL: where numbers, effect sizes, or sample sizes
    appear in quoted evidence, are they cited correctly?

For each issue: type (1-10), description, severity (critical/moderate/minor),
fix_guidance.

Respond with ONLY JSON:
{{
    "issues": [{{"type": "...", "description": "...", "severity": "...", "fix_guidance": "..."}}],
    "overall_assessment": "1-2 sentence judgment"
}}

If genuinely solid, return only real issues. Do not invent problems."""

        print(f"  Reading own review...", flush=True)
        result = self.agent_manager.run_primary(
            prompt, as_json=True, task="self_review")
        if result.success and result.json_response:
            issues = result.json_response.get("issues", [])
            assessment = result.json_response.get("overall_assessment", "")
            state["self_review_issues"] = issues if isinstance(issues, list) else []
            state["self_review_done"] = True
            if assessment:
                print(f"  {Fore.WHITE}Assessment: {assessment}{Style.RESET_ALL}")
            if not issues:
                print(f"  {Fore.GREEN}{SYM_CHECK} No issues.{Style.RESET_ALL}")
            else:
                crit = [i for i in issues if i.get("severity") == "critical"]
                mod = [i for i in issues if i.get("severity") == "moderate"]
                minor = [i for i in issues if i.get("severity") == "minor"]
                print(f"  {Fore.YELLOW}{SYM_EYES} {len(issues)} issues "
                      f"({len(crit)} critical, {len(mod)} moderate, {len(minor)} minor){Style.RESET_ALL}")
                for iss in issues[:8]:
                    sev = iss.get("severity", "?")
                    color = Fore.RED if sev == "critical" else (Fore.YELLOW if sev == "moderate" else Fore.WHITE)
                    print(f"    {color}[{sev}] {iss.get('description', '')}{Style.RESET_ALL}")
        else:
            state["self_review_issues"] = []
            state["self_review_done"] = True
        return state

    # ---------- Self-fix ----------

    def node_self_fix(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 11: SELF-FIX")
        review = state.get("literature_review")
        issues = state.get("self_review_issues", [])
        actionable = [i for i in issues if i.get("severity") in ("critical", "moderate")]

        if not review or not actionable:
            print(f"  {Fore.YELLOW}Nothing to fix.{Style.RESET_ALL}")
            return state

        evidence = self._get_evidence(state)
        issues_text = self._annotate_issues_in_review(review, actionable)
        rules_block = self._super_critical_quote_rules()

        # When the EVIDENCE section is code-assembled and locked, hide it from the
        # fixer so it can only touch the analysis sections; it is swapped back in
        # verbatim afterwards. This makes a fix incapable of corrupting evidence.
        locked = state.get("locked_evidence")
        review_for_prompt = self._mask_evidence(review) if locked else review
        lock_note = ""
        if locked and self._EVIDENCE_MASK in review_for_prompt:
            lock_note = (
                f"\nIMPORTANT: The EVIDENCE section is already correct and LOCKED. It is "
                f"shown only as the marker {self._EVIDENCE_MASK}. Reproduce that marker "
                f"EXACTLY where the EVIDENCE section belongs and write NO evidence content "
                f"yourself — fix only the other sections.\n")

        prompt = f"""Rewrite this APA 7th, quote-driven academic literature review to fix
specific issues.

RESEARCH QUESTION: "{state['original_query']}"

EVIDENCE BASE (with verified quotes per study):
{evidence}

{rules_block}
{lock_note}
PREVIOUS DRAFT:
{review_for_prompt}

ISSUES TO FIX (with exactly where each occurs and how to fix it):
{issues_text}

INSTRUCTIONS:
1. Produce the complete revised review with the same APA 7th section
   structure: INTRODUCTION, METHOD, EVIDENCE, DISCUSSION, LIMITATIONS,
   CONCLUSION, REFERENCES.
2. Resolve every listed issue. Preserve accurate unrelated content.
3. Direct quotes are inserted as TOKENS like [[Q7]], and each study's intro is
   inserted as a STUDY-INTRO TOKEN like [[S3]] (a paraphrase pasted as plain
   prose, not a quote). PRESERVE every [[Qn]] and [[Sn]] token you keep EXACTLY as
   written — do not expand, rewrite, renumber, or alter tokens, and never type
   quotation-marked quote text yourself. To remove a quote, delete its whole
   token; to add one, insert a token that exists in the evidence base. Keep each
   study's [[Sn]] intro before that study's first quote.
   Keep each token's "CITE AS" citation immediately after it.
4. APA 7th in-text citations and reference entries.
5. DISCUSSION reasons about quoted evidence; do NOT introduce new quote tokens
   there.
6. Only studies in the evidence base, and only studies you actually use via a
   token — i.e. you quote it with a [[Qn]] token and/or introduce it with its
   [[Sn]] token. Do not cite or discuss a study you are not using via a token.
7. CONCLUSION gives the evidence-based answer.

OUTPUT ONLY THE REVISED REVIEW — no preamble or markdown fences."""

        print(f"  Rewriting...", flush=True)
        result = self.agent_manager.run_primary(prompt, task="self_fix")
        if result.success and result.response:
            fixed = self._unmask_evidence(result.response, state)
            state["literature_review"] = self._sanitize_review(fixed, state.get("quote_registry"))
            state["evidence_base_cache"] = None
            print(f"  {Fore.GREEN}{SYM_WRENCH} Rewrite complete: "
                  f"{len(result.response.split())} words ({result.elapsed_time:.0f}s){Style.RESET_ALL}")
        return state

    # ---------- Verification convergence helpers ----------

    def _super_critical_quote_rules(self) -> str:
        """The strongest possible framing (per James's spec): hand-typing any
        quotation, or copying evidence-base display text, is a SUPER-CRITICAL,
        review-breaking error that blocks the review from passing."""
        return (
            "================ SUPER-CRITICAL, REVIEW-BREAKING RULES ================\n"
            "These are the MOST IMPORTANT rules in this task. Breaking ANY of them is a\n"
            "SUPER-CRITICAL, review-breaking error \u2014 NOT minor, NOT cosmetic \u2014 and the\n"
            "review CANNOT and WILL NOT pass verification until it is fixed:\n"
            "1. DO NOT use quotation marks (\" \", ' ', \u201c \u201d, \u2018 \u2019) around ANY source text,\n"
            "   and DO NOT type, paste, or otherwise insert ANY quotation WHATSOEVER\n"
            "   outside the verified token system. The ONLY way a quotation may appear\n"
            "   is by writing a [[Qn]] token; the exact verified text is pasted in for\n"
            "   you automatically. If you write quotation marks around source text, that\n"
            "   is a fabricated/hand-typed quote and it FAILS verification.\n"
            "2. DO NOT copy ANY text from the EVIDENCE BASE display. NEVER write the\n"
            "   words 'expands to:', the words 'CITE AS:', a '(NN words) \u2014' word-count, a\n"
            "   '(context: ...)' note, the 'VERIFIED QUOTES' header, or the literal quote\n"
            "   text or summary text shown after them. Those are scaffolding for you to\n"
            "   READ \u2014 they must NEVER appear in the review.\n"
            "3. Introduce each study ONLY by writing its [[Sn]] token (pasted in as plain\n"
            "   prose, no quotation marks). NEVER hand-type a study's summary/introduction.\n"
            "4. A study's [[Sn]] intro MUST appear BEFORE that study's first [[Qn]] quote.\n"
            "5. Do NOT restate, paraphrase, summarise, or 'preview' the CONTENT of any\n"
            "   [[Qn]] quote or [[Sn]] summary in your own prose. The tokens already\n"
            "   contain that content verbatim; repeating it makes the review unreadable\n"
            "   and is a review-breaking error. In the EVIDENCE section your prose is ONLY\n"
            "   brief connective sentences between tokens; save analysis for DISCUSSION.\n"
            "If you are ever tempted to write a sentence in quotation marks: STOP. Either\n"
            "place the correct [[Qn]] token instead, or state the point in your OWN words\n"
            "with NO quotation marks.\n"
            "======================================================================"
        )

    def _extract_offending_text(self, issue: Dict) -> str:
        desc = issue.get("description", "") or ""
        m = re.search(r'(?:Quote|Offending text|quote)\s*:?\s*"([^"]+)"', desc)
        if m:
            return m.group(1).strip().strip(". ")
        m = re.search(r'"([^"]{12,})"', desc)
        if m:
            return m.group(1).strip()
        return ""

    def _flexible_find_in_review(self, review: str, needle: str, window: int = 160) -> str:
        """Best-effort: locate `needle` in the raw review tolerating whitespace
        differences, and return a context window around it (or "" if not found)."""
        if not review or not needle:
            return ""
        head = needle[:48].strip()
        parts = [re.escape(p) for p in re.split(r"\s+", head) if p]
        if not parts:
            return ""
        pat = r"\s+".join(parts)
        try:
            m = re.search(pat, review, flags=re.IGNORECASE)
        except re.error:
            return ""
        if not m:
            return ""
        s = max(0, m.start() - 40)
        e = min(len(review), m.end() + window)
        return review[s:e].replace("\n", " ")

    def _locate_snippet_for_issue(self, review: str, issue: Dict) -> str:
        """Return a human-readable pointer to exactly where in the review this
        issue lives (#4: 'show it what specifically in the report is causing the
        error')."""
        review = review or ""
        off = self._extract_offending_text(issue)
        ctx = self._flexible_find_in_review(review, off) if off else ""
        if not ctx:
            si_m = re.search(r'Study (\d+)', issue.get("description", "") or "")
            if si_m:
                si = si_m.group(1)
                m = re.search(r'\[\[S' + re.escape(si) + r'\]\]', review)
                if not m:
                    m = re.search(r'\[\[Q\d+\]\]', review)
                if m:
                    s = max(0, m.start() - 40)
                    e = min(len(review), m.end() + 160)
                    ctx = review[s:e].replace("\n", " ")
        if off and ctx:
            return f'offending text: "{off}"\n      seen in your review near: "...{ctx}..."'
        if off:
            return f'offending text: "{off}"'
        if ctx:
            return f'seen in your review near: "...{ctx}..."'
        return ""

    def _annotate_issues_in_review(self, review: str, issues: List[Dict]) -> str:
        """Render each issue with its located span and fix guidance, so the model
        can see precisely what to change."""
        lines = []
        for idx, iss in enumerate(issues, 1):
            lines.append(f"ISSUE {idx} [{iss.get('severity','?')}] ({iss.get('type','?')}):")
            lines.append(f"  Problem: {iss.get('description','')}")
            loc = self._locate_snippet_for_issue(review, iss)
            if loc:
                lines.append(f"  Where in YOUR review: {loc}")
            fg = iss.get("fix_guidance")
            if fg:
                lines.append(f"  How to fix: {fg}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _render_refinement_memory(self, state: ReviewState) -> str:
        """Summarise the previous fix attempt's outcome (#4: show what was tried,
        what resolved, what newly appeared) for the next fix prompt."""
        mem = state.get("verification_refinement_memory", []) or []
        if not mem:
            return ""
        last = mem[-1]
        parts = [f"RESULT OF YOUR PREVIOUS FIX ATTEMPT (pass #{last.get('attempt')}, "
                 f"{last.get('mode')} mode):"]
        if last.get("targeted"):
            parts.append(f"  You were asked to fix: {last['targeted']}")
        parts.append(f"  After that attempt {last.get('critical',0)} critical, "
                     f"{last.get('moderate',0)} moderate, {last.get('minor',0)} minor "
                     f"issue(s) still remain.")
        if last.get("resolved"):
            parts.append(f"  RESOLVED ({len(last['resolved'])}) \u2014 you fixed these last time, "
                         f"keep them fixed:")
            for d in last["resolved"][:6]:
                parts.append(f"    - {d}")
        if last.get("newly_introduced"):
            parts.append(f"  NEWLY INTRODUCED ({len(last['newly_introduced'])}) \u2014 your last edit "
                         f"CREATED these new problems; do not repeat that mistake:")
            for d in last["newly_introduced"][:6]:
                parts.append(f"    - {d}")
        return "\n".join(parts)

    # ---------- Surgical (one-issue-at-a-time) fix ----------

    # ---------- Splice-based surgical targeting ----------

    # Matches any verified-quote or study-intro token, e.g. [[Q7]] or [[S3]].
    _ANY_TOKEN_RE = re.compile(r"\[\[[A-Za-z]?\d+\]\]")
    # Issue types that are LOCAL text errors and can be fixed in a small window.
    _LOCAL_ISSUE_TYPES = {
        "fabricated_quote", "misattributed_quote", "missing_citation",
        "unknown_citation", "scaffolding_leak", "invalid_quote_placeholder",
        "invalid_summary_placeholder",
    }
    _MARK_OPEN = "\u27e6ERROR\u25b6 "
    _MARK_CLOSE = " \u25c4ERROR\u27e7"

    def _flexible_find_span(self, review: str, needle: str, max_words: int = 60):
        """Locate `needle` in the raw review tolerating whitespace differences and
        return its (start, end) char offsets, or None."""
        if not review or not needle:
            return None
        words = [w for w in re.split(r"\s+", needle.strip()) if w][:max_words]
        if not words:
            return None
        pat = r"\s+".join(re.escape(w) for w in words)
        try:
            m = re.search(pat, review, flags=re.IGNORECASE)
        except re.error:
            m = None
        if not m:
            head = words[:8]
            if not head:
                return None
            try:
                m = re.search(r"\s+".join(re.escape(w) for w in head), review,
                              flags=re.IGNORECASE)
            except re.error:
                return None
        return (m.start(), m.end()) if m else None

    def _issue_offsets(self, review: str, issue: Dict):
        """Best-effort (start, end) char offsets of an issue's offending span."""
        cs, ce = issue.get("char_start"), issue.get("char_end")
        if isinstance(cs, int) and isinstance(ce, int) and 0 <= cs < ce <= len(review):
            return (cs, ce)
        off = self._extract_offending_text(issue)
        if off:
            return self._flexible_find_span(review, off)
        return None

    # ---------- Deterministic structural fixes (no LLM) ----------

    def _first_quote_pos_for_study(self, review: str, registry: Dict, si: int):
        """Char offset of the first [[Qn]] token belonging to study `si`, or None."""
        best = None
        for m in self._PLACEHOLDER_RE.finditer(review):
            entry = registry.get(str(m.group(1)))
            if entry and entry.get("study_index") == si:
                if best is None or m.start() < best:
                    best = m.start()
        return best

    def _deterministic_intro_fix(self, review: str, si: int, registry: Dict):
        """Ensure study `si`'s [[Sn]] intro token sits at the start of the
        sentence containing that study's FIRST [[Qn]] quote. Inserts the token if
        missing, or moves it forward if it currently sits after the first quote.
        The [[Sn]] token expands to a frozen, grounded paraphrase, so this can
        never introduce a hallucination. Returns the new review, or None if it
        cannot anchor (no quote for the study)."""
        qpos = self._first_quote_pos_for_study(review, registry, si)
        if qpos is None:
            return None
        token = f"[[S{si}]]"
        sm = re.search(r"\[\[S" + str(si) + r"\]\]", review)
        if sm and sm.start() < qpos:
            return None  # already correctly placed
        work = review
        if sm:
            # Remove the misplaced (after-quote) token; it sits after qpos, so
            # qpos stays valid for the (now shorter) string before it.
            work = work[:sm.start()] + work[sm.end():]
        cands = [c for c in (work.rfind(". ", 0, qpos), work.rfind("\n", 0, qpos))
                 if c != -1]
        if not cands:
            insert_at = 0
        else:
            sb = max(cands)
            insert_at = sb + 2 if work[sb:sb + 2] == ". " else sb + 1
        return work[:insert_at] + token + " " + work[insert_at:]

    def _fix_all_structural_intros(self, review: str, registry: Dict,
                                   issues: List[Dict]):
        """Apply _deterministic_intro_fix for EVERY study flagged with a missing
        or misordered intro, in one pass (re-deriving positions on the running
        string after each edit). Returns (new_review, n_fixed)."""
        sis: List[int] = []
        for iss in issues:
            if iss.get("type") in ("missing_study_intro", "study_intro_after_quote"):
                si = iss.get("study_index")
                if si is None:
                    m = re.search(r"Study (\d+)", iss.get("description", "") or "")
                    si = int(m.group(1)) if m else None
                if si is not None and si not in sis:
                    sis.append(si)
        work = review
        n = 0
        for si in sis:
            nr = self._deterministic_intro_fix(work, si, registry)
            if nr is not None and nr != work:
                work = nr
                n += 1
        return work, n

    def _snap_left(self, review: str, pos: int) -> int:
        """Snap a left window edge back to the nearest clean boundary."""
        b = max(review.rfind("\n\n", 0, pos), review.rfind("\n", 0, pos),
                review.rfind(". ", 0, pos))
        return (b + 1) if b != -1 else 0

    def _snap_right(self, review: str, pos: int) -> int:
        """Snap a right window edge forward to the nearest clean boundary."""
        cands = [c for c in (review.find("\n\n", pos), review.find("\n", pos),
                             review.find(". ", pos)) if c != -1]
        if not cands:
            return len(review)
        c = min(cands)
        return c + 1 if review[c:c + 2] == ". " else c

    def _expand_window(self, review: str, start: int, end: int, words: int = 150):
        """Expand [start,end] by ~`words` words on each side, snap to clean
        boundaries, and ensure neither edge falls INSIDE a [[...]] token."""
        left_words = re.findall(r"\S+\s*", review[:start])
        back = "".join(left_words[-words:]) if len(left_words) > words else review[:start]
        w_start = start - len(back)
        right_words = re.findall(r"\S+\s*", review[end:])
        fwd = "".join(right_words[:words]) if len(right_words) > words else review[end:]
        w_end = end + len(fwd)
        w_start = self._snap_left(review, w_start)
        w_end = self._snap_right(review, w_end)
        # Never split a token: pull edges outside any token they land inside.
        for m in self._ANY_TOKEN_RE.finditer(review):
            if m.start() < w_start < m.end():
                w_start = m.start()
            if m.start() < w_end < m.end():
                w_end = m.end()
        return max(0, w_start), min(len(review), w_end)

    def _strip_error_markers(self, text: str) -> str:
        return text.replace(self._MARK_OPEN, "").replace(self._MARK_CLOSE, "")

    def node_surgical_fix(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 11b: SURGICAL FIX (one issue at a time)")
        review = state.get("literature_review")
        target = state.get("_surgical_target")
        if not review or not target:
            print(f"  {Fore.YELLOW}Nothing to fix surgically.{Style.RESET_ALL}")
            return state

        key = target.get("description", "")
        attempts = state.get("_surgical_attempts", {}) or {}
        retry_cap = int(self.config.get("surgical_window_retry_cap", 2))

        # ---- Deterministic structural fixes FIRST (no LLM, instant, reliable).
        # Missing / misordered [[Sn]] study-intro tokens were the dominant
        # critical class. The [[Sn]] token is a frozen grounded paraphrase, so
        # inserting/moving it can never hallucinate. We fix ALL such issues in
        # this single pass to avoid burning a verification round per study.
        if target.get("type") in ("missing_study_intro", "study_intro_after_quote"):
            registry = state.get("quote_registry") or self._assign_registry(
                state.get("study_analyses", []))
            state["quote_registry"] = registry
            fixed_review, n = self._fix_all_structural_intros(
                review, registry, state.get("last_verification_issues", []))
            if n > 0:
                state["literature_review"] = self._sanitize_review(fixed_review, state.get("quote_registry"))
                state["evidence_base_cache"] = None
                print(f"  {Fore.GREEN}{SYM_WRENCH} Deterministically inserted/moved "
                      f"{n} study-intro token(s) — no LLM call needed.{Style.RESET_ALL}")
                return state
            # Could not anchor deterministically — fall through to a whole-doc LLM fix.
            print(f"  {Fore.YELLOW}Could not place intro token(s) deterministically; "
                  f"using a whole-document fix.{Style.RESET_ALL}")
            return self._surgical_fix_wholedoc(state, target)

        # Decide window vs whole-doc. Local text errors use a fast window splice;
        # structural errors (and anything we can't locate, or that has failed the
        # window approach too many times) fall back to a whole-document fix.
        offs = None
        if target.get("type") in self._LOCAL_ISSUE_TYPES and attempts.get(key, 0) < retry_cap:
            offs = self._issue_offsets(review, target)

        if offs is not None:
            attempts[key] = attempts.get(key, 0) + 1
            state["_surgical_attempts"] = attempts
            return self._surgical_fix_window(state, target, offs)
        return self._surgical_fix_wholedoc(state, target)

    def _surgical_fix_window(self, state: ReviewState, target: Dict, offs) -> ReviewState:
        review = state["literature_review"]
        words = int(self.config.get("surgical_window_words", 150))
        e_start, e_end = offs
        w_start, w_end = self._expand_window(review, e_start, e_end, words=words)
        window = review[w_start:w_end]

        # Visually mark the exact error inside the window (for display only).
        rel_s = max(0, e_start - w_start)
        rel_e = min(len(window), e_end - w_start)
        marked = (window[:rel_s] + self._MARK_OPEN + window[rel_s:rel_e]
                  + self._MARK_CLOSE + window[rel_e:])

        rules = self._super_critical_quote_rules()
        fix_guidance = target.get("fix_guidance", "Fix the marked text.")
        prompt = f"""You are fixing ONE error in a SMALL EXCERPT of an APA 7th, quote-driven
literature review. Edit ONLY this excerpt. Everything outside it is already
correct and will be preserved automatically.

RESEARCH QUESTION: "{state['original_query']}"

{rules}

THE ERROR (the text between {self._MARK_OPEN.strip()} and {self._MARK_CLOSE.strip()} markers is the problem):
- What is wrong: {target.get('description','')}
- How to fix it: {fix_guidance}

EXCERPT TO FIX (markers show the exact error; remove the markers in your output):
\"\"\"
{marked}
\"\"\"

INSTRUCTIONS:
- Output ONLY the corrected excerpt — the same passage with the one error fixed.
- Remove the {self._MARK_OPEN.strip()} / {self._MARK_CLOSE.strip()} markers.
- Change as little as possible; do not touch anything except the marked error.
- PRESERVE every [[Qn]] and [[Sn]] token in the excerpt EXACTLY as written.
- Never hand-type a quotation or copy evidence-base display text; to remove a
  bad quote, delete it or replace it with a [[Qn]] token; to introduce a study
  use its [[Sn]] token.
- No preamble, no markdown fences, no commentary — just the corrected passage."""

        print(f"  {Fore.MAGENTA}Window fix{Style.RESET_ALL} (~{len(window.split())} words) for: "
              f"{Fore.MAGENTA}[{target.get('severity','?')}] {target.get('description','')}"
              f"{Style.RESET_ALL}", flush=True)
        result = self.agent_manager.run_primary(prompt, task="self_fix")
        if result.success and result.response:
            fixed = self._strip_error_markers(result.response.strip())
            # Guard: a runaway response (model re-emitted far more than the window)
            # is rejected to avoid corrupting the document; we keep the review and
            # let the loop retry / escalate.
            if len(fixed) > len(window) * 4 + 400:
                print(f"  {Fore.YELLOW}Window fix returned far more text than the excerpt — "
                      f"discarding this edit and will retry/escalate.{Style.RESET_ALL}")
                return state
            new_review = review[:w_start] + fixed + review[w_end:]
            new_review = self._reimpose_evidence_by_span(new_review, state)
            state["literature_review"] = self._sanitize_review(new_review, state.get("quote_registry"))
            state["evidence_base_cache"] = None
            print(f"  {Fore.GREEN}{SYM_WRENCH} Spliced fix into the review "
                  f"({result.elapsed_time:.0f}s).{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}Window fix LLM call failed: {result.error}{Style.RESET_ALL}")
        return state

    def _surgical_fix_wholedoc(self, state: ReviewState, target: Dict) -> ReviewState:
        review = state["literature_review"]
        evidence = self._get_evidence(state)
        annotated = self._annotate_issues_in_review(review, [target])
        memory = self._render_refinement_memory(state)
        rules = self._super_critical_quote_rules()

        # Hide the locked, code-assembled EVIDENCE section so a structural fix can
        # only alter the analysis; it is restored verbatim afterwards.
        locked = state.get("locked_evidence")
        review_for_prompt = self._mask_evidence(review) if locked else review
        lock_note = ""
        if locked and self._EVIDENCE_MASK in review_for_prompt:
            lock_note = (
                f"\nIMPORTANT: The EVIDENCE section is correct and LOCKED; it appears only as "
                f"the marker {self._EVIDENCE_MASK}. Reproduce that marker EXACTLY and write no "
                f"evidence content yourself.\n")

        prompt = f"""You are fixing ONE specific issue in an APA 7th, quote-driven literature
review. Change ONLY what is needed to fix THIS single issue and leave everything
else EXACTLY as it is \u2014 same wording, same tokens, same order, same sections \u2014
unchanged apart from the targeted fix.

RESEARCH QUESTION: "{state['original_query']}"

EVIDENCE BASE (with verified quotes per study):
{evidence}

{rules}
{lock_note}
{memory}

THE SINGLE ISSUE TO FIX NOW:
{annotated}

CURRENT REVIEW (fix ONLY the issue above; change NOTHING else):
{review_for_prompt}

INSTRUCTIONS:
- Make the smallest edit that fully resolves the one issue above.
- Do NOT touch any other sentence, citation, heading, or token.
- PRESERVE every [[Qn]] and [[Sn]] token elsewhere EXACTLY as written.
- Never hand-type a quotation or copy evidence-base display text.
- Output the COMPLETE revised review (all sections) with only this one issue
  fixed. No preamble, no markdown fences."""

        print(f"  {Fore.CYAN}Whole-document fix{Style.RESET_ALL} (structural / unlocatable) for: "
              f"{Fore.MAGENTA}[{target.get('severity','?')}] {target.get('description','')}"
              f"{Style.RESET_ALL}", flush=True)
        result = self.agent_manager.run_primary(prompt, task="self_fix")
        if result.success and result.response:
            fixed = self._unmask_evidence(result.response, state)
            state["literature_review"] = self._sanitize_review(fixed, state.get("quote_registry"))
            state["evidence_base_cache"] = None
            print(f"  {Fore.GREEN}{SYM_WRENCH} Surgical edit complete "
                  f"({len(result.response.split())} words, {result.elapsed_time:.0f}s)"
                  f"{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}Surgical fix LLM call failed: {result.error}{Style.RESET_ALL}")
        return state

    def _sanitize_review(self, review: str, registry: Dict = None) -> str:
        """Remove internal provenance labels that must never appear in the
        published review (#3), strip transcribed evidence text, dedupe tokens,
        and fix run-on spacing.

        When `registry` is provided, this is operating on the model's RAW output
        (before token expansion), so it also strips any hand-copied transcription
        of a registry summary/quote (the root cause of the duplicated EVIDENCE
        section). It must NOT be given a registry when called on already-expanded
        text, or it would delete the legitimately pasted quotes.
        """
        if not review:
            return review
        # Merged into a citation, e.g. "(de la Monte & Kril, 2014, verified
        # quote from full text)" -> "(de la Monte & Kril, 2014)".
        review = re.sub(
            r",\s*verified quote from (?:full text|abstract)\s*(?=\))",
            "", review, flags=re.IGNORECASE)
        # Standalone parenthetical, e.g. " (verified quote from abstract)".
        review = re.sub(
            r"\s*\((?:verified quote from (?:full text|abstract))\)",
            "", review, flags=re.IGNORECASE)
        # Any residual bare phrase.
        review = re.sub(
            r"\s*,?\s*verified quote from (?:full text|abstract)",
            "", review, flags=re.IGNORECASE)
        # Remove hand-copied transcriptions of the evidence text (pre-expansion).
        if registry:
            review = self._strip_transcribed_registry_text(review, registry)
        # Strip invalid example tokens the model copied from the instructions
        # verbatim (literal letter 'n', or a '?') — these never expand and were
        # seen rendered raw in the output, e.g. a literal [[Qn]].
        review = re.sub(r"\[\[[SQ]n\]\]", "", review)
        review = re.sub(r"\[\[[SQ]\?\]\]", "", review)
        # Deterministically dedupe tokens: each [[Sn]] / [[Qn]] must paste at most
        # once. A model that uses a token twice (seen in the wild) would otherwise
        # render the same grounded summary or quote two or three times, which is a
        # major readability problem. Keep the FIRST occurrence, drop the rest.
        review = self._dedupe_tokens(review)
        # Remove orphaned quote-mark + duplicate-citation residue left when a model
        # hand-typed quotes and the verbatim body was transcription-stripped (the
        # `" (Author, Year) " (Author, Year)` junk seen at the end of EVIDENCE).
        # Safe pre- and post-expansion: it only targets an empty quoted span.
        review = self._strip_orphan_citations(review)
        # Fix a missing space after a sentence-ending period before the next
        # capitalised word, e.g. "Crossref.The initial" -> "Crossref. The initial".
        # Guarded so it won't touch abbreviations (al., e.g., U.S.) or URLs/decimals.
        review = re.sub(r"([a-z]{3,})\.([A-Z])", r"\1. \2", review)
        return review

    def _normalize_with_map(self, s: str):
        """Return (normalized_string, index_map) where index_map[i] is the index
        in the ORIGINAL string of the i-th normalized char. Normalisation unifies
        smart quotes, lowercases, and collapses each whitespace run to one space —
        so a transcription that differs only in line-wrapping still matches."""
        out, idx = [], []
        prev_space = False
        for i, ch in enumerate(s):
            c = ch
            if c in "\u201c\u201d\u2033":
                c = '"'
            elif c in "\u2018\u2019\u2032":
                c = "'"
            if c.isspace():
                if prev_space:
                    continue
                out.append(" ")
                idx.append(i)
                prev_space = True
            else:
                out.append(c.lower())
                idx.append(i)
                prev_space = False
        return "".join(out), idx

    def _strip_transcribed_registry_text(self, review: str, registry: Dict) -> str:
        """Remove any contiguous, near-verbatim TRANSCRIPTION of a registry
        summary/quote from the review, keeping the [[token]] itself. Before token
        expansion the review should hold ONLY tokens + the model's own connective
        prose; any long span matching a registry text is the model having
        hand-copied the evidence it was shown. The token still expands to the
        canonical text, so removing the copy de-duplicates losslessly."""
        if not review or not registry:
            return review
        import difflib
        present = set(re.findall(r"\[\[([SQ]\d+)\]\]", review))
        texts = []
        for key in present:
            entry = registry.get(key) if key.startswith("S") else registry.get(key[1:])
            t = (entry or {}).get("text", "") if entry else ""
            if t and t.strip():
                texts.append(t.strip())
        # Longest first, so a big summary is removed before any of its substrings.
        texts.sort(key=len, reverse=True)
        removed_any = False
        for T in texts:
            norm_T, _ = self._normalize_with_map(T)
            norm_T = norm_T.strip()
            if len(norm_T) < 50:
                continue
            for _ in range(4):  # a text may be transcribed more than once
                norm_R, idx_map = self._normalize_with_map(review)
                sm = difflib.SequenceMatcher(None, norm_R, norm_T, autojunk=False)
                blocks = [b for b in sm.get_matching_blocks() if b.size >= 12]
                if not blocks:
                    break
                total = sum(b.size for b in blocks)
                if total < 0.6 * len(norm_T):
                    break
                a_start = blocks[0].a
                a_end = blocks[-1].a + blocks[-1].size
                if (a_end - a_start) > 1.6 * len(norm_T):
                    break  # matches scattered across the doc -> not a transcription
                raw_start = idx_map[a_start]
                raw_end = idx_map[a_end - 1] + 1
                # Swallow a tiny leftover lead-in fragment (e.g. the model's
                # paraphrased "The" where the summary said "This") sitting between
                # the token and the transcription — but only a short, period-free
                # gap, so a genuine connector sentence is never eaten.
                pre = review[:raw_start]
                tok_pos = pre.rfind("]]")
                if tok_pos != -1 and 0 <= raw_start - (tok_pos + 2) <= 30 \
                        and "." not in review[tok_pos + 2:raw_start]:
                    raw_start = tok_pos + 2
                review = review[:raw_start] + review[raw_end:]
                removed_any = True
        if removed_any:
            review = re.sub(r"\(\s*\)", "", review)
            review = re.sub(r"[ \t]{2,}", " ", review)
            review = re.sub(r"[ \t]+([.,;:])", r"\1", review)
            review = re.sub(r"\n[ \t]+\n", "\n\n", review)
            review = re.sub(r"\n{3,}", "\n\n", review)
        return review

    def _strip_orphan_citations(self, review: str) -> str:
        """Remove orphaned quote-mark + in-text-citation residue — e.g. a run of
        `" (Maiwald et al., 2024) " (Maiwald et al., 2024)` left at the end of the
        EVIDENCE section. This residue appears when a model hand-types verified
        quotes (with quotation marks + citations) and _strip_transcribed_registry_text
        deletes the verbatim BODY (it matches the registry) but not the model's
        surrounding quote mark or the duplicate citation.

        An orphan is a quote mark that (a) FOLLOWS whitespace/line-start — so it is
        NOT the closing mark of a real quote, whose closing mark follows the quote's
        text — and (b) is immediately followed by ONLY an in-text citation
        '(Author, YEAR)' with NO quoted text in between. Legitimate quotes,
        `“real body…” (Author, Year)`, are never matched (their closing mark
        follows text; their opening mark is followed by text, not a bare citation),
        so this is safe to run both before and after token expansion.
        """
        if not review:
            return review
        # quote mark(s) preceded by whitespace, then optional spaces, then a bare
        # parenthetical citation containing a 4-digit year and nothing else.
        pat = re.compile(
            r'(?<=\s)["\u201c\u201d\u2018\u2019]+[ \t]*'
            r'\([^()]*\b(?:18|19|20)\d{2}\b[^()]*\)')
        new = pat.sub("", review)
        if new != review:
            # Tidy the whitespace/blank lines the removal leaves behind.
            new = re.sub(r"[ \t]{2,}", " ", new)
            new = re.sub(r"[ \t]+\n", "\n", new)
            new = re.sub(r"\n[ \t]+", "\n", new)
            new = re.sub(r"\n{3,}", "\n\n", new)
        return new

    def _dedupe_tokens(self, review: str) -> str:
        """Remove 2nd+ occurrences of any [[Sn]] / [[Qn]] token (keep the first).
        Collapses any double spaces / stray blank lines the removal leaves."""
        if not review:
            return review
        seen = set()

        def repl(m):
            tok = m.group(0)
            if tok in seen:
                return ""
            seen.add(tok)
            return tok

        review = re.sub(r"\[\[[SQ]\d+\]\]", repl, review)
        review = re.sub(r"[ \t]{2,}", " ", review)
        review = re.sub(r"\n{3,}", "\n\n", review)
        return review

    def _strip_invalid_tokens(self, review: str, registry: Dict):
        """Deterministically REMOVE any [[Qn]]/[[Sn]] token that is not in the
        registry (a hallucinated / out-of-range token the model invented). An
        invalid token has no verified backing, so the only correct action is to
        delete it — and crucially the LLM cannot do this reliably, which is what
        sent the verification loop into dozens of non-converging passes. Returns
        (review, n_removed)."""
        if not review:
            return review, 0
        registry = registry or {}
        removed = 0

        def q_repl(m):
            nonlocal removed
            if str(m.group(1)) in registry:
                return m.group(0)
            removed += 1
            return ""

        def s_repl(m):
            nonlocal removed
            if f"S{m.group(1)}" in registry:
                return m.group(0)
            removed += 1
            return ""

        review = re.sub(r"\[\[Q(\d+)\]\]", q_repl, review)
        review = re.sub(r"\[\[S(\d+)\]\]", s_repl, review)
        if removed:
            review = re.sub(r"[ \t]{2,}", " ", review)
            review = re.sub(r"[ \t]+([.,;:])", r"\1", review)
            review = re.sub(r"\n{3,}", "\n\n", review)
        return review, removed

    def _normalize_for_match(self, s: str) -> str:
        s = s.replace("\u201c", "\"").replace("\u201d", "\"")
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _extract_quotes_and_cites(self, review: str,
                                  research_question: str = "") -> List[Dict]:
        text = self._normalize_for_match(review)
        results = []

        # Build an exclusion set of normalised strings the checker must NOT
        # treat as study quotes (#1). The research question is quoted verbatim
        # in the Introduction; without this it is flagged every pass as a
        # "quote without citation", which can never be fixed and prevents the
        # self-fix loop from ever converging.
        rq_norm = self._normalize_for_match(research_question or "").rstrip("?. ").lower()

        def _is_excluded(q: str) -> bool:
            qn = q.rstrip("?. ").lower()
            if not rq_norm:
                return False
            # Exact, or research-question-contains-quote, or quote-contains-RQ
            # (the model sometimes wraps the RQ in a lead-in sentence).
            return qn == rq_norm or qn in rq_norm or rq_norm in qn

        inline_pattern = re.compile(r'"([^"]{20,})"')
        for m in inline_pattern.finditer(text):
            qtext = m.group(1).strip()
            if _is_excluded(qtext):
                continue
            # A genuine study quote does not itself contain an APA citation;
            # if it does, this is narrative prose that happened to be quoted
            # (or a mis-segmented span), not a verbatim source quote.
            if re.search(r"\([^()]*\b(1[7-9]\d{2}|20\d{2}|21\d{2})\b[^()]*\)", qtext):
                continue
            tail = text[m.end():m.end()+200]
            cite_m = re.search(r"\(([^()]+?\d{4}[^()]*)\)", tail)
            citation_text = cite_m.group(0) if cite_m else ""
            results.append({"quote": qtext, "citation_text": citation_text, "kind": "inline"})

        paragraphs = re.split(r"\n\s*\n", review)
        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue
            leading_ws = len(para) - len(para.lstrip(" \t"))
            if leading_ws < 2:
                continue
            normalized = self._normalize_for_match(stripped)
            word_count = len(normalized.split())
            if word_count < 40:
                continue
            cite_m = re.search(r"\(([^()]+?\d{4}[^()]*)\)\s*\.?\s*$", normalized)
            if not cite_m:
                continue
            citation_text = cite_m.group(0)
            quote_body = normalized[:cite_m.start()].strip().rstrip(".")
            if len(quote_body) < 40:
                continue
            if _is_excluded(quote_body):
                continue
            results.append({"quote": quote_body, "citation_text": citation_text, "kind": "block"})

        return results

    def _parse_authors_year_from_citation(self, citation_text: str) -> Tuple[List[str], Optional[str]]:
        inner = citation_text.strip().strip("()")
        year_m = re.search(r"\b(1[7-9]\d{2}|20\d{2}|21\d{2})\b", inner)
        year = year_m.group(1) if year_m else None
        head = inner
        if year_m:
            head = inner[:year_m.start()].rstrip(", ")
        head = head.strip()
        if " et al" in head.lower():
            base = re.split(r"\s+et\s+al", head, flags=re.IGNORECASE)[0].strip()
            return ([base.strip()] if base else []), year
        if "&" in head:
            names = [p.strip() for p in head.split("&") if p.strip()]
            return names, year
        if re.search(r"\band\b", head, flags=re.IGNORECASE):
            names = [p.strip() for p in re.split(r"\band\b", head, flags=re.IGNORECASE) if p.strip()]
            return names, year
        if head:
            return [head], year
        return [], year

    def _last_name(self, author: str) -> str:
        # Delegate to the shared parser so citation matching uses the same
        # surname logic as the in-text and reference formatters (handles
        # 'Surname JM' trailing-initials format too).
        return _apa_last_name(author).lower()

    def _study_matches_citation(self, analysis: Dict, cite_names: List[str],
                                cite_year: Optional[str]) -> bool:
        if not cite_names:
            return False
        study_authors = analysis.get("paper_authors") or []
        if not study_authors:
            return False
        first_last = self._last_name(study_authors[0])
        cite_first_last = self._last_name(cite_names[0])
        if first_last != cite_first_last:
            return False
        if cite_year and analysis.get("paper_year"):
            if str(analysis.get("paper_year")).strip() != str(cite_year).strip():
                return False
        return True

    def _load_paper_into_attr_store(self, analysis: Dict) -> Optional[str]:
        if not self._attr_doc_store:
            return None
        try:
            paper_id = analysis.get("paper_id") or analysis.get("source_paper_id") or "unknown"
            text_type = analysis.get("deep_analysis_text_type", "")
            text = None

            paper = self.discovery.paper_catalog.get(paper_id)
            if paper:
                if text_type.startswith("ABSTRACT"):
                    text = paper.abstract
                elif paper.full_text_available and paper.full_text_content:
                    text = paper.full_text_content
                elif paper.abstract:
                    text = paper.abstract

            if not text:
                verified_qs = [q.get("quote", "") for q in (analysis.get("key_quotes") or [])
                               if isinstance(q, dict) and q.get("verified")]
                text = "\n\n".join(q for q in verified_qs if q)
            if not text:
                return None

            self._attr_doc_store.clear()
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"attr_{hashlib.md5(str(paper_id).encode()).hexdigest()[:10]}.txt")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(text)
            doc = self._attr_doc_store.load_document(tmp_path)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return doc.document_id if doc else None
        except Exception as e:
            logger.warning(f"Attribution store load failed: {e}")
            return None

    def _quote_found_in_paper(self, quote_text: str, analysis: Dict) -> bool:
        if not self._attr_text_matcher or not self._attr_doc_store:
            return False
        doc_id = self._load_paper_into_attr_store(analysis)
        if not doc_id:
            return False
        chunks = self._attr_doc_store.all_chunks
        if not chunks:
            return False
        best_conf = 0.0
        for ch in chunks:
            words = quote_text.split()
            key_phrases = []
            if len(words) >= 6:
                key_phrases.append(" ".join(words[:5]))
                key_phrases.append(" ".join(words[-5:]))
                if len(words) >= 12:
                    mid = len(words) // 2
                    key_phrases.append(" ".join(words[mid-2:mid+3]))
            match = self._attr_text_matcher.find_quote_in_chunk(
                quote_text, ch.text, ch.start_char, key_phrases)
            if match.success and match.confidence > best_conf:
                best_conf = match.confidence
        return best_conf >= 0.6

    def _verify_attribution(self, review: str,
                            analyses: List[Dict],
                            research_question: str = "",
                            registry: Optional[Dict[str, Dict]] = None) -> List[Dict]:
        issues: List[Dict] = []
        if not self._attr_text_matcher or not self._attr_doc_store:
            return issues

        registry = registry or {}

        # --- Placeholder-token checks (new) -------------------------------
        # 1) Every [[Qn]] quote token AND every [[Sn]] study-intro token in the
        #    review must exist in the registry.
        # 2) Determine which studies are actually USED (have >=1 quote token or
        #    their intro token). A study that is cited/discussed but used by no
        #    token must be removed.
        token_ids = [m.group(1) for m in self._PLACEHOLDER_RE.finditer(review or "")]
        unknown_tokens = sorted({tid for tid in token_ids if str(tid) not in registry},
                                key=lambda x: int(x))
        for tid in unknown_tokens:
            issues.append({
                "type": "invalid_quote_placeholder",
                "description": f"Review uses token [[Q{tid}]] which is not a valid quote "
                               f"in the evidence base.",
                "severity": "critical",
                "fix_guidance": "Use only quote tokens shown in the evidence base, or remove "
                                "this token.",
            })

        # Study-intro [[Sn]] tokens: validate against registry key "S{n}".
        summary_ids = [m.group(1) for m in self._SUMMARY_RE.finditer(review or "")]
        unknown_summaries = sorted({sid for sid in summary_ids if f"S{sid}" not in registry},
                                   key=lambda x: int(x))
        for sid in unknown_summaries:
            issues.append({
                "type": "invalid_summary_placeholder",
                "description": f"Review uses study-intro token [[S{sid}]] which is not a valid "
                               f"study in the evidence base.",
                "severity": "critical",
                "fix_guidance": "Use only [[Sn]] tokens shown in the evidence base, or remove "
                                "this token.",
            })

        # --- Scaffolding-leak guard (root cause of hand-typed quotes) --------
        # The model must NEVER transcribe the evidence-base DISPLAY text into the
        # review. Strings like 'expands to:', 'CITE AS:', '(NN words) —' and
        # '(context: ...)' exist ONLY in the scaffolding shown to the model; if
        # they appear in the review the model has copied display text verbatim,
        # which is the usual cause of hand-typed / fabricated quotes. Flag each
        # occurrence as critical, with the exact offending span, so the fix loop
        # removes it (we do NOT silently rewrite the model's quotes).
        rv = review or ""
        scaffold_patterns = [
            (r'expands to\s*:', "the '[[Sn]] expands to: \"...\"' study-summary display line"),
            (r'CITE AS\s*:', "the 'CITE AS:' citation display line"),
            (r'\(\s*\d+\s+words?\s*\)\s*[\u2014-]', "the '(NN words) —' quote display line"),
            (r'\(context\s*:', "the '(context: ...)' quote-context display line"),
            (r'VERIFIED QUOTES?\s*[\u2014-]', "the 'VERIFIED QUOTES —' display header"),
        ]
        for pat, label in scaffold_patterns:
            for m in re.finditer(pat, rv, flags=re.IGNORECASE):
                ctx_start = max(0, m.start() - 80)
                ctx_end = min(len(rv), m.end() + 160)
                snippet = rv[ctx_start:ctx_end].replace("\n", " ")
                issues.append({
                    "type": "scaffolding_leak",
                    "description": f"The review contains evidence-base DISPLAY text that must never "
                                   f"appear in prose — {label}. Offending text: \"...{snippet}...\". "
                                   f"This means quote or summary text was transcribed by hand instead "
                                   f"of using a token.",
                    "severity": "critical",
                    "char_start": m.start(),
                    "char_end": m.end(),
                    "fix_guidance": "Delete the copied display text entirely. Introduce a study ONLY "
                                    "with its [[Sn]] token and insert a quote ONLY with its [[Qn]] "
                                    "token; never write quote text, summary text, 'expands to:', "
                                    "'CITE AS:', word-counts, or '(context: ...)' yourself.",
                })

        # --- (#1) Study-intro ordering guard ---------------------------------
        # Every quoted study MUST be introduced by its [[Sn]] token BEFORE its
        # first [[Qn]] quote, so a study's finding can never appear without — or
        # ahead of — its own introduction, and an introduction can never be
        # mixed up with a different study's findings. (The token binding already
        # guarantees each quote's text+citation come from exactly one study;
        # this adds the deterministic presence+ordering guarantee.)
        s_first_pos: Dict[str, int] = {}
        for m in self._SUMMARY_RE.finditer(rv):
            sid = m.group(1)
            if sid not in s_first_pos:
                s_first_pos[sid] = m.start()
        q_first_pos_by_study: Dict[int, int] = {}
        for m in self._PLACEHOLDER_RE.finditer(rv):
            entry = registry.get(str(m.group(1)))
            if not entry:
                continue
            si = entry.get("study_index")
            if si is None:
                continue
            if si not in q_first_pos_by_study or m.start() < q_first_pos_by_study[si]:
                q_first_pos_by_study[si] = m.start()
        for si, qpos in q_first_pos_by_study.items():
            cite = self._apa_in_text_citation(analyses[si - 1]) if 1 <= si <= len(analyses) else ""
            intro_pos = s_first_pos.get(str(si))
            if intro_pos is None:
                issues.append({
                    "type": "missing_study_intro",
                    "study_index": si,
                    "description": f"Study {si} {cite} is quoted (its [[Q]] token appears) but its "
                                   f"study-intro token [[S{si}]] is missing entirely. Every quoted "
                                   f"study MUST be introduced with its own [[S{si}]] token before its "
                                   f"first quote, so a finding is never shown without introducing the "
                                   f"study it came from.",
                    "severity": "critical",
                    "fix_guidance": f"Insert the [[S{si}]] token (followed by the citation {cite}) "
                                    f"where this study is first introduced, BEFORE its first [[Q]] "
                                    f"token. Do not type the introduction yourself — use the token.",
                })
            elif intro_pos > qpos:
                issues.append({
                    "type": "study_intro_after_quote",
                    "study_index": si,
                    "description": f"Study {si} {cite} is quoted BEFORE it is introduced: its "
                                   f"[[S{si}]] intro token appears AFTER its first [[Q]] quote token. "
                                   f"The introduction must come first so a finding is never detached "
                                   f"from — or shown ahead of — its own study introduction.",
                    "severity": "critical",
                    "fix_guidance": f"Move the [[S{si}]] token (with citation {cite}) so it appears "
                                    f"BEFORE this study's first [[Q]] token.",
                })

        used_paper_ids = set()
        for tid in token_ids:
            entry = registry.get(str(tid))
            if entry and entry.get("source_paper_id"):
                used_paper_ids.add(entry.get("source_paper_id"))
        # A study introduced via its [[Sn]] token also counts as used.
        for sid in summary_ids:
            entry = registry.get(f"S{sid}")
            if entry and entry.get("source_paper_id"):
                used_paper_ids.add(entry.get("source_paper_id"))

        # Body text only (exclude an existing REFERENCES section) for the
        # cited-but-not-quoted heuristic, so reference entries don't count as
        # in-body citations.
        ref_idx = self._find_references_index(review or "")
        body_for_cite = (review or "")[:ref_idx] if ref_idx is not None else (review or "")
        cited_not_quoted = 0
        for a in analyses:
            pid = a.get("paper_id") or a.get("source_paper_id")
            if pid in used_paper_ids:
                continue
            if self._study_cited_in_body(a, body_for_cite):
                cited_not_quoted += 1
                if cited_not_quoted <= 8:
                    issues.append({
                        "type": "cited_but_not_quoted",
                        "description": f"Study {self._apa_in_text_citation(a)} is cited or "
                                       f"discussed but is not quoted via any token, so it will "
                                       f"be dropped from the review.",
                        "severity": "moderate",
                        "fix_guidance": "Either insert at least one quote token from this study "
                                        "(if it genuinely supports a point) or remove every "
                                        "mention and citation of it.",
                    })

        # --- Existing hand-typed-quote attribution checks -----------------
        quotes = self._extract_quotes_and_cites(review, research_question=research_question)

        print(f"  {Fore.CYAN}Deterministic attribution check: "
              f"{len(quotes)} hand-typed quote(s), {len(token_ids)} token(s) "
              f"({len(used_paper_ids)} studies quoted) against {len(analyses)} "
              f"studies...{Style.RESET_ALL}")

        for entry in quotes:
            qtext = entry["quote"]
            cite = entry["citation_text"]
            if not cite:
                issues.append({
                    "type": "missing_citation",
                    "description": f"Direct quote without an APA in-text citation: \"{qtext}\"",
                    "severity": "critical",
                    "fix_guidance": "Add an APA author-date citation immediately after the quote.",
                })
                continue

            cite_names, cite_year = self._parse_authors_year_from_citation(cite)
            cited_studies = [a for a in analyses
                             if self._study_matches_citation(a, cite_names, cite_year)]

            if not cited_studies:
                issues.append({
                    "type": "unknown_citation",
                    "description": f"Quote cites \"{cite}\" but no matching study exists in the "
                                   f"evidence base for quote \"{qtext}\"",
                    "severity": "critical",
                    "fix_guidance": "Either correct the citation to a study that exists in the "
                                    "evidence base, or remove the quote.",
                })
                continue

            cited_study = cited_studies[0]
            found_in_cited = self._quote_found_in_paper(qtext, cited_study)
            if found_in_cited:
                continue

            true_source = None
            for a in analyses:
                if a is cited_study:
                    continue
                if self._quote_found_in_paper(qtext, a):
                    true_source = a
                    break

            if true_source:
                correct_cite = self._apa_in_text_citation(true_source)
                issues.append({
                    "type": "misattributed_quote",
                    "description": f"Quote attributed to {cite} but actually appears in "
                                   f"{true_source.get('paper_title','?')}. "
                                   f"Quote: \"{qtext}\"",
                    "severity": "critical",
                    "fix_guidance": f"Change the citation to {correct_cite} so the attribution "
                                    f"matches the true source.",
                })
            else:
                issues.append({
                    "type": "fabricated_quote",
                    "description": f"Direct quote not found in the cited study or any other "
                                   f"study in the evidence base. Citation: {cite}. "
                                   f"Quote: \"{qtext}\"",
                    "severity": "critical",
                    "fix_guidance": "Remove the quote; replace with a verified quote from the "
                                    "evidence base or paraphrase the finding in your own words.",
                })

        crit = sum(1 for i in issues if i.get("severity") == "critical")
        if crit == 0:
            print(f"  {Fore.GREEN}{SYM_CHECK} All quotations correctly attributed.{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}{crit} attribution issue(s) detected.{Style.RESET_ALL}")
            for i in issues:
                print(f"    {Fore.RED}[{i.get('severity','?')}] {i.get('description','')}"
                      f"{Style.RESET_ALL}")
        return issues

    # ---------- Quote-token expansion & deterministic references ----------

    def _find_references_index(self, review: str) -> Optional[int]:
        """Return the character index where the REFERENCES heading line starts,
        or None if not found. Tolerates numbering and markdown markers."""
        if not review:
            return None
        m = re.search(r'(?im)^[ \t]*[#>*\s]*(?:7\.\s*)?REFERENCES\b.*$', review)
        return m.start() if m else None

    def _expand_quote_placeholders(self, review: str,
                                   registry: Dict[str, Dict]):
        """Replace every [[Qn]] token with its exact verified quote text.

        Returns (expanded_review, used_paper_ids, stats). EVERY quote token
        renders as a set-apart, indented evidence block whose verbatim text is
        wrapped in curly quotation marks (“…”) so the reader can always tell
        verbatim study text from the model's connective prose; the in-text
        citation trails the closing quote. (Any leftover token rendered inline is
        also wrapped in curly quotation marks.) Model-added quotation marks
        hugging a token are stripped first so quotes are never doubled. Unknown
        tokens (should not occur post-verification) are removed and counted.
        """
        if not review:
            return review, set(), {"expanded": 0, "unknown": 0, "block": 0, "inline": 0, "summary": 0}

        registry = registry or {}
        threshold = self.config.get("block_quote_word_threshold", 40)
        try:
            threshold = int(threshold)
        except (TypeError, ValueError):
            threshold = 40

        used_paper_ids = set()
        stats = {"expanded": 0, "unknown": 0, "block": 0, "inline": 0, "summary": 0}

        # Strip quotation marks the model may have wrapped around a token
        # (straight or curly), so inline expansion does not double-quote.
        review = re.sub(
            r'["\u201c\u2018]\s*(\[\[Q\d+\]\])\s*["\u201d\u2019]', r'\1', review)
        # A model may also wrap a [[Sn]] intro token in quotes; strip those too,
        # because a study summary is paraphrase prose and must NOT be quoted.
        review = re.sub(
            r'["\u201c\u2018]\s*(\[\[S\d+\]\])\s*["\u201d\u2019]', r'\1', review)

        # Pass 0 — STUDY-INTRO summaries: expand [[Sn]] to the frozen grounded
        # paraphrase as PLAIN PROSE (no quotation marks, no block indent). This
        # introduces the study; it is not a verbatim quote.
        def summary_repl(m):
            sid = m.group(1)
            entry = registry.get(f"S{sid}")
            if not entry:
                stats["unknown"] += 1
                return ""
            if entry.get("source_paper_id"):
                used_paper_ids.add(entry["source_paper_id"])
            stats["expanded"] += 1
            stats["summary"] += 1
            return entry.get("text", "")

        review = self._SUMMARY_RE.sub(summary_repl, review)

        # Pass 1 — EVERY quote token renders as a clearly set-apart EVIDENCE
        # BLOCK (indented, its own paragraph), pulling an immediately-following
        # in-text citation into the block. APA's 40-word inline rule is
        # intentionally NOT applied: this is a machine-built evidence review, and
        # rendering every verified quote as a distinct block is what keeps the
        # quoted evidence visually separate from the connective prose. Quotes may
        # be long — that is fine.
        cite_after = r'(?:[ \t]*(\([^()\n]*\d{4}[^()\n]*\)))?'

        def quote_block_repl(m):
            qid = m.group(1)
            cite = (m.group(2) or "").strip()
            entry = registry.get(str(qid))
            if not entry:
                stats["unknown"] += 1
                return ""
            if entry.get("source_paper_id"):
                used_paper_ids.add(entry["source_paper_id"])
            stats["expanded"] += 1
            stats["block"] += 1
            qt = (entry.get("text", "") or "").strip()
            # Wrap the verbatim text in curly quotation marks so it is always
            # visually distinct from the model's connective/analytical prose.
            # This is a deterministic, compile-time render of code-held verified
            # text — the model never typed it — so adding marks adds zero
            # hallucination risk. The citation trails the closing quote.
            out = f"\n\n    \u201c{qt}\u201d"
            if cite:
                out += f" {cite}"
            return out + "\n\n"

        review = re.sub(r'\[\[Q(\d+)\]\]' + cite_after, quote_block_repl, review)
        # Collapse the extra blank lines the block breaks may create, and remove
        # any orphaned leading punctuation a block split may leave at the start of
        # the following line (e.g. a stray ". " when a token sat mid-sentence).
        review = re.sub(r'\n{3,}', '\n\n', review)
        review = re.sub(r'\n\n[ \t]*[.,;:]+[ \t]*', '\n\n', review)
        review = review.strip() + "\n"

        def inline_repl(m):
            qid = m.group(1)
            entry = registry.get(str(qid))
            if not entry:
                stats["unknown"] += 1
                return ""
            if entry.get("source_paper_id"):
                used_paper_ids.add(entry["source_paper_id"])
            stats["expanded"] += 1
            stats["inline"] += 1
            return '\u201c' + entry.get("text", "") + '\u201d'

        review = self._PLACEHOLDER_RE.sub(inline_repl, review)

        used_paper_ids.discard(None)
        return review, used_paper_ids, stats

    def _apa_ref_sort_key(self, analysis: Dict) -> str:
        authors = analysis.get("paper_authors") or []
        surname = _apa_last_name(authors[0]) if authors else "zzzz"
        year = str(analysis.get("paper_year") or "9999")
        return f"{surname.lower()}|{year}"

    def _study_cited_in_body(self, analysis: Dict, body: str) -> bool:
        """Heuristic: is this study cited in the body (parenthetically or
        narratively)? Used so the deterministic reference rebuild never drops a
        study that is still referenced in the prose (avoids orphan citations)."""
        if not body:
            return False
        nb = self._normalize_for_match(body)
        # Exact parenthetical in-text citation present?
        cite = self._normalize_for_match(self._apa_in_text_citation(analysis))
        if cite and cite in nb:
            return True
        authors = analysis.get("paper_authors") or []
        year = str(analysis.get("paper_year") or "").strip()
        if not authors or not year:
            return False
        surname = _apa_last_name(authors[0])
        if not surname:
            return False
        # Narrative form: surname followed (within a short window) by the year,
        # e.g. "Smith et al. (2020)".
        pat = re.escape(surname) + r"[^\n]{0,40}?\b" + re.escape(year) + r"\b"
        return re.search(pat, nb, flags=re.IGNORECASE) is not None

    def _backfill_missing_venues(self, analyses: List[Dict]) -> int:
        """Deterministically recover a missing journal/venue for reference
        entries, so an APA entry is never left as a bare title + DOI when the
        journal name is actually knowable.

        Sources, in order, all REAL (never fabricated):
          1) the already-fetched discovery catalog, matched by paper_id then by
             normalised DOI — no network call, always tried;
          2) only if still missing AND a DOI exists AND
             backfill_venue_via_doi_lookup is on (default): the existing OpenAlex
             by-DOI lookup, reusing tested infrastructure.

        Never overwrites a venue that is already present. If nothing real is
        found the entry is left exactly as-is (honest incompleteness, no guess).
        Returns the number of venues filled. Fully defensive: any failure leaves
        the analyses untouched.
        """
        if not analyses:
            return 0
        # Build local indexes from the catalog (paper_id and normalised DOI).
        by_id: Dict[str, Any] = {}
        by_doi: Dict[str, Any] = {}
        try:
            catalog = list(self.discovery.get_all_papers())
        except Exception:
            catalog = []
        for p in catalog:
            pid = getattr(p, "paper_id", None)
            if pid:
                by_id[pid] = p
            d = (getattr(p, "doi", "") or "").replace("https://doi.org/", "").strip().lower()
            if d:
                by_doi.setdefault(d, p)

        use_net = bool(self.config.get("backfill_venue_via_doi_lookup", True))
        oa = None
        if use_net:
            try:
                oa = self.discovery.clients.get("openalex")
            except Exception:
                oa = None

        def _norm_doi(d: str) -> str:
            return (d or "").replace("https://doi.org/", "").strip().lower()

        filled = 0
        for a in analyses:
            if (a.get("paper_venue") or "").strip():
                continue
            venue = ""
            # 1a) Local catalog by paper_id.
            pid = a.get("paper_id") or a.get("source_paper_id")
            pm = by_id.get(pid) if pid else None
            # 1b) Local catalog by DOI.
            if pm is None:
                dn = _norm_doi(a.get("paper_doi") or "")
                if dn:
                    pm = by_doi.get(dn)
            if pm is not None:
                venue = (getattr(pm, "venue", "") or "").strip()
            # 2) Network fallback: OpenAlex by-DOI lookup, only if still empty.
            if not venue and oa is not None:
                doi_raw = (a.get("paper_doi") or "").strip()
                if doi_raw:
                    try:
                        meta = oa.lookup_by_doi(doi_raw)
                        if meta is not None:
                            venue = (getattr(meta, "venue", "") or "").strip()
                    except Exception:
                        venue = ""
            if venue:
                a["paper_venue"] = venue
                filled += 1
        return filled

    def _rebuild_references_section(self, review: str, analyses: List[Dict],
                                    used_paper_ids: set) -> str:
        """Rebuild REFERENCES to contain exactly the studies that were quoted
        (have >=1 used token) plus any study still cited in the body. Studies the
        model chose not to quote (and does not cite) are dropped. Controlled by
        rebuild_references_from_used_quotes; if the include-set is empty, the
        original review is returned unchanged as a safety net."""
        if not review:
            return review
        if not self.config.get("rebuild_references_from_used_quotes", True):
            return review

        ref_idx = self._find_references_index(review)
        body_text = review[:ref_idx] if ref_idx is not None else review

        include = []
        seen = set()
        for a in analyses:
            pid = a.get("paper_id") or a.get("source_paper_id")
            if pid in used_paper_ids or self._study_cited_in_body(a, body_text):
                key = pid if pid is not None else id(a)
                if key in seen:
                    continue
                seen.add(key)
                include.append(a)

        if not include:
            # Nothing recognised as used/cited — do not wipe the references.
            return review

        # Recover any missing journal/venue from real metadata (catalog first,
        # then DOI lookup) so entries are complete WITHOUT asking the LLM — which
        # would risk an invented journal name. Pure no-op when nothing is missing.
        self._backfill_missing_venues(include)

        entries = sorted(
            ((self._apa_ref_sort_key(a), self._apa_reference_entry(a)) for a in include),
            key=lambda t: t[0])
        refs_block = "REFERENCES\n\n" + "\n\n".join(e for _, e in entries)

        if ref_idx is not None:
            return body_text.rstrip() + "\n\n" + refs_block + "\n"
        return review.rstrip() + "\n\n" + refs_block + "\n"

    # ---------- LLM-side verification (still runs, but now after attribution) ----------

    # Phrases that mark an LLM-verification claim as belonging to the
    # DETERMINISTIC domain (token presence / quote attribution / treating tokens
    # as unexpanded errors). These are checked authoritatively elsewhere; LLM
    # claims matching them are dropped (the model raises persistent false
    # positives here that otherwise make the review thrash forever).
    _LLM_DETERMINISTIC_DOMAIN_PATTERNS = [
        r"no\s+quote\s+token",
        r"no\s+\w*\s*token\s+(from|for)\b",
        r"token\s+from\s+(this|that)\s+study\s+was\s+(not\s+)?inserted",
        r"(not|never)\s+(quoted|tokeni[sz]ed)\b",
        r"cited[^.]{0,90}\bbut\b[^.]{0,90}(no\b|not\b|without\b)[^.]{0,40}token",
        r"\bwithout\s+(using\s+)?the\s+token",
        r"paraphrase[^.]{0,70}(instead of|without)[^.]{0,25}token",
        r"token[^.]{0,30}(missing|not\s+used|not\s+inserted|not\s+present|absent)",
        r"\[\[[sq]\d+\]\][^.]{0,90}(missing|not\s+used|not\s+inserted|not\s+expanded|"
        r"not\s+replaced|\braw\b|placeholder|metadata)",
        r"raw\s+display\s+metadata",
        r"placeholder[s]?\s+(for|not)",
        r"not\s+expanded",
        r"every\s+(cited\s+)?study\s+must\s+be\s+(quoted|tokeni[sz]ed)",
        r"no\s+corresponding\s+(quote\s+)?token",
    ]
    # Hand-typed-quote / scaffolding claims: honoured ONLY if the deterministic
    # check also found a hand-typed quote this pass (corroboration), since the
    # deterministic check is exhaustive for these.
    _LLM_HANDTYPED_PATTERNS = [
        r"hand-?typed", r"expands\s+to", r"\bcite\s+as\b", r"\(\s*\d+\s*words?",
        r"verified\s+quotes", r"quotation\s+marks[^.]{0,40}instead",
    ]
    # REFERENCES are rebuilt deterministically at COMPILE, which runs AFTER
    # verification — so at verification time the REFERENCES section is still an
    # empty placeholder and the LLM verifier predictably complains that it is
    # missing / incomplete / lacks the journal name / is not APA-formatted. Every
    # such complaint is moot: the deterministic rebuild (controlled by
    # rebuild_references_from_used_quotes) overwrites the section afterwards with
    # exactly the quoted/cited studies, formatted by apa_reference(). Honouring
    # these complaints only burns verification passes and tempts the model to
    # INVENT journal names. They are suppressed ONLY while the deterministic
    # rebuild is the authority (the config flag below), so turning the rebuild off
    # restores the LLM's say over references.
    _LLM_REFERENCE_DOMAIN_PATTERNS = [
        r"\breferences?\b[^.]{0,90}(missing|incomplete|empty|absent|not\s+present|"
        r"lacks?|omit|blank|abruptly|no\s+(citations|entries))",
        r"(missing|incomplete|empty|absent|no)\b[^.]{0,40}\breferences?\b\s*(section|list)?",
        r"\breference(s)?\s+(list|section|entry|entries)\b[^.]{0,90}(journal|volume|"
        r"issue|page|doi|title|hanging\s+indent|format|apa|order|alphabe)",
        r"(omit|omits|lacks?|missing|without|absent)\b[^.]{0,40}(journal\s+name|"
        r"volume(/| or |\s+and\s+|, )?\s*(pages?|page\s+numbers?)?|page\s+numbers?|"
        r"source\s+title)",
        r"must\s+include\s+the\s+(source\s+title|journal)",
        r"hanging\s+indent",
        r"full\s+citation\s+for\b",
        r"\breferences?\b[^.]{0,90}(apa\s*7|apa\s+7th|properly\s+format|not\s+"
        r"(properly\s+)?format|requires?\s+.*format)",
    ]

    def _filter_llm_verification_issues(self, llm_issues, attribution_issues, review):
        """Drop LLM-verifier issues that contradict the deterministic attribution
        check. Returns (kept_issues, n_suppressed)."""
        if not llm_issues:
            return [], 0
        det_handtyped_types = {"fabricated_quote", "misattributed_quote",
                               "missing_citation", "unknown_citation", "scaffolding_leak"}
        det_has_handtyped = any(i.get("type") in det_handtyped_types
                                for i in (attribution_issues or []))
        dom = [re.compile(p, re.IGNORECASE) for p in self._LLM_DETERMINISTIC_DOMAIN_PATTERNS]
        hand = [re.compile(p, re.IGNORECASE) for p in self._LLM_HANDTYPED_PATTERNS]
        # Reference-section complaints are deterministic-domain too, but ONLY when
        # the deterministic rebuild is the authority (it runs after verification
        # and replaces the whole section). Compile the set once and gate on the
        # rebuild flag so the suppression vanishes if the rebuild is disabled.
        refs_authoritative = bool(
            self.config.get("rebuild_references_from_used_quotes", True))
        ref = [re.compile(p, re.IGNORECASE) for p in self._LLM_REFERENCE_DOMAIN_PATTERNS]
        kept, dropped = [], 0
        for iss in llm_issues:
            desc = iss.get("description", "") or ""
            itype = (iss.get("type", "") or "").lower()
            # Type-based fast path for obvious deterministic-domain echoes.
            if itype in ("cited_but_not_quoted", "missing_token", "no_token",
                         "token_missing", "unexpanded_token"):
                dropped += 1
                continue
            if any(p.search(desc) for p in dom):
                dropped += 1
                continue
            # Moot reference-section complaints (rebuilt deterministically later).
            if refs_authoritative and (
                    itype in ("missing_section", "references_missing",
                              "reference_format", "incomplete_reference")
                    or any(p.search(desc) for p in ref)):
                dropped += 1
                continue
            if any(p.search(desc) for p in hand):
                if det_has_handtyped:
                    kept.append(iss)
                else:
                    dropped += 1
                continue
            kept.append(iss)
        return kept, dropped

    def node_verify_review(self, state: ReviewState) -> ReviewState:
        self._print_phase_banner("PHASE 12: VERIFICATION (attribution + LLM)")
        review = state.get("literature_review")
        if not review:
            state["verification_passed"] = True
            return state
        # The no-evidence report contains no quotes, no citations and no claims
        # about the literature, so there is nothing for either the deterministic
        # attribution check or the LLM pass to verify.
        if state.get("_no_evidence_report"):
            print(f"  {Fore.YELLOW}No-evidence report — no quotes or claims to "
                  f"verify.{Style.RESET_ALL}")
            state["verification_passed"] = True
            return state
        state["verification_attempts"] = state.get("verification_attempts", 0) + 1

        analyses = state.get("study_analyses", [])
        # Defend against a missing/empty registry at verify time (the symptom that
        # made EVERY token read "invalid" and spun the loop): rebuild it from the
        # analyses if needed so token validity is judged against the real set.
        registry = state.get("quote_registry") or {}
        if not registry and analyses:
            registry = self._assign_registry(analyses)
            state["quote_registry"] = registry
        # An invalid (hallucinated / out-of-range) token is a DETERMINISTIC
        # problem — the LLM cannot reliably remove it, which is exactly what sent
        # this loop into dozens of passes. Strip such tokens here, once, so they
        # never become an unfixable critical. (A token with no registry entry has
        # no verified backing, so deleting it loses nothing real.)
        review, n_bad_tokens = self._strip_invalid_tokens(review, registry)
        if n_bad_tokens:
            review = self._sanitize_review(review)
            state["literature_review"] = review
            print(f"  {Fore.YELLOW}Removed {n_bad_tokens} invalid/hallucinated token(s) "
                  f"deterministically (not in the evidence base) — these are unfixable by "
                  f"the LLM and were stripped to prevent a verification loop.{Style.RESET_ALL}")

        attribution_issues = self._verify_attribution(
            review, analyses, research_question=state.get("original_query", ""),
            registry=registry)

        evidence = self._get_evidence(state)
        current_date = datetime.now().strftime("%B %Y")

        prompt = f"""Final SEMANTIC accuracy check on an APA 7th, quote-driven literature review.

CURRENT DATE: {current_date}
(Papers dated this year or before are NOT future-dated.)

RESEARCH QUESTION: "{state['original_query']}"

EVIDENCE BASE (with verified quotes per study):
{evidence}

LITERATURE REVIEW:
{review}

CRITICAL CONTEXT — HOW TOKENS WORK (read carefully):
- [[Qn]] and [[Sn]] are PLACEHOLDER TOKENS. They are SUPPOSED to remain in the
  text exactly as written; a later automated step expands them into the verified
  quote / grounded study introduction. Their presence is CORRECT and REQUIRED.
- Whether each study has its tokens, whether quotes are attributed to the right
  study, and whether any quote was hand-typed are ALL checked deterministically
  and are GUARANTEED correct. That is NOT your job.

THEREFORE YOU MUST NOT FLAG (these are false positives — a separate exact check
owns them, and doing so causes the review to thrash):
- "Study X is cited but not quoted" / "no [[Qn]] token inserted" / "token
  missing" / "not tokenized".
- "[[Sn]] / [[Qn]] is raw metadata / a placeholder that should be replaced / was
  not expanded." (They are meant to stay literal — do not touch them.)
- "a paraphrase was used instead of the token" for token presence.
Treat every study as correctly tokenized and every quote as correctly attributed.

YOUR CHECKS (semantic only):
1. DIRECT ANSWER — does the CONCLUSION actually answer the research question?
2. RELEVANCE — no off-topic content or claims unsupported by the evidence base.
3. APA STRUCTURE — the 7 sections (Introduction, Method, Evidence, Discussion,
   Limitations, Conclusion, References) are all present and correctly ordered;
   flag any extra non-standard section.
4. HONESTY ON LIMITATIONS — where indirect/tangential or low-quality evidence is
   used, the Limitations section acknowledges it.
5. REFERENCES — the References section is present and not empty.

Report ONLY genuine semantic problems. If you are unsure, do NOT flag it. If the
review reads as a coherent answer with the right sections, it PASSES.

Respond with ONLY JSON:
{{
    "passes_verification": true/false,
    "critical_issues": <count>,
    "moderate_issues": <count>,
    "issues": [{{"type": "...", "description": "...", "severity": "..."}}]
}}"""

        print(f"  LLM verification pass {state['verification_attempts']}...", flush=True)
        result = self.agent_manager.run_primary(
            prompt, as_json=True, task="verification")

        llm_issues: List[Dict] = []
        if result.success and result.json_response:
            data = result.json_response
            ll = data.get("issues", []) if isinstance(data.get("issues"), list) else []
            llm_issues = ll

        # Suppress LLM claims that stray into the DETERMINISTIC domain (token
        # presence, quote attribution, treating tokens as errors). The model
        # repeatedly raises false positives here that contradict the exact check
        # above and cause the loop to oscillate. The deterministic check is the
        # sole authority on quote/token integrity.
        llm_issues, suppressed = self._filter_llm_verification_issues(
            llm_issues, attribution_issues, review)
        if suppressed:
            print(f"  {Fore.CYAN}Suppressed {suppressed} LLM claim(s) that contradict the "
                  f"deterministic attribution check (token presence & attribution are verified "
                  f"deterministically, not by the LLM).{Style.RESET_ALL}")

        all_issues = attribution_issues + llm_issues
        critical_count = sum(1 for i in all_issues if i.get("severity") == "critical")
        moderate_count = sum(1 for i in all_issues if i.get("severity") == "moderate")
        minor_count = sum(1 for i in all_issues if i.get("severity") == "minor")

        # Convergence configuration.
        holistic_attempts = int(self.config.get("verification_holistic_attempts", 3))
        surgical_max = int(self.config.get("verification_surgical_max_passes", 60))
        require_zero_moderate = bool(self.config.get("verification_require_zero_moderate", True))
        attempt = state["verification_attempts"]

        # --- Refinement memory: diff this pass against the previous one so the
        # next fix prompt can show the model what it resolved, what still
        # remains, and what it newly introduced.
        prev = state.get("verification_refinement_memory", []) or []
        prev_descs = set(prev[-1].get("issues_after_descs", [])) if prev else set()
        cur_descs = [i.get("description", "") for i in all_issues]
        cur_set = set(cur_descs)
        resolved = sorted(prev_descs - cur_set) if prev else []
        newly = sorted(cur_set - prev_descs) if prev else []
        record = {
            "attempt": attempt,
            "mode": state.get("_verification_mode", "holistic"),
            "targeted": ((state.get("_surgical_target") or {}).get("description", "")
                         if state.get("_verification_mode") == "surgical" else ""),
            "critical": critical_count, "moderate": moderate_count, "minor": minor_count,
            "issues_after_descs": cur_descs,
            "resolved": resolved,
            "newly_introduced": newly,
        }
        state["verification_refinement_memory"] = prev + [record]

        # Live issue list, used by the holistic + surgical fix prompts.
        state["last_verification_issues"] = all_issues
        state["evidence_base_cache"] = None

        clean = (critical_count == 0 and (moderate_count == 0 or not require_zero_moderate))

        self._print_verification_result(critical_count, moderate_count, minor_count,
                                        all_issues, resolved, newly, attempt)

        if clean:
            state["verification_passed"] = True
            state["_surgical_target"] = None
            return state

        # Not clean — keep going. NEVER ship with a critical issue.
        state["verification_passed"] = False

        # Absolute safety cap so the loop cannot run literally forever. If hit
        # with issues remaining we stop and report FAILED loudly — we do NOT
        # fabricate success and we do NOT auto-mangle the model's quotes.
        if attempt >= holistic_attempts + surgical_max:
            state["verification_exhausted"] = True
            state["verification_passed"] = True  # allow the graph to terminate
            print(f"  {Fore.RED}{Style.BRIGHT}VERIFICATION NOT CONVERGED after {attempt} passes: "
                  f"{critical_count} critical, {moderate_count} moderate issue(s) REMAIN. "
                  f"These UNRESOLVED issues are listed in the run summary and saved data file."
                  f"{Style.RESET_ALL}")
            return state

        if attempt <= holistic_attempts:
            state["_verification_mode"] = "holistic"
            state["_surgical_target"] = None
            print(f"  {Fore.YELLOW}{SYM_LOOP} Will rewrite holistically "
                  f"(holistic attempt {attempt}/{holistic_attempts}).{Style.RESET_ALL}")
        else:
            # Surgical mode: fix exactly ONE issue (criticals first), leaving the
            # rest of the review untouched, then re-verify.
            ordered = ([i for i in all_issues if i.get("severity") == "critical"] +
                       [i for i in all_issues if i.get("severity") == "moderate"] +
                       [i for i in all_issues if i.get("severity") not in ("critical", "moderate")])
            state["_verification_mode"] = "surgical"
            state["_surgical_target"] = ordered[0] if ordered else None
            remaining = critical_count + (moderate_count if require_zero_moderate else 0)
            print(f"  {Fore.YELLOW}{SYM_LOOP} SURGICAL mode: fixing ONE issue at a time "
                  f"({remaining} issue(s) left). Targeting now:{Style.RESET_ALL}")
            if state["_surgical_target"]:
                print(f"    {Fore.MAGENTA}[{state['_surgical_target'].get('severity','?')}] "
                      f"{state['_surgical_target'].get('description','')}{Style.RESET_ALL}")
        return state

    def _print_verification_result(self, critical, moderate, minor, all_issues,
                                   resolved, newly, attempt):
        """Print the FULL, un-truncated verification outcome (#3): every issue,
        its full description and fix guidance, plus the resolved/new diff."""
        if critical == 0 and moderate == 0:
            print(f"  {Fore.GREEN}{SYM_CHECK} PASSED — no critical or moderate issues "
                  f"(minor: {minor}).{Style.RESET_ALL}")
            return
        head_color = Fore.RED if critical else Fore.YELLOW
        print(f"  {head_color}{critical} critical, {moderate} moderate, {minor} minor "
              f"after pass {attempt}.{Style.RESET_ALL}")
        if resolved:
            print(f"  {Fore.GREEN}Resolved since last pass: {len(resolved)}{Style.RESET_ALL}")
        if newly:
            print(f"  {Fore.RED}Newly introduced since last pass: {len(newly)}{Style.RESET_ALL}")
        for idx, iss in enumerate(all_issues, 1):
            sev = iss.get("severity", "?")
            color = (Fore.RED if sev == "critical"
                     else (Fore.YELLOW if sev == "moderate" else Fore.WHITE))
            print(f"    {color}{idx}. [{sev}] {iss.get('description', '')}{Style.RESET_ALL}")
            fg = iss.get("fix_guidance")
            if fg:
                print(f"       {Fore.CYAN}\u2192 fix: {fg}{Style.RESET_ALL}")

    # ---------- Compile output ----------

    def node_compile_output(self, state: ReviewState) -> ReviewState:
        review = state.get("literature_review")
        if review:
            # Paste the exact verified quotes into the review in place of their
            # [[Qn]] tokens, then rebuild REFERENCES to exactly the quoted (and
            # any still-cited) studies, then strip internal provenance labels.
            registry = state.get("quote_registry") or {}
            analyses = state.get("study_analyses", [])
            expanded, used_paper_ids, stats = self._expand_quote_placeholders(review, registry)
            expanded = self._rebuild_references_section(expanded, analyses, used_paper_ids)
            expanded = self._sanitize_review(expanded)
            state["literature_review"] = expanded
            review = expanded

            total_studies = len(analyses)
            used_studies = len(used_paper_ids)
            print(f"  {Fore.GREEN}{SYM_QUOTE} Pasted {stats['expanded']} verified quote(s) "
                  f"({stats['block']} block, {stats['inline']} inline) into the review."
                  f"{Style.RESET_ALL}")
            if stats.get("unknown"):
                print(f"  {Fore.YELLOW}{stats['unknown']} unknown token(s) were removed "
                      f"(not found in the evidence base).{Style.RESET_ALL}")
            dropped = total_studies - used_studies
            if dropped > 0:
                print(f"  {Fore.YELLOW}{dropped} of {total_studies} analysed studies were not "
                      f"quoted and were dropped from the review.{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}{used_studies} study(ies) quoted in the final review."
                  f"{Style.RESET_ALL}")

            title = self._generate_title_from_review(state)
            state["review_title"] = title
            print(f"\n  {Fore.WHITE}Review title: {Fore.CYAN}{title}{Style.RESET_ALL}")
            self._rename_papers_dir(state)
        state["final_output"] = review
        self._save(state)
        return state

    def _generate_title_from_review(self, state: ReviewState) -> str:
        review = state.get("literature_review", "")
        query = state["original_query"]

        conclusion_match = re.search(
            r'(?:^|\n)\s*(?:7\.\s*)?CONCLUSION.*?(?=\n\s*(?:8\.|REFERENCES)|\Z)',
            review, re.DOTALL | re.IGNORECASE)
        conclusion_excerpt = ""
        if conclusion_match:
            conclusion_excerpt = conclusion_match.group(0)[:1500]
        else:
            ref_idx = review.upper().rfind("REFERENCES")
            if ref_idx > 0:
                conclusion_excerpt = review[max(0, ref_idx - 2000):ref_idx]
            else:
                conclusion_excerpt = review[-2000:]

        prompt = f"""Generate a single concise academic title for a literature review.

THE LITERATURE REVIEW ANSWERED THIS RESEARCH QUESTION:
"{query}"

CONCLUSION SECTION:
{conclusion_excerpt}

Generate ONE title that describes the topic of THIS REVIEW.

RULES:
- 6-15 words.
- Professional academic tone.
- Describes the REVIEW topic, NOT any individual study.
- DO NOT copy a study title.
- DO NOT include question marks or quotation marks.
- DO NOT rephrase as a question.
- Colons OK.

Respond with ONLY the title text."""

        result = self.agent_manager.run_primary(prompt, task="title_generation")
        if result.success and result.response:
            title = result.response.strip().strip('"\'')
            for line in title.split('\n'):
                line = line.strip().strip('"\'')
                if line and not line.lower().startswith(('title:', 'note:', 'here')):
                    title = line
                    break
            title = re.sub(r'^(title|review)\s*:\s*', '', title, flags=re.IGNORECASE)
            title = re.sub(r'[^\w\s:-]', '', title).strip()
            title = re.sub(r'\s+', ' ', title)
            if len(title) > 120:
                title = title[:120].rsplit(' ', 1)[0]
            if len(title.split()) > 20 or len(title.split()) < 3:
                title = state["original_query"].strip().rstrip('?').title()
                title = re.sub(r'[^\w\s:-]', '', title).strip()
                if len(title) > 120:
                    title = title[:120].rsplit(' ', 1)[0]
            return title

        title = state["original_query"].strip().rstrip('?').title()
        title = re.sub(r'[^\w\s:-]', '', title).strip()
        if len(title) > 120:
            title = title[:120].rsplit(' ', 1)[0]
        return title

    def _rename_papers_dir(self, state: ReviewState):
        title = state.get("review_title", "")
        if not title:
            return
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        if not safe_title:
            return
        old_dir = self.discovery.papers_dir
        new_dir = os.path.join(self.paths.get("papers_directory", "Papers"), safe_title)
        if old_dir == new_dir or not os.path.exists(old_dir):
            return
        try:
            if os.path.exists(new_dir):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_dir = f"{new_dir}_{ts}"
            os.rename(old_dir, new_dir)
            self.discovery.papers_dir = new_dir
            print(f"  {Fore.GREEN}Papers dir renamed to: {os.path.basename(new_dir)}{Style.RESET_ALL}")
        except OSError as e:
            logger.warning(f"Could not rename papers dir: {e}")

    def _save(self, state):
        out = self.paths.get("output_directory", "Reviews")
        title = state.get("review_title", "untitled")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        if not safe:
            safe = "untitled"

        review = state.get("literature_review")
        if review:
            p = os.path.join(out, f"{safe}_{ts}.txt")
            with open(p, 'w', encoding='utf-8') as f:
                f.write(f"ACADEMIC LITERATURE REVIEW (APA 7th)\n{'='*60}\n")
                f.write(f"Title: {title}\n")
                f.write(f"Question: {state['original_query']}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Papers: {len(self.discovery.paper_catalog)}\n")
                f.write(f"Studies (post-curation): {len(state.get('study_analyses',[]))}\n")
                f.write(f"Search rounds: {state.get('discovery_round', 0)}\n")
                f.write(f"Tangential engagements: {state.get('tangential_engagement_count', 0)}\n")
                f.write(f"{'='*60}\n\n{review}")
            print(f"  {Fore.GREEN}{SYM_CHECK} Review saved: {p}{Style.RESET_ALL}")

        ap = os.path.join(out, f"{safe}_{ts}_data.json")
        with open(ap, 'w', encoding='utf-8') as f:
            json.dump({"query": state["original_query"], "title": title,
                       "catalog": self.discovery.get_catalog_summary(),
                       "analyses": state.get("study_analyses", []),
                       "methodology": state.get("methodology_assessment"),
                       "methodology_gap_fill_done": state.get("methodology_gap_fill_done", False),
                       "methodology_weaknesses": state.get("methodology_weaknesses", []),
                       "methodology_gap_queries": state.get("methodology_gap_queries", []),
                       "self_review_issues": state.get("self_review_issues", []),
                       "final_verification_issues": state.get("last_verification_issues", []),
                       "verification_refinement_memory": state.get("verification_refinement_memory", []),
                       "verification_exhausted": state.get("verification_exhausted", False),
                       "strategy_memo": state.get("strategy_memo", ""),
                       "search_history": state.get("search_history", []),
                       "curation_history": state.get("curation_history", []),
                       "sufficiency_decision": state.get("last_sufficiency_decision"),
                       "tangential_engagements": state.get("tangential_engagement_count", 0),
                       "tangential_round_count": state.get("tangential_round_count", 0),
                       "tangential_papers_added": state.get("tangential_papers_added", 0)},
                      f, indent=2, default=str)
        print(f"  {Fore.GREEN}{SYM_CHECK} Data saved: {ap}{Style.RESET_ALL}")

    def _display_review_colored(self, review: str, title: str):
        print(f"\n{Fore.CYAN}{'═'*70}")
        print(f"  {SYM_BOOK} {title}")
        print(f"{'═'*70}{Style.RESET_ALL}\n")
        lines = review.split('\n')
        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()
            colored = False
            for section, color in SECTION_COLORS.items():
                if upper.startswith(section) or any(upper.startswith(f"{n}. {section}") for n in range(1,8)) or \
                   (f"# {section}" in upper):
                    print(f"\n{color}{Style.BRIGHT}{stripped}{Style.RESET_ALL}")
                    colored = True
                    break
            if not colored:
                if stripped.startswith('#') or (stripped.isupper() and 3 < len(stripped) < 60):
                    print(f"{Fore.WHITE}{Style.BRIGHT}{stripped}{Style.RESET_ALL}")
                else:
                    print(line)
        print(f"\n{Fore.CYAN}{'═'*70}{Style.RESET_ALL}")

    def run(self, query: str) -> Dict:
        self._interrupted = False
        clear_interrupt()
        start = time.time()

        # Begin the per-run verbose log. Everything printed from here until
        # stop() is mirrored into Logs/<Review_Name>.txt (named at the end).
        session_log = SessionLogger(self.paths.get("logs_directory", "Logs"))
        session_log.start(query=query)

        final: Dict = {}
        result: Dict = {"query": query, "review": None, "title": None,
                        "papers_found": 0, "papers_with_full_text": 0,
                        "studies_analyzed": 0, "elapsed_time": 0.0}
        try:
            print(f"\n{'='*70}\n{Fore.CYAN}{SYM_BOOK} ACADEMIC LITERATURE REVIEW{Style.RESET_ALL}\n"
                  f"{Fore.WHITE}Query: {query}{Style.RESET_ALL}\n{'='*70}")

            initial: ReviewState = {
                "original_query": query, "review_title": None,
                "research_plan": None, "focus_areas_completed": [],
                "discovery_round": 0,
                "max_discovery_rounds": self.config.get("max_discovery_rounds", 15),
                "study_summaries": [], "read_paper_ids": [], "study_analyses": [],
                "methodology_assessment": None, "evidence_base_cache": None,
                "methodology_gap_fill_done": False,
                "methodology_gaps_significant": False,
                "methodology_weaknesses": [],
                "methodology_gap_queries": [],
                "quote_registry": {},
                "locked_evidence": None,
                "evidence_plan": None,
                "literature_review": None,
                "self_review_issues": [], "self_review_done": False,
                "verification_attempts": 0,
                "max_verification_attempts": self.config.get("verification_max_retries", 2),
                "verification_passed": True,
                "last_verification_issues": [],
                "_verification_mode": "holistic",
                "verification_refinement_memory": [],
                "_surgical_target": None,
                "_surgical_attempts": {},
                "verification_exhausted": False,
                "ready_to_write": False,
                "final_output": None, "interrupted": False, "errors": [],
                "search_history": [],
                "strategy_memo": "",
                "distillations_done": 0,
                "tangential_mode_active": False,
                "tangential_engagement_count": 0,
                "tangential_round_count": 0,
                "stagnant_round_count": 0,
                "all_study_summaries": [],
                "filter_dropped_total": 0,
                "curation_excluded_total": 0,
                "executed_queries": [],
                "_papers_added_this_round": 0,
                "tangential_papers_added": 0,
                "tangential_distillations_done": 0,
                "in_tangential_round": False,
                "last_sufficiency_decision": None,
                "sufficiency_reasoning": "",
                "curated_paper_ids": [],
                "curation_history": [],
                "rounds_since_last_curation": 0,
                "tangential_rounds_since_last_curation": 0,
                "_no_evidence_report": False,
                "_insufficient_evidence_reason": None,
                # Run-bookkeeping for the Phase 8b gate (declared in ReviewState so
                # they persist across nodes). run_start_time is also set in
                # node_setup_review; initialising it here is harmless and keeps the
                # channel populated from the very first node.
                "run_start_time": time.time(),
                "post_review_retries": 0,
                "_post_review_action": None,
                "_tangential_curated_count_this_engagement": 0,
            }

            final = self.graph.invoke(initial)
            elapsed = time.time() - start
            self._display(final, elapsed)
            self._log_run_summary(final, elapsed)

            result = {"query": query, "review": final.get("final_output"),
                      "title": final.get("review_title"),
                      "papers_found": len(self.discovery.paper_catalog),
                      "papers_with_full_text": len(self.discovery.get_full_text_papers()),
                      "studies_analyzed": len(final.get("study_analyses", [])),
                      "elapsed_time": elapsed}
        finally:
            # Always close + name the log, even if the run errored or was
            # interrupted, so partial runs are still captured for debugging.
            title = final.get("review_title", "") if isinstance(final, dict) else ""
            session_log.stop(review_title=title or "")

        return result

    def _log_run_summary(self, state: Dict, elapsed: float):
        """Print a compact, greppable end-of-run summary so any problems are
        easy to find in the log: errors raised, unresolved verification issues,
        and key counts. All of this is captured by the session log."""
        print(f"\n{Fore.WHITE}{'='*70}")
        print(f"  RUN SUMMARY")
        print(f"{'='*70}{Style.RESET_ALL}")
        m, sec = int(elapsed // 60), int(elapsed % 60)
        print(f"  Elapsed: {m}m {sec}s")
        print(f"  Studies in final review: {len(state.get('study_analyses', []))}")
        print(f"  Verification attempts: {state.get('verification_attempts', 0)} | "
              f"passed: {state.get('verification_passed', '?')}")
        if state.get("verification_exhausted"):
            print(f"  {Fore.RED}{Style.BRIGHT}WARNING: verification did NOT fully converge — "
                  f"the review below still contains the unresolved issues listed beneath."
                  f"{Style.RESET_ALL}")

        errors = state.get("errors", []) or []
        if errors:
            print(f"\n  {Fore.RED}ERRORS ({len(errors)}):{Style.RESET_ALL}")
            for e in errors:
                print(f"    {Fore.RED}- {e}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.GREEN}No errors recorded.{Style.RESET_ALL}")

        unresolved = state.get("last_verification_issues", []) or []
        if unresolved:
            print(f"\n  {Fore.YELLOW}UNRESOLVED VERIFICATION ISSUES "
                  f"({len(unresolved)}):{Style.RESET_ALL}")
            for i in unresolved:
                print(f"    {Fore.YELLOW}[{i.get('severity','?')}] "
                      f"{i.get('description','')}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.GREEN}No unresolved verification issues.{Style.RESET_ALL}")

        if not state.get("final_output"):
            print(f"\n  {Fore.RED}NOTE: No final review was produced this run.{Style.RESET_ALL}")

    def _display(self, state, elapsed):
        s = self.discovery.get_catalog_summary()
        m, sec = int(elapsed // 60), int(elapsed % 60)
        title = state.get("review_title", "Untitled Review")
        review = state.get("final_output")

        print(f"\n{Fore.WHITE}{'─'*70}")
        print(f"  Time: {m}m {sec}s | Papers in catalog: {s['total_papers']} "
              f"(full text: {s['with_full_text']}) | Studies in review: {len(state.get('study_analyses',[]))}")
        print(f"  Rounds: {state.get('discovery_round',0)} | "
              f"Distillations: {state.get('distillations_done',0)} | "
              f"Tangential engagements: {state.get('tangential_engagement_count',0)}")
        print(f"{'─'*70}{Style.RESET_ALL}")

        if review:
            self._display_review_colored(review, title)
            print(f"\n  {Fore.WHITE}{len(review.split())} words{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}No review generated.{Style.RESET_ALL}")


def print_banner():
    p = get_primary_llm_config().get("model_name", "?")
    print(f"""
{Fore.CYAN}\u2554{'═'*66}\u2557
\u2551  {Fore.WHITE}ACADEMIC LITERATURE REVIEW SYSTEM{Fore.CYAN}                                \u2551
\u2551  {Fore.YELLOW}APA 7th — Quote-Driven with Attribution Verification{Fore.CYAN}             \u2551
\u2551                                                                    \u2551
\u2551  {Fore.YELLOW}\u2022 Model: {p:<46}{Fore.CYAN}     \u2551
\u2551  {Fore.YELLOW}\u2022 Main mode requires full text; abstracts only in tangential{Fore.CYAN}     \u2551
\u2551  {Fore.YELLOW}\u2022 Tangential gated against unrelated cross-domain picks{Fore.CYAN}          \u2551
\u2551                                                                    \u2551
\u255A{'═'*66}\u255D{Style.RESET_ALL}
""")


def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

    # Per-run verbose logging is now handled by SessionLogger inside
    # pipeline.run(), which writes a complete Logs/<Review_Name>.txt transcript
    # for EACH review (named like the Papers/<Review_Name> folder). The old
    # global logs/academic_*.log handler has been removed in favour of these
    # per-run logs. We only set a sane root level here; with --debug, logging
    # records are also echoed to the console.
    logging.getLogger().setLevel(logging.INFO)
    if '--debug' in sys.argv:
        _console_h = logging.StreamHandler(sys.stdout)
        _console_h.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(_console_h)

    print_banner()

    try:
        cfg = get_primary_llm_config()
        resp = requests.get(f"{cfg['base_url']}/api/tags", timeout=5)
        if resp.status_code != 200:
            print(f"{Fore.RED}Cannot connect to Ollama.{Style.RESET_ALL}")
            sys.exit(1)
        models = [m['name'] for m in resp.json().get('models', [])]
        c = get_primary_llm_config()
        if not any(c['model_name'] in m for m in models):
            print(f"{Fore.YELLOW}Warning: model '{c['model_name']}' not in Ollama.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Ollama: {e}{Style.RESET_ALL}")
        sys.exit(1)

    pipeline = AcademicReviewPipeline()
    while True:
        try:
            print(f"\n{Fore.GREEN}Enter research question (or 'quit'):{Style.RESET_ALL}")
            query = input(f"{Fore.GREEN}> {Style.RESET_ALL}").strip()
            if not query:
                continue
            if query.lower() in ('quit', 'exit', 'q'):
                break
            results = pipeline.run(query)
            # Q&A MODE TOGGLE — when qa_mode_enabled is False in the config, the
            # program skips the post-review interactive Q&A and returns straight to
            # the research-question prompt (type 'quit' there to exit). When True
            # (default) it behaves exactly as before.
            qa_enabled = bool(pipeline.config.get("qa_mode_enabled", True))
            if results.get("review") and qa_enabled:
                print(f"\n{Fore.CYAN}Q&A mode ('new'/'quit'){Style.RESET_ALL}")
                while True:
                    fu = input(f"{Fore.GREEN}Q&A> {Style.RESET_ALL}").strip()
                    if not fu or fu.lower() in ('new', 'quit', 'exit', 'q'):
                        break
                    r = pipeline.agent_manager.run_primary(
                        f'Answer from this review: "{fu}"\n\n{results["review"][:10000]}',
                        task="qa_mode")
                    if r.success:
                        print(f"\n{Fore.WHITE}{r.response}{Style.RESET_ALL}")
                if fu and fu.lower() in ('quit', 'exit', 'q'):
                    break
            elif results.get("review") and not qa_enabled:
                print(f"\n{Fore.WHITE}Q&A mode is disabled in config — review complete. "
                      f"Enter another question or 'quit'.{Style.RESET_ALL}")
        except EOFError:
            break
        except KeyboardInterrupt:
            print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
            break
    pipeline.agent_manager.shutdown()


if __name__ == "__main__":
    main()
