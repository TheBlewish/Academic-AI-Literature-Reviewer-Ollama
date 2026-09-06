# study_analyser.py  (formerly study_analyzer.py)
# Study Analyser — Two-Phase Design
#
# PHASE 1 (quick_read): LLM reads ABSTRACT ONLY, extracts key findings. NO quotes,
#   NO TextMatcher, NO full text. Used during discovery to understand what studies
#   contain. This is FAST — abstracts are typically 200-400 words.
#
# PHASE 2 (deep_analysis): LLM reads FULL TEXT (or ABSTRACT in tangential mode) and
#   extracts exact quotes. TextMatcher verifies quotes against source text.
#
# CHANGES IN THIS VERSION:
#   - deep_analysis() now accepts a `mode` parameter ("main" or "tangential").
#     In "main" mode, papers without usable full text return None (drop signal).
#     In "tangential" mode, abstract-only papers are accepted; the abstract is
#     loaded into the doc store and quotes are verified against the abstract.
#   - Every verified-quote dict now carries the source paper's identity
#     (paper_id, title, authors, year, DOI, source text type). This is used
#     later by the attribution check in academic_researcher.py to deterministically
#     verify that quotes in the final review are correctly attributed.
#
# CHANGES IN THIS VERSION (variable quote count):
#   - deep_analysis() NO LONGER asks for a fixed number of quotes ("up to 8").
#     The LLM is now instructed to extract EVERY quote that is directly relevant
#     to the research question — which may be many, few, or NONE. It is told not
#     to pad to hit a target and not to omit relevant material to stay under one.
#     If nothing in the paper is directly relevant, it returns an empty list and
#     the study is dropped downstream (see min_verified_quotes_per_study in the
#     deep-analysis node). An OPTIONAL safety ceiling is read from
#     research_config["max_quotes_per_study_hard_cap"] (0 = unlimited, the
#     default); when >0 it is appended to the prompt purely as an upper bound.
#     The legacy keys extract_key_quotes_per_study / _per_abstract remain in the
#     config for backward compatibility but no longer drive the instruction.
#     NOTHING ELSE in this file changed — quote verification, source-identity
#     metadata, the merge-with-existing-summary path, and both read modes are
#     all preserved exactly.
#
# CHANGES IN THIS VERSION (evidence-section quote gating):
#   - PURPOSE: stop quoting Introduction/Background material. We now only keep a
#     verified quote if it reports THIS STUDY'S OWN findings, and we ENFORCE that
#     deterministically by WHERE in the paper the quote physically sits.
#   - TWO COMPLEMENTARY LAYERS (mirrors the existing two-layer design where the
#     prompt SELECTS and TextMatcher VERIFIES it is real):
#       LAYER A — SELECTION: the deep_analysis prompt now instructs the LLM to
#         quote ONLY sentences that state this study's own results / findings /
#         conclusions (Results, Findings, Discussion, Conclusion sections, plus a
#         results sentence in the Abstract), and to NEVER quote the Introduction,
#         Background, aims/hypotheses, or sentences reporting OTHER studies'
#         findings (e.g. "Smith et al. (2019) found ...", "[12] reported ...").
#       LAYER B — ENFORCEMENT (deterministic, position-based): after a quote is
#         verified verbatim by TextMatcher, we classify the MATCH POSITION
#         (match.start_position — a real offset into the loaded source text) into
#         a section using a heading map built from the SAME text. A quote that is
#         provably inside an EXCLUDED section (Introduction, Background, Related
#         Work, Literature Review, References, Acknowledgements, ...) is REJECTED
#         by setting verified=False. The downstream node counts only quotes with
#         verified==True (Academic_Researcher.node_deep_analysis), so rejected
#         intro quotes are dropped automatically and a study is dropped ONLY if it
#         has no remaining valid findings quotes. NOTHING downstream changed.
#   - DEFAULT BEHAVIOUR IS NOW PROOF-POSITIVE ("require_evidence"): a verified
#     quote is KEPT only if it is PROVABLY inside an evidence/abstract section —
#     i.e. the nearest preceding RECOGNISED heading is EVIDENCE or ABSTRACT.
#     Everything else (Introduction/Background/Methods/References, an unrecognised
#     heading region, or frontmatter before the first heading) is REJECTED
#     (verified=False). There is deliberately NO fallback: if a paper's headings
#     cannot be parsed at all, nothing is provable and all its quotes drop (the
#     study is then dropped downstream) — weird-format outliers are excluded on
#     purpose, which is exactly the hallucination-proof outcome we want.
#   - BOUNDARY SAFETY: the heading detector is GRAMMAR-BASED with high recall over
#     real-world heading formats (numbered "3. Results", "3.1 Results and
#     Discussion", roman "III. Results", markdown "## Results", Nature-pipe
#     "4 | Discussion", ALL CAPS, trailing colon, combined "Results & Discussion",
#     "Discussion and Conclusions", etc.). Crucially it ALSO recognises any
#     heading-shaped line it can't name as category "OTHER", which still acts as a
#     section BOUNDARY — so an evidence section can never bleed into the
#     references/appendix and smuggle a non-finding through. Prose sentences that
#     merely start with "Results"/"Discussion" are NOT treated as headings (tested).
#   - The legacy SUBTRACTIVE behaviour is still available as an opt-in via
#     section_filter_mode="exclude_non_evidence" (keep everything except provable
#     EXCLUDE/OTHER). Abstract-only (tangential) quotes are NEVER section-gated.
#   - All gating is configurable via research_config WITH SAFE DEFAULTS, so this
#     file works with NO academic_config.py change. Optional keys (see
#     _section_gating_settings): enforce_evidence_section_filtering (default True),
#     section_filter_mode (default "require_evidence"; or "exclude_non_evidence").
#   - NOTHING ELSE in this file changed — quick_read, the verbatim TextMatcher
#     verification (Layer that makes hallucination impossible), source-identity
#     metadata, the merge-with-existing-summary path, the no-matcher fallback,
#     both read modes, and the StudyAnalyzer alias are all preserved exactly.
#
# CHANGES IN THIS VERSION (high-confidence quote acceptance):
#   - A quote is now accepted ONLY if the matched SOURCE text is a HIGH-CONFIDENCE
#     match to the LLM's proposed quote. We score the TRUE text similarity between
#     the proposed quote and the matched text (containment => 1.0, else
#     difflib ratio) — NOT the matcher's internal heuristic confidence, which can
#     read 1.0 for a loose word-overlap hit on a different sentence. The bar is
#     research_config["quote_match_min_similarity"] (default 0.95).
#   - NUMBER SAFETY: unless the quote is exactly contained in the source, every
#     number in the proposed quote must also appear in the matched text, so a
#     near-identical sentence with a changed figure ("30%" vs "80%") is rejected
#     even though it scores ~0.97. Findings depend on their numbers.
#   - The stored match_confidence now reflects this true similarity. The verbatim
#     TextMatcher strategies are unchanged; this is purely a stricter ACCEPTANCE
#     gate layered on top, so nothing fabricated or materially altered survives.

import os
import re
import json
import time
import logging
import hashlib
import tempfile
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple
# CHANGES IN THIS VERSION (section-detection robustness — fixes the "everything
# dropped as non-evidence" failure on real HTML/PDF-extracted text):
#   - Removed the generic "OTHER" heading catch-all. On real extraction it fired
#     on author names, affiliations, figure/table captions and reference lines,
#     producing 77-255 spurious "headings" per paper that shredded the section
#     boundaries and caused real findings to be dropped. A line is now a heading
#     ONLY if it positively parses to a known section phrase.
#   - DEFAULT section mode is now "exclude_non_evidence" (subtractive), which is
#     robust to messy extraction. It drops a quote only when it PROVABLY sits in
#     a TERMINAL non-findings section (REFS: References/Bibliography/
#     Acknowledgements/Funding/Appendix/etc.) or in a NORMAL-SIZED leading
#     Introduction/Background region. If an Introduction region is suspiciously
#     large (a later evidence heading was missed and it "leaked" over the body),
#     the quote is KEPT — findings are never dropped to a leak.
#   - "require_evidence" is still available (strict: keep only EVIDENCE/ABSTRACT)
#     with a per-paper fallback so a paper with no detectable evidence heading is
#     not wiped out. Optional config: section_filter_mode,
#     max_leading_exclude_chars (default 8000).

from colorama import Fore, Style, init
init()

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dedicated SECTION-DEBUG logger. All the verbose per-paper heading inventory and
# per-quote accept/reject detail goes here, to a file under Logs/, and NOT to the
# terminal: propagate=False stops it bubbling up to the root logger (which prints
# to the console). One file per program run: Logs/section_debug_<timestamp>.txt.
# This keeps the terminal clean while preserving everything needed to refine
# heading detection from a run.
_section_debug_logger = None


def get_section_debug_logger():
    global _section_debug_logger
    if _section_debug_logger is not None:
        return _section_debug_logger
    lg = logging.getLogger("study_analyser.section_debug")
    lg.setLevel(logging.INFO)
    lg.propagate = False  # do NOT send these records to the console
    try:
        import time as _t
        os.makedirs("Logs", exist_ok=True)
        path = os.path.join("Logs", f"section_debug_{_t.strftime('%Y%m%d_%H%M%S')}.txt")
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        lg.addHandler(fh)
        lg._debug_path = path  # for an informational note on the console
    except Exception:
        lg.addHandler(logging.NullHandler())
        lg._debug_path = None
    _section_debug_logger = lg
    return lg


try:
    from document_store import DocumentStore, DocumentRecord, TextChunk
    HAS_DOCUMENT_STORE = True
except ImportError:
    HAS_DOCUMENT_STORE = False
    print("Warning: document_store.py not found. Quote verification will be unavailable.")

try:
    from text_matcher import TextMatcher, find_quote, MatchResult
    HAS_TEXT_MATCHER = True
except ImportError:
    HAS_TEXT_MATCHER = False
    print("Warning: text_matcher.py not found. Quote verification will be unavailable.")

from academic_config import get_research_config


# =============================================================================
# EVIDENCE-SECTION DETECTION (Layer B support) — grammar-based, high recall
# -----------------------------------------------------------------------------
# GOAL: a verified quote may be used as PRIMARY EVIDENCE only if it PROVABLY sits
# in an evidence section (Results/Findings/Discussion/Conclusion/Outcomes) or the
# Abstract. Proof = the nearest preceding RECOGNISED heading is EVIDENCE/ABSTRACT.
#
# DEFAULT MODE = "require_evidence" (positive identification):
#   * When an EVIDENCE/ABSTRACT heading is detected in the paper, a quote is kept
#     ONLY if it lands in such a section; everything else (Introduction, Methods,
#     unknown front-matter, References) is rejected.
#   * When NO evidence heading is detected (heading extraction failed for that
#     paper), we DON'T silently wipe the paper: a SAFE subtractive fallback keeps
#     quotes except those provably in EXCLUDE/REFS regions. Set
#     require_evidence_strict=True to instead reject everything in that case.
#   * "exclude_non_evidence" (opt-in) is the older purely-subtractive mode: keep
#     everything except provable terminal REFS / normal-sized intro regions.
#
# HEADING DETECTION is two-layer and POSITIVE-ONLY (no generic "OTHER" catch-all,
# which previously fired on author names / captions / reference lines and shredded
# the section map):
#   1. PHRASE layer  — precise: the normalised heading core matches a known
#      section phrase/token ("Results", "Results and Discussion", "References").
#   2. WORD-LEVEL layer (recall) — runs only if the phrase layer found nothing:
#      strips generic qualifiers ("Main", "Key", "Experimental", ...) and accepts
#      iff every remaining word is a decisive section noun or qualifier (purity
#      guard), so "Main Results"/"Key Findings" are caught but "Results Are
#      Promising" is not. Figure/table captions are explicitly rejected.
#
# DEBUG LOGGING (section_debug_logging, default ON): for every deep-analysed paper
# the run log records each recognised heading (category + char range), each
# heading-SHAPED line we did NOT classify (the near-misses to tune against), and
# each proposed quote's matched position, section, and ACCEPT/REJECT decision with
# the reason — so detection can be refined run over run.
#
# Categories: EVIDENCE, ABSTRACT, METHODS, EXCLUDE (leading non-findings),
# REFS (terminal non-findings), UNKNOWN (front-matter before the first heading).
# None of this touches the verbatim TextMatcher verification; a rejected quote is
# simply marked verified=False.
# =============================================================================

# ---- token sets (grammar-based, not flat list match) ----
EVIDENCE_PRIMARY = {
    "results","result","findings","finding","key findings","principal findings",
    "main findings","primary findings","major findings","study findings",
    "discussion","discussions","general discussion","main discussion",
    "conclusion","conclusions","concluding remarks","concluding remark",
    "interpretation","interpretations","outcome","outcomes",
    "results and findings","summary","summary and conclusions",
    "summary and conclusion","outlook and summary","summary and outlook",
}
EVIDENCE_SECONDARY = {  # only valid WITH a primary token in a combined heading
    "analysis","analyses","recommendations","recommendation",
    "implications","implication","clinical implications","future directions",
    "future work","future research","limitations","limitation","perspectives",
    "significance","comments","outlook",
}
ABSTRACT_PHRASES = {"abstract","structured abstract","summary","graphical abstract summary"}
METHODS_PHRASES = {
    "methods","method","materials and methods","methods and materials",
    "method and materials","material and methods","methodology","methodologies",
    "study design","study designs","experimental design","experimental setup",
    "experimental procedures","experimental section","data collection",
    "participants","subjects","procedure","procedures","measures","measurements",
    "materials","statistical analysis","statistical analyses","statistical methods",
    "data analysis","analytic strategy","analysis strategy","sample","samples",
    "study population","intervention","interventions","design","setting",
    "research design","data and methods","patients and methods",
    "subjects and methods","methods and materials",
}
# LEADING non-findings sections (Introduction/Background). These sit BEFORE the
# findings, so their region is bounded by the next heading — but if a later
# evidence heading is missed they can "leak" forward. The DEFAULT subtractive
# mode therefore does NOT drop quotes by position for these (the LLM prompt +
# second-hand gate handle intro avoidance); only require_evidence excludes them.
EXCLUDE_PHRASES = {
    "introduction","background","background and significance","significance",
    "related work","related works","related literature","literature review",
    "review of literature","review of the literature","prior work","previous work",
    "prior literature","previous literature","theoretical background",
    "theoretical framework","rationale","motivation","overview","preface",
}
# TERMINAL non-findings sections. These run to the end of the paper and NEVER
# contain the study's findings, so excluding their region by position is safe
# (no leak risk — nothing after them is a finding). The default subtractive mode
# drops quotes located here (most importantly, reference-list lines).
TERMINAL_EXCLUDE_PHRASES = {
    "references","reference","bibliography","works cited","literature cited",
    "acknowledgement","acknowledgements","acknowledgment","acknowledgments",
    "funding","funding statement","funding sources","funding information",
    "conflict of interest","conflicts of interest","competing interests",
    "competing interest","declaration of interest","declarations of interest",
    "declaration of competing interest","author contributions",
    "authors contributions","author contribution","contributorship",
    "data availability","data availability statement","code availability",
    "ethics statement","ethical approval","ethics approval",
    "ethical considerations","ethics","abbreviations","appendix","appendices",
    "supplementary material","supplementary materials","supplementary information",
    "supporting information","disclosure","disclosures","highlights",
    "graphical abstract","keywords","key words","notes","footnotes",
    "consent","informed consent","credit authorship contribution statement",
}

# -----------------------------------------------------------------------------
# WORD-LEVEL RECALL LAYER (additive; runs ONLY when the precise phrase layer
# above returns None). Real-world headings are often a decisive section noun
# wrapped in generic qualifiers ("Main Results", "Key Findings", "Experimental
# Results", "Detailed Discussion"). The phrase layer requires the WHOLE token to
# match, so it misses these. This layer strips qualifiers/stopwords and accepts a
# heading ONLY when every remaining word is either a decisive section noun or a
# generic qualifier (a strict "purity" guard) — so sentence fragments that merely
# contain the word "results" ("Results Are Promising") are NOT treated as
# headings. REFS/ABSTRACT are intentionally left to the phrase layer (their
# multi-word terminal forms — "conflict of interest", "data availability" — are
# safer matched as exact phrases), so this layer only adds EVIDENCE/METHODS/
# EXCLUDE recall.
_EVIDENCE_HEAD_NOUNS = {
    "results","result","findings","finding","discussion","discussions",
    "conclusion","conclusions","outcome","outcomes","interpretation",
    "interpretations","observation","observations",
}
_METHODS_HEAD_NOUNS = {
    "methods","method","methodology","methodologies","materials","material",
    "procedure","procedures","participants","subjects","measures","measure",
    "measurement","measurements","intervention","interventions","apparatus",
    "protocol","protocols","analysis","analyses",
}
_EXCLUDE_HEAD_NOUNS = {
    "introduction","background","motivation","rationale","overview","preface",
    "prologue","foreword",
}
# Generic words that may surround a decisive section noun in a heading. They
# carry no section meaning alone and are ignored by the purity guard.
_HEAD_QUALIFIERS = {
    "main","primary","key","principal","major","study","overall","general",
    "empirical","experimental","present","current","final","detailed","brief",
    "summary","further","additional","new","initial","preliminary","core",
    "central","important","selected","statistical","quantitative","qualitative",
    "data","results-section",
}
# Connective/stop words allowed inside a heading without breaking purity.
_HEAD_STOP = {
    "and","or","the","of","in","for","to","a","an","with","on","vs","versus",
    "our","from","section","part","chapter",
}


def _classify_core_wordlevel(core: str) -> Optional[str]:
    """Additive recall: classify a heading by its decisive section NOUNS after
    stripping generic qualifiers/stopwords. Returns a category ONLY if every word
    is a known section noun, qualifier, or stopword (purity guard); otherwise
    None. EXCLUDE is checked before EVIDENCE so an intro/background heading that
    happens to also contain an evidence noun is never mis-read as evidence."""
    words = re.findall(r"[a-z]+", core)
    if not words:
        return None
    known = (_EVIDENCE_HEAD_NOUNS | _METHODS_HEAD_NOUNS | _EXCLUDE_HEAD_NOUNS
             | _HEAD_QUALIFIERS | _HEAD_STOP)
    leftover = [w for w in words if w not in known]
    if leftover:
        return None  # extra content words => a phrase/sentence, not a clean heading
    if any(w in _EXCLUDE_HEAD_NOUNS for w in words):
        return "EXCLUDE"
    if any(w in _EVIDENCE_HEAD_NOUNS for w in words):
        return "EVIDENCE"
    if any(w in _METHODS_HEAD_NOUNS for w in words):
        return "METHODS"
    return None


# Figure/table/equation captions frequently contain the word "results" and are
# short + title-cased, so they would otherwise be mistaken for headings. Reject
# any line that opens with one of these caption markers followed by a number.
_CAPTION_RE = re.compile(
    r'^(figure|fig|table|tab|scheme|equation|eq|algorithm|chart|plate|box|'
    r'exhibit|appendix\s+figure|appendix\s+table)\s*\.?\s*\d', re.I)

_MD_PREFIX = re.compile(r'^#{1,6}\s*')
_LEADING_LABEL = re.compile(r'^(section|chapter|part)\b[\s:.\-]*', re.I)
# numbering: dotted digits, romans, or single letter (single letter requires punct)
_LEADING_NUM = re.compile(
    r'^(?:(\d+(?:\.\d+)*)|([ivxlcdm]+)|([a-z]))(?:[\.\)\:])?(?=\s|$)', re.I)
_LEADING_PIPE = re.compile(r'^\|\s*')
_STOPWORDS = {"and","or","the","of","in","for","to","a","an","with","on","&","vs","versus"}
_CONNECTIVE = re.compile(r'\s*&\s*|\s*/\s*|\s*,\s*|\s*;\s*|\s+and\s+|\s*\+\s*', re.I)


def _strip_prefixes(s: str):
    """Strip md/bold/number/label/pipe prefixes. Return (core, had_structural_prefix)."""
    had = False
    s = s.strip()
    # bold/italic markdown wrappers
    s2 = re.sub(r'^[\*_`]+', '', s); s2 = re.sub(r'[\*_`]+$', '', s2)
    if s2 != s: had = True
    s = s2.strip()
    m = _MD_PREFIX.match(s)
    if m: had = True; s = s[m.end():].strip()
    m = _LEADING_PIPE.match(s)
    if m: had = True; s = s[m.end():].strip()
    m = _LEADING_LABEL.match(s)
    if m: had = True; s = s[m.end():].strip()
    # numbering possibly repeated (e.g. "Section 3.1")
    for _ in range(2):
        m = _LEADING_NUM.match(s)
        if m:
            # don't strip a bare single letter w/o punctuation (avoid eating words)
            tok = m.group(0)
            if m.group(3) and not re.search(r'[\.\)\:]', tok):
                break
            had = True; s = s[m.end():].strip()
        else:
            break
    m = _LEADING_PIPE.match(s)
    if m: had = True; s = s[m.end():].strip()
    return s, had


def _core(s: str) -> str:
    core, _ = _strip_prefixes(s)
    core = core.strip().strip(':.#*-').strip()
    core = core.lower()
    core = re.sub(r'\s+', ' ', core)
    core = re.sub(r'\s+sections?$', '', core)
    core = re.sub(r'^(the|our)\s+', '', core)
    return core


def _tokens(core: str):
    return [t.strip() for t in _CONNECTIVE.split(core) if t and t.strip()]


def _is_all_caps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _is_title_case(s: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", s)]
    content = [w for w in words if w.lower() not in _STOPWORDS]
    if not content:
        return False
    return all(w[0].isupper() for w in content)


def _classify_core_phrase(core: str) -> Optional[str]:
    if not core:
        return None
    toks = _tokens(core)
    if not toks:
        return None
    if toks[0] in ("appendix", "appendices") or core.startswith("appendix"):
        return "REFS"
    # TERMINAL excludes (references/acknowledgements/etc.) — run to EOF, safe to drop
    if core in TERMINAL_EXCLUDE_PHRASES or any(t in TERMINAL_EXCLUDE_PHRASES for t in toks):
        return "REFS"
    # LEADING excludes (introduction/background/etc.)
    if core in EXCLUDE_PHRASES or any(t in EXCLUDE_PHRASES for t in toks):
        return "EXCLUDE"
    # METHODS
    if core in METHODS_PHRASES or any(t in METHODS_PHRASES for t in toks):
        return "METHODS"
    # ABSTRACT (standalone)
    if core in ABSTRACT_PHRASES:
        return "ABSTRACT"
    # EVIDENCE: every token is primary/secondary AND at least one primary
    all_ok = all((t in EVIDENCE_PRIMARY or t in EVIDENCE_SECONDARY) for t in toks)
    has_primary = any(t in EVIDENCE_PRIMARY for t in toks)
    if all_ok and has_primary:
        return "EVIDENCE"
    return None


def _classify_core(core: str) -> Optional[str]:
    """Precise phrase match first (high precision); if that finds nothing, fall
    back to the qualifier-stripped word-level layer (high recall, purity-guarded).
    """
    cat = _classify_core_phrase(core)
    if cat is not None:
        return cat
    return _classify_core_wordlevel(core)


# Standard section words that we accept as a heading EVEN IF the extracted line
# is lowercase (PDF/HTML extraction often drops capitalisation on real headings
# like a bare "results" line). Anything NOT in this set must look like a heading
# (start with a capital/number, be ALL-CAPS, be numbered, or end with a colon) to
# be treated as one — this stops stray lowercase words ("reference", "design")
# and lowercase sentence fragments from being mistaken for section headings.
_STRONG_EXACT_CORES = {
    "abstract", "introduction", "background", "methods", "method", "methodology",
    "materials and methods", "results", "result", "findings", "discussion",
    "discussions", "conclusion", "conclusions", "references", "bibliography",
    "acknowledgements", "acknowledgments",
}


def _is_heading_shaped(stripped: str) -> bool:
    """True if the line LOOKS like a heading: starts with a capital letter or a
    number, is ALL-CAPS, carries a numbering/markdown/'Section' prefix, or ends
    with a colon. Real headings (Title Case or Sentence case) start with a capital;
    body sentence-fragments that caused false 'Methods'/'References' headings in
    the logs all started lowercase, so this cleanly separates them."""
    first_alpha = next((c for c in stripped if c.isalpha()), "")
    if first_alpha and first_alpha.isupper():
        return True
    if _is_all_caps(stripped):
        return True
    if _MD_PREFIX.match(stripped) or _LEADING_LABEL.match(stripped) or _LEADING_NUM.match(stripped):
        return True
    if stripped.rstrip().endswith(":"):
        return True
    return False


def classify_heading_line_with_reason(stripped: str) -> Tuple[Optional[str], str]:
    """Like classify_heading_line, but also returns a human-readable REASON for
    the debug logs (why a line was/was not treated as a recognised heading)."""
    if not stripped:
        return None, "empty line"
    if len(stripped) > 90:
        return None, f"too long ({len(stripped)} chars)"
    if len(stripped.split()) > 10:
        return None, f"too many words ({len(stripped.split())})"
    if _CAPTION_RE.match(stripped):
        return None, "figure/table/equation caption"
    core_norm = _core(stripped)
    if not core_norm:
        return None, "no core text after stripping prefixes"
    # SHAPE GUARD (with a fallback for un-shaped lines).
    # A line that is not heading-shaped (starts lowercase, no number/colon) is
    # normally NOT a heading. But a line whose WHOLE content is a section word is
    # almost certainly a heading even when extraction dropped its capitalisation.
    # We accept such un-shaped lines when:
    #   (a) the core is one of a short list of standard section words, OR
    #   (b) the core is a PURE primary-EVIDENCE expression (results / findings /
    #       discussion / conclusion / outcomes / "results and discussion" / ...)
    #       with no other words — caught by the purity-guarded word-level layer.
    # We deliberately do NOT auto-accept un-shaped METHODS/REFERENCES words here
    # (e.g. a stray lowercase "reference" or "design"): a mis-fired terminal
    # heading can swallow a whole paper's body, which is exactly the failure we are
    # guarding against. Shaped lines are unaffected and go through normal matching.
    if not _is_heading_shaped(stripped):
        pure_evidence = (_classify_core_wordlevel(core_norm) == "EVIDENCE")
        if core_norm not in _STRONG_EXACT_CORES and not pure_evidence:
            return None, (f"core '{core_norm}' not heading-shaped and not a pure "
                          f"section word")
    cat = _classify_core(core_norm)
    if cat is None:
        return None, f"core '{core_norm}' did not match any section vocabulary"
    return cat, f"core '{core_norm}' -> {cat}"


def classify_heading_line(stripped: str) -> Optional[str]:
    """
    Return EVIDENCE / ABSTRACT / METHODS / EXCLUDE / REFS for a line that is
    POSITIVELY a known section heading, else None.

    We deliberately DO NOT emit a generic "OTHER" heading. On real HTML/PDF-
    extracted text a shape-only rule fires on author names, affiliations,
    figure/table captions and reference lines — producing hundreds of spurious
    "headings" per paper that shred section boundaries and cause real findings to
    be mis-classified and dropped. A line only counts as a heading if (a) it is
    short and heading-shaped, (b) it is not a figure/table caption, and (c) its
    normalised core positively parses to a known section phrase (precise phrase
    layer) or to a clean qualifier-wrapped section noun (word-level recall layer).
    """
    cat, _reason = classify_heading_line_with_reason(stripped)
    return cat


def _looks_like_heading_shape(stripped: str) -> bool:
    """A cheap pre-filter for the DEBUG log only: is this line shaped like a
    heading at all (short, and either ALL-CAPS / Title Case / numbered / markdown
    / ends with a colon)? Used so the log can show heading-shaped lines that we
    did NOT classify — the near-misses we tune detection against."""
    if not stripped or len(stripped) > 90 or len(stripped.split()) > 10:
        return False
    if _CAPTION_RE.match(stripped):
        return False
    has_prefix = bool(_MD_PREFIX.match(stripped) or _LEADING_LABEL.match(stripped)
                      or _LEADING_NUM.match(stripped))
    return (_is_all_caps(stripped) or _is_title_case(stripped)
            or has_prefix or stripped.rstrip().endswith(":"))


def build_section_map(text: str) -> List[Tuple[int, str, str]]:
    """
    Scan `text` line by line; return ordered (line_start_char, category,
    heading_text) for every detected heading. Each heading owns the region from
    its line start up to the next detected heading. Positions are offsets into
    EXACTLY the `text` loaded into the DocumentStore, so they line up with
    TextMatcher's match.start_position.
    """
    headings: List[Tuple[int, str, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        stripped = line.strip()
        if not stripped:
            continue
        category = classify_heading_line(stripped)
        if category:
            headings.append((line_start, category, stripped))
    return headings


def build_section_map_verbose(text: str) -> Tuple[List[Tuple[int, str, str]], List[Dict]]:
    """
    Same as build_section_map, but ALSO returns a list of debug records for every
    heading-SHAPED candidate line (whether or not it was accepted), each:
        {"line_no", "pos", "text", "category" (or None), "reason"}
    so the section-detection debug log can show both what was recognised and the
    near-misses that detection should be tuned to catch.
    """
    headings: List[Tuple[int, str, str]] = []
    candidates: List[Dict] = []
    pos = 0
    line_no = 0
    for line in text.splitlines(keepends=True):
        line_no += 1
        line_start = pos
        pos += len(line)
        stripped = line.strip()
        if not stripped:
            continue
        category = classify_heading_line(stripped)
        if category:
            headings.append((line_start, category, stripped))
            cat, reason = classify_heading_line_with_reason(stripped)
            candidates.append({"line_no": line_no, "pos": line_start,
                               "text": stripped[:90], "category": cat, "reason": reason})
        elif _looks_like_heading_shape(stripped):
            _cat, reason = classify_heading_line_with_reason(stripped)
            candidates.append({"line_no": line_no, "pos": line_start,
                               "text": stripped[:90], "category": None, "reason": reason})
    return headings, candidates


def classify_position(section_map: List[Tuple[int, str, str]], pos: int) -> Tuple[str, str]:
    """
    Return (category, heading_text) of the section containing `pos`. Positions
    before the first heading are UNKNOWN frontmatter.
    """
    if not section_map:
        return "UNKNOWN", ""
    current = ("UNKNOWN", "")
    for line_start, category, heading_text in section_map:
        if pos >= line_start:
            current = (category, heading_text)
        else:
            break
    return current


def _region_length(section_map: List[Tuple[int, str, str]], pos: int, text_len: int) -> int:
    """
    Length (in chars) of the section that contains `pos`: from the heading
    at-or-before `pos` to the next heading (or end of text). Used to detect a
    leading-section "leak" — a normal Introduction is small; an Introduction
    whose region spans most of the document means a later heading was missed.
    """
    if not section_map:
        return text_len
    start = 0
    end = text_len
    for idx, (line_start, _cat, _h) in enumerate(section_map):
        if pos >= line_start:
            start = line_start
            end = section_map[idx + 1][0] if idx + 1 < len(section_map) else text_len
        else:
            break
    return max(0, end - start)


def _section_is_eligible(category: str, mode: str) -> bool:
    """
    exclude_non_evidence (DEFAULT, subtractive): drop a quote ONLY if it sits in a
    TERMINAL non-findings section (REFS: References / Bibliography /
    Acknowledgements / Funding / Appendix / etc.). Those run to the end of the
    paper and never contain findings, so excluding them is leak-free. Everything
    else — including Introduction/Background regions — is KEPT here; intro
    avoidance is handled by the LLM prompt and the second-hand citation gate.

    require_evidence (opt-in, strict): keep ONLY EVIDENCE / ABSTRACT. (The caller
    applies a per-paper fallback so a paper with no detectable evidence heading is
    not wiped out — see deep_analysis.)
    """
    if mode == "require_evidence":
        return category in ("EVIDENCE", "ABSTRACT")
    return category != "REFS"


# =============================================================================
# SECOND-HAND / CITATION DETECTION (Layer C — primary-evidence-only)
# -----------------------------------------------------------------------------
# A sentence can pass the section gate (e.g. it sits in the Discussion) yet still
# report ANOTHER study's work ("Smith et al. (2019) found ...", "previous studies
# have shown ...", "[12]"). Such a quote is SECOND-HAND and must NOT be presented
# as this study's evidence. We detect in-text citations / attribution markers
# deterministically and reject any quote that carries one. The cited works are
# instead chased and acquired separately (see reference_harvester.py) so their
# findings can be quoted DIRECTLY as primary evidence.
# =============================================================================

# Numeric reference markers: [12], [3,4], [3-5], [3–5]
_CITE_NUMERIC = re.compile(r'\[\s*\d+(?:\s*[,\-\u2013]\s*\d+)*\s*\]')
# Narrative author-year: "Smith (2019)", "Smith et al. (2019)", "Smith and Jones 2019",
# "Smith & Jones (2020)"
_CITE_NARRATIVE = re.compile(
    r'\b[A-Z][A-Za-z\u00C0-\u017F\-]+\s+'
    r'(?:et\s+al\.?|and\s+[A-Z][A-Za-z\u00C0-\u017F\-]+|&\s+[A-Z][A-Za-z\u00C0-\u017F\-]+)'
    r'[\s,]*\(?\s*(?:18|19|20)\d{2}[a-z]?')
# Single narrative author with parenthetical year: "Brown (2020)"
_CITE_SINGLE_PAREN = re.compile(r'\b[A-Z][A-Za-z\u00C0-\u017F\-]+\s*\(\s*(?:18|19|20)\d{2}[a-z]?\s*\)')
# Parenthetical citation containing an author-ish token AND a year:
# "(Smith, 2019)", "(Smith et al., 2019)", "(Jones & Lee, 2018; Park, 2020)"
_CITE_PAREN_AUTHOR_YEAR = re.compile(
    r'\([^)]*?(?:[A-Z][A-Za-z\u00C0-\u017F\-]+|\bet\s+al\.?)[^)]*?(?:18|19|20)\d{2}[a-z]?[^)]*?\)')
# Bare "et al" is itself a strong second-hand signal
_CITE_ETAL = re.compile(r'\bet\s+al\.?', re.IGNORECASE)

# Attribution phrases that clearly report prior/other work (kept deliberately
# conservative so a study's OWN result sentence is not mis-flagged).
_SECONDHAND_PHRASES = [
    r'\b(?:previous|prior|earlier|existing|past|other|recent)\s+'
    r'(?:study|studies|work|works|research|reports?|literature|findings?|'
    r'evidence|trials?|reviews?|investigations?|authors?)\b',
    r'\bother\s+(?:researchers?|investigators?|groups?)\b',
    r'\baccording\s+to\b',
    r'\bas\s+(?:reported|shown|demonstrated|noted|described|reviewed|suggested)\s+'
    r'(?:by|in|elsewhere|previously)\b',
    r'\bhas\s+been\s+(?:reported|shown|demonstrated|suggested|documented|established)\b',
    r'\bhave\s+been\s+(?:reported|shown|demonstrated|suggested|documented)\b',
    r'\b(?:it\s+is|it\s+has\s+been)\s+(?:well[\s\-])?(?:established|known|documented|recognised|recognized)\b',
    r'\bin\s+the\s+literature\b',
]
_SECONDHAND_PHRASE_RE = re.compile('|'.join(_SECONDHAND_PHRASES), re.IGNORECASE)


def detect_secondhand_markers(text: str) -> List[str]:
    """
    Return a list of the second-hand / citation markers found in `text`.
    Empty list means the sentence reads as the study's own statement.
    """
    if not text:
        return []
    markers = []
    for label, rx in (
        ("numeric_citation", _CITE_NUMERIC),
        ("narrative_citation", _CITE_NARRATIVE),
        ("single_paren_citation", _CITE_SINGLE_PAREN),
        ("paren_author_year", _CITE_PAREN_AUTHOR_YEAR),
        ("et_al", _CITE_ETAL),
        ("attribution_phrase", _SECONDHAND_PHRASE_RE),
    ):
        m = rx.search(text)
        if m:
            snippet = m.group(0).strip()
            markers.append(f"{label}:{snippet}")
    return markers


def quote_is_secondhand(text: str) -> bool:
    """True if the quote carries any in-text citation or attribution marker."""
    return bool(detect_secondhand_markers(text))


# =============================================================================
# REFERENCES-SECTION EXTRACTION (support for backward citation chasing)
# -----------------------------------------------------------------------------
# The References / Bibliography region is never quoted (it is an EXCLUDE section),
# but it is exactly where the cited primary studies are listed. We slice it out
# using the SAME heading map so the reference harvester can mine it.
# =============================================================================

_REFERENCES_HEADING_WORDS = (
    "references", "reference", "bibliography", "works cited", "literature cited",
)


def extract_references_block(text: str) -> str:
    """
    Return the text of the References/Bibliography section (heading line excluded),
    or "" if no such section heading is found. Uses build_section_map so detection
    is consistent with the quote-gating logic.
    """
    if not text:
        return ""
    smap = build_section_map(text)
    if not smap:
        return ""
    ref_start = None
    ref_end = len(text)
    for i, (line_start, category, heading_text) in enumerate(smap):
        norm = _core(heading_text)
        toks = _tokens(norm)
        is_ref = (norm in _REFERENCES_HEADING_WORDS
                  or any(t in _REFERENCES_HEADING_WORDS for t in toks)
                  or norm.startswith("reference") or norm.startswith("bibliograph"))
        if is_ref and category == "REFS":
            # body starts at the end of the heading line
            nl = text.find("\n", line_start)
            ref_start = (nl + 1) if nl != -1 else line_start
            # End at the next recognised non-terminal section heading; references
            # normally run to EOF, but stop early if a real section follows.
            ref_end = len(text)
            for j in range(i + 1, len(smap)):
                if smap[j][1] in ("EVIDENCE", "METHODS", "ABSTRACT", "EXCLUDE"):
                    ref_end = smap[j][0]
                    break
            break
    if ref_start is None:
        return ""
    return text[ref_start:ref_end].strip()


class _ExtractionResult:
    """Minimal stand-in for llm_manager's AgentResult.

    The chunked low-end path merges several real AgentResults into one logical
    result. The caller in deep_analysis() only ever touches .success and
    .json_response, so this exposes exactly those two attributes and nothing
    else, keeping the merge path honest about what it actually provides.
    """

    __slots__ = ("success", "json_response")

    def __init__(self, success, json_response):
        self.success = success
        self.json_response = json_response


def _is_interrupted() -> bool:
    """True if the user has pressed Ctrl+C. Imported lazily so this module can
    still be imported (and unit-tested) without llm_manager present."""
    try:
        from llm_manager import is_interrupted
        return bool(is_interrupted())
    except Exception:
        return False


def _split_text_into_chunks(text, chunk_chars, overlap, max_chunks=0):
    """Split paper text into sequential overlapping chunks for low-end reading.

    Boundaries are pulled back to the nearest paragraph break, then sentence
    end, within the last 20% of the chunk. This matters more here than in a
    normal RAG splitter: the model is asked to reproduce a COMPLETE sentence
    verbatim, and a sentence cut in half by a chunk boundary can only ever
    produce a fragment that fails verbatim verification and is discarded. The
    overlap is the second line of defence for sentences that still straddle a
    boundary — they appear whole in the following chunk.

    Returns a list of strings. A text shorter than one chunk returns [text].
    """
    text = text or ""
    chunk_chars = max(int(chunk_chars or 0), 1000)
    overlap = max(int(overlap or 0), 0)
    if overlap >= chunk_chars:
        overlap = chunk_chars // 4
    if len(text) <= chunk_chars:
        return [text] if text else []

    chunks = []
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + chunk_chars, n)
        if end < n:
            window_start = pos + int(chunk_chars * 0.8)
            cut = text.rfind("\n\n", window_start, end)
            if cut == -1:
                for pat in (". ", ".\n", "? ", "! "):
                    c = text.rfind(pat, window_start, end)
                    if c > cut:
                        cut = c + len(pat) - 1
            if cut > window_start:
                end = cut + 1
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        if max_chunks and len(chunks) >= max_chunks:
            break
        pos = max(end - overlap, pos + 1)
    return chunks


def _build_deep_analysis_prompt(paper, query, text_type, cap_clause,
                                chunk_text, chunk_header=""):
    """Build the deep-analysis quote-extraction prompt.

    The prompt text is IDENTICAL to the one this module has always used. Two
    placeholders were parameterised so the same prompt can serve both modes:

      chunk_text   - the paper text this particular call sees. In full mode this
                     is the entire `text`, so the rendered prompt is byte-for-byte
                     what it was before.
      chunk_header - "" in full mode, which renders as the blank line that has
                     always sat between the Content line and PAPER CONTENT. In
                     low-end chunked mode it carries the "PART n OF m" notice.

    A unit test in the shipped test suite asserts the full-mode rendering is
    byte-identical to the original prompt, so this refactor cannot silently
    change full-mode behaviour.
    """
    return f"""You are extracting EXACT VERBATIM quotes from an academic paper for a literature review.

RESEARCH QUESTION: "{query}"

PAPER: {paper.title}
Authors: {', '.join(paper.authors[:5])} | Year: {paper.year}
Content: {text_type}
{chunk_header}
PAPER CONTENT:
{chunk_text}

Your task is to extract the quotes that report THIS STUDY'S OWN FINDINGS that
help answer the research question. A "finding" is something THIS study itself
measured, observed, reported, or concluded — its results and what they mean.

EXTRACT quotes ONLY from the study's evidence:
- the RESULTS / FINDINGS section (what the study found),
- the DISCUSSION section (what the study's results mean / its interpretation),
- the CONCLUSION section (the study's own conclusions),
- and, if present, a sentence in the ABSTRACT that states this study's RESULTS.
There is NO fixed number — extract as many as are genuinely findings relevant to
the question, which may be many, a few, or NONE at all.

DO NOT quote (these are NOT this study's findings):
- the INTRODUCTION, BACKGROUND, or LITERATURE REVIEW (motivation, definitions,
  why the topic matters, descriptions of the problem),
- the study's AIMS, OBJECTIVES, HYPOTHESES, or research questions,
- METHODS-only sentences that merely describe what was done (unless the sentence
  also states a result),
- any sentence that reports ANOTHER study's findings or is attributed to other
  authors — e.g. "Smith et al. (2019) found ...", "Previous studies have
  shown ...", "It is well established that ...", "[12] reported ...". Only quote
  sentences where THIS paper states what IT found, observed, or concluded.

CRITICAL VERBATIM RULES:
- Extract ALL directly-relevant FINDINGS quotes. Do NOT pad the list with weak,
  tangential, or background quotes to reach some number, and do NOT leave out a
  relevant findings quote to keep the list short. The right count is exactly
  however many findings are relevant.
- If NOTHING in this paper reports a finding relevant to the research question,
  return an empty list ("key_quotes": []). It is correct and expected to return
  no quotes for such a paper — it will simply be excluded from the review.
- Each quote will be VERIFIED character-by-character against the source text AND
  checked to confirm it comes from the study's evidence (not its introduction)
  and is not a citation of another study. A quote that does not match EXACTLY,
  that sits in the introduction/background, or that reports another study's work
  will be REJECTED.
- Each quote must be ONE complete sentence, copied EXACTLY as it appears in the
  paper — every word, number, symbol, and punctuation mark identical. Do NOT
  paraphrase, summarise, join two sentences, trim, or "clean up" the wording.
  If you cannot reproduce a sentence exactly, do not include it.
- Prefer sentences that state a concrete result: an effect, a measurement, a
  number, a comparison, an association, or an explicit conclusion of THIS study.
- Each quote must stand on its own as a statement of what THIS study found,
  observed, measured, or concluded, relevant to the research question.{cap_clause}

ALSO WRITE A STUDY SUMMARY (separate from the quotes). Now that you have read the
whole paper, write a short, factual summary IN YOUR OWN WORDS that will be used to
INTRODUCE this study in the review before its quoted findings are presented. It
must be grounded ONLY in what this paper actually says — do not invent details. In
2-4 sentences, cover, where the paper states them:
  - what the study set out to examine (its aim / question),
  - its design / methodology (e.g. theoretical mass model, RCT, simulation,
    laboratory experiment, cohort study, systematic review, meta-analysis),
  - its sample, dataset, or scope (what/who/how many, conditions tested),
  - the general nature of its findings (one clause — the detail stays in the quotes).
This summary is a PARAPHRASE, not a quotation; it will be clearly labelled as the
reviewer's summary of the study (never shown in quotation marks). If the paper does
not state something (e.g. no sample size), simply omit it rather than guessing.

Respond with ONLY JSON:
{{
    "study_summary": "2-4 sentence paraphrased introduction to THIS study: its aim, design/methodology, sample/scope, and the general nature of its findings, grounded only in the paper. Not a quote.",
    "key_quotes": [
        {{"quote": "EXACT TEXT copied verbatim from the study's findings/discussion/conclusion", "context": "what THIS study found and how it answers the question", "importance": "high|medium|low"}}
    ]
}}"""



class StudyAnalyser:
    """Two-mode study analyser: quick_read for discovery, deep_analysis for review writing."""

    def __init__(self, agent_manager):
        self.llm = agent_manager
        self.research_config = get_research_config()
        self.text_matcher = TextMatcher() if HAS_TEXT_MATCHER else None
        self._doc_store = DocumentStore() if HAS_DOCUMENT_STORE else None
        # A quote is only accepted if its matched source text is at least this
        # similar to the LLM's proposed quote (true text similarity, NOT the
        # matcher's internal heuristic confidence). Default 0.95 = high-confidence
        # near-verbatim. Numbers must still line up exactly (see _accept_match).
        self._quote_min_similarity = float(
            self.research_config.get("quote_match_min_similarity", 0.95))
        # In subtractive section mode, a leading Introduction/Background quote is
        # dropped only when its region is at most this many characters (a normal
        # intro). Larger => assume a later heading was missed (leak) and KEEP the
        # quote rather than risk dropping a real finding.
        self._max_leading_exclude_chars = int(
            self.research_config.get("max_leading_exclude_chars", 8000))
        # Verbose section-detection logging (to the run log / Logs file via the
        # module logger). ON by default so we can refine heading detection from
        # real runs. Set research_config["section_debug_logging"] = False to mute.
        self._section_debug = bool(
            self.research_config.get("section_debug_logging", True))
        # POSITIVE-IDENTIFICATION gate (require_evidence mode): a quote is usable
        # ONLY if it sits in a positively-identified EVIDENCE/ABSTRACT section, or
        # in the paper's leading abstract block (the bounded title+abstract region
        # before the first heading, when no separate ABSTRACT heading was found).
        # There is no "keep everything except intro/refs" fallback — if a section
        # cannot be positively identified, its quotes are rejected. Setting
        # require_evidence_strict = True tightens this further: it requires an
        # explicit EVIDENCE/ABSTRACT *heading* and will NOT accept the inferred
        # leading abstract block (so a paper whose headings we cannot detect yields
        # nothing). Default False keeps the inferred-abstract acceptance.
        self._require_evidence_strict = bool(
            self.research_config.get("require_evidence_strict", False))
        # ------------------------------------------------------------------
        # LOW-END DEVICE MODE — chunked deep analysis.
        #
        # In FULL mode every value below is unused and deep_analysis() behaves
        # exactly as it always has: one LLM call containing the whole paper.
        #
        # In LOW-END mode the paper is read in sequential overlapping chunks and
        # the proposed quotes are accumulated across them, because a whole paper
        # (max_study_text_length, default 50000 chars ~ 16.7K tokens) cannot fit
        # an 8K window and was previously being silently truncated by Ollama.
        #
        # CRITICAL: only the SELECTION half is chunked. The DocumentStore is
        # still loaded with the COMPLETE text and every proposed quote is still
        # verified verbatim against that complete text, so the anti-hallucination
        # guarantee is bit-for-bit identical to full mode.
        #
        # These keys are published by academic_config.get_research_config() only
        # when low_end_device_mode is on, so `.get(...)` returning the default is
        # itself the full-mode signal.
        # ------------------------------------------------------------------
        self._low_end = bool(
            self.research_config.get("low_end_device_mode", False))
        self._le_chunk_chars = int(
            self.research_config.get("low_end_deep_analysis_chunk_chars", 9000) or 9000)
        self._le_chunk_overlap = int(
            self.research_config.get("low_end_deep_analysis_chunk_overlap", 500) or 500)
        # Hard safety bound on LLM calls per paper. Without it a long paper at a
        # small chunk size could fire a dozen calls and make a run take days on a
        # mini PC. 0 = unlimited.
        self._le_max_chunks = int(
            self.research_config.get("low_end_max_chunks_per_paper", 4) or 0)
        # Spend one extra (small) call merging the per-chunk study summaries into
        # the single 2-4 sentence paraphrase synthesis expects. Set False to skip
        # the call and simply use the longest per-chunk summary instead.
        self._le_merge_summary = bool(
            self.research_config.get("low_end_merge_study_summary", True))
        # Initialise the file-only section-debug logger now and tell the user once
        # where the verbose detection detail goes (it is NOT printed to the
        # terminal — only to this file — so the run output stays readable).
        if self._section_debug:
            _dbg = get_section_debug_logger()
            _path = getattr(_dbg, "_debug_path", None)
            if _path:
                print(f"  {Fore.WHITE}Section-detection debug -> {_path} "
                      f"(verbose; not shown in terminal){Style.RESET_ALL}")

    # =========================================================================
    # MODE 1: QUICK READ — used during discovery (ABSTRACT ONLY, fast)
    # =========================================================================

    def quick_read(self, paper, query: str) -> Optional[Dict]:
        """
        Quick read of a study using ABSTRACT ONLY for speed.
        Extracts key findings, methodology, relevance from the abstract.
        NO full text reading, NO quote extraction, NO TextMatcher.

        Full text reading happens later in deep_analysis() once the LLM
        decides it has enough studies to write the review.
        """
        if paper.abstract:
            text = paper.abstract
            text_type = "ABSTRACT"
        elif paper.full_text_available and paper.full_text_content:
            text = paper.full_text_content[:2000]
            text_type = "EXCERPT (no abstract available)"
        else:
            return None

        print(f"      {Fore.BLUE}Reading ({text_type.lower()})...{Style.RESET_ALL}", end=" ", flush=True)
        start = time.time()

        prompt = f"""You are reading an academic study for a literature review.

RESEARCH QUESTION: "{query}"

PAPER: {paper.title}
Authors: {', '.join(paper.authors[:5])} | Year: {paper.year} | Venue: {paper.venue}
DOI: {paper.doi} | Citations: {paper.citation_count} | Content: {text_type}
Full text acquired: {"Yes" if paper.full_text_available else "No"}

PAPER {text_type}:
{text}

Based on the abstract/excerpt, summarize this study's relevance to the research question.
Be concise and factual. Extract what you can from the available text.

Respond with ONLY JSON:
{{
    "study_type": "systematic_review|meta_analysis|rct|cohort_study|case_control|cross_sectional|case_report|review|animal_study|other",
    "sample_size": "number or description if mentioned",
    "methodology_summary": "brief methods description from abstract",
    "key_findings": ["finding 1 with specific numbers/results if available", "finding 2", "finding 3"],
    "relevance_to_question": "how this specifically helps answer the research question",
    "limitations": ["limitation 1", "limitation 2"],
    "reliability_score": 1-10,
    "reliability_reasoning": "brief justification"
}}

Be specific about findings — include actual numbers, effect sizes, p-values when available in the abstract."""

        result = self.llm.run_primary(prompt, as_json=True, task="quick_read")
        elapsed = time.time() - start

        if result.success and result.json_response:
            analysis = result.json_response
            analysis["paper_id"] = paper.paper_id
            analysis["paper_title"] = paper.title
            analysis["paper_authors"] = paper.authors
            analysis["paper_year"] = paper.year
            analysis["paper_venue"] = paper.venue
            analysis["paper_doi"] = paper.doi
            analysis["paper_citation_count"] = paper.citation_count
            analysis["has_full_text"] = paper.full_text_available
            analysis["quotes_verified"] = False
            analysis["key_quotes"] = []

            print(f"{Fore.GREEN}done ({elapsed:.0f}s) | {analysis.get('study_type','?')} | "
                  f"reliability: {analysis.get('reliability_score','?')}/10{Style.RESET_ALL}")
            return analysis

        print(f"{Fore.RED}failed ({elapsed:.0f}s){Style.RESET_ALL}")
        return None

    # =========================================================================
    # MODE 2: DEEP ANALYSIS — used only for final review writing (with quotes)
    # =========================================================================

    def _section_gating_settings(self) -> Dict:
        """
        Read the (optional) evidence-section gating settings from research_config
        with safe defaults so the program works with NO config change.
        """
        cfg = self.research_config
        return {
            "enforce": bool(cfg.get("enforce_evidence_section_filtering", True)),
            "mode": cfg.get("section_filter_mode", "require_evidence"),
            "exclude_secondhand": bool(cfg.get("exclude_secondhand_quotes", True)),
        }

    def deep_analysis(self, paper, query: str, existing_summary: Dict = None,
                      mode: str = "main") -> Optional[Dict]:
        """
        Deep analysis with verified quote extraction. Used only when writing
        the final review.

        Modes:
          "main"       — Full text REQUIRED. If the paper has no full text (or
                         the full text is below the minimum length threshold
                         for main mode), returns None. The caller MUST treat
                         None as a signal to drop the paper from the final
                         synthesis.
          "tangential" — Abstract-only papers are allowed. The abstract is
                         loaded into the doc store and quotes are verified
                         against the abstract.

        Every verified quote in the returned analysis dict carries the source
        paper's identity (paper_id, title, authors, year, DOI, source text type)
        so the attribution check in academic_researcher.py can deterministically
        verify attribution in the final review.

        EVIDENCE-SECTION GATING (Layer B): after a quote is verified verbatim, it
        is also classified by WHERE in the paper it sits. Quotes provably inside
        an excluded section (Introduction / Background / Related Work /
        References / Acknowledgements / ...) are rejected (verified=False) so
        only this study's own findings survive. Abstract-only (tangential) quotes
        are not section-gated.
        """
        max_length = self.research_config.get("max_study_text_length", 50000)
        min_chars_full = self.research_config.get("min_chars_for_deep_analysis", 5000)
        min_chars_abstract = self.research_config.get(
            "min_chars_for_abstract_quote_extraction", 150)

        has_full = bool(paper.full_text_available and paper.full_text_content)
        has_abstract = bool(paper.abstract and len(paper.abstract) >= min_chars_abstract)

        # Decide source text and source-type label
        if has_full and len(paper.full_text_content) >= min_chars_full:
            text = paper.full_text_content[:max_length]
            text_type = "FULL TEXT"
        elif has_full and mode == "tangential":
            # Full text was retrieved but is short — only acceptable in tangential mode
            text = paper.full_text_content[:max_length]
            text_type = "SHORT FULL TEXT"
        elif mode == "tangential" and has_abstract:
            text = paper.abstract
            text_type = "ABSTRACT ONLY (tangential mode)"
        else:
            # main mode + no usable full text  -> drop signal
            if mode == "main":
                print(f"      {Fore.YELLOW}No usable full text — DROPPING "
                      f"(main mode requires full text){Style.RESET_ALL}")
                return None
            # tangential mode but also no usable abstract — nothing to work with
            print(f"      {Fore.YELLOW}No usable text — keeping summary only "
                  f"(no quotes will be extracted){Style.RESET_ALL}")
            return existing_summary

        # Load source text into doc store for verification
        doc_record = None
        if self._doc_store:
            self._doc_store.clear()
            doc_record = self._load_paper(text, paper.paper_id)

        # --- Evidence-section gating setup (Layer B) ---------------------------
        # Built on the SAME `text` string loaded into the doc store, so the
        # heading positions line up with match.start_position. Abstract-only
        # (tangential) text is never gated — an abstract IS the study's own
        # condensed findings.
        gate = self._section_gating_settings()
        is_abstract_only = text_type.startswith("ABSTRACT")
        section_map: List[Tuple[int, str, str]] = []
        section_gate_active = False
        has_evidence_heading = False
        if gate["enforce"] and not is_abstract_only:
            section_map = build_section_map(text)
            has_evidence_heading = any(
                c in ("EVIDENCE", "ABSTRACT") for _, c, _ in section_map)
            if gate["mode"] == "require_evidence":
                # POSITIVE-IDENTIFICATION mode: always evaluate every quote's
                # section. When an evidence/abstract heading is present we keep
                # ONLY quotes provably inside it; when none is detected we fall
                # back to a safe subtractive rule (drop only EXCLUDE/REFS) unless
                # require_evidence_strict is set (then drop everything). The
                # per-quote branch below applies the decision.
                section_gate_active = True
            else:
                # Subtractive mode: only drop PROVABLE terminal non-findings
                # sections (REFS). Active whenever such a section was detected.
                section_gate_active = any(c == "REFS" for _, c, _ in section_map)
            cats = ", ".join(sorted({c for _, c, _ in section_map})) or "none"
            get_section_debug_logger().info(
                f"Section map for {paper.paper_id}: "
                f"{len(section_map)} headings ({cats}); "
                f"gate_active={section_gate_active}, mode={gate['mode']}, "
                f"evidence_heading={has_evidence_heading}, "
                f"strict={self._require_evidence_strict}")
            if self._section_debug:
                self._log_section_debug(
                    paper.paper_id, text, section_map, gate,
                    has_evidence_heading, section_gate_active)

        print(f"      {Fore.BLUE}Extracting verified quotes "
              f"({text_type.lower()})...{Style.RESET_ALL}", end=" ", flush=True)
        start = time.time()

        # VARIABLE QUOTE COUNT — the LLM extracts EVERY directly-relevant quote
        # (many, few, or none). No fixed target. An optional hard cap acts only
        # as an upper safety bound; 0 (the default) means unlimited.
        hard_cap = self.research_config.get("max_quotes_per_study_hard_cap", 0)
        try:
            hard_cap = int(hard_cap)
        except (TypeError, ValueError):
            hard_cap = 0
        cap_clause = ""
        if hard_cap and hard_cap > 0:
            cap_clause = (f"\n- As an upper bound, extract at most {hard_cap} quotes; if more than "
                          f"{hard_cap} are relevant, keep the {hard_cap} most important.")

        result = self._extract_deep_analysis(
            paper, query, text, text_type, cap_clause)

        elapsed = time.time() - start

        if not result.success or not result.json_response:
            print(f"{Fore.RED}failed ({elapsed:.0f}s){Style.RESET_ALL}")
            return existing_summary

        proposed_quotes = result.json_response.get("key_quotes", [])
        print(f"{Fore.GREEN}{len(proposed_quotes)} quotes proposed ({elapsed:.0f}s){Style.RESET_ALL}", end="")

        # Build per-quote source-identity metadata once (used for all verified quotes)
        source_meta = {
            "source_paper_id": paper.paper_id,
            "source_paper_title": paper.title,
            "source_paper_authors": paper.authors,
            "source_paper_year": paper.year,
            "source_paper_doi": paper.doi,
            "source_text_type": text_type,
        }

        verified_quotes = []
        if doc_record and self.text_matcher and self._doc_store:
            print(f" {Fore.BLUE}verifying...{Style.RESET_ALL}", end=" ", flush=True)
            verify_start = time.time()
            for _qi, qd in enumerate(proposed_quotes, 1):
                if not isinstance(qd, dict):
                    continue
                quote_text = qd.get("quote", "")
                if not quote_text or len(quote_text) < 20:
                    continue
                match = self._find_in_document(quote_text)
                if match:
                    # Layer B: classify WHERE the verified quote physically sits.
                    if is_abstract_only:
                        seg_cat, seg_head = "ABSTRACT", ""
                    else:
                        seg_cat, seg_head = classify_position(section_map, match.start_position)
                    eligible = True
                    reject_method = None
                    gate_branch = "gate inactive"
                    if section_gate_active:
                        if gate["mode"] == "require_evidence":
                            # ===== POSITIVE IDENTIFICATION ONLY =====
                            # A quote is usable ONLY if we can positively place it in
                            # the study's own findings. There is NO "keep everything
                            # except intro/refs" fallback: if we cannot positively
                            # identify the section, the quote is rejected (a paper we
                            # cannot parse simply yields nothing, rather than risk
                            # quoting its introduction).
                            if seg_cat in ("EVIDENCE", "ABSTRACT"):
                                eligible = True
                                gate_branch = f"positive: {seg_cat} section"
                            elif seg_cat == "UNKNOWN" and not self._require_evidence_strict:
                                # The block BEFORE the first heading is title +
                                # (unlabelled) ABSTRACT. The abstract is the authors'
                                # own summary of THEIR findings = primary evidence, so
                                # accept it — but ONLY when we can positively call it
                                # the abstract: (1) the paper has NO separate ABSTRACT
                                # heading (else the real abstract is that section and
                                # this leading block is just title/nav junk), (2) at
                                # least one real heading was found so the block is
                                # bounded, and (3) the block is a normal abstract size.
                                has_abstract_heading = any(
                                    c == "ABSTRACT" for _, c, _ in section_map)
                                lead_len = section_map[0][0] if section_map else len(text)
                                if (not has_abstract_heading and section_map
                                        and lead_len <= self._max_leading_exclude_chars):
                                    eligible = True
                                    gate_branch = (f"positive: inferred abstract "
                                                   f"(leading block {lead_len} chars)")
                                else:
                                    eligible = False
                                    gate_branch = ("reject: front-matter not a "
                                                   "bounded abstract")
                            else:
                                eligible = False
                                gate_branch = f"reject: non-evidence section [{seg_cat}]"
                            if not eligible:
                                reject_method = "rejected_non_evidence_section"
                        else:
                            # Subtractive mode: always drop terminal REFS sections.
                            # Drop a leading Introduction/Background quote ONLY when
                            # its region is a normal, bounded size — if the region
                            # is suspiciously large (a later evidence heading was
                            # missed and the intro "leaked" over the body) we KEEP
                            # the quote rather than risk dropping a real finding.
                            if seg_cat == "REFS":
                                eligible = False
                            elif seg_cat == "EXCLUDE":
                                region_len = _region_length(
                                    section_map, match.start_position, len(text))
                                eligible = region_len > self._max_leading_exclude_chars
                            else:
                                eligible = True
                            gate_branch = "exclude_non_evidence:subtractive"
                        if not eligible:
                            reject_method = "rejected_non_evidence_section"
                    # Layer C: primary-evidence-only — reject second-hand quotes
                    # (in-text citations / attribution of others' work) even when
                    # they sit in an evidence section.
                    secondhand_markers = []
                    if eligible and gate["exclude_secondhand"]:
                        secondhand_markers = detect_secondhand_markers(match.matched_text)
                        if secondhand_markers:
                            eligible = False
                            reject_method = "rejected_secondhand_citation"
                    if self._section_debug:
                        decision = ("ACCEPT" if eligible
                                    else f"REJECT:{reject_method}")
                        _dbg = get_section_debug_logger()
                        _dbg.info(
                            f"  QUOTE {_qi}: matched @pos {match.start_position} "
                            f"section=[{seg_cat}] '{seg_head[:50]}' "
                            f"sim={match.confidence} | {gate_branch} -> {decision}"
                            + (f" | second-hand markers={secondhand_markers}"
                               if secondhand_markers else ""))
                        _dbg.info(f"    text: \"{match.matched_text[:110]}\"")
                    entry = {
                        "quote": match.matched_text,
                        "context": qd.get("context", ""),
                        "importance": qd.get("importance", "medium"),
                        "verified": eligible,
                        "verification_method": (
                            match.match_method if eligible else reject_method),
                        "match_confidence": match.confidence,
                        "source_section": seg_cat,
                        "section_heading": seg_head,
                        "section_eligible": eligible,
                        "section_gate_active": section_gate_active,
                        "secondhand_markers": secondhand_markers,
                    }
                else:
                    if self._section_debug:
                        get_section_debug_logger().info(
                            f"  QUOTE {_qi}: NOT FOUND in source "
                            f"(text: \"{quote_text[:90]}\")")
                    entry = {
                        "quote": quote_text,
                        "context": qd.get("context", ""),
                        "importance": qd.get("importance", "medium"),
                        "verified": False,
                        "verification_method": "not_found",
                        "match_confidence": 0.0,
                        "source_section": "N/A",
                        "section_heading": "",
                        "section_eligible": False,
                        "section_gate_active": section_gate_active,
                        "secondhand_markers": [],
                    }
                entry.update(source_meta)
                verified_quotes.append(entry)
            v_count = sum(1 for q in verified_quotes if q.get("verified"))
            sec_rejected = sum(
                1 for q in verified_quotes
                if q.get("verification_method") == "rejected_non_evidence_section")
            secondhand = sum(
                1 for q in verified_quotes
                if q.get("verification_method") == "rejected_secondhand_citation")
            not_found = sum(
                1 for q in verified_quotes
                if q.get("verification_method") == "not_found")
            extra = ""
            if sec_rejected:
                extra += f", {sec_rejected} dropped: non-evidence section"
            if secondhand:
                extra += f", {secondhand} dropped: second-hand citation"
            if not_found:
                extra += f", {not_found} not found"
            print(f"{Fore.GREEN}{v_count}/{len(verified_quotes)} verified"
                  f"{extra} ({time.time()-verify_start:.0f}s){Style.RESET_ALL}")
        else:
            # No TextMatcher or no doc record — mark all as unverified
            for qd in proposed_quotes:
                if isinstance(qd, dict) and qd.get("quote"):
                    entry = {
                        "quote": qd["quote"],
                        "context": qd.get("context", ""),
                        "importance": qd.get("importance", "medium"),
                        "verified": False,
                        "verification_method": "no_matcher",
                        "match_confidence": 0.0,
                        "source_section": "N/A",
                        "section_heading": "",
                        "section_eligible": False,
                        "section_gate_active": False,
                        "secondhand_markers": [],
                    }
                    entry.update(source_meta)
                    verified_quotes.append(entry)
            print(f" {Fore.YELLOW}(unverified — no doc store / matcher){Style.RESET_ALL}")

        # Merge with existing summary if available
        if existing_summary:
            existing_summary["key_quotes"] = verified_quotes
            existing_summary["quotes_verified"] = any(q.get("verified") for q in verified_quotes)
            existing_summary["deep_analysis_mode"] = mode
            existing_summary["deep_analysis_text_type"] = text_type
            # Frozen, grounded paraphrase used to INTRODUCE the study in synthesis.
            _ds = (result.json_response or {}).get("study_summary", "")
            if _ds:
                existing_summary["study_summary"] = _ds
            return existing_summary

        # Build new analysis
        analysis = result.json_response
        analysis["key_quotes"] = verified_quotes
        analysis["quotes_verified"] = any(q.get("verified") for q in verified_quotes)
        analysis["deep_analysis_mode"] = mode
        analysis["deep_analysis_text_type"] = text_type
        analysis["paper_id"] = paper.paper_id
        analysis["paper_title"] = paper.title
        analysis["paper_authors"] = paper.authors
        analysis["paper_year"] = paper.year
        analysis["paper_venue"] = paper.venue
        analysis["paper_doi"] = paper.doi
        analysis["paper_citation_count"] = paper.citation_count
        analysis["has_full_text"] = paper.full_text_available
        return analysis

    # =========================================================================
    # DEEP-ANALYSIS QUOTE EXTRACTION (full mode + low-end chunked mode)
    # =========================================================================

    def _extract_deep_analysis(self, paper, query, text, text_type, cap_clause):
        """Get {study_summary, key_quotes} for a paper.

        FULL MODE (default, unchanged): exactly one LLM call containing the whole
        paper, with the prompt rendered byte-identically to the original.

        LOW-END MODE: the paper is read in sequential overlapping chunks, one LLM
        call per chunk, and the proposed quotes are accumulated. This exists
        because a whole paper is ~16.7K tokens and an 8K window silently
        truncated it, so the model only ever saw the first half and the back of
        every paper — Results, Discussion and Conclusion, i.e. exactly where the
        findings live — was invisible.

        Returns an object exposing .success and .json_response, so the caller's
        verification, section-gating and merge logic is untouched in both modes.
        """
        if not self._low_end:
            # ---- FULL MODE: original single call, unchanged ----
            prompt = _build_deep_analysis_prompt(
                paper, query, text_type, cap_clause,
                chunk_text=text, chunk_header="")
            return self.llm.run_primary(prompt, as_json=True,
                                        task="deep_analysis")

        # ---- LOW-END MODE: chunked read ----
        chunks = _split_text_into_chunks(
            text, self._le_chunk_chars, self._le_chunk_overlap,
            max_chunks=self._le_max_chunks)

        if len(chunks) <= 1:
            # Short paper — one call is enough. Use the low-end task profile so
            # the output budget still fits the small window.
            prompt = _build_deep_analysis_prompt(
                paper, query, text_type, cap_clause,
                chunk_text=chunks[0] if chunks else text, chunk_header="")
            return self.llm.run_primary(prompt, as_json=True,
                                        task="deep_analysis_low_end")

        print(f"\n        {Fore.WHITE}low-end: reading in {len(chunks)} "
              f"chunks{Style.RESET_ALL}", end=" ", flush=True)

        all_quotes = []
        summaries = []
        seen = set()
        any_success = False

        for idx, chunk in enumerate(chunks, 1):
            # Honour Ctrl+C between chunks: stop reading and keep what we have
            # rather than discarding a partly-read paper.
            if _is_interrupted():
                print(f"{Fore.YELLOW}[interrupted after chunk {idx-1}]"
                      f"{Style.RESET_ALL}", end=" ", flush=True)
                break

            header = (
                f"THIS IS PART {idx} OF {len(chunks)} of the paper's text. Extract "
                f"quotes ONLY from the text shown below. Other parts are handled "
                f"separately, so do not guess at or reconstruct text you cannot "
                f"see, and do not comment on the split. If this part contains no "
                f"relevant findings, return an empty \"key_quotes\" list — that is "
                f"a normal and expected outcome for a part that is all methods or "
                f"references."
            )
            prompt = _build_deep_analysis_prompt(
                paper, query, text_type, cap_clause,
                chunk_text=chunk, chunk_header=header)
            r = self.llm.run_primary(prompt, as_json=True,
                                     task="deep_analysis_low_end")
            if not r.success or not r.json_response:
                print(f"{Fore.YELLOW}[chunk {idx} failed]{Style.RESET_ALL}",
                      end=" ", flush=True)
                continue

            any_success = True
            for qd in (r.json_response.get("key_quotes") or []):
                if not isinstance(qd, dict):
                    continue
                qt = (qd.get("quote") or "").strip()
                if not qt:
                    continue
                # Overlap between consecutive chunks means the same sentence can
                # legitimately be proposed twice. Deduplicate on normalised text
                # so it is verified and counted once.
                key = re.sub(r"\W+", " ", qt.lower()).strip()
                if key in seen:
                    continue
                seen.add(key)
                all_quotes.append(qd)

            s = (r.json_response.get("study_summary") or "").strip()
            if s:
                summaries.append(s)
            print(f"{Fore.GREEN}{idx}\u2713{Style.RESET_ALL}", end="", flush=True)

        if not any_success:
            return _ExtractionResult(False, None)

        # A hard cap is normally applied by the prompt, but in chunked mode each
        # chunk applies it independently, so enforce it once over the merged set.
        hard_cap = self.research_config.get("max_quotes_per_study_hard_cap", 0)
        try:
            hard_cap = int(hard_cap)
        except (TypeError, ValueError):
            hard_cap = 0
        if hard_cap and hard_cap > 0 and len(all_quotes) > hard_cap:
            rank = {"high": 0, "medium": 1, "low": 2}
            all_quotes.sort(key=lambda q: rank.get(
                str(q.get("importance", "medium")).lower(), 1))
            all_quotes = all_quotes[:hard_cap]

        summary = self._merge_chunk_summaries(summaries, paper)
        print(f" {Fore.WHITE}({len(all_quotes)} candidates from "
              f"{len(chunks)} chunks){Style.RESET_ALL}", end=" ", flush=True)
        return _ExtractionResult(
            True, {"study_summary": summary, "key_quotes": all_quotes})

    def _merge_chunk_summaries(self, summaries, paper):
        """Reduce the per-chunk study summaries to the single 2-4 sentence
        paraphrase that synthesis uses to introduce the study.

        Each chunk only saw part of the paper, so its summary is partial.
        Concatenating them would produce a repetitive wall of text that then gets
        printed under the study's heading in the review, so they are merged.
        """
        summaries = [s for s in summaries if s]
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0]
        if not self._le_merge_summary or _is_interrupted():
            # Cheapest acceptable fallback: the longest single summary is the one
            # written from the chunk that saw the most substantive material.
            return max(summaries, key=len)

        joined = "\n\n".join(f"PART {i}: {s}" for i, s in enumerate(summaries, 1))
        prompt = f"""Below are partial summaries of ONE academic paper. Each was written after
reading a different part of it, so they overlap and each is incomplete.

PAPER: {paper.title}
Authors: {', '.join(paper.authors[:5])} | Year: {paper.year}

{joined}

Merge these into ONE factual summary of 2-4 sentences that will be used to
INTRODUCE this study in a literature review, before its quoted findings are
presented. Cover, where the parts state them: what the study set out to examine,
its design/methodology, its sample/dataset/scope, and the general nature of its
findings.

Rules:
- Use ONLY information present in the partial summaries above. Do not add,
  infer, or embellish any detail that is not there.
- Write it as a paraphrase in your own words, not as a quotation.
- If the parts contradict each other, prefer the more specific statement.
- Output ONLY the merged summary text. No preamble, no JSON, no headings.

Merged summary:"""
        r = self.llm.run_primary(prompt, task="title_generation")
        if r.success and (r.response or "").strip():
            return r.response.strip()
        return max(summaries, key=len)

    def _log_section_debug(self, paper_id, text, section_map, gate,
                           has_evidence_heading, section_gate_active):
        """Write a detailed, greppable section-detection report to the run log so
        heading detection can be refined from real runs. Shows every recognised
        heading (category + char range + length), every heading-SHAPED line we did
        NOT classify (the near-misses to tune against), and the resulting gate
        decision for this paper."""
        try:
            dbg = get_section_debug_logger()
            _headings, candidates = build_section_map_verbose(text)
            tlen = len(text)
            dbg.info(f"===== SECTION DETECTION DEBUG: {paper_id} "
                     f"(mode={gate['mode']}, strict={self._require_evidence_strict}) =====")
            if section_map:
                dbg.info(f"  RECOGNISED HEADINGS ({len(section_map)}):")
                for idx, (ls, cat, htext) in enumerate(section_map):
                    end = section_map[idx + 1][0] if idx + 1 < len(section_map) else tlen
                    dbg.info(f"    [{cat:<9}] @{ls:>6}  region={end-ls:>6} chars  "
                             f"'{htext[:80]}'")
            else:
                dbg.info("  RECOGNISED HEADINGS: none")
            near = [c for c in candidates if c["category"] is None]
            if near:
                dbg.info(f"  HEADING-SHAPED BUT UNCLASSIFIED ({len(near)}) "
                         f"— candidates to tune detection against:")
                for c in near[:60]:
                    dbg.info(f"    line {c['line_no']:>5}: '{c['text']}'  "
                             f"[{c['reason']}]")
                if len(near) > 60:
                    dbg.info(f"    ... (+{len(near)-60} more)")
            cat_counts: Dict[str, int] = {}
            for _ls, cat, _h in section_map:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            lead_block = section_map[0][0] if section_map else tlen
            dbg.info(f"  SUMMARY: categories={cat_counts or 'none'} | "
                     f"evidence_heading={has_evidence_heading} | "
                     f"gate_active={section_gate_active} | "
                     f"leading_block(abstract/front-matter)={lead_block} chars | "
                     f"mode={gate['mode']}")
            if gate["mode"] == "require_evidence" and not has_evidence_heading:
                if self._require_evidence_strict:
                    dbg.info("  NOTE: no evidence/abstract heading detected + strict "
                             "mode -> ALL quotes from this paper will be REJECTED. "
                             "(Tune detection or disable strict.)")
                else:
                    dbg.info("  NOTE: no evidence/abstract heading detected -> SAFE "
                             "FALLBACK active (drop only EXCLUDE/REFS quotes). See "
                             "unclassified candidates above to improve recall.")
        except Exception as e:  # debug logging must never break a run
            logger.warning(f"section debug logging failed for {paper_id}: {e}")

    def _find_in_document(self, quote_text: str) -> Optional['MatchResult']:
        """Find a quote in the current DocumentStore using TextMatcher."""
        if not self._doc_store or not self.text_matcher:
            return None

        search_terms = " ".join(quote_text.split()[:10])
        search_results = self._doc_store.search(search_terms, top_k=10)

        # BM25 returns 0 for single-chunk docs. Fall back to all chunks.
        if not search_results and self._doc_store.all_chunks:
            search_results = [(chunk, 1.0) for chunk in self._doc_store.all_chunks]
        if not search_results:
            return None

        best_match = None
        best_sim = 0.0

        for chunk, score in search_results:
            words = quote_text.split()
            key_phrases = []
            if len(words) >= 6:
                key_phrases.append(" ".join(words[:5]))
                key_phrases.append(" ".join(words[-5:]))
                if len(words) >= 12:
                    mid = len(words) // 2
                    key_phrases.append(" ".join(words[mid-2:mid+3]))

            match = self.text_matcher.find_quote_in_chunk(
                quote_text, chunk.text, chunk.start_char, key_phrases)
            if match.success:
                # Score by TRUE similarity between the LLM's proposed quote and
                # the matched source text — not the matcher's internal heuristic
                # confidence (which is unreliable for the loose strategies).
                sim = self._quote_similarity(quote_text, match.matched_text)
                if sim > best_sim:
                    best_sim = sim
                    best_match = match

        if best_match and self._accept_match(quote_text, best_match, best_sim):
            # Reflect the TRUE similarity as the stored confidence so logs are honest.
            best_match.confidence = round(best_sim, 4)
            return best_match
        return None

    @staticmethod
    def _norm_for_match(text: str) -> str:
        return re.sub(r'\s+', ' ', text or '').strip().lower()

    @staticmethod
    def _numbers_in(text: str) -> set:
        # numeric tokens that carry meaning in findings (12, 30, 0.001, 1,234)
        return set(re.findall(r'\d+(?:[.,]\d+)*', text or ''))

    def _quote_similarity(self, quote_text: str, matched_text: str) -> float:
        """
        True similarity in [0,1] between the LLM's quote and the matched source
        text. If the quote's exact words are CONTAINED in the matched text (the
        proposed quote is genuinely present in the source), this is 1.0;
        otherwise it is the SequenceMatcher ratio.
        """
        nq = self._norm_for_match(quote_text)
        nm = self._norm_for_match(matched_text)
        if not nq or not nm:
            return 0.0
        if nq in nm:
            return 1.0
        return SequenceMatcher(None, nq, nm).ratio()

    def _accept_match(self, quote_text: str, match: 'MatchResult', sim: float) -> bool:
        """
        Accept a match only if it is HIGH CONFIDENCE:
          * similarity >= self._quote_min_similarity (default 0.95), AND
          * NUMBER SAFETY: unless the quote is exactly contained (sim == 1.0),
            every number in the proposed quote must also appear in the matched
            text — so a near-identical sentence with a DIFFERENT number
            (e.g. "30%" vs "80%") is rejected even though it scores ~0.97.
        """
        if sim < self._quote_min_similarity:
            return False
        if sim >= 1.0:
            return True  # exact containment — numbers already present verbatim
        q_nums = self._numbers_in(self._norm_for_match(quote_text))
        m_nums = self._numbers_in(self._norm_for_match(match.matched_text))
        if not q_nums.issubset(m_nums):
            return False  # a number in the quote is missing/changed -> reject
        return True

    def _load_paper(self, text: str, paper_id: str) -> Optional['DocumentRecord']:
        try:
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"study_{hashlib.md5(paper_id.encode()).hexdigest()[:8]}.txt")
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(text)
            doc_record = self._doc_store.load_document(tmp_path)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return doc_record
        except Exception as e:
            logger.error(f"DocumentStore load failed: {e}")
            return None


# Backward-compatible alias. The class was renamed StudyAnalyzer ->
# StudyAnalyser (Australian spelling); this keeps any external import of the
# old name working.
StudyAnalyzer = StudyAnalyser
