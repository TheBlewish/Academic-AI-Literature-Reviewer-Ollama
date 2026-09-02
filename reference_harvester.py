# reference_harvester.py
# Backward citation chasing ("snowballing") for the academic literature reviewer.
#
# WHY: the literature review must rest on PRIMARY evidence only. We never quote a
# study's Introduction/Discussion sentence that reports ANOTHER study's findings
# (those are rejected as second-hand by study_analyser's Layer C). Instead, this
# module takes the References section of each RETAINED study, asks the LLM which
# cited works look directly relevant, ACQUIRES those works through the existing
# PaperDiscoveryEngine, and runs them through the SAME deep_analysis quote
# pipeline — so their findings can be quoted DIRECTLY as primary evidence.
#
# DEPTH-N SNOWBALLING + GLOBAL BUDGET:
#   The harvest is a BREADTH-FIRST walk backwards through the citation graph.
#     depth 0 = the retained "seed" studies
#     depth 1 = relevant works they cite
#     depth 2 = relevant works THOSE cite ... up to `max_depth`.
#   Three brakes keep it bounded:
#     * max_depth          — how many generations to chase (1 = original behaviour)
#     * max_total          — GLOBAL budget: total acquisition ATTEMPTS across ALL
#                            depths combined; when it hits 0 the whole walk stops.
#     * max_per_study      — per-paper cap on how many references it may contribute.
#   A global `seen` set (by DOI/normalised-title) prevents cycles and re-acquiring
#   the same work at any depth.
#
# DESIGN PRINCIPLES:
#   * Additive & safe: a single new graph node calls harvest(); every external
#     step is wrapped so any failure degrades to "no new papers" and never harms
#     the existing corpus.
#   * Reuses existing machinery: discovery (search/lookup/acquire/catalog),
#     study_analyzer.deep_analysis (with all three quote gates), and the section
#     map in study_analyser for slicing the References block.
#   * FULLY TRACED: every reference considered is logged with its outcome and the
#     reason it was kept or dropped, to the run log (print + logging) and to a
#     dedicated Logs/harvest_<ts>.json / .txt report, so you can see exactly what
#     the harvester is catching and missing.
#
# The pure-logic helpers (reference parsing, DOI extraction, candidate de-dup,
# selection-mapping) are import-safe and unit-tested independently of any network.

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from colorama import Fore, Style, init
    init()
except Exception:  # pragma: no cover - colour is cosmetic
    class _N:
        def __getattr__(self, _):
            return ""
    Fore = Style = _N()

# Reuse the section detector + references slicer from the analyser so heading
# logic stays consistent across the program.
from study_analyser import extract_references_block

# PaperMetadata is the discovery engine's paper object.
try:
    from paper_discovery import PaperMetadata
    HAS_PAPER_META = True
except Exception:  # pragma: no cover
    HAS_PAPER_META = False
    PaperMetadata = None


# =============================================================================
# PURE HELPERS (no network) — unit-tested
# =============================================================================

# A DOI as it appears in reference lists / URLs.
_DOI_RE = re.compile(r'10\.\d{4,9}/[^\s,;\]\)>"\']+', re.IGNORECASE)
# Leading entry markers: "1.", "12)", "[3]", "(4)"
_ENTRY_MARKER = re.compile(r'(?m)^\s*(?:\[\d+\]|\(\d+\)|\d+[\.\)])\s+')
# A 4-digit year, used as a fallback entry boundary in author-year bibliographies.
_YEAR_RE = re.compile(r'(?:18|19|20)\d{2}')


def clean_doi(raw: str) -> str:
    """Normalise a DOI string (strip URL prefix and trailing punctuation)."""
    if not raw:
        return ""
    d = raw.strip()
    d = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', d, flags=re.IGNORECASE)
    d = re.sub(r'^doi:\s*', '', d, flags=re.IGNORECASE)
    d = d.rstrip(' ).,;>"\']')
    return d


def extract_dois(text: str) -> List[str]:
    """Return all DOIs found in `text`, de-duplicated (order preserved)."""
    seen, out = set(), []
    for m in _DOI_RE.finditer(text or ""):
        d = clean_doi(m.group(0)).lower()
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def parse_reference_entries(refs_block: str, max_entries: int = 80) -> List[Dict]:
    """
    Split a References block into individual entries.

    Strategy (robust to messy PDF extraction):
      1. If numbered/bracketed markers exist, split on them.
      2. Else split on blank lines.
      3. Else split on single newlines.
    Each entry -> {"raw": <text>, "doi": <doi or "">}. Very short fragments are
    dropped. Returns at most `max_entries` entries.
    """
    if not refs_block or not refs_block.strip():
        return []
    block = refs_block.strip()

    parts: List[str]
    if _ENTRY_MARKER.search(block):
        pieces = _ENTRY_MARKER.split(block)
        parts = [p for p in pieces if p and p.strip()]
    elif "\n\n" in block:
        parts = [p for p in block.split("\n\n") if p.strip()]
    else:
        parts = [p for p in block.split("\n") if p.strip()]

    entries: List[Dict] = []
    for p in parts:
        raw = re.sub(r'\s+', ' ', p).strip()
        if len(raw) < 15:  # too short to be a real reference
            continue
        dois = extract_dois(raw)
        entries.append({"raw": raw, "doi": dois[0] if dois else ""})
        if len(entries) >= max_entries:
            break
    return entries


def candidate_key(doi: str = "", title: str = "") -> str:
    """A stable de-dup key for a candidate reference."""
    if doi:
        return "doi:" + clean_doi(doi).lower()
    norm = re.sub(r'[^a-z0-9]+', ' ', (title or "").lower()).strip()
    return "title:" + norm[:120]


def map_selection_to_entries(selection: List[Dict], entries: List[Dict]) -> List[Dict]:
    """
    Map an LLM selection (list of {index, title, doi}) back onto parsed entries.
    `index` is 1-based as presented to the LLM. Falls back to the entry's own DOI
    when the LLM omitted it. Invalid indices with no verifiable DOI are dropped.
    De-dups by candidate_key.
    """
    out, seen = [], set()
    for sel in selection or []:
        if not isinstance(sel, dict):
            continue
        idx = sel.get("index")
        title = (sel.get("title") or "").strip()
        doi = clean_doi(sel.get("doi") or "")
        entry = None
        if isinstance(idx, int) and 1 <= idx <= len(entries):
            entry = entries[idx - 1]
        elif isinstance(idx, str) and idx.isdigit() and 1 <= int(idx) <= len(entries):
            entry = entries[int(idx) - 1]
        if entry is not None:
            if not doi:
                doi = clean_doi(entry.get("doi", ""))
            if not title:
                title = entry.get("raw", "")[:200]
        else:
            # Index did not resolve to a real reference entry. Only trust the
            # candidate if it carries a verifiable DOI; otherwise drop it (a
            # free-form title with no anchor could be hallucinated).
            if not doi:
                continue
        if not title and not doi:
            continue
        key = candidate_key(doi, title)
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "doi": doi, "raw": (entry or {}).get("raw", "")})
    return out


# =============================================================================
# HARVEST TRACE — detailed record of what was caught / missed
# =============================================================================

class HarvestTrace:
    """Accumulates a structured, inspectable record of the whole snowball walk."""

    def __init__(self, query: str, settings: Dict):
        self.query = query
        self.settings = settings
        self.started = datetime.now().isoformat()
        self.selection_events: List[Dict] = []   # per frontier paper, per depth
        self.candidate_events: List[Dict] = []    # per acquired/attempted candidate
        self.depth_summaries: List[Dict] = []     # per depth roll-up
        self.notes: List[str] = []

    def note(self, msg: str):
        self.notes.append(msg)
        logger.info("HARVEST: %s", msg)

    def record_selection(self, depth, paper_title, refs_parsed, refs_selected, selected_list):
        ev = {"depth": depth, "paper": paper_title, "refs_parsed": refs_parsed,
              "refs_selected": refs_selected,
              "selected": [{"title": s.get("title", "")[:120], "doi": s.get("doi", "")}
                           for s in selected_list]}
        self.selection_events.append(ev)
        logger.info("HARVEST depth %d: '%s' -> parsed %d refs, LLM selected %d",
                    depth, paper_title[:60], refs_parsed, refs_selected)

    def record_candidate(self, ev: Dict):
        self.candidate_events.append(ev)
        # file-only granular line
        logger.info(
            "HARVEST depth %s candidate '%s' (doi=%s): acq=%s | proposed=%s "
            "verified_primary=%s dropped[nonEvid=%s,secondHand=%s,notFound=%s] -> %s%s",
            ev.get("depth"), (ev.get("title") or "")[:60], ev.get("doi") or "-",
            ev.get("acquire_status"), ev.get("proposed"), ev.get("verified_primary"),
            ev.get("dropped_non_evidence"), ev.get("dropped_secondhand"),
            ev.get("dropped_not_found"), "KEPT" if ev.get("kept") else "DROPPED",
            "" if ev.get("kept") else f" ({ev.get('drop_reason')})")
        for dq in ev.get("dropped_quotes", []):
            logger.debug("HARVEST   dropped-quote [%s/%s]%s :: %s",
                         dq.get("reason"), dq.get("section"),
                         (" markers=" + ",".join(dq.get("markers", []))) if dq.get("markers") else "",
                         (dq.get("quote") or "")[:160])

    def record_depth(self, depth, frontier_size, candidates, kept, budget_left):
        summ = {"depth": depth, "frontier_size": frontier_size,
                "candidates": candidates, "kept": kept, "budget_left": budget_left}
        self.depth_summaries.append(summ)

    def as_dict(self) -> Dict:
        return {
            "query": self.query, "started": self.started,
            "finished": datetime.now().isoformat(), "settings": self.settings,
            "depth_summaries": self.depth_summaries,
            "selection_events": self.selection_events,
            "candidate_events": self.candidate_events,
            "notes": self.notes,
        }

    def write(self, log_dir: str) -> Optional[str]:
        """Write JSON + human-readable .txt report into log_dir. Returns txt path."""
        if not log_dir:
            return None
        try:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join(log_dir, f"harvest_{ts}.json")
            txt_path = os.path.join(log_dir, f"harvest_{ts}.txt")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.as_dict(), f, indent=2, ensure_ascii=False)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(self._render_text())
            return txt_path
        except Exception as e:  # pragma: no cover
            logger.warning("Could not write harvest report: %s", e)
            return None

    def _render_text(self) -> str:
        L = []
        L.append("=" * 78)
        L.append("REFERENCE HARVEST REPORT (backward citation chasing)")
        L.append("=" * 78)
        L.append(f"Query:    {self.query}")
        L.append(f"Started:  {self.started}")
        L.append(f"Settings: {self.settings}")
        L.append("")
        L.append("PER-DEPTH SUMMARY")
        L.append("-" * 78)
        for d in self.depth_summaries:
            L.append(f"  depth {d['depth']}: frontier={d['frontier_size']} "
                     f"candidates={d['candidates']} kept={d['kept']} "
                     f"budget_left={d['budget_left']}")
        if self.notes:
            L.append("")
            L.append("NOTES")
            L.append("-" * 78)
            for n in self.notes:
                L.append(f"  - {n}")
        L.append("")
        L.append("REFERENCE SELECTION (what each paper offered vs what was chosen)")
        L.append("-" * 78)
        for ev in self.selection_events:
            L.append(f"  [depth {ev['depth']}] {ev['paper'][:70]}")
            L.append(f"      parsed {ev['refs_parsed']} refs, selected {ev['refs_selected']}")
            for s in ev["selected"]:
                L.append(f"        + {s['doi'] or '(no doi)'} :: {s['title']}")
        L.append("")
        L.append("CANDIDATE OUTCOMES (what was caught and what was missed, with reasons)")
        L.append("-" * 78)
        for ev in self.candidate_events:
            status = "KEPT" if ev.get("kept") else f"DROPPED ({ev.get('drop_reason')})"
            L.append(f"  [depth {ev.get('depth')}] {status}: "
                     f"{ev.get('doi') or '(no doi)'} :: {(ev.get('title') or '')[:70]}")
            L.append(f"      acquire={ev.get('acquire_status')} | proposed={ev.get('proposed')} "
                     f"verified_primary={ev.get('verified_primary')} "
                     f"dropped[nonEvidence={ev.get('dropped_non_evidence')}, "
                     f"secondHand={ev.get('dropped_secondhand')}, "
                     f"notFound={ev.get('dropped_not_found')}]")
            for dq in ev.get("dropped_quotes", []):
                mk = (" markers=" + ",".join(dq.get("markers", []))) if dq.get("markers") else ""
                L.append(f"        - dropped[{dq.get('reason')}/{dq.get('section')}]{mk}: "
                         f"{(dq.get('quote') or '')[:140]}")
        L.append("")
        L.append("=" * 78)
        return "\n".join(L)


# =============================================================================
# HARVESTER (network-touching, fully guarded)
# =============================================================================

class ReferenceHarvester:
    """
    Mines references of retained studies, acquires relevant cited works, and
    deep-analyses them as primary evidence — recursively to `max_depth`, bounded
    by a global acquisition budget. Construct with the orchestrator's existing
    components so no new global state is introduced.
    """

    def __init__(self, discovery, agent_manager, study_analyzer, config: Dict):
        self.discovery = discovery
        self.llm = agent_manager
        self.study_analyzer = study_analyzer
        self.config = config or {}
        self.last_trace: Optional[HarvestTrace] = None

    # ----- config accessors (safe defaults; NO academic_config.py change needed)
    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enable_reference_harvesting", True))

    @property
    def max_depth(self) -> int:
        return max(1, int(self.config.get("reference_harvest_max_depth", 1)))

    @property
    def max_per_study(self) -> int:
        return int(self.config.get("reference_harvest_max_per_study", 5))

    @property
    def max_total(self) -> int:
        return int(self.config.get("reference_harvest_max_total", 20))

    @property
    def min_verified(self) -> int:
        return int(self.config.get("reference_harvest_min_verified_quotes",
                                   self.config.get("min_verified_quotes_per_study", 1)))

    @property
    def write_report(self) -> bool:
        return bool(self.config.get("reference_harvest_log", True))

    # -------------------------------------------------------------------------
    def harvest(self, seed_analyses: List[Dict], query: str,
                log_dir: str = None) -> List[Dict]:
        """
        Depth-N backward citation chase. Returns a list of NEW deep_analysis dicts
        (same shape node_deep_analysis produces) for acquired cited works that
        yielded >= min_verified primary quotes, across all depths. Never raises.
        """
        settings = {"max_depth": self.max_depth, "max_total": self.max_total,
                    "max_per_study": self.max_per_study, "min_verified": self.min_verified}
        trace = HarvestTrace(query, settings)
        self.last_trace = trace
        new_analyses_all: List[Dict] = []

        if not self.enabled or not seed_analyses:
            trace.note("disabled or no seed studies")
            if self.write_report:
                trace.write(log_dir)
            return new_analyses_all

        # Global de-dup across the entire walk (seeds + everything acquired).
        seen = self._already_seen_keys(seed_analyses)
        budget = self.max_total
        frontier = list(seed_analyses)  # depth 0

        for depth in range(1, self.max_depth + 1):
            if budget <= 0:
                trace.note(f"global budget exhausted before depth {depth}")
                break
            if not frontier:
                trace.note(f"no papers to expand at depth {depth}")
                break

            print(f"  {Fore.MAGENTA}Snowball depth {depth}/{self.max_depth} — "
                  f"expanding {len(frontier)} paper(s), budget left {budget}{Style.RESET_ALL}")

            # Refresh so papers acquired in the previous depth (now in the catalog
            # WITH full text) can themselves be mined.
            try:
                papers_map = {p.paper_id: p for p in self.discovery.get_all_papers()}
            except Exception as e:  # pragma: no cover
                trace.note(f"could not list papers at depth {depth}: {e}")
                break

            # ----- gather candidates from the current frontier -----
            candidates: List[Dict] = []
            for analysis in frontier:
                if budget - len(candidates) <= 0:
                    break
                pid = analysis.get("paper_id", "")
                paper = papers_map.get(pid)
                full_text = getattr(paper, "full_text_content", None) if paper else None
                if not full_text:
                    continue
                remaining = budget - len(candidates)
                try:
                    picked, refs_parsed = self._select_references_for_paper(
                        paper, query, remaining=remaining)
                except Exception as e:
                    trace.note(f"selection failed for {pid}: {e}")
                    picked, refs_parsed = [], 0
                trace.record_selection(depth, getattr(paper, "title", pid),
                                       refs_parsed, len(picked), picked)
                for cand in picked:
                    key = candidate_key(cand.get("doi", ""), cand.get("title", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    cand["_depth"] = depth
                    candidates.append(cand)
                    if budget - len(candidates) <= 0:
                        break

            if not candidates:
                trace.record_depth(depth, len(frontier), 0, 0, budget)
                trace.note(f"no new relevant references at depth {depth}")
                break

            print(f"  {Fore.WHITE}  acquiring {len(candidates)} cited work(s) "
                  f"at depth {depth}...{Style.RESET_ALL}")

            # ----- acquire + analyse each candidate -----
            depth_kept: List[Dict] = []
            for i, cand in enumerate(candidates, 1):
                budget -= 1  # one acquisition attempt spent
                title_disp = (cand.get("title") or cand.get("doi") or "?")[:68]
                print(f"  {Fore.CYAN}  [d{depth} {i}/{len(candidates)}] "
                      f"{title_disp}{Style.RESET_ALL}")
                ev = {"depth": depth, "title": cand.get("title", ""),
                      "doi": cand.get("doi", ""), "acquire_status": "not_attempted",
                      "proposed": 0, "verified_primary": 0,
                      "dropped_non_evidence": 0, "dropped_secondhand": 0,
                      "dropped_not_found": 0, "dropped_quotes": [],
                      "kept": False, "drop_reason": ""}

                paper, status = self._acquire_candidate(cand)
                ev["acquire_status"] = status
                if not paper:
                    ev["drop_reason"] = f"not acquired ({status})"
                    print(f"      {Fore.YELLOW}skip — {status}{Style.RESET_ALL}")
                    trace.record_candidate(ev)
                    continue

                try:
                    analysis = self.study_analyzer.deep_analysis(paper, query, mode="main")
                except Exception as e:
                    ev["acquire_status"] = f"{status}; analysis_error"
                    ev["drop_reason"] = f"deep_analysis error: {e}"
                    trace.record_candidate(ev)
                    continue
                if not analysis:
                    ev["drop_reason"] = "deep_analysis returned None (no usable full text)"
                    trace.record_candidate(ev)
                    continue

                self._fill_quote_stats(ev, analysis)
                if ev["verified_primary"] < self.min_verified:
                    ev["drop_reason"] = (f"{ev['verified_primary']} primary quotes "
                                         f"(< {self.min_verified})")
                    print(f"      {Fore.YELLOW}dropped — {ev['drop_reason']}{Style.RESET_ALL}")
                    trace.record_candidate(ev)
                    continue

                analysis["acquired_via"] = "reference_harvest"
                analysis["harvest_depth"] = depth
                new_analyses_all.append(analysis)
                depth_kept.append(analysis)
                ev["kept"] = True
                print(f"      {Fore.GREEN}kept — {ev['verified_primary']} primary "
                      f"quotes{Style.RESET_ALL}")
                trace.record_candidate(ev)

            trace.record_depth(depth, len(frontier), len(candidates),
                               len(depth_kept), budget)
            frontier = depth_kept  # next generation = what we just kept

        print(f"  {Fore.GREEN}Reference harvest complete: {len(new_analyses_all)} new "
              f"primary studies across {len(trace.depth_summaries)} depth(s).{Style.RESET_ALL}")
        report_path = trace.write(log_dir) if self.write_report else None
        if report_path:
            print(f"  {Fore.WHITE}Harvest report: {report_path}{Style.RESET_ALL}")
        return new_analyses_all

    # ----- internals ---------------------------------------------------------
    @staticmethod
    def _fill_quote_stats(ev: Dict, analysis: Dict):
        quotes = analysis.get("key_quotes", []) or []
        ev["proposed"] = len(quotes)
        for q in quotes:
            if not isinstance(q, dict):
                continue
            method = q.get("verification_method")
            if q.get("verified") and q.get("quote"):
                ev["verified_primary"] += 1
            elif method == "rejected_non_evidence_section":
                ev["dropped_non_evidence"] += 1
                ev["dropped_quotes"].append({"quote": q.get("quote", ""),
                                             "reason": "non_evidence_section",
                                             "section": q.get("source_section", ""),
                                             "markers": q.get("secondhand_markers", [])})
            elif method == "rejected_secondhand_citation":
                ev["dropped_secondhand"] += 1
                ev["dropped_quotes"].append({"quote": q.get("quote", ""),
                                             "reason": "secondhand_citation",
                                             "section": q.get("source_section", ""),
                                             "markers": q.get("secondhand_markers", [])})
            elif method == "not_found":
                ev["dropped_not_found"] += 1
                ev["dropped_quotes"].append({"quote": q.get("quote", ""),
                                             "reason": "not_found",
                                             "section": q.get("source_section", ""),
                                             "markers": q.get("secondhand_markers", [])})

    def _already_seen_keys(self, study_analyses: List[Dict]) -> set:
        seen = set()
        for a in study_analyses:
            seen.add(candidate_key(a.get("paper_doi") or "", a.get("paper_title") or ""))
        try:
            for p in self.discovery.get_all_papers():
                seen.add(candidate_key(getattr(p, "doi", "") or "",
                                       getattr(p, "title", "") or ""))
        except Exception:  # pragma: no cover
            pass
        return seen

    def _select_references_for_paper(self, paper, query: str,
                                     remaining: int) -> Tuple[List[Dict], int]:
        """Slice references, parse entries, ask the LLM which look relevant.
        Returns (picked_candidates, refs_parsed_count)."""
        refs_block = extract_references_block(paper.full_text_content or "")
        if not refs_block:
            return [], 0
        entries = parse_reference_entries(refs_block)
        if not entries:
            return [], 0

        cap = max(1, min(self.max_per_study, remaining))
        listing = "\n".join(f"{i+1}. {e['raw'][:300]}" for i, e in enumerate(entries))
        prompt = f"""You are selecting CITED references to chase as PRIMARY evidence for a literature review.

RESEARCH QUESTION: "{query}"

The following are references cited by the study "{paper.title}". Select ONLY the
references that look like PRIMARY STUDIES (original empirical research: trials,
cohorts, experiments, etc. — not reviews, editorials, or textbooks) whose
FINDINGS would directly help answer the research question. If a reference is not
clearly relevant, do NOT select it. Select at most {cap}.

REFERENCES:
{listing}

Respond with ONLY JSON (an empty list is valid):
{{
  "selected": [
    {{"index": <number from the list above>, "title": "the work's title", "doi": "DOI if present else empty string"}}
  ]
}}"""
        result = self.llm.run_primary(prompt, as_json=True, task="paper_selection")
        if not result or not getattr(result, "success", False) or not result.json_response:
            return [], len(entries)
        selection = result.json_response.get("selected", [])
        picked = map_selection_to_entries(selection, entries)
        return picked[:cap], len(entries)

    def _acquire_candidate(self, cand: Dict) -> Tuple[Optional["PaperMetadata"], str]:
        """Resolve a candidate to a PaperMetadata with full text.
        Returns (paper_or_None, status_string)."""
        if not HAS_PAPER_META:
            return None, "paper_metadata_unavailable"
        doi = clean_doi(cand.get("doi", ""))
        title = (cand.get("title") or "").strip()
        paper = None
        route = ""

        if doi:
            route = "doi_lookup"
            oa_client = self.discovery.clients.get("openalex")
            if oa_client and hasattr(oa_client, "lookup_by_doi"):
                try:
                    paper = oa_client.lookup_by_doi(doi)
                except Exception:
                    paper = None
            if paper is None:
                paper = PaperMetadata(title=title or doi, authors=[], year=None,
                                      abstract=None, doi=doi)
        elif title:
            route = "title_search"
            try:
                results = self.discovery.search_all_apis(title, limit_per_api=5)
            except Exception:
                results = []
            paper = self._best_title_match(title, results)
            if paper is None:
                return None, "title_search:no_confident_match"
        else:
            return None, "no_doi_or_title"

        try:
            ok = self.discovery.acquire_full_text(paper)
        except Exception:
            ok = False
        if not ok or not getattr(paper, "full_text_content", None):
            return None, f"{route}:no_full_text"

        try:
            self.discovery.add_selected_papers([paper])
        except Exception:
            pass
        return paper, f"{route}:full_text_ok"

    @staticmethod
    def _best_title_match(title: str, results: List) -> Optional["PaperMetadata"]:
        if not results:
            return None
        def toks(s):
            return set(re.findall(r'[a-z0-9]{3,}', (s or "").lower()))
        want = toks(title)
        if not want:
            return results[0]
        best, best_score = None, 0.0
        for p in results:
            got = toks(getattr(p, "title", ""))
            if not got:
                continue
            overlap = len(want & got) / max(1, len(want))
            if overlap > best_score:
                best, best_score = p, overlap
        return best if best_score >= 0.6 else None
