# research_planner.py
# LLM-Driven Research Planning — Scope-Aware + Iterative Self-Refining
#
# CHANGES IN THIS VERSION:
#   - distill_search_strategy() prompt forbids empty new_memo output and
#     provides a floor instruction (fixes "Distillation produced empty memo").
#   - refine_plan_tangential() rewritten with HARD GATES to stop unrelated
#     (not merely tangential) searches: a domain anchor, a one-hop rule, and
#     a "would a domain expert recognise this as evidence" test, with worked
#     good/bad examples. Indirect routes must stay within the question's real
#     subject domain — no cross-domain analogy chains.
#   - Identified scope is still persisted (unchanged).

import json
import logging
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from colorama import Fore, Style, init
init()

logger = logging.getLogger(__name__)


@dataclass
class FocusArea:
    area: str
    priority: int
    search_query: str = ""
    papers_found: int = 0
    investigated: bool = False
    notes: str = ""

    def to_dict(self) -> Dict:
        return {"area": self.area, "priority": self.priority,
                "search_query": self.search_query,
                "papers_found": self.papers_found,
                "investigated": self.investigated, "notes": self.notes}


@dataclass
class ResearchPlan:
    original_query: str
    query_type: str
    academic_field: str
    focus_areas: List[FocusArea]
    methodology_notes: str
    search_strategy: str
    iteration: int = 0
    is_tangential: bool = False
    identified_scope: str = ""

    def to_dict(self) -> Dict:
        return {"original_query": self.original_query,
                "query_type": self.query_type,
                "academic_field": self.academic_field,
                "focus_areas": [fa.to_dict() for fa in self.focus_areas],
                "methodology_notes": self.methodology_notes,
                "search_strategy": self.search_strategy,
                "iteration": self.iteration,
                "is_tangential": self.is_tangential,
                "identified_scope": self.identified_scope}


class ResearchPlanner:
    def __init__(self, agent_manager):
        self.llm = agent_manager
        self.all_plans: List[ResearchPlan] = []
        self.all_focus_areas_used: List[str] = []
        self.all_search_queries_used: List[str] = []
        self._query_lock = threading.Lock()

    # =========================================================================
    # Internal helper: fetch the most-recently-set scope, if any
    # =========================================================================

    def _get_current_scope(self) -> str:
        """Return the scope from the latest plan that has one set."""
        for p in reversed(self.all_plans):
            if p.identified_scope:
                return p.identified_scope
        return ""

    # =========================================================================
    # INITIAL PLAN — scope extraction + scope-respecting focus areas
    # =========================================================================

    def create_initial_plan(self, query: str, max_retries: int = 3) -> Optional[ResearchPlan]:
        print(f"\n{Fore.CYAN}\U0001F9E0 Analyzing research query...{Style.RESET_ALL}")

        prompt = f"""You are planning an academic literature review.

RESEARCH QUESTION: "{query}"

================ STEP 1: SCOPE EXTRACTION ================

Before designing any focus areas, identify what the research question is
specifically asking about. The question may carry constraints — population,
intervention, outcome domain, time horizon, methodology, or none — that
define the SCOPE of an answer that would actually satisfy the user.

Write out the scope you identify. Be specific. For example, if the question
mentions "healthy adults," that is the population scope and ALL searches must
respect it. If it specifies a particular drug, treatment, or technique, that
is the intervention scope. If it implies a particular outcome, that is the
outcome scope.

If the question is genuinely broad and has no explicit constraints, say so —
then you can vary more freely. But do not invent constraints that aren't
in the question, and do not ignore constraints that ARE in the question.

================ STEP 2: DESIGN 5 FOCUS AREAS ================

Each focus area targets a DIFFERENT aspect of the research question WITHIN
the scope you identified. Each gets ONE academic-database search query.

CRITICAL: every focus area must stay within the scope. Do not introduce
populations, interventions, or outcomes the user did not ask about.

What dimensions to vary across depends on the QUESTION. Choose the
dimensions that are actually relevant. Possible angles include:
- Different specific outcomes within the asked-about domain
- Different mechanisms or pathways that inform the answer
- Different time horizons if relevant
- Different study types (RCT, observational, systematic review)
- Different sub-aspects of the intervention itself

Do not force the same dimensions onto every question.

================ STEP 3: OUTPUT ================

Respond with ONLY a JSON object:
{{
    "identified_scope": "explicit description of the scope/constraints the question carries",
    "query_type": "broad|niche|specific|comparative",
    "academic_field": "primary field of study",
    "methodology_notes": "what study types are most relevant",
    "search_strategy": "your overall approach in 1-2 sentences, anchored to the scope",
    "focus_areas": [
        {{"area": "what aspect this targets (within scope)", "priority": 5, "search_query": "3-8 word academic search"}},
        {{"area": "what aspect this targets (within scope)", "priority": 4, "search_query": "3-8 word academic search"}},
        {{"area": "what aspect this targets (within scope)", "priority": 3, "search_query": "3-8 word academic search"}},
        {{"area": "what aspect this targets (within scope)", "priority": 2, "search_query": "3-8 word academic search"}},
        {{"area": "what aspect this targets (within scope)", "priority": 1, "search_query": "3-8 word academic search"}}
    ]
}}

QUERY-WRITING RULES:
- Priority 5 is the MOST DIRECT query for the research question.
- Each query must use the specific entities/concepts from the question itself.
- Queries should be 3-8 words, written like you would type into PubMed.
- Do NOT include dosages, ages, or other numeric specifics.
- Use your knowledge of how real papers in this field are titled.
- Each of the 5 queries must use substantively different terms."""

        for attempt in range(max_retries):
            result = self.llm.run_primary(prompt, as_json=True, task="planning")
            if result.success and result.json_response:
                plan = self._parse_plan(query, result.json_response, iteration=0)
                if plan and len(plan.focus_areas) >= 3:
                    self.all_plans.append(plan)
                    if plan.identified_scope:
                        print(f"  {Fore.WHITE}Scope: {Fore.CYAN}{plan.identified_scope}{Style.RESET_ALL}")
                    self._display_plan(plan)
                    return plan
            print(f"  {Fore.YELLOW}Retry ({attempt + 2}/{max_retries})...{Style.RESET_ALL}")
        return self._fallback_plan(query)

    # =========================================================================
    # REFINE PLAN — sees strategy memo + recent search history
    # =========================================================================

    def refine_plan(self, query: str, existing_studies_summary: str,
                    focus_areas_used: List[str],
                    strategy_memo: str = "",
                    recent_search_history: str = "") -> Optional[ResearchPlan]:
        print(f"\n{Fore.CYAN}\U0001F504 Refining search plan...{Style.RESET_ALL}")

        previously_used = "\n".join(f"  - {fa}" for fa in focus_areas_used)
        prev_queries = "\n".join(f"  - {q}" for q in self.all_search_queries_used[-15:])

        memo_block = ""
        if strategy_memo.strip():
            memo_block = f"""

YOUR ACCUMULATED SEARCH STRATEGY MEMO (what you've learned about searching
this topic effectively — informed by previous rounds):
{strategy_memo}
"""

        history_block = ""
        if recent_search_history.strip():
            history_block = f"""

RECENT SEARCH HISTORY (queries and what they returned — use this to judge
which phrasings and angles actually find relevant papers):
{recent_search_history}
"""

        scope_block = ""
        current_scope = self._get_current_scope()
        if current_scope:
            scope_block = f"""

IDENTIFIED SCOPE OF THE QUESTION (from the initial plan — refinement must
stay within this scope):
{current_scope}
"""

        prompt = f"""You are refining the search plan for an academic literature review.

RESEARCH QUESTION: "{query}"
{scope_block}
================ SCOPE REMINDER ================

The research question defines the scope. Stay within whatever population,
intervention, or outcome the user actually asked about. Do NOT drift to
related-but-different populations or topics unless the user's question
allows it.
{memo_block}{history_block}

FOCUS AREAS ALREADY INVESTIGATED (don't repeat these EXACTLY, but you may
reword them or come at the same aspect from a different angle):
{previously_used}

SEARCH QUERIES ALREADY USED (do NOT submit identical duplicates; you ARE
encouraged to reword previous queries that returned poor results):
{prev_queries}

STUDIES ALREADY FOUND:
{existing_studies_summary}

================ YOUR TASK ================

Based on the studies found so far AND what you've learned from past searches:

1. What is still MISSING from the evidence base WITHIN SCOPE?
2. Are any past search queries clearly underperforming? Reword them — change
   specificity, change terminology, change framing — not abandon the aspect.
3. Are there in-scope aspects you haven't searched for yet at all?

If existing studies comprehensively answer the question within scope and
there are no useful refinements left to try, set should_continue to false.

Otherwise generate 5 NEW focus areas that fill the gaps you identified.

Respond with ONLY JSON:
{{
    "should_continue": true/false,
    "gap_analysis": "specific in-scope gaps",
    "strategy_reflection": "1-2 sentence note on what your past searches taught you",
    "search_strategy": "how the new searches address the gaps",
    "focus_areas": [
        {{"area": "what aspect this targets (in scope)", "priority": 5, "search_query": "different 3-8 word query"}},
        {{"area": "what aspect this targets (in scope)", "priority": 4, "search_query": "different 3-8 word query"}},
        {{"area": "what aspect this targets (in scope)", "priority": 3, "search_query": "different 3-8 word query"}},
        {{"area": "what aspect this targets (in scope)", "priority": 2, "search_query": "different 3-8 word query"}},
        {{"area": "what aspect this targets (in scope)", "priority": 1, "search_query": "different 3-8 word query"}}
    ]
}}

QUERY RULES:
- All 5 queries must respect the user's question's scope.
- Apply what your strategy memo says about effective phrasing.
- If previous round returned few results, try BROADER queries this round.
- If previous round returned many off-topic results, try MORE SPECIFIC queries.
- Each query must be substantively different from queries already used."""

        result = self.llm.run_primary(prompt, as_json=True, task="refine_planning")
        if not result.success or not result.json_response:
            return None

        data = result.json_response
        if not data.get("should_continue", True):
            gap_info = data.get("gap_analysis", "")
            print(f"  {Fore.GREEN}\U00002705 LLM: research comprehensive enough.{Style.RESET_ALL}")
            if gap_info:
                print(f"  {Fore.WHITE}  {gap_info}{Style.RESET_ALL}")
            return None

        gap_info = data.get("gap_analysis", "")
        if gap_info:
            print(f"  {Fore.YELLOW}Gaps: {gap_info}{Style.RESET_ALL}")
        reflection = data.get("strategy_reflection", "")
        if reflection:
            print(f"  {Fore.WHITE}Strategy reflection: {reflection}{Style.RESET_ALL}")

        new_plan = self._parse_plan(
            query,
            {"query_type": "refined",
             "academic_field": "same",
             "methodology_notes": "",
             "search_strategy": data.get("search_strategy", "Refined based on gaps"),
             "focus_areas": data.get("focus_areas", [])},
            iteration=len(self.all_plans))

        if new_plan and new_plan.focus_areas:
            new_plan.identified_scope = current_scope
            self.all_plans.append(new_plan)
            self._display_plan(new_plan)
            return new_plan
        return None

    # =========================================================================
    # TANGENTIAL REFINE — iterative
    # =========================================================================
    # CHANGED IN THIS VERSION: hard gates against UNRELATED (vs. tangential)
    # routes. Indirect evidence must stay in the question's real subject domain,
    # connect via a single inferential hop, and pass a domain-expert test.

    def refine_plan_tangential(self, query: str, existing_studies_summary: str,
                                focus_areas_used: List[str],
                                strategy_memo: str = "",
                                recent_search_history: str = "",
                                sufficiency_reasoning: str = "",
                                tangential_round: int = 1) -> Optional[ResearchPlan]:
        print(f"\n{Fore.MAGENTA}\U0001F504 TANGENTIAL MODE — Round {tangential_round}: refining...{Style.RESET_ALL}")

        previously_used = "\n".join(f"  - {fa}" for fa in focus_areas_used)
        prev_queries = "\n".join(f"  - {q}" for q in self.all_search_queries_used[-20:])

        memo_block = ""
        if strategy_memo.strip():
            memo_block = f"""

YOUR ACCUMULATED SEARCH STRATEGY MEMO:
{strategy_memo}
"""

        history_block = ""
        if recent_search_history.strip():
            history_block = f"""

RECENT SEARCH HISTORY:
{recent_search_history}
"""

        sufficiency_block = ""
        if sufficiency_reasoning.strip():
            sufficiency_block = f"""

EVIDENCE SUFFICIENCY ASSESSMENT (your own reasoning for engaging tangential mode):
{sufficiency_reasoning}
"""

        scope_block = ""
        current_scope = self._get_current_scope()
        if current_scope:
            scope_block = f"""

IDENTIFIED SCOPE OF THE ORIGINAL QUESTION:
{current_scope}

In tangential mode, you may design searches that step OUTSIDE the narrowest
reading of this scope to gather indirect evidence — but every indirect search
must still belong to the SAME REAL-WORLD SUBJECT DOMAIN as the question and
inform an answer to it within a single inferential step.
"""

        prompt = f"""You are designing TANGENTIAL searches for an academic literature review.

RESEARCH QUESTION: "{query}"
{scope_block}
You have determined that direct evidence on this exact topic is sparse, and
you are now in tangential mode — searching for indirect evidence that can
still inform an answer to the original question. This is tangential round
{tangential_round}.
{sufficiency_block}{memo_block}{history_block}

FOCUS AREAS ALREADY INVESTIGATED (across standard AND tangential rounds):
{previously_used}

SEARCH QUERIES ALREADY USED (don't duplicate):
{prev_queries}

STUDIES ALREADY FOUND:
{existing_studies_summary}

================ WHAT "TANGENTIAL" MEANS HERE ================

Tangential means a CLOSE NEIGHBOUR of the question that a researcher IN THE
QUESTION'S OWN FIELD would still recognise as relevant evidence. It does NOT
mean "any paper I can connect to the topic through a clever chain of
reasoning." Most over-reaching happens when a model takes a generic physical,
chemical, or statistical concept that merely shares a WORD with the question
(e.g. "pressure", "contact", "transfer", "area") and builds an analogy chain
to the topic. That is UNRELATED, not tangential. Do not do this.

================ THREE HARD GATES — A SEARCH MUST PASS ALL THREE ============

GATE 1 — DOMAIN ANCHOR:
The search must stay inside the SAME real-world subject domain as the
question (e.g. if the question is about human personal-hygiene behaviour, the
search must be about human hygiene behaviour, bathroom practices, or directly
related human health behaviour — NOT about industrial surfaces, machine
tribology, chemical exposure modelling, or abstract physics, even if those
share vocabulary like "contact pressure" or "surface area").

GATE 2 — ONE-HOP CONNECTION:
The link from the paper's actual findings to the original question must be a
SINGLE, DIRECT inferential step. If explaining the relevance requires a chain
("this measures X, which could model Y, which lets us infer Z, which relates
to the question"), it FAILS. If you need the words "infer", "model",
"estimate", "extrapolate" stacked together to justify it, reject it.

GATE 3 — DOMAIN-EXPERT TEST:
Would a researcher who actually studies the question's topic nod and say "yes,
that's relevant adjacent evidence" — or would they say "that has nothing to do
with my field"? If the latter, reject it.

================ INDIRECT ROUTE STRATEGIES (within the gates) ===============

Use ONLY those that keep you inside the question's domain:

A. CLASS-LEVEL EVIDENCE — the broader class of the SAME KIND of thing the
   question asks about (same domain, one level up).
B. MECHANISTIC EVIDENCE — the underlying mechanism OF THE SAME PHENOMENON,
   studied within the same field (not a generic physical analogue).
C. ADJACENT POPULATIONS / SETTINGS — a closely related human population or
   setting where findings about the SAME behaviour/outcome plausibly transfer.
D. PROXY OUTCOMES — a validated proxy FOR THE SAME OUTCOME, used in the same
   field — not a loosely analogous measurement from another field.
E. CONVERSE / NEGATIVE EVIDENCE — studies on why the same effect does or does
   not occur, within the same domain.

================ WORKED EXAMPLES ================

GOOD tangential (passes all gates), for a question on a specific medication's
effect on sleep in adults:
  - "sedative class effects on sleep architecture" (class-level, same domain)
  - "insomnia self-report measures validity adults" (proxy outcome, same field)

BAD over-reaching (REJECT), for a question on female toilet-paper folding
behaviour:
  - "contact pressure between rough surfaces" (machine tribology — fails Gate 1)
  - "oil transfer to surface sample media" (industrial contamination — fails 1)
  - "dermal exposure during spraying activities" (occupational chemistry —
    fails Gate 1; the connection to paper folds needs a 4-step chain — fails 2)
A domain expert in hygiene behaviour would not call any of those evidence.

================ YOUR TASK ================

Generate up to 5 focus areas that ALL pass the three gates. It is BETTER to
return FEWER focus areas (even 1-2) that genuinely pass than to pad to 5 with
over-reaching ones. If you cannot find any genuinely in-domain indirect route,
return an empty focus_areas list and say so.

For EACH focus area, in 'connection_to_question', state the SINGLE hop that
connects it to the question, and explicitly note which domain it belongs to so
the gate is auditable.

Respond with ONLY JSON:
{{
    "tangential_strategy": "which in-domain indirect routes you're using and why they pass the gates",
    "search_strategy": "1-2 sentence overall approach for this tangential round",
    "focus_areas": [
        {{"area": "what indirect aspect this targets", "priority": 5, "search_query": "3-8 word search", "connection_to_question": "the SINGLE hop + the domain this belongs to"}}
    ]
}}

If no in-domain indirect route genuinely passes the three gates:
{{"tangential_strategy": "why no valid indirect route exists", "search_strategy": "", "focus_areas": []}}

QUERY RULES:
- Every query must stay in the question's real subject domain (Gate 1).
- Every connection must be a single hop (Gate 2) and survive the expert test (Gate 3).
- Do NOT build analogy chains from shared vocabulary.
- Apply what your strategy memo taught you about phrasing in this field.
- Each query must be substantively different from queries already used."""

        result = self.llm.run_primary(prompt, as_json=True, task="tangential_refine")
        if not result.success or not result.json_response:
            return None

        data = result.json_response
        strategy = data.get("tangential_strategy", "")
        if strategy:
            print(f"  {Fore.MAGENTA}Indirect strategy: {strategy}{Style.RESET_ALL}")

        focus_areas_data = data.get("focus_areas", [])
        if not focus_areas_data:
            print(f"  {Fore.YELLOW}No in-domain indirect routes passed the gates this round.{Style.RESET_ALL}")
            return None

        for fa in focus_areas_data:
            connection = fa.get("connection_to_question", "")
            if connection and isinstance(fa, dict):
                fa["area"] = f"{fa.get('area','?')} [link: {connection}]"

        new_plan = self._parse_plan(
            query,
            {"query_type": "tangential",
             "academic_field": "same",
             "methodology_notes": "",
             "search_strategy": data.get("search_strategy", "Indirect-route search"),
             "focus_areas": focus_areas_data},
            iteration=len(self.all_plans))

        if new_plan and new_plan.focus_areas:
            new_plan.is_tangential = True
            new_plan.identified_scope = current_scope
            self.all_plans.append(new_plan)
            self._display_plan(new_plan, tangential=True)
            return new_plan
        return None

    # =========================================================================
    # DISTILL SEARCH STRATEGY
    # =========================================================================

    def distill_search_strategy(self, query: str,
                                 full_search_history: str,
                                 previous_memo: str = "") -> Dict:
        prev_block = ""
        if previous_memo.strip():
            prev_block = f"""

PREVIOUS STRATEGY MEMO (your prior advice to yourself — evaluate whether it
turned out to be correct based on the new searches):
{previous_memo}
"""
        else:
            prev_block = """

PREVIOUS STRATEGY MEMO: (none — this is the first distillation. You MUST
still produce a starter memo based on the search history below. Even if
patterns are weak, write down what you observed so far so future rounds
have something to evaluate.)
"""

        prompt = f"""You are analysing your own search history to improve your future searches.

RESEARCH QUESTION: "{query}"
{prev_block}

SEARCH HISTORY (every query you've run, what each query returned, how many
papers were selected, and how many survived later relevance filtering):
{full_search_history}

================ YOUR TASK ================

Look at the search history and reason about:

1. WHICH QUERIES WORKED: which phrasings, terminologies, or angles returned
   genuinely relevant papers? What do successful queries have in common?

2. WHICH QUERIES FAILED: which queries returned irrelevant papers, no
   papers, or papers that didn't survive filtering?

3. WHAT THIS TOPIC NEEDS: what kind of search phrasing is effective for
   this specific topic?

4. WHAT TO TRY NEXT: what should future searches do differently?

If you have a previous memo, decide whether its advice held up — keep what
turned out correct, revise what turned out wrong, add new insights.

================ ABSOLUTE OUTPUT REQUIREMENT ================

The "new_memo" field MUST be a non-empty multi-line string containing AT
LEAST 3 numbered points. Empty output is NEVER acceptable. Specifically:

- If your previous memo is still valid, REPEAT it verbatim (you may add
  small refinements). Do not return empty.
- If this is the first distillation and the history is thin, write your
  best initial observations about the most-successful query phrasings and
  the most-failed ones, even if patterns are tentative.
- If absolutely nothing useful can be observed yet, output AT MINIMUM:
  "1. Search history is limited — continue with the current query
   phrasings and re-evaluate after more rounds.
   2. Watch for which phrasings produce hits vs. zero results.
   3. Note which databases (Semantic Scholar, OpenAlex, CORE, Europe PMC,
      Crossref) are most productive for this topic."

Respond with ONLY JSON:
{{
    "new_memo": "Updated multi-line strategy memo for your future self. MUST contain at least 3 numbered points. NEVER empty. Reference specific successful and failed queries when possible.",
    "reasoning": "1-3 sentence summary of why the memo changed (or stayed the same)"
}}"""

        result = self.llm.run_primary(prompt, as_json=True, task="distill_strategy")
        if result.success and result.json_response:
            data = result.json_response
            new_memo = (data.get("new_memo") or "").strip()
            reasoning = (data.get("reasoning") or "").strip()

            if not new_memo:
                if previous_memo.strip():
                    new_memo = previous_memo.strip()
                    reasoning = (reasoning or
                                 "LLM returned empty memo — keeping previous memo unchanged.")
                else:
                    new_memo = (
                        "1. Search history is limited — continue with the current "
                        "query phrasings and re-evaluate after more rounds.\n"
                        "2. Watch which phrasings produce hits vs. zero results so "
                        "future rounds can lean on what works.\n"
                        "3. Note which databases (Semantic Scholar, OpenAlex, CORE, "
                        "Europe PMC, Crossref) are most productive for this topic."
                    )
                    reasoning = (reasoning or
                                 "LLM returned empty memo on first distillation — "
                                 "using starter floor memo.")
            return {
                "new_memo": new_memo,
                "reasoning": reasoning,
                "previous_memo": previous_memo.strip(),
            }
        return {"new_memo": previous_memo,
                "reasoning": "Distillation call failed; keeping previous memo.",
                "previous_memo": previous_memo}

    # =========================================================================
    # READINESS CHECK
    # =========================================================================

    def check_readiness(self, query: str, studies_summary: str) -> Dict:
        print(f"\n{Fore.CYAN}\U0001F50D Checking if ready to write review...{Style.RESET_ALL}")

        prompt = f"""You are evaluating whether enough evidence has been collected to write
a comprehensive academic literature review WITHIN THE SCOPE of the user's
research question.

RESEARCH QUESTION: "{query}"

STUDIES COLLECTED:
{studies_summary}

EVALUATION:

1. Identify the 2-4 specific things a reader would need to know to consider
   this research question well answered (within its scope).
2. For each thing, check whether at least one collected study addresses it
   directly. Tangentially-related studies do not count as fully addressing it.
3. Decide: is the evidence sufficient to write a credible review that
   directly answers the question?

Be honest. If studies are largely tangential, say not ready. If multiple solid
studies cover the core question from different angles, say ready.

Respond with ONLY JSON:
{{
    "ready_to_write": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "name what's covered and what isn't",
    "remaining_gaps": ["specific gap 1", "specific gap 2"]
}}

In the "reasoning" and "remaining_gaps" strings, refer to any study using its
APA 7th in-text citation (the "APA in-text" value shown for each study, e.g.
(Smith et al., 2020), or narratively as Smith et al. (2020)) — do NOT refer to
studies by their number."""

        result = self.llm.run_primary(prompt, as_json=True, task="readiness_check")
        if result.success and result.json_response:
            data = result.json_response
            ready = data.get("ready_to_write", False)
            conf = data.get("confidence", 0)
            if ready:
                print(f"  {Fore.GREEN}\U00002705 Ready ({conf:.0%} confidence){Style.RESET_ALL}")
            else:
                print(f"  {Fore.YELLOW}\U000026A0 Not ready ({conf:.0%}){Style.RESET_ALL}")
                for gap in data.get("remaining_gaps", [])[:3]:
                    print(f"    {Fore.YELLOW}Gap: {gap}{Style.RESET_ALL}")
            return data
        return {"ready_to_write": False, "confidence": 0, "reasoning": "Assessment failed",
                "remaining_gaps": []}

    def tangential_readiness_check(self, query: str, studies_summary: str,
                                    tangential_paper_count: int) -> Dict:
        print(f"\n{Fore.MAGENTA}\U0001F50D Tangential readiness check ({tangential_paper_count} papers collected this engagement)...{Style.RESET_ALL}")

        prompt = f"""You are in TANGENTIAL MODE for a literature review. You have been searching
indirect routes because direct evidence on the user's question is sparse.

RESEARCH QUESTION: "{query}"

PAPERS COLLECTED IN THIS TANGENTIAL ENGAGEMENT: {tangential_paper_count}

STUDIES SO FAR (combined standard + tangential):
{studies_summary}

EVALUATION:

1. Do the indirect-route studies you've gathered so far cover the main angles
   that would inform an answer to the original question?
2. Are there still meaningful indirect routes (class-level, mechanistic,
   adjacent populations, proxy outcomes) you have NOT yet explored that
   would add new informative angles?
3. Is the indirect evidence repeatedly returning the same kinds of findings,
   suggesting diminishing returns from further tangential searching?

Be honest. If a few more rounds would meaningfully expand the indirect
evidence base, say not ready. If you've covered the indirect angles well or
are hitting diminishing returns, say ready.

Respond with ONLY JSON:
{{
    "ready_to_write": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "what tangential angles you've covered well vs. what's still missing or showing diminishing returns",
    "remaining_indirect_routes": ["route 1", "route 2"]
}}

In the "reasoning" and "remaining_indirect_routes" strings, refer to any study
using its APA 7th in-text citation (the "APA in-text" value shown for each
study, e.g. (Smith et al., 2020), or narratively as Smith et al. (2020)) — do
NOT refer to studies by their number."""

        result = self.llm.run_primary(prompt, as_json=True, task="readiness_check")
        if result.success and result.json_response:
            data = result.json_response
            ready = data.get("ready_to_write", False)
            conf = data.get("confidence", 0)
            if ready:
                print(f"  {Fore.GREEN}\U00002705 Tangential evidence sufficient ({conf:.0%}){Style.RESET_ALL}")
            else:
                print(f"  {Fore.YELLOW}\U000026A0 More tangential searches warranted ({conf:.0%}){Style.RESET_ALL}")
                for route in data.get("remaining_indirect_routes", [])[:3]:
                    print(f"    {Fore.MAGENTA}Untried route: {route}{Style.RESET_ALL}")
            return data
        return {"ready_to_write": False, "confidence": 0,
                "reasoning": "Tangential readiness assessment failed",
                "remaining_indirect_routes": []}

    # =========================================================================
    # INTERNALS
    # =========================================================================

    def _parse_plan(self, query: str, data: Dict, iteration: int) -> Optional[ResearchPlan]:
        try:
            focus_areas = []
            for fa_data in data.get("focus_areas", []):
                if not isinstance(fa_data, dict):
                    continue
                area = fa_data.get("area", "")
                if not area:
                    continue
                priority = fa_data.get("priority", 3)
                if isinstance(priority, str):
                    try:
                        priority = int(priority)
                    except ValueError:
                        priority = 3
                priority = max(1, min(5, priority))

                search_query = fa_data.get("search_query", "")
                if isinstance(search_query, list):
                    search_query = search_query[0] if search_query else ""

                focus_areas.append(FocusArea(area=area, priority=priority,
                                             search_query=search_query))
            if not focus_areas:
                return None
            focus_areas.sort(key=lambda x: x.priority, reverse=True)
            return ResearchPlan(
                original_query=query, query_type=data.get("query_type", "broad"),
                academic_field=data.get("academic_field", "general"),
                focus_areas=focus_areas,
                methodology_notes=data.get("methodology_notes", ""),
                search_strategy=data.get("search_strategy", ""),
                iteration=iteration,
                identified_scope=data.get("identified_scope", ""))
        except Exception as e:
            logger.error(f"Plan parse error: {e}")
            return None

    def _fallback_plan(self, query: str) -> ResearchPlan:
        words = [w for w in query.split() if len(w) > 3]
        fas = [
            FocusArea(area=f"Direct: {query[:50]}", priority=5,
                      search_query=" ".join(words[:5])),
            FocusArea(area=f"Reviews: {query[:50]}", priority=4,
                      search_query=f"{' '.join(words[:3])} systematic review"),
            FocusArea(area=f"Mechanisms: {query[:50]}", priority=3,
                      search_query=f"{' '.join(words[:3])} mechanism"),
            FocusArea(area=f"Outcomes: {query[:50]}", priority=2,
                      search_query=f"{' '.join(words[:3])} outcomes"),
            FocusArea(area=f"Recent: {query[:50]}", priority=1,
                      search_query=f"{' '.join(words[:3])} recent findings"),
        ]
        return ResearchPlan(original_query=query, query_type="broad",
                            academic_field="general", focus_areas=fas,
                            methodology_notes="Fallback", search_strategy="Direct query",
                            iteration=0)

    def _display_plan(self, plan: ResearchPlan, tangential: bool = False):
        label = f" (Iteration {plan.iteration})" if plan.iteration > 0 else ""
        marker = " — TANGENTIAL MODE" if tangential else ""
        color = Fore.MAGENTA if tangential else Fore.CYAN
        print(f"\n{color}\U0001F4CB Research Plan{label}{marker}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}Field: {plan.academic_field} | Type: {plan.query_type}{Style.RESET_ALL}")
        if plan.search_strategy:
            print(f"  {Fore.BLUE}Strategy: {plan.search_strategy}{Style.RESET_ALL}")
        for fa in plan.focus_areas:
            c = Fore.GREEN if fa.priority >= 4 else Fore.YELLOW if fa.priority >= 2 else Fore.WHITE
            print(f"  {c}P{fa.priority}: {fa.area}{Style.RESET_ALL}")
            if fa.search_query:
                print(f"  {Fore.BLUE}    \U0001F50E {fa.search_query}{Style.RESET_ALL}")

    def record_query_used(self, query: str):
        with self._query_lock:
            if query not in self.all_search_queries_used:
                self.all_search_queries_used.append(query)

    def record_focus_area_used(self, area: str):
        if area not in self.all_focus_areas_used:
            self.all_focus_areas_used.append(area)
