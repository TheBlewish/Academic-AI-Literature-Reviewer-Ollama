# academic_config.py
# Configuration for Academic Literature Review System
# Designed for 128GB VRAM system running Ollama
#
# SINGLE MODEL ARCHITECTURE (this version):
#   PRIMARY MODEL: the one model used for all phases — planning, searching,
#                  analysis, synthesis, verification.
#   The parallel search-agent feature is DORMANT in this version. Its config
#   was removed from this file; the accessor functions below still return safe
#   defaults so the dormant scaffolding in llm_manager.py keeps working and can
#   be re-enabled later without code changes.

import os

# =============================================================================
# PRIMARY LLM CONFIG (Analysis, Planning, Synthesis, Verification)
# =============================================================================

# ---- SET THESE TWO VALUES BEFORE YOUR FIRST RUN ----
#
# base_url : where your Ollama server is listening.
#     * Ollama running on THIS machine   -> "http://localhost:11434"  (default)
#     * Ollama on another box on your LAN -> "http://192.168.1.50:11434"
#     * Ollama over Tailscale/VPN         -> "http://your-host.your-tailnet.ts.net:11434"
#   You can also leave this alone and set the OLLAMA_BASE_URL environment
#   variable instead, which takes precedence:
#       export OLLAMA_BASE_URL="http://192.168.1.50:11434"
#
# model_name : the EXACT tag of a model you have already pulled in Ollama.
#   Check what you have with:  ollama list
#   This program is context-hungry. A model with a large context window and
#   strong reasoning is strongly recommended — a 30B-class model at ~64K
#   context is the sweet spot on a high-memory machine. See the README for
#   sizing guidance and for the low-end (small machine) mode below.
PRIMARY_LLM_CONFIG = {
    "llm_type": "ollama",
    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "model_name": os.environ.get("OLLAMA_MODEL", "qwen3.5:35b"),
    "temperature": 0.1,
    "top_p": 0.9,
    "n_ctx": 65536,
    "max_tokens": 8192,
    "stop": [],
}

# =============================================================================
# THINKING MODEL SETTINGS
# =============================================================================

THINKING_CONFIG = {
    "show_model_thinking": True,
    "thinking_token_multiplier": 2,

    # --- Thinking vs non-thinking model handling ---
    # The program supports BOTH reasoning ("thinking") models that emit a
    # <think>...</think> block (e.g. qwen3, deepseek-r1, qwq) AND plain
    # non-thinking models that answer directly. It auto-detects which kind the
    # configured model is and adapts seamlessly:
    #   * thinking models  -> the `think` flag is sent to Ollama, the output
    #     budget is multiplied (thinking_token_multiplier), <think> blocks are
    #     stripped from the answer, and the truncated-inside-think retry applies.
    #   * non-thinking models -> none of that overhead is applied; the model is
    #     driven as a direct-answer model (a task profile that requests
    #     think:True is simply ignored because the model cannot think).
    #
    # auto_detect=True queries the Ollama model's real capabilities first
    # (/api/show), then falls back to known name patterns. Set force_thinking to
    # True or False to override detection entirely; leave it None for auto.
    "auto_detect": True,
    "force_thinking": None,
}

# =============================================================================
# LOW-END DEVICE MODE  (chunked execution for small / low-memory machines)
# =============================================================================
# The FULL program is unchanged by default. Set low_end_device_mode to "yes" to
# run complete reviews on a small-context device: every LLM call is capped to
# n_ctx below, and the two content-heavy stages are chunked so their INPUT fits:
#   * deep_analysis  — a long paper is split into SEQUENTIAL chunks (in reading
#     order, NO similarity selection) and quotes are extracted from each chunk,
#     then accumulated. Quote verification is unchanged (against the full text).
#   * synthesis      — studies are written up in BATCHES (map step), then the
#     batch drafts are assembled into the final APA review (reduce step). Quotes
#     remain [[Qn]] tokens throughout, so nothing is ever retyped and the
#     deterministic compile-time paste/verification is identical to full mode.
# The review-checking stages (self-review / self-fix / verification) are fed a
# COMPACT token index instead of the full quote text — safe because quote
# integrity is already guaranteed deterministically by the token system.
#
# Tradeoff (stated honestly): to keep the discovery-stage holistic calls
# (curation / sufficiency, which view all paper summaries at once) within a
# small window, low-end mode also caps the corpus size below. A small device
# therefore reviews a focused corpus rather than a 200-paper sweep. Raise these
# caps if your small device has more headroom.
LOW_END_CONFIG = {
    # MASTER TOGGLE — simple yes/no. "no" (default) = full behaviour, unchanged,
    # for the big machine. "yes" = chunked low-end mode for a small device.
    "low_end_device_mode": "no",

    # Context window of the small device's model. Every call is capped here and
    # the chunk budgets below keep each call's input comfortably under it.
    # 8192 is a safe value for a 16GB mini-PC / laptop class machine running a
    # ~14B model at a low quant, which is fast at 8K. Raise to 16384 if your
    # small device has more headroom — the chunk sizes below still fit a 16K
    # window.
    "n_ctx": 8192,

    # OPTIONAL endpoint / model overrides for the small device. Leave BLANK ("")
    # to keep the PRIMARY_LLM_CONFIG values (e.g. if the same Ollama endpoint
    # serves the small model). Fill these if the small device differs.
    "base_url": "",
    "model_name": "",

    # ---- Chunk budgets (characters; ~3 chars/token) ----
    # deep_analysis: sequential paper chunk size + overlap between chunks.
    # Sized for an 8K window: ~6000 chars (~2000 tok) + prompt scaffolding
    # (~470 tok) + the 2048-tok output budget + safety/headroom all fit under
    # 8192 with room to spare (computed need ~7200 tok). A 500-char overlap
    # keeps sentences that straddle a chunk boundary recoverable in one chunk.
    # If you raise n_ctx above to 16384 you may raise this to 9000 to read each
    # paper in fewer passes.
    "deep_analysis_chunk_chars": 6000,
    "deep_analysis_chunk_overlap": 500,
    # synthesis MAP: max evidence characters per study-batch.
    "synthesis_map_batch_chars": 6000,
    # synthesis REDUCE: max combined draft characters per assembly group; if the
    # batch drafts exceed this they are merged hierarchically in groups first.
    "synthesis_reduce_group_chars": 7000,

    # ---- Corpus caps so discovery-stage holistic calls fit the small window ----
    "max_total_papers": 60,
    "target_papers_per_focus_area": 3,
}

# =============================================================================
# TASK PROFILES — per-task context size, token budget, and thinking mode
# =============================================================================

TASK_PROFILES = {
    # --- Planning-style structured-JSON tasks ---
    "planning": {
        "num_ctx": 16384,
        "max_tokens": 2048,
        "think": False,
    },
    "refine_planning": {
        "num_ctx": 49152,
        "max_tokens": 2048,
        "think": False,
    },
    "readiness_check": {
        "num_ctx": 32768,
        "max_tokens": 1024,
        "think": False,
    },
    "paper_selection": {
        "num_ctx": 32768,
        "max_tokens": 1024,
        "think": False,
    },

    # --- Per-paper reading ---
    "quick_read": {
        "num_ctx": 16384,
        "max_tokens": 2048,
        "think": False,
    },
    "deep_analysis": {
        "num_ctx": 49152,
        "max_tokens": 4096,
        "think": False,
    },
    # Low-end sibling of deep_analysis. Used ONLY when low_end_device_mode is on,
    # for the per-chunk quote extraction in study_analyser.deep_analysis(). The
    # num_ctx here is just a ceiling; the model's hard window (low_end n_ctx,
    # e.g. 8192) caps it lower anyway. The output budget is small (2048) so that
    # one ~6000-char chunk + the prompt scaffolding + this output all fit inside
    # an 8K window without the no-truncation guarantee having to push the window
    # above the device's limit. think is off (a small instruct model on a low-end box
    # should not burn the tiny budget on a <think> block). This profile is NEVER
    # consulted in full mode — deep_analysis only requests it under low-end.
    "deep_analysis_low_end": {
        "num_ctx": 8192,
        "max_tokens": 2048,
        "think": False,
    },

    # --- Methodology assessment ---
    "methodology_assessment": {
        "num_ctx": 32768,
        "max_tokens": 8192,
        "think": False,
    },

    # --- Synthesis and verification (full context + thinking) ---
    "synthesis": {
        "num_ctx": 65536,
        "max_tokens": 16384,
        "think": True,
    },
    # CHANGED (think True -> False): with thinking ON, Ollama's num_predict
    # budget is shared by the <think> block AND the answer. On this large-input
    # tail stage (~39K-token input in a 65K window) the think block exhausted
    # the budget — and there was no room left for the auto-retry to raise it —
    # leaving an empty answer (the Phase 12 hang). Verification is a checking
    # task, not search-refinement reasoning, so disabling thinking is safe and
    # gives the full 16384 budget to the actual answer.
    "verification": {
        "num_ctx": 65536,
        "max_tokens": 16384,
        "think": False,
    },

    # --- Small finishing tasks ---
    "title_generation": {
        "num_ctx": 8192,
        "max_tokens": 300,
        "think": False,
    },
    "qa_mode": {
        "num_ctx": 32768,
        "max_tokens": 4096,
        "think": True,
    },

    # --- Relevance filter ---
    "relevance_filter": {
        "num_ctx": 49152,
        "max_tokens": 2048,
        "think": False,
    },

    # --- Self-review of the produced literature review ---
    # CHANGED (think True -> False): same shared-budget issue as verification.
    # Self-review only survived in the failing run via the budget-raising retry
    # (8192 -> 23951) and is fragile as input grows. It emits a compact issues
    # JSON, so the full 4096 answer budget is more than enough without thinking.
    "self_review": {
        "num_ctx": 65536,
        "max_tokens": 4096,
        "think": False,
    },

    # --- Self-fix — rewrites the review based on identified issues ---
    # CHANGED (think True -> False): this is the task that ENDED the failing run.
    # With think ON it ran ~16 min, LOOPED inside the <think> block (repeating
    # the same line), hit num_predict=32768 with a 0-char answer, and — because
    # the ~40K input left <24K of window — the auto-retry could not raise the
    # budget, so it failed permanently. Self-fix is mechanical (apply the listed
    # issues while preserving [[Qn]] tokens); thinking was actively harmful here.
    # Disabling it frees the full 16384 budget for the rewrite and removes the
    # loop. Fits the window with no truncation (~40K in + 16384 out + headroom).
    "self_fix": {
        "num_ctx": 65536,
        "max_tokens": 16384,
        "think": False,
    },

    # --- Strategy distillation — reasons about search effectiveness ---
    "distill_strategy": {
        "num_ctx": 65536,
        "max_tokens": 4096,
        "think": True,
    },

    # --- Evidence sufficiency decision (3-way) ---
    "evidence_sufficiency": {
        "num_ctx": 65536,
        "max_tokens": 3072,
        "think": True,
    },

    # --- Tangential-mode refinement (one-shot, indirect routes) ---
    "tangential_refine": {
        "num_ctx": 49152,
        "max_tokens": 2048,
        "think": True,
    },

    # --- Curate evidence — standard mode ---
    # max_tokens raised 8192 -> 12288 (num_predict 24576 with the x2 thinking
    # multiplier) so a verbose thinking model can finish its reasoning AND emit
    # the decisions JSON in the first attempt. Still fits the 65K window with
    # room to spare. The empty/truncated-response retry in llm_manager.py is the
    # safety net if even this is exceeded.
    #
    # NOTE: This whole-batch profile is retained for backward compatibility but
    # is NO LONGER USED by node_curate_evidence, which now curates ONE paper per
    # call (see "curate_study" below). Curating all papers in a single call is
    # what exhausted the output budget inside the thinking block on large sets.
    "curate_evidence": {
        "num_ctx": 65536,
        "max_tokens": 12288,
        "think": True,
    },

    # --- Curate evidence — PER-STUDY (sequential) ---
    # node_curate_evidence now decides INCLUDE/EXCLUDE one paper at a time, with
    # a quick-read-style UI (blue while deciding, green INCLUDE, yellow EXCLUDE,
    # with an ETA). Each call sees only the single paper under review plus a
    # compact one-line catalog of the other candidates (for redundancy
    # judgement), so the context is small and the output budget can never be
    # exhausted the way the whole-batch call could. Fast, no thinking budget.
    "curate_study": {
        "num_ctx": 16384,
        "max_tokens": 1024,
        "think": False,
    },

    # --- Tangential curation ---
    "tangential_curate_evidence": {
        "num_ctx": 65536,
        "max_tokens": 12288,
        "think": True,
    },
}

# =============================================================================
# ACADEMIC SEARCH API CONFIG (all FREE)
# =============================================================================
#
# ---- YOU MUST SUPPLY YOUR OWN EMAIL ADDRESS BELOW ----
#
# OpenAlex, Unpaywall and Crossref are free and need no signup, but they ask
# that you identify yourself with a contact email so they can reach you if your
# script misbehaves. Unpaywall REJECTS requests from "example.com" outright, so
# leaving the placeholder in will disable full-text discovery via Unpaywall.
#
# Use ANY real address you own — a personal Gmail/Outlook/Proton address is
# fine, it does not have to be a university one. Two ways to set it:
#
#   1. Environment variables (recommended — keeps your address out of the repo,
#      so you can never accidentally commit it):
#         export OPENALEX_EMAIL="you@yourdomain.com"
#         export UNPAYWALL_EMAIL="you@yourdomain.com"
#         export CROSSREF_EMAIL="you@yourdomain.com"
#
#   2. Or just edit the three "your.email@example.com" strings below directly.
#      If you do this, be careful not to commit the change back to a public
#      repository.
#
# ---- OPTIONAL API KEYS ----
#
# Both of these are optional. The program works without them; they simply
# raise your rate limits / unlock an extra source.
#
#   SEMANTIC_SCHOLAR_API_KEY — free key, request one at:
#         https://www.semanticscholar.org/product/api
#      Without a key, Semantic Scholar is heavily rate-limited and the program
#      will disable it after repeated 429s (this is expected, not a bug).
#
#   CORE_API_KEY — free key, register at:
#         https://core.ac.uk/services/api
#      Without a key, the CORE source is skipped entirely.
#
# Set them the same way:
#         export SEMANTIC_SCHOLAR_API_KEY="..."
#         export CORE_API_KEY="..."
#
# NEVER paste an API key directly into this file if you intend to push your
# copy of the repo anywhere public.

SEARCH_APIS = {
    "semantic_scholar": {
        "enabled": True,
        "base_url": "https://api.semanticscholar.org/graph/v1",
        "api_key": os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        "fields": "title,authors,year,abstract,externalIds,openAccessPdf,citationCount,venue,publicationTypes,publicationDate,journal",
        "results_per_page": 20,
        "max_pages": 3,
        "rate_limit_delay": 1.0,
    },
    "openalex": {
        "enabled": True,
        "base_url": "https://api.openalex.org",
        "email": os.environ.get("OPENALEX_EMAIL", "your.email@example.com"),
        "results_per_page": 25,
        "max_pages": 3,
        "rate_limit_delay": 0.2,
    },
    "core": {
        "enabled": True,
        "base_url": "https://api.core.ac.uk/v3",
        "api_key": os.environ.get("CORE_API_KEY", ""),
        "results_per_page": 25,
        "max_pages": 2,
        "rate_limit_delay": 2.5,
    },
    "unpaywall": {
        "enabled": True,
        "base_url": "https://api.unpaywall.org/v2",
        "email": os.environ.get("UNPAYWALL_EMAIL", "your.email@example.com"),
        "rate_limit_delay": 0.1,
    },
    "europe_pmc": {
        "enabled": True,
        "base_url": "https://www.ebi.ac.uk/europepmc/webservices/rest",
        "results_per_page": 25,
        "max_pages": 3,
        "rate_limit_delay": 0.15,
    },
    "crossref": {
        "enabled": True,
        "base_url": "https://api.crossref.org",
        "email": os.environ.get("CROSSREF_EMAIL", "your.email@example.com"),
        "results_per_page": 20,
        "max_pages": 2,
        "rate_limit_delay": 0.15,
    },
}

# =============================================================================
# RESEARCH PIPELINE CONFIG
# =============================================================================

RESEARCH_CONFIG = {
    # Post-review interactive Q&A. True (default) keeps the existing behaviour:
    # after a review is written you can ask follow-up questions about it. False:
    # the program skips Q&A on completion and returns to the research-question
    # prompt (type 'quit' there to exit).
    "qa_mode_enabled": True,
    "num_focus_areas": 5,
    "max_search_terms_per_area": 4,
    # With curation gating, more rounds may run productively
    "max_discovery_rounds": 15,
    "min_papers_before_sufficiency_check": 5,
    "target_papers_per_focus_area": 5,
    "max_total_papers": 200,
    "dedup_by_doi": True,
    "papers_directory": "Papers",
    "max_pdf_download_retries": 3,
    "pdf_download_timeout": 30,
    "accept_abstract_only": True,    # legacy flag; the new gate is below
    "prefer_open_access": True,
    "max_study_text_length": 50000,
    "extract_key_quotes_per_study": 8,
    "max_quote_correction_attempts": 2,
    "min_chars_for_deep_analysis": 5000,

    # --- NEW: Variable quote count (no fixed per-study target) ---
    # deep_analysis() now extracts EVERY directly-relevant quote rather than a
    # fixed number. This is an OPTIONAL upper safety bound only:
    #   0  = unlimited (default; the LLM uses as many or as few as are relevant)
    #   >0 = never extract more than this many quotes from a single study
    # The legacy keys extract_key_quotes_per_study / _per_abstract above and
    # below are retained for backward compatibility but no longer drive the
    # extraction prompt.
    "max_quotes_per_study_hard_cap": 0,

    # --- NEW: Quote-placeholder synthesis settings ---
    # The synthesis stage no longer has the LLM retype quotes. Each verified
    # quote gets a stable token like [[Q7]]; the model places the token and the
    # exact verified text is pasted in deterministically at compile time. A
    # quote of this many words or more is rendered as an APA block quote when
    # the model places its token alone on a line.
    "block_quote_word_threshold": 40,
    # TWO-PHASE EVIDENCE SYNTHESIS (default on). When True, the EVIDENCE section
    # is built by code from an LLM-produced PLAN (thematic grouping + chosen quote
    # order) rather than written as free prose: the model emits only study/quote
    # NUMBERS, so it can never duplicate or corrupt the evidence text. Short
    # navigational connectors are then added in a separate, locked pass, and the
    # analysis sections are written in a SEPARATE call with a fresh context.
    # Set False to fall back to the original single-shot prose synthesis.
    "two_phase_synthesis": True,
    # Add short LLM navigational connectors between evidence blocks (locked pass).
    # If False, the evidence is assembled with no connective sentences at all.
    "evidence_connectors_enabled": True,
    # If True (default), the REFERENCES section is rebuilt deterministically at
    # compile time to contain exactly the studies whose quotes were actually
    # used (plus any study still cited in the body), so studies the LLM chose
    # not to quote are dropped from the review entirely. Set False to keep the
    # LLM-authored references verbatim.
    "rebuild_references_from_used_quotes": True,

    # --- NEW: Main-mode full-text gate ---
    # In main (non-tangential) mode, papers must have full text to make it
    # into the final synthesis. Abstract-only papers are dropped at
    # deep_analysis time. In tangential mode this flag is overridden and
    # abstract-only papers ARE allowed (with abstract-quote verification).
    "require_full_text_main_mode": True,

    # --- NEW: Tangential-mode abstract-quote extraction ---
    # When tangential mode allows abstract-only papers, the abstract is
    # loaded into the doc store as the verification source. Quotes are
    # verified against the abstract text via TextMatcher.
    "min_chars_for_abstract_quote_extraction": 150,
    "extract_key_quotes_per_abstract": 3,

    # --- NEW: Verified-quote requirement per study ---
    # A paper is only retained for the final synthesis if it has at least
    # this many verified quotes. Set to 0 to keep all papers regardless.
    "min_verified_quotes_per_study": 1,

    # Strategy distillation schedule
    # CHANGED: first distillation now runs after round 1 (was 2). With
    # distill_every_n_rounds=1 this means distillation runs every round.
    "first_distill_after_round": 1,
    "distill_every_n_rounds": 1,

    # Curation schedule
    # Curation runs every N search rounds OR when readiness check says ready.
    "curate_every_n_rounds": 5,

    # Tangential mode thresholds (PER ENGAGEMENT)
    "tangential_paper_target": 75,
    "tangential_round_cap": 100,
    "tangential_curate_every_n_rounds": 5,
    "tangential_min_curated_papers": 15,

    "study_type_weights": {
        "systematic_review": 10, "meta_analysis": 10,
        "randomized_controlled_trial": 8, "cohort_study": 6,
        "case_control_study": 5, "cross_sectional_study": 4,
        "case_report": 2, "expert_opinion": 1,
    },
    # --- (#2) Methodology gap-fill (one-time, weakness-targeted) ---
    # When the comprehensive methodology assessment judges the evidence too weak
    # to confidently answer the question AND names fillable gaps, run ONE
    # targeted gap-fill pass (one search per weakness), preserving all previously
    # verified quotes and deep-analysing only the new studies, then re-assess.
    "methodology_gap_fill_enabled": True,
    # Hard cap on how many weaknesses get a search in the single gap-fill pass.
    "methodology_gap_max_weaknesses": 6,
    "review_max_tokens": 16384,
    # --- Verification convergence (no-give-up loop) ---
    # The pipeline now NEVER ships a review with a critical issue. It loops:
    #   * the first `verification_holistic_attempts` failed passes do a full
    #     holistic rewrite (given every issue + a refinement memory + the exact
    #     offending snippets annotated with how to fix them);
    #   * after that it switches to SURGICAL mode — fixing ONE issue at a time,
    #     showing the model the precise span causing it and instructing it to
    #     change only that, re-verifying after each, until the review is clean.
    # `verification_surgical_max_passes` is a generous absolute safety cap so the
    # loop cannot run forever; if it is ever hit with issues remaining the run
    # stops and reports FAILED loudly (it does not pretend success).
    # `verification_require_zero_moderate` keeps the loop going on moderate
    # issues too (aim for a fully issue-free review); criticals ALWAYS block.
    # Holistic rewrites regenerate the WHOLE review from scratch, which tends to
    # multiply issues before resolving any. Default 0 = go STRAIGHT to surgical
    # (edit in place, one issue at a time). Raise this only if you specifically
    # want full-rewrite passes first.
    "verification_holistic_attempts": 0,
    "verification_surgical_max_passes": 15,
    # Criticals are the hard gate (a critical NEVER ships). Moderates are
    # advisory: chasing debatable LLM-raised moderates caused oscillation, so by
    # default we ship once the review is critical-free. Set True to also block on
    # moderate issues.
    "verification_require_zero_moderate": False,
    # Surgical mode edits a small WINDOW around each error (this many words on
    # each side of the offending span) and splices the fix back into the full
    # review, instead of re-emitting the whole document — faster and far less
    # prone to drift. If a window fix fails this many times on the same issue it
    # escalates to a whole-document single-issue fix.
    "surgical_window_words": 150,
    "surgical_window_retry_cap": 2,
    # Legacy key, retained for backward compatibility. No longer used to stop the
    # loop early (the holistic/surgical settings above govern convergence).
    "verification_max_retries": 2,
    "show_search_progress": True,
    "show_api_calls": True,
    "show_reasoning": True,
    "verbose": True,
}

# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

DOCUMENT_CONFIG = {
    "supported_formats": [".txt", ".pdf", ".md", ".docx", ".html"],
    "max_file_size_mb": 100,
    "encoding": "utf-8",
    "preserve_formatting": True,
    "normalize_whitespace": False,
}

RAG_CONFIG = {
    "chunk_size": 4000,
    "chunk_overlap": 500,
    "top_k_chunks": 50,
    "min_relevance_score": 0.05,
    "max_search_iterations": 10,
    "max_chunks_per_search": 50,
    "bm25_top_k": 20,
    "max_quote_retry_attempts": 3,
    "quote_similarity_threshold": 0.80,
    "show_character_diff_on_failure": True,
    "require_exhaustive_search": True,
    "min_search_terms_per_iteration": 3,
    "show_reasoning": True,
    "show_search_progress": True,
    "verbose_verification": True,
}

LOG_CONFIG = {
    "log_directory": "Logs",
    "log_level": "INFO",
    "log_llm_prompts": True,
    "log_llm_responses": True,
    "log_api_calls": True,
}

PATHS_CONFIG = {
    "papers_directory": "Papers",
    "logs_directory": "Logs",
    "output_directory": "Reviews",
}

# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================

def get_primary_llm_config():
    # In low-end mode, cap the model context window (so the dynamic sizer caps
    # every call) and optionally redirect to the small device's endpoint/model.
    if is_low_end_enabled():
        cfg = dict(PRIMARY_LLM_CONFIG)
        le = LOW_END_CONFIG
        if le.get("n_ctx"):
            cfg["n_ctx"] = le["n_ctx"]
        if le.get("base_url"):
            cfg["base_url"] = le["base_url"]
        if le.get("model_name"):
            cfg["model_name"] = le["model_name"]
        return cfg
    return PRIMARY_LLM_CONFIG

def get_agent_llm_config():
    # Parallel agents are dormant; if ever re-enabled they fall back to the
    # primary model so behaviour is well-defined without a separate agent model.
    return PRIMARY_LLM_CONFIG

def get_llm_config():
    return PRIMARY_LLM_CONFIG

def get_thinking_config():
    return THINKING_CONFIG

def get_parallel_config():
    # Dormant defaults — kept so llm_manager.py's parallel scaffolding stays
    # functional and can be re-enabled later without config changes.
    return {
        "num_search_agents": 2,
        "agent_timeout": 600,
        "agent_start_delay": 2.0,
    }

def get_search_api_config(api_name=None):
    if api_name:
        return SEARCH_APIS.get(api_name, {})
    return SEARCH_APIS

def get_research_config():
    # In low-end mode, cap the corpus so discovery-stage holistic calls fit the
    # small window, and propagate the chunking flags/sizes that study_analyser
    # and the pipeline read from the research config. Returns a MERGED COPY so
    # the module-level RESEARCH_CONFIG is never mutated; full mode returns the
    # original object unchanged.
    if is_low_end_enabled():
        cfg = dict(RESEARCH_CONFIG)
        le = LOW_END_CONFIG
        for k in ("max_total_papers", "target_papers_per_focus_area"):
            if le.get(k) is not None:
                cfg[k] = le[k]
        cfg["low_end_device_mode"] = True
        cfg["low_end_n_ctx"] = le.get("n_ctx", 16384)
        cfg["low_end_deep_analysis_chunk_chars"] = le.get("deep_analysis_chunk_chars", 9000)
        cfg["low_end_deep_analysis_chunk_overlap"] = le.get("deep_analysis_chunk_overlap", 500)
        cfg["low_end_synthesis_map_batch_chars"] = le.get("synthesis_map_batch_chars", 9000)
        cfg["low_end_synthesis_reduce_group_chars"] = le.get("synthesis_reduce_group_chars", 11000)
        return cfg
    return RESEARCH_CONFIG


def get_low_end_config():
    return LOW_END_CONFIG


def _truthy_yes_no(value) -> bool:
    """Parse a simple yes/no (also accepts true/1/on/enabled) into a bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("yes", "y", "true", "1", "on", "enabled")


def is_low_end_enabled() -> bool:
    return _truthy_yes_no(LOW_END_CONFIG.get("low_end_device_mode", "no"))

def get_rag_config():
    return RAG_CONFIG

def get_document_config():
    return DOCUMENT_CONFIG

def get_paths_config():
    return PATHS_CONFIG

def get_task_profile(task_name):
    return TASK_PROFILES.get(task_name)
