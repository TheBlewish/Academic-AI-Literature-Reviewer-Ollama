# Low-End Device Branch

Runs the full pipeline on a modest machine — an 8GB GPU with a 7B/8B model at Q4
— with **no features removed**.

> This README only covers what is **different** on this branch. For what the
> program does, how it works, the full feature list, project structure and the
> general configuration reference, see the
> [main branch README](https://github.com/TheBlewish/Academic-AI-Literature-Reviewer-Ollama).

Every stage still runs: research planning, multi-round discovery across the
academic APIs, quick reads, deep analysis with verbatim quote verification,
methodology assessment, evidence curation, two-phase synthesis, self-review,
self-fix, verification and Q&A.

---

## Quick start

```bash
# Ollama + a small model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b

git clone -b low-end-mode https://github.com/TheBlewish/Academic-AI-Literature-Reviewer-Ollama.git
cd Academic-AI-Literature-Reviewer-Ollama

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # put your real contact email in it
set -a; source .env; set +a

python Academic_Researcher.py
```

Low-end mode is **already on** — no config editing needed to get started.

Setting a real contact email matters more than it sounds: Unpaywall rejects
`example.com` addresses, and Unpaywall is a main route to full-text PDFs. See
the main README for detail.

---

## What is different from main

Two files differ: `academic_config.py` and `study_analyser.py`.

| | main | this branch |
|---|---|---|
| `low_end_device_mode` | `"no"` | **`"yes"`** |
| Context window | 65536 | **16384** |
| Deep analysis | whole paper, one call | **sequential chunks** |
| Per-call output budget | up to 16384 | **capped at 3072** |
| Thinking blocks | per profile | **off** |
| Search results per API per round | 20 × 3 pages | **5 × 1 page** |
| Corpus ceiling | 200 papers | **40 papers** |

### Why each change exists

**Chunked deep analysis.** `deep_analysis` sent up to 50,000 characters
(~16,700 tokens) in one prompt. On a small window Ollama silently truncated it,
so the model saw the abstract, introduction and start of the methods and then
ran out — while being asked to quote from the Results, Discussion and Conclusion
it never received. Papers are now read in sequential overlapping chunks, in
reading order, with the whole paper covered.

Measured on a 42,967-character paper at `n_ctx = 8192`:

| | Before | After |
|---|---|---|
| Verified quotes | **0** | **42** |
| Discussion quote found | No | Yes |
| Conclusion quote found | No | Yes |
| Survives `min_verified_quotes_per_study: 1` | **No — paper dropped** | Yes |

**Capped output budgets.** `num_predict` and `num_ctx` come out of the same
window. Nine task profiles requested output budgets at or beyond a small window,
leaving zero or negative room for the prompt — the model generating into a
window that had evicted its own input:

| Profile | Effective output | Input room at 8K |
|---|---|---|
| `synthesis` | 32,768 | −26,624 |
| `curate_evidence` | 24,576 | −18,432 |
| `verification`, `self_fix` | 16,384 | −10,240 |
| `methodology_assessment` | 8,192 | −2,048 |

`get_task_profile()` now caps `max_tokens`, forces `think=False` and caps
`num_ctx` — only when low-end mode is on. It is the single chokepoint every LLM
call passes through, so this fixes all call sites without touching
`llm_manager.py`.

**Reduced search paging.** Paper selection shows the model every candidate from
a search round in one call. A full-mode round pulled ~120 candidates (~28,000
tokens). Paging is now capped at the source, which also cuts wall-clock time and
API load a lot on a slow machine. Discovery quality is preserved by the round
structure: fewer candidates per round, same number of rounds.

---

## Quote integrity is unchanged

Chunking changes **which passages the model proposes**. It cannot affect whether
a surviving quote is real.

The `DocumentStore` still holds the **complete** paper text, and every proposed
quote is still matched verbatim against that complete text at
`quote_match_min_similarity: 0.95`. The `[[Qn]]` token system means the model
never retypes a quote — code pastes the verified string. Section gating
(Results/Discussion/Conclusion only) and second-hand citation rejection also run
unchanged, against the full text.

The test suite asserts this directly: every verified quote is checked to be a
literal substring of the source paper.

---

## Configuration

Everything lives in `LOW_END_CONFIG` in `academic_config.py`.

### Two verified operating points

Every stage was measured against the real dynamic context sizer in
`llm_manager.py`. Neither of these truncates at any stage.

**DEFAULT — 8GB GPU, 7B/8B at Q4** (what ships):

```python
"n_ctx": 16384,
"max_output_tokens": 3072,
"max_total_papers": 40,
"search_results_per_page": 5,
```

A 7B at Q4_K_M is about 4.5GB of weights; a 16K KV cache adds roughly 1–2GB, so
this fits an 8GB card. Prefer it — the holistic stages (methodology assessment,
evidence curation) need room to see the whole corpus at once, and that is where
review quality comes from.

**MINIMAL — very tight memory, or a model you can only run at 8K:**

```python
"n_ctx": 8192,
"max_output_tokens": 2048,
"max_total_papers": 15,
"search_results_per_page": 3,
```

Also verified end to end. The corpus has to drop because methodology assessment
grades every study in one holistic call.

**Change all four together — they were measured as a set.**

### Model choice

| Model | Notes |
|---|---|
| `qwen3:8b` | Ships as the default. Strong instruction-following and reliable JSON, which this pipeline lives on. |
| `qwen3:4b` | Noticeably faster, still holds the JSON contract. Use if 8b is too slow. |
| `llama3.1:8b` | Solid alternative if you prefer the Llama family. |

Verbatim quoting is the one thing a small model must do well here. If your
verified-quote counts look low, try a **larger quant (Q5_K_M) before a different
model** — quantisation hurts exact reproduction more than parameter count does.

Set the model with `LOW_END_OLLAMA_MODEL` / `LOW_END_OLLAMA_BASE_URL`, or edit
`LOW_END_CONFIG` directly.

### Speed

Chunking means more calls: a 43,000-character paper becomes about 9 calls
instead of 1. That is the cost of actually reading the paper. Expect a full
review to take hours on this class of hardware.

`"max_chunks_per_paper"` bounds calls per paper. It defaults to `0`
(unlimited — read everything, which is the point of this mode). When you do set
it, chunks are sampled **evenly across the paper with the first and last always
kept**, so the abstract and conclusion are never what gets dropped.

Reducing `max_discovery_rounds` and `max_total_papers` saves far more time than
capping chunks, and costs less quality.

---

## Switching back to full mode

Set `"low_end_device_mode": "no"`. Nothing else in `LOW_END_CONFIG` is read,
`PRIMARY_LLM_CONFIG` takes over completely, and behaviour matches main. You do
not need to switch branches if you later move to a bigger machine.

Verified with the toggle off: `TASK_PROFILES` and search paging are identical to
main, `get_research_config()` returns the *identical object*, deep analysis
makes exactly one call with the whole paper uncut, and the prompt renders
byte-identically to main's.

---

## Tests

```bash
python test_low_end.py
```

41 assertions: the chunk splitter (coverage, boundaries, overlap, termination on
pathological input), cap sampling, end-to-end chunked deep analysis on a 43K
paper, context-window fit at 8K and 16K, output-budget caps, resilience (a
failed chunk, an exception mid-chunk, total failure) and the full-mode
regression.

---

## Known limits on this hardware class

- Quality tracks model capability. An 8B completes the pipeline but writes a
  weaker review than a 30B-class model — curation, methodology assessment and
  synthesis are where it shows.
- The corpus is smaller by design (40 papers, not 200), because the holistic
  curation and assessment calls have to fit the window.
- A full review is slow. Chunking makes deep analysis several times more work
  than in full mode.
- The code paths are tested, but throughput on any specific GPU is not something
  the tests can measure. Start with a narrow question and a low
  `max_discovery_rounds` for your first run.
