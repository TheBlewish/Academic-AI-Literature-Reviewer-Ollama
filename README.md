# Academic-AI-Literature-Reviewer-Ollama

## Description

**Academic-AI-Literature-Reviewer** is an automated literature review system that turns a locally run large language model, served through Ollama, into a research assistant that writes real, citable literature reviews.

Unlike a general web researcher, this program does not search the open internet. It searches **academic databases**, downloads the **actual full text** of open-access studies, reads them, judges them on methodological quality, and writes an **APA 7th edition literature review** in which every quotation is verified verbatim against the paper it came from.

The defining feature is that **the model is structurally prevented from fabricating quotations**. Quotes are never retyped by the LLM. Each verified quote is assigned a stable token like `[[Q7]]`; the model places the token, and the exact verified text is pasted in deterministically by code at compile time. A quote that does not match its source at high similarity never enters the document at all. The same applies to study introductions (`[[Sn]]`), so a finding can never be attached to the wrong study.

This is the successor to my earlier project, [Automated-AI-Web-Researcher-Ollama](https://github.com/TheBlewish/Automated-AI-Web-Researcher-Ollama). Where that program scraped the web, this one goes to the primary scientific literature — trading breadth for reliability, verifiability, and academic rigour.

Everything runs locally. No paid APIs, no cloud model, no data leaving your machine except the academic database queries themselves.

## Here's How It Works

1. You provide a research question (e.g. *"What are the health impacts of plastic water bottle use?"*).
2. The LLM analyses the question and generates focus areas with prioritised search queries.
3. It searches six academic databases in parallel, deduplicates by DOI, and **selects** which papers are worth pursuing rather than grabbing everything.
4. It acquires the **full text** of the selected papers through redundant open-access channels — Unpaywall, CORE, Europe PMC, arXiv, and publisher-hosted open-access PDFs.
5. It performs a fast **quick read** of each paper to extract methodology, sample size, study type, and key findings, then filters out papers that turned out to be irrelevant.
6. It runs a **gap analysis**: re-shown your original question, it decides whether the evidence base can actually answer it. If not, it distils what it has learned, refines its search strategy, and loops back to searching — a self-improving research cycle.
7. It **curates** the corpus one paper at a time, judging relevance, recency, methodological quality, and redundancy against every other candidate.
8. Once the evidence is sufficient, it performs **deep analysis** on each surviving study, extracting every directly relevant finding as a candidate quote. Each candidate is checked verbatim against the source text and discarded if it does not match.
9. It performs **backward citation chasing** (snowballing), following the reference lists of good papers to find the primary studies they rely on.
10. It writes a **methodology assessment** of its own evidence base — strengths, weaknesses, and whether the corpus is actually strong enough to answer your question. If it identifies weaknesses, it can run a targeted **gap-fill search** aimed specifically at those weaknesses before proceeding.
11. It **synthesises** the review in APA 7th format: an evidence plan groups quotes thematically, the evidence section is assembled deterministically from verified quotes, and the analytical sections are written in a separate context so analysis can never corrupt the evidence.
12. It then **self-reviews, self-fixes, and verifies** the finished document — checking every in-text citation against the study it claims to cite, surgically repairing individual issues rather than rewriting the whole review.
13. Finally you can enter **Q&A mode** and ask questions about the review and its findings.

The output is a complete literature review with an introduction, methodology, evidence section, discussion, limitations, conclusion, and a reference list rebuilt from the quotes that were actually used — plus every downloaded paper and a complete run log, saved to disk.

## Features

- **Six academic databases searched in parallel** — Semantic Scholar, OpenAlex, CORE, Europe PMC, Crossref, and Unpaywall
- **Real full-text acquisition** through redundant open-access channels, not abstract-scraping
- **Hallucination-proof quotation** — verbatim matching with a deterministic token-paste system; the model never retypes a quote
- **Deterministic attribution checking** — a study's findings cannot be attributed to another study, and a study must be introduced before it is quoted
- **Self-improving search loop** — gap analysis, strategy distillation, and plan refinement across multiple discovery rounds
- **Evidence curation** — papers are judged one at a time on relevance, recency, methodological quality, and redundancy
- **Methodology weighting** — meta-analyses and systematic reviews are weighted above RCTs, above cohort studies, above case reports
- **Methodology self-assessment with targeted gap-fill searching** when the evidence base is judged weak
- **Backward citation chasing** to reach the primary studies behind good reviews
- **Two-phase synthesis** — evidence assembly and analytical writing happen in separate contexts
- **Surgical verification** — issues are repaired one at a time rather than by regenerating the document
- **Tangential engagement mode** for exploring adjacent literature, gated against unrelated cross-domain drift
- **APA 7th edition output** with in-text citations, block quotes, and a rebuilt reference list
- **Low-end device mode** — chunked deep analysis and map-reduce synthesis so the full pipeline runs on a small machine
- **Per-run logging** — every review gets its own complete transcript and its own papers folder
- **Post-review Q&A** about the findings (toggleable)
- **Graceful interrupt** — Ctrl+C aborts the active LLM call cleanly; press again to force-exit

## Limitations

I think it's important to note due to the amount of information processing this program does and the compute required not only are people without at least mid-level consumer hardware unable to use this, but for all systems the program does take a while to run.

This is primarily as it was designed from the ground up to have any findings when run verified as accurate to the original author cited, and to achieve that additional steps requiring longer run times were instated so this is just me being upfront.

So expect to leave it on in the background for a couple hours if doing some heavy research! 

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TheBlewish/Academic-AI-Literature-Reviewer-Ollama
cd Academic-AI-Literature-Reviewer-Ollama
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows, use `venv\Scripts\activate` instead.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and configure Ollama

Install Ollama from [https://ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull qwen3.5:35b
```

**Model choice matters a great deal here.** This program is context-hungry — synthesis and verification stages want a large window. A 30B-class reasoning model at roughly 64K context is the sweet spot on a high-memory machine. Smaller models will run the pipeline, but the quality of curation, methodology assessment, and synthesis degrades noticeably. If you are on modest hardware, use **low-end device mode** (below) rather than simply shrinking the context.

Both reasoning ("thinking") models and plain instruct models are supported. The program auto-detects which kind you have configured and adapts — thinking budgets, `<think>` block stripping, and truncation retries are applied only where they are needed.

### 5. Point the program at your Ollama server and model

Open `academic_config.py` and edit the block at the top:

```python
PRIMARY_LLM_CONFIG = {
    "llm_type": "ollama",
    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "model_name": os.environ.get("OLLAMA_MODEL", "qwen3.5:35b"),
    ...
}
```

- **`base_url`** — where Ollama is listening. Use `http://localhost:11434` if Ollama runs on the same machine, or the address of another box on your network if it doesn't.
- **`model_name`** — the exact tag of a model you have already pulled. Run `ollama list` to see what you have.

You can also set these with environment variables instead of editing the file:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="qwen3.5:35b"
```

### 6. Provide your own contact email for the academic APIs

**This step is easy to skip and will quietly cost you papers.** OpenAlex, Unpaywall and Crossref are free and require no signup, but they ask you to identify yourself with a contact email. Unpaywall **rejects** requests using an `example.com` address, and Unpaywall is one of the main routes to full-text PDFs.

Use any real address you own — a personal Gmail, Outlook or Proton address is fine, it does **not** have to be a university one.

The recommended way is environment variables, which keeps your address out of the repository entirely:

```bash
cp .env.example .env
# edit .env, put in your real email
set -a; source .env; set +a
```

Or edit the three `your.email@example.com` placeholders directly in the `SEARCH_APIS` block of `academic_config.py`. If you do this, take care not to commit the change back to a public fork.

If you leave the placeholder in, the program will warn you at startup:

```
⚠ Unpaywall: email is 'example.com' — will reject requests.
```

### 7. (Optional) Add free API keys

Both of these are genuinely optional — the program works without them and the other sources carry the search.

- **Semantic Scholar** — [request a free key](https://www.semanticscholar.org/product/api). Without one you are heavily rate-limited, and the program will disable this source after repeated 429 responses. That is expected behaviour, not a bug.
- **CORE** — [register for a free key](https://core.ac.uk/services/api). Without one, this source is skipped entirely.

```bash
export SEMANTIC_SCHOLAR_API_KEY="..."
export CORE_API_KEY="..."
```

**Never paste an API key directly into `academic_config.py` if you intend to push your copy of the repo anywhere public.** `.gitignore` is configured to keep `.env` out of git for exactly this reason.

## Usage

### Start Ollama

```bash
ollama serve
```

### Run the researcher

```bash
python Academic_Researcher.py
```

Add `--debug` to echo detailed logging to the console.

### Start a review

Enter your research question at the prompt:

```
Enter research question (or 'quit'):
> What are the documented health impacts of microplastic exposure from plastic water bottles?
```

Then leave it running. A full review makes a great many LLM calls and downloads a great many papers; on a large model this can take a long while. The terminal shows every phase, every API call, every paper acquired, and the model's reasoning as it curates and analyses.

### During the run

- **Ctrl+C once** — aborts the active LLM call and requests a graceful stop
- **Ctrl+C twice** — force-exits immediately

### After the review

The review prints in full to the terminal with coloured section headings, and the following are written to disk:

- **`Reviews/<Review_Title>_<timestamp>.txt`** — the finished literature review, with a header recording the question, paper count, curated study count, and search rounds
- **`Reviews/<Review_Title>_<timestamp>_data.json`** — structured run data: the paper catalogue, every study analysis, and the methodology assessment
- **`Papers/<Review_Title>/`** — every paper downloaded for that specific review
- **`Logs/<Review_Title>.txt`** — a complete transcript of the run

Each review gets its own titled papers folder and log, so you can always tell which papers produced which review.

You then enter **Q&A mode**, where you can ask questions answered from the review's contents. Type `new` to start another review or `quit` to exit. Q&A can be turned off entirely by setting `"qa_mode_enabled": False` in `academic_config.py`.

## Configuration

All settings live in `academic_config.py`, which is heavily commented. The ones most worth knowing about:

### Model and connection

| Setting | Purpose |
|---|---|
| `PRIMARY_LLM_CONFIG["base_url"]` | Ollama server address |
| `PRIMARY_LLM_CONFIG["model_name"]` | Model tag — **you must set this to a model you have pulled** |
| `PRIMARY_LLM_CONFIG["n_ctx"]` | Model context ceiling (default 65536) |
| `THINKING_CONFIG["force_thinking"]` | Override thinking-model auto-detection (`None` = auto) |

Context sizes are chosen **dynamically per call** from the actual input size, not fixed at the ceiling — `n_ctx` and the per-task `num_ctx` values in `TASK_PROFILES` are upper caps, not allocations.

### Research scope

| Setting | Default | Purpose |
|---|---|---|
| `num_focus_areas` | 5 | Focus areas generated per research plan |
| `max_discovery_rounds` | 15 | Maximum search-and-refine loops |
| `max_total_papers` | 200 | Corpus size ceiling |
| `target_papers_per_focus_area` | 5 | Papers sought per focus area |
| `require_full_text_main_mode` | True | Main mode requires full text; abstracts are only accepted in tangential mode |

### Quality gates

| Setting | Default | Purpose |
|---|---|---|
| `min_verified_quotes_per_study` | 1 | A study with no verified quotes is dropped |
| `quote_match_min_similarity` | 0.95 | Verbatim threshold a quote must clear to survive. Not listed in `RESEARCH_CONFIG` by default — add the key there to override it. **Lowering this weakens the core guarantee of the program; don't.** |
| `study_type_weights` | — | Methodological hierarchy used when weighing evidence |
| `methodology_gap_fill_enabled` | True | Run targeted searches against identified methodological weaknesses |

### Output behaviour

| Setting | Default | Purpose |
|---|---|---|
| `qa_mode_enabled` | True | Post-review interactive Q&A |
| `two_phase_synthesis` | True | Separate evidence assembly from analytical writing |
| `block_quote_word_threshold` | 40 | Word count at which a quote is rendered as an APA block quote |
| `rebuild_references_from_used_quotes` | True | Reference list is rebuilt from quotes actually cited |

### Low-end device mode

If you are running on a small machine, set:

```python
LOW_END_CONFIG = {
    "low_end_device_mode": "yes",
    "n_ctx": 8192,
    ...
}
```

This does **not** remove features. Deep analysis reads each paper in sequential chunks and accumulates quotes; synthesis writes studies up in batches and then assembles the batch drafts. Quote verification is unchanged and still runs against the full text, so the anti-hallucination guarantees hold exactly as they do in full mode. The one honest tradeoff is corpus size: low-end mode caps the number of papers so that the holistic curation calls fit a small context window, so a small device reviews a focused corpus rather than a 200-paper sweep.

## Project Structure

| File | Role |
|---|---|
| `Academic_Researcher.py` | Main entry point; the LangGraph pipeline and all 12 phases |
| `academic_config.py` | All configuration — **the file you edit** |
| `llm_manager.py` | Ollama interface, dynamic context sizing, thinking-model handling, retries |
| `paper_discovery.py` | The six academic API clients and full-text acquisition |
| `research_planner.py` | Focus-area planning and search-strategy refinement |
| `study_analyser.py` | Quick read, deep analysis, and verified quote extraction |
| `text_matcher.py` | Verbatim quote matching against source text |
| `document_store.py` | PDF/HTML text extraction and BM25 retrieval |
| `reference_harvester.py` | Backward citation chasing (snowballing) |
| `json_parser.py` | Robust JSON recovery from LLM output |
| `rag_config.py` | Compatibility shim so the reused RAG modules find their config |

## Current Status

This is a working system that produces real literature reviews, and it has been through a great deal of iteration on reliability — particularly on hallucination-proofing, which is where most of the engineering effort has gone. It is still a personal project rather than polished software, and how well it performs depends heavily on the model you run it with and on how much open-access literature exists for your question.

Known rough edges, stated honestly:

- Some papers simply cannot be acquired. Paywalled work with no open-access version will be dropped, and for some questions that meaningfully limits the corpus.
- Semantic Scholar rate-limits unauthenticated users aggressively; without a key it will often disable itself mid-run.
- A full review is slow. This is a deliberate tradeoff — the verification passes are the point.
- Quality tracks model capability closely. Small models will complete the pipeline but write weak reviews.

## Dependencies

- **Ollama** — install separately from [https://ollama.com](https://ollama.com)
- **Python 3.10+**
- Python packages listed in `requirements.txt`
- A pulled Ollama model with a large context window; a 30B-class reasoning model is recommended

## Contributing

Contributions are welcome. This is an ambitious project with plenty of room for improvement — additional database integrations, better acquisition fallbacks, and improved synthesis quality are all fair game. Please don't submit changes that weaken the verification layer; the anti-hallucination guarantees are the reason this project exists.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Ollama team for their local LLM runtime
- **OpenAlex**, **Semantic Scholar**, **CORE**, **Europe PMC**, **Crossref**, and **Unpaywall** for providing free, open scholarly infrastructure — this project would be impossible without them
- The LangGraph project for the orchestration framework

## Personal Note

This grew out of my previous project, *Automated-AI-Web-Researcher-Ollama*, and out of a frustration with it. That program searched the web well, but the web is full of confident nonsense, and an LLM summarising confident nonsense produces confident nonsense with citations. I wanted something that went to the actual studies.

The hard part turned out not to be the searching. It was trust. An LLM asked to write a literature review will happily invent a plausible quotation and attach it to a real paper, and you would never know unless you checked every one by hand. So the architecture is built around a simple principle: **the model is never allowed to type a quote.** It chooses which verified quote to place, and code does the placing. If a quote cannot be found verbatim in the source, it does not exist as far as the document is concerned.

That constraint drove most of the design — the token system, the deterministic attribution checks, the two-phase synthesis that keeps analysis away from evidence, the surgical verification that repairs one issue at a time instead of regenerating and reintroducing errors. It made the program much more complicated than it needed to be, and I think it is the only reason the output is worth reading.

I'm still a fairly new programmer and this is easily the most complex thing I've built. I hope it's useful to someone else.

## Disclaimer

This project is for educational and research purposes. Ensure you comply with the terms of service of all APIs and services used, and respect the rate limits of the free academic infrastructure this program depends on — those services are a public good and are easy to abuse accidentally.

Reviews produced by this program are a **research aid, not a substitute for reading the literature yourself**. The verification layer guarantees that quotations are genuine and correctly attributed; it does not and cannot guarantee that the model's interpretation of them is sound. Check the papers before you rely on anything.
