"""Low-end branch test suite.

Runs the REAL StudyAnalyser and the REAL config accessors against a mock LLM and
a paper-shaped text, so chunking, pooling, dedup, verification, context-window
fit and the full-mode regression are all exercised end to end.

    python test_low_end.py
"""
import sys, os, re, json, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import academic_config as ac

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))

# --------------------------------------------------------------------------
# A realistic paper: quotable findings spread across the WHOLE document,
# including the Discussion and Conclusion at the very end.
# --------------------------------------------------------------------------
def make_paper_text():
    p = ["Effects of Widget Calibration on Throughput\n\n",
         "Abstract\n\nThis study examined whether widget calibration improves throughput. "
         "We ran a randomised controlled trial with 240 participants over 12 weeks.\n\n",
         "Introduction\n\n" + ("Prior work has long suggested a link. " * 120) + "\n\n",
         "Methods\n\n" + ("Participants were randomly assigned to conditions. " * 150) + "\n\n",
         "Results\n\n"]
    for i in range(40):
        p.append(f"Throughput in condition {i} increased by {10+i} percent relative to baseline. ")
        p.append("X" * 650 + ". ")
    p.append("\n\nDiscussion\n\n")
    p.append("These findings indicate that calibration is the dominant factor in throughput gains. ")
    p.append("We interpret this cautiously. " * 20)
    p.append("\n\nConclusion\n\n")
    p.append("We conclude that routine widget calibration yields a reliable throughput improvement. ")
    p.append("\n\nReferences\n\n" + ("Smith J. 2019. Some Paper. Journal. " * 20))
    return "".join(p)

PAPER = make_paper_text()

class FakePaper:
    def __init__(self, text):
        self.paper_id = "p1"; self.title = "Effects of Widget Calibration on Throughput"
        self.authors = ["Smith, J.", "Doe, A."]; self.year = 2022
        self.venue = "Journal of Widgets"; self.doi = "10.1/abc"; self.citation_count = 7
        self.full_text_available = True; self.full_text_content = text
        self.abstract = "This study examined whether widget calibration improves throughput."

class Result:
    def __init__(self, ok, js):
        self.success = ok; self.json_response = js
        self.response = json.dumps(js) if js else ""
        self.error = None; self.elapsed_time = 0.1; self.thinking = None

QUOTE_RE = re.compile(r"[^.]*?(?:increased by \d+ percent|indicate that|We conclude that)[^.]*\.")

class MockLLM:
    """Quotes VERBATIM from whatever text it is shown — like an honest model."""
    def __init__(self, fail=(), raise_on=()):
        self.calls = []; self.n = 0
        self.fail = set(fail); self.raise_on = set(raise_on)
    def run_primary(self, prompt, as_json=False, task=None, **kw):
        self.calls.append({"task": task, "len": len(prompt), "prompt": prompt})
        self.n += 1
        if self.n in self.raise_on: raise RuntimeError("simulated network error")
        if self.n in self.fail: return Result(False, None)
        if "PAPER CONTENT:" in prompt:
            body = prompt.split("PAPER CONTENT:", 1)[1]
            q = [m.strip() for m in QUOTE_RE.findall(body) if len(m.strip()) >= 20][:12]
            return Result(True, {"study_summary": "An RCT of 240 participants.",
                                 "key_quotes": [{"quote": x, "context": "c", "importance": "high"} for x in q]})
        return Result(True, {"study_summary": "An RCT of 240 participants.", "key_quotes": []})

def build(low_end):
    ac.LOW_END_CONFIG["low_end_device_mode"] = "yes" if low_end else "no"
    sys.modules.pop("study_analyser", None)
    import study_analyser as sa
    llm = MockLLM()
    an = sa.StudyAnalyser.__new__(sa.StudyAnalyser)
    sa.StudyAnalyser.__init__(an, llm)
    return sa, an, llm

# ==========================================================================
print("\n=== 1. Chunk splitter ===")
sa, an, llm = build(True)
split = sa._split_text_into_chunks
chunks = split(PAPER, 6000, 500)
check("produces multiple chunks", len(chunks) > 1, str(len(chunks)))
check("every chunk within budget", all(len(c) <= 6000 for c in chunks), str(max(len(c) for c in chunks)))
check("the CONCLUSION is in the final chunk", "We conclude that routine widget calibration" in chunks[-1])
check("the DISCUSSION appears in some chunk", any("dominant factor in throughput" in c for c in chunks))
joined = "".join(chunks)
missing = [s for s in ["Throughput in condition 0 increased", "Throughput in condition 39 increased",
                       "dominant factor in throughput", "We conclude that routine"] if s not in joined]
check("no findings lost between chunks", not missing, str(missing))
check("empty text -> no chunks", split("", 6000, 500) == [])
check("short text -> single chunk", split("abc", 6000, 500) == ["abc"])
check("overlap >= chunk does not hang", len(split("x"*20000, 1000, 5000)) > 0)
check("text with no sentence breaks terminates", len(split("x"*200000, 6000, 500)) > 1)

print("\n=== 2. Chunk cap SAMPLES instead of truncating ===")
capped = split(PAPER, 6000, 500, max_chunks=4)
check("cap returns at most N", len(capped) <= 4, str(len(capped)))
check("cap keeps the FIRST chunk", capped[0] == chunks[0])
check("cap keeps the LAST chunk (conclusion)", capped[-1] == chunks[-1])
check("CAPPED read still reaches the conclusion",
      "We conclude that routine widget calibration" in "".join(capped))
check("cap 0 = unlimited", split(PAPER, 6000, 500, max_chunks=0) == chunks)
check("cap above count is a no-op", split(PAPER, 6000, 500, max_chunks=999) == chunks)

print("\n=== 3. End-to-end low-end deep_analysis ===")
sa, an, llm = build(True)
out = an.deep_analysis(FakePaper(PAPER), "Does calibration improve throughput?", mode="main")
check("returns an analysis", isinstance(out, dict))
chunk_calls = [c for c in llm.calls if c["task"] == "deep_analysis_low_end"]
check("used the chunked low-end path", len(chunk_calls) > 1, f"{len(chunk_calls)} calls")
check("never used the full-mode profile", not any(c["task"] == "deep_analysis" for c in llm.calls))
vq = [q for q in out.get("key_quotes", []) if q.get("verified")]
check("produced VERIFIED quotes", len(vq) > 0, f"{len(vq)}/{len(out.get('key_quotes', []))}")
check("quotes_verified flag set", out.get("quotes_verified") is True)
check("study_summary present", bool(out.get("study_summary")))
check("QUOTE FROM THE CONCLUSION SURVIVES (the bug this fixes)",
      any("conclude that routine widget calibration" in q["quote"].lower() for q in vq))
check("QUOTE FROM THE DISCUSSION SURVIVES",
      any("dominant factor in throughput" in q["quote"].lower() for q in vq))
check("every verified quote is verbatim in the SOURCE", all(q["quote"] in PAPER for q in vq))
keys = [re.sub(r"\s+", " ", q["quote"]).strip().lower() for q in out["key_quotes"]]
check("no duplicate quotes across overlaps", len(keys) == len(set(keys)), f"{len(keys)} vs {len(set(keys))}")

print("\n=== 4. Every low-end call fits the context window ===")
spec = importlib.util.spec_from_file_location("lm", os.path.join(HERE, "llm_manager.py"))
lm = importlib.util.module_from_spec(spec); spec.loader.exec_module(lm)
worst = max(c["len"] for c in chunk_calls)
for N in (8192, 16384):
    p = ac.get_task_profile("deep_analysis_low_end")
    c = lm._compute_dynamic_ctx("x"*worst, None, p["max_tokens"], N, p["num_ctx"])
    room = c["chosen"] - p["max_tokens"]
    check(f"worst chunk prompt ({worst}c) fits n_ctx={N}", c["input_tokens"] <= room,
          f"in={c['input_tokens']} room={room}")

print("\n=== 5. No task profile starves the input ===")
N = ac.LOW_END_CONFIG["n_ctx"]; mult = ac.THINKING_CONFIG.get("thinking_token_multiplier", 3)
bad = []
for k in ac.TASK_PROFILES:
    p = ac.get_task_profile(k)
    eff = p["max_tokens"] * mult if p.get("think") else p["max_tokens"]
    if N - eff - lm.DYNAMIC_CTX_HEADROOM <= 1500: bad.append(k)
check("no profile leaves zero room for the prompt", not bad, str(bad))
check("thinking is suppressed in low-end mode",
      not any(ac.get_task_profile(k).get("think") for k in ac.TASK_PROFILES))
check("search paging is capped at the source",
      all(v["results_per_page"] <= ac.LOW_END_CONFIG["search_results_per_page"]
          for v in ac.get_search_api_config().values() if v.get("results_per_page")))

print("\n=== 6. Resilience ===")
sa, an, llm = build(True)
an.llm = MockLLM(fail=(2,))
o = an.deep_analysis(FakePaper(PAPER), "Does calibration improve throughput?", mode="main")
check("a failed chunk does NOT lose the paper", isinstance(o, dict) and len(o.get("key_quotes", [])) > 0)

sa, an, llm = build(True)
an.llm = MockLLM(raise_on=(1,))
o = an.deep_analysis(FakePaper(PAPER), "Does calibration improve throughput?", mode="main")
check("an exception mid-chunk is survivable", o is not None)

class AllFail(MockLLM):
    def run_primary(self, prompt, as_json=False, task=None, **kw):
        self.calls.append({"task": task, "len": len(prompt), "prompt": prompt})
        return Result(False, None)
sa, an, llm = build(True)
an.llm = AllFail()
o = an.deep_analysis(FakePaper(PAPER), "q", mode="main", existing_summary={"x": 1})
check("total failure returns the existing summary", o == {"x": 1})

print("\n=== 7. FULL-MODE REGRESSION ===")
sa_f, an_f, llm_f = build(False)
check("_low_end is False in full mode", an_f._low_end is False)
out_f = an_f.deep_analysis(FakePaper(PAPER), "Does calibration improve throughput?", mode="main")
check("full mode returns an analysis", isinstance(out_f, dict))
check("full mode makes EXACTLY ONE llm call", len(llm_f.calls) == 1, str(len(llm_f.calls)))
check("full mode uses the 'deep_analysis' task", llm_f.calls[0]["task"] == "deep_analysis")
body = llm_f.calls[0]["prompt"].split("PAPER CONTENT:\n", 1)[1]
check("full mode sends the WHOLE paper uncut", PAPER in body, f"{len(body)} vs {len(PAPER)}")
check("full mode profiles are uncapped", ac.get_task_profile("synthesis")["max_tokens"] > 3072)
check("full mode search paging is uncapped",
      ac.get_search_api_config("semantic_scholar")["results_per_page"] == 20)
check("RESEARCH_CONFIG is the identical object in full mode",
      ac.get_research_config() is ac.RESEARCH_CONFIG)

print("\n" + "=" * 64)
print(f"PASSED: {len(PASS)}    FAILED: {len(FAIL)}")
if FAIL:
    print("FAILURES:")
    for f in FAIL: print("   -", f)
    sys.exit(1)
print("ALL TESTS PASSED")
