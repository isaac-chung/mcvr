"""Option A (tab:fact-types), step 2 of 3: the text-only fact judge (gpt-6-sol@low, judge_v1) on
the requests of fact_types_build.py, chunked (<= 40 facts per call), max_tokens 16000. PAID.

    uv run --env-file .env python pipeline/fact_types_judge.py --test   # one call (largest source, chunk 0)
    uv run --env-file .env python pipeline/fact_types_judge.py --all    # every chunk

reads and writes only results/full/fact_filter_judge_optA/ (requests.jsonl in, chunks.json and
judged.json / judged_test.json out). Answers are cached, so a rerun from the cache is free.
"""
import argparse, json, math, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import importlib.util, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = ROOT/"results/full/fact_filter_judge_optA"; D = OUT; OUT.mkdir(parents=True, exist_ok=True)
from src.cache import shared_cache
from src.judge import JUDGE_PROMPT, JUDGE_SPEC
from src.llm import make_llm
from src.types import Cost
def load(n, rel):
    s = importlib.util.spec_from_file_location(n, ROOT/rel); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
vi = load("variant_index", "dev/description_length_dev.py"); pv = vi.pilot_module()
ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--all", action="store_true"); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
reqs = [json.loads(l) for l in (D/"requests.jsonl").open()]
calls = []
reqs = [r for r in reqs if r["facts"]]
for r in reqs:
    n = len(r["facts"]); k = math.ceil(n/40); size = math.ceil(n/k)
    for i in range(k):
        calls.append({"source": r["source"], "chunk": i, "description": r["description"], "facts": r["facts"][i*size:(i+1)*size], "offset": i*size})
(OUT/"chunks.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "description"} for c in calls]))
print("calls", len(calls), "max facts/call", max(len(c["facts"]) for c in calls))
if a.test:
    big = max(reqs, key=lambda r: len(r["facts"]))["source"]
    calls = [c for c in calls if c["source"] == big and c["chunk"] == 0]
elif not a.all: sys.exit("pass --test or --all")
cfg = yaml.safe_load((ROOT/"configs/mc_full.yaml").read_text()); errors = []
llm = pv.SafeCalls(pv.more_retries(make_llm(JUDGE_SPEC, shared_cache("llm", ROOT/cfg["results_dir"]))), errors)
llm.inner.inner.max_tokens = 16000
template = (ROOT/JUDGE_PROMPT).read_text()
def one(c):
    j = vi.judge_description(llm, template, c["description"], c["facts"])
    raw = j["raw"].strip()
    try: n = len(json.loads(raw)["facts"])
    except Exception: n = -1
    j["complete"] = n == len(c["facts"])
    return f'{c["source"]}||{c["chunk"]}', j
with ThreadPoolExecutor(a.workers) as p: res = dict(p.map(one, calls))
path = OUT/("judged_test.json" if a.test else "judged.json")
path.write_text(json.dumps({"judge": JUDGE_SPEC, "max_tokens": 16000, "judged": res, "errors": errors}, indent=1))
cost = sum((Cost(**j["cost"]) for j in res.values()), Cost())
print("calls", len(res), "complete", sum(j["complete"] for j in res.values()), "errors", len(errors), "tokens_in", cost.tokens_in, "tokens_out", cost.tokens_out, "per-call out max", max(j["cost"]["tokens_out"] for j in res.values()))
