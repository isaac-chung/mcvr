"""Option A (tab:fact-types), step 1 of 3: build the text-only judge requests for ALL mined facts of
the 100 test sources. FREE: no model calls. Same judge prompt, description file and shuffle as
pipeline/fact_judge.py.

    uv run python pipeline/fact_types_build.py [OUT]   # default results/full/fact_filter_judge_optA

writes OUT/{requests.jsonl, key.jsonl}; then pipeline/fact_types_judge.py (paid) and tab_fact_types.py.
"""
import ast, collections, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.judge import JUDGE_PROMPT, judge_prompt
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"results/full/fact_filter_judge_optA"
SEED = "optA"
src = set(json.load(open(ROOT/"results/full/select/sources100.json"))["sources"])
template = (ROOT/JUDGE_PROMPT).read_text()
# use same MAIN_PROMPT filter as the existing script
import importlib.util
spec = importlib.util.spec_from_file_location("ffj", ROOT/"pipeline/fact_judge.py"); ffj = importlib.util.module_from_spec(spec); spec.loader.exec_module(ffj)
desc = ffj.load_descriptions(ROOT, "descriptions_fps05.jsonl")
prev = {}  # (source,text)->group from the existing judge key
for l in open(ROOT/"results/full/fact_filter_judge/key.jsonl"):
    k = json.loads(l); prev[(k["source"], k["text"])] = k["group"]
reqs, key, miss = [], [], []
for l in open(ROOT/"data/mc/full/constraints.jsonl"):
    r = json.loads(l)
    if r["source"] not in src: continue
    s = r["source"]
    if not desc.get(s): miss.append(s); continue
    facts = r["facts"] if isinstance(r["facts"], list) else ast.literal_eval(r["facts"])
    order = list(facts); random.Random(f"{SEED}:{s}").shuffle(order)
    texts = [f["text"] for f in order]
    reqs.append({"source": s, "dataset": r["dataset"], "description": desc[s], "facts": texts,
                 "prompt": judge_prompt(template, desc[s].strip(), [{"text": t} for t in texts])})
    grounded = {int(g["key"].lstrip("n")) for g in ast.literal_eval(r["grounding"]) if g.get("role")=="natural"} if isinstance(r["grounding"], str) else {int(g["key"].lstrip("n")) for g in r["grounding"] if g.get("role")=="natural"}
    for pos, f in enumerate(order, 1):
        key.append({"source": s, "pos": pos, "fact_id": f["id"], "text": f["text"], "type": f["type"],
                    "selectable": f.get("selectable"), "grounded": f["id"] in grounded,
                    "prev_group": prev.get((s, f["text"]))})
OUT.mkdir(parents=True, exist_ok=True)
(OUT/"requests.jsonl").write_text("".join(json.dumps(x)+"\n" for x in reqs))
(OUT/"key.jsonl").write_text("".join(json.dumps(x)+"\n" for x in key))
c = collections.Counter(k["type"] for k in key)
print("sources", len(reqs), "missing desc", miss)
print("facts", len(key), dict(c))
print("already judged:", sum(k["prev_group"] is not None for k in key), " new:", sum(k["prev_group"] is None for k in key))
words = sum(len(r["prompt"].split()) for r in reqs)
print("prompt words total", words, "~tokens", int(words*1.4), "max facts/call", max(len(r["facts"]) for r in reqs))
