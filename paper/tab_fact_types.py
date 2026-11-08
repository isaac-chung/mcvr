"""Option A (tab:fact-types), step 3 of 3: judge "yes" rates by fact type and group, with 95%
cluster-bootstrap intervals over sources, and agreement with the fact judge of tab:fact-judge on
the overlapping facts. FREE: no model calls. Prints the numbers typed into tables/fact_filter_types.tex.

    uv run python paper/tab_fact_types.py

reads results/full/fact_filter_judge_optA/{key.jsonl, chunks.json, judged.json} and
results/full/fact_filter_judge/{key.jsonl, judged.json}; writes rows_with_answers.json there.
"""
import json, random, collections, sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]; D=R/"results/full/fact_filter_judge_optA"
key=[json.loads(l) for l in (D/"key.jsonl").open()]
ch=json.load(open(R/"results/full/fact_filter_judge_optA/chunks.json"))
J=json.load(open(R/"results/full/fact_filter_judge_optA/judged.json"))["judged"]
ans={}
for c in ch:
    a=J[f'{c["source"]}||{c["chunk"]}']["answers"]
    for i,x in enumerate(a): ans[(c["source"],c["offset"]+i+1)]=x
rows=[dict(k,ans=ans[(k["source"],k["pos"])]) for k in key if (k["source"],k["pos"]) in ans]
print("rows",len(rows),"of",len(key))
KEPT={"attribute","object","on_screen_text","count","scene"}
def grp(r):
    if r["type"] not in KEPT: return "excluded type"
    return "kept type, grounded-yes/yes" if r["prev_group"]=="kept" else ("kept type, grounder-split" if r["prev_group"]=="dropped" else ("kept type, grounder-rejected/other" if r["grounded"] else "kept type, selectable never grounded"))
def rate(rs,lab="yes"):
    return sum(r["ans"]==lab for r in rs)/len(rs) if rs else float("nan")
def boot(rs,lab="yes",B=2000):
    by=collections.defaultdict(list)
    for r in rs: by[r["source"]].append(r)
    ks=list(by); rng=random.Random(0); v=[]
    for _ in range(B):
        s=[x for k in (rng.choice(ks) for _ in ks) for x in by[k]]
        v.append(rate(s,lab))
    v.sort(); return v[int(.025*B)],v[int(.975*B)]
def show(title,keyf,order=None):
    print("\n##",title); g=collections.defaultdict(list)
    for r in rows: g[keyf(r)].append(r)
    print(f'{"cell":44s} {"n":>5s} {"yes":>6s} {"95% CI":>14s} {"no":>6s} {"unclear":>7s}')
    for k in (order or sorted(g)):
        rs=g[k]
        if not rs: continue
        lo,hi=boot(rs)
        print(f'{str(k):44s} {len(rs):5d} {rate(rs):6.3f} [{lo:.3f},{hi:.3f}] {rate(rs,"no"):6.3f} {rate(rs,"unclear"):7.3f}')
show("by type",lambda r:r["type"],["attribute","object","on_screen_text","count","scene","action","spatial_relation","temporal_order","other"])
show("excluded vs kept types (all facts)",lambda r:"excluded" if r["type"] not in KEPT else "kept types")
show("excluded vs kept-type facts never sent to grounders (clean comparison)",lambda r: grp(r) if grp(r) in("excluded type","kept type, selectable never grounded") else None,["excluded type","kept type, selectable never grounded"])
show("by group",grp)
# overlap agreement with Table 8
ov=[r for r in rows if r["prev_group"]]
T8=json.load(open(R/"results/full/fact_filter_judge/judged.json"))["judged"]; K8=[json.loads(l) for l in open(R/"results/full/fact_filter_judge/key.jsonl")]
t8={(k["source"],k["text"]):T8[k["source"]]["answers"][k["pos"]-1] for k in K8 if k["source"] in T8}
m=collections.Counter((t8[(r["source"],r["text"])],r["ans"]) for r in ov if (r["source"],r["text"]) in t8)
print("\n## overlap with Table 8 (rows: Table8 answer, cols: optA answer)"); n=sum(m.values())
for a in("yes","no","unclear"): print(a,[m[(a,b)] for b in("yes","no","unclear")])
print("agreement",sum(m[(a,a)] for a in("yes","no","unclear"))/n,"n",n)
json.dump({"rows":rows},open(D/"rows_with_answers.json","w"))
