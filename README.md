# MCVR

Code for the paper *How Should We Rerank Multi-Constraint Video Search? Whole-Query Judgments,
Frames, and Their Cost*. The dataset (corpus, queries, facts and labels) is on Hugging Face:
<https://anonymous-hf.com/a/1yr7dq5i07kv>.

The repo holds scripts, prompts and configs only: no videos, labels or model outputs. Scripts read from `data/` and write to `results/`, both git-ignored, and every path
is relative to the repo root, so run everything from there.

## Layout

| Folder | What is in it |
|---|---|
| `download/` | Fetch the four video sources and the MCVR release into `data/`; restore the model-call caches |
| `pipeline/` | The test-set run: index, queries, systems, labels. Most stages call models (paid) |
| `dev/` | The 100-source development set (S6 threshold, description-length ablation, cost figures). The pipeline also loads its runners |
| `human/` | Build the human label sheets |
| `paper/` | Offline scripts that write the paper's tables, figures and numbers from saved runs. No model calls |
| `src/` | The Python package: datasets, index, systems, verifiers, judges, metrics, statistics |
| `prompts/` | Every prompt in the Prompts appendix, one file each |
| `configs/` | `mc_full.yaml` (test run: models, frames, judge), `mc_tier1.yaml` (graded verifiers, trigger), dev configs |
| `tests/` | Offline unit tests (`uv run --group dev pytest`) |

### Names

The code and the saved results still use the run keys from before the paper's names were settled:

| Paper | Run key in `results/` |
|---|---|
| S2 (graded per-constraint, descriptions) | `S2g@50` |
| S3 (graded per-constraint, frames) | `S3g` |
| S2b, S3b (binary versions) | `S2@50`, `S3` |
| S4@50, S5 (whole-query rerank on descriptions, frames) | `S4@50`, `S5` |
| S4L (S4 on long descriptions) | `S4L_s4pool`, `S4L_s0pool` |
| S5 on S0's pool | `S5_s0top10` |
| S6 (frames when the S4@50 top-two margin is at most 2) | computed in the eval, no run file |
| nested / near-miss sets | `natural` / `twin` |
| HN video | `twin` |

## Setup

Requires Python 3.11 to 3.13, [uv](https://docs.astral.sh/uv/) and `ffmpeg` (frames and Vript spans).

```bash
uv sync --extra llm
cp .env.example .env
```

Fill in `.env` only for paid stages: `OPENAI_API_KEY` (gpt-6-sol, gpt-6-luna, gpt-6-astra, gpt-5.5,
text-embedding-3-small), `GEMINI_API_KEY` (gemini-3.5-flash-lite, gemini-3.8-flash) and `HF_TOKEN`
(the release while it is private, and Vript). Scripts that call models are run with
`uv run --env-file .env ...`; the rest with plain `uv run`. Scripts under `paper/` need matplotlib:
`uv run --with matplotlib python paper/<script>.py`.

## Data

`data/mc/<part>/` holds one folder per corpus source (`shot2story`, `carebench`, `vript_short`,
`vript_long`), `data/mc/full/` the test pool's queries and labels, `data/mc/smoke*/` the dev set.
`results/full/` holds the test-set runs and reports, `results/smoke*/` the dev runs, `results/_cache/`
the model-call caches and `results/paper/` the paper's tables and figures.

What you need depends on what you want to run:

1. **Videos and captions** (everything starts here):

   ```bash
   uv run python -I download/shot2story.py
   uv run python -I download/carebench.py
   uv run python -I download/vript_short.py
   uv run --env-file .env python -I download/vript_long.py
   ```

   Vript's licence forbids redistribution: keep `data/mc/vript_*/videos/` local.

2. **The MCVR release** (queries, facts, labels, index descriptions), at
   <https://anonymous-hf.com/a/1yr7dq5i07kv>. Pass its dataset id as `--repo`, or download the
   parquet folders yourself and use `--from-dir`:

   ```bash
   uv run --env-file .env python download/release.py --repo <namespace>/MCVR
   ```

   It saves the parquet configs in `data/release/` and rebuilds `descriptions_fps05.jsonl` per part,
   `data/mc/full/{sample.json, queries.jsonl, queries_twin.jsonl}`, the strict and expanded qrels in
   `data/mc/full/judge/` and `results/full/select/sources100.json` (the 100 test sources). Checked
   against the originals: qrels, descriptions and the test-set sources are identical; query rows
   match on every field the release carries and lack `query_template`, `d_index` and two fact flags,
   which no evaluation reads. This is enough to score a new system against the paper's labels.

3. **Saved runs and model-call caches** (to rerun `paper/` and the eval without paying): the
   per-query outputs of every system and the cache shards will be released with the paper. Put the
   runs under `results/full/` and the shards under `results/_cache/shards/`, then:

   ```bash
   uv run python download/caches.py import
   ```

   With the caches in place every paid stage below replays from them at no cost, with the exact
   answers of the paper's run. A cold rerun calls the models again and will not match exactly:
   the closed models change, and `gemini-3.8-flash` was called by alias.

## Reproducing the paper

### 1. Test pool and index (paid)

```bash
uv run python pipeline/sample_test_pool.py                      # 280 sources, seed 1 -> data/mc/full/sample.json
uv run python pipeline/build_index.py frames                    # 0.5 fps frames, at least 8 per video
uv run --env-file .env python pipeline/build_index.py describe  # describe_v4, gemini-3.5-flash-lite
uv run --env-file .env python pipeline/build_index.py embed     # text-embedding-3-small
```

### 2. Queries (paid)

```bash
uv run --env-file .env python pipeline/build_queries.py --estimate
uv run --env-file .env python pipeline/build_queries.py         # facts, HN videos, grounding, queries
uv run python pipeline/build_queries_report.py
```

### 3. Systems, labels and the test set (paid)

```bash
uv run --env-file .env python pipeline/run_systems.py --stage retrieve       # S0, splitter, S1
uv run python pipeline/run_systems.py --stage select                         # frozen subsets
uv run --env-file .env python pipeline/run_systems.py --stage desc           # S2b@50, S4@50
uv run --env-file .env python pipeline/run_systems.py --stage second         # gpt-5.5 second verifier
uv run --env-file .env python pipeline/run_systems.py --stage frames         # S3b, S5 on the frame subsets
uv run --env-file .env python pipeline/run_systems.py --stage repeats        # run-to-run repeats
uv run --env-file .env python pipeline/run_graded_systems.py --stage desc-graded    # S2
uv run --env-file .env python pipeline/run_graded_systems.py --stage frames-graded  # S3
uv run --env-file .env python pipeline/run_graded_systems.py --stage frames-all     # S5 on all queries
uv run --env-file .env python pipeline/run_graded_systems.py --stage second-graded  # S2 with gpt-5.5
uv run --env-file .env python pipeline/judge_labels.py --stage run                  # frame judge
uv run python pipeline/run_graded_systems.py --stage fill-plan
uv run --env-file .env python pipeline/judge_labels.py --stage run --fill-queries results/full/tier1/fill_ids.json
uv run python pipeline/draw_test_set.py --seed 11 --out results/full/select/sources100.json
```

Each paid stage prints its projected tokens and stops at its cap; `--project-only` makes no call.

Review runs and checks:

```bash
SRC="--sources-file results/full/select/sources100.json"
uv run --env-file .env python pipeline/run_systems.py --stage frames --frame-pool s0top10 --run-suffix _s0top10 $SRC   # S5 on S0's pool
uv run --env-file .env python pipeline/long_descriptions.py describe                                                 # describe_v4d
uv run --env-file .env python pipeline/run_systems.py --stage s4long --frame-pool s4at50 --desc-file descriptions_fps05_v4d.jsonl $SRC
uv run --env-file .env python pipeline/run_systems.py --stage s4long --frame-pool s0top10 --desc-file descriptions_fps05_v4d.jsonl $SRC
uv run --env-file .env python pipeline/run_systems.py --stage frames --frame-verifier gemini:gemini-3.8-flash@low \
    --frame-media-resolution medium --frame-stage frame_verifier_gemini38                                            # Gemini frame verifier
uv run --env-file .env python pipeline/fact_judge.py --set test --stage build
uv run --env-file .env python pipeline/fact_judge.py --set test --stage judge                                        # text-only fact judge
uv run python pipeline/fact_types_build.py
uv run --env-file .env python pipeline/fact_types_judge.py --all                                                     # fact types (Option A)
```

### 4. Evaluation (free)

```bash
uv run python pipeline/run_systems.py --stage eval --sources-file results/full/select/sources100.json
uv run python pipeline/run_graded_systems.py --stage eval
```

These write `results/full/eval/summary.json` and `results/full/report.md` (all systems, strict and
expanded labels, cluster-bootstrap intervals, costs, S6) and `results/full/tier1/{summary.json,report.md}`
(S2 vs S4@50, the S3/S5 equivalence test, the frame gain, the trigger, length).

### 5. Development set (paid)

The dev set sets the S6 threshold (margin at most 2) and gives the description-length ablation and
several cost figures. Configs: `configs/mc_smoke*.yaml`.

```bash
uv run python dev/sample_dev.py
uv run --env-file .env python dev/build_index_dev.py
uv run --env-file .env python dev/build_queries_dev.py
uv run --env-file .env python dev/run_systems_dev.py --stage retrieve
uv run --env-file .env python dev/run_systems_dev.py --stage desc
uv run --env-file .env python dev/run_systems_dev.py --stage frames
uv run --env-file .env python dev/depth50_dev.py --stage rerank
uv run --env-file .env python dev/judge_labels_dev.py
for v in v4b v4c v4d; do                                                   # 70, 101, 202 words
  uv run --env-file .env python dev/description_length_dev.py --stage all --prompt prompts/describe_$v.txt
done
# near-miss rows (v4c, v4d): the same with --queries on the dev near-miss queries and
# --out results/smoke/variant_<tag>_twin; the exact flags of that run were not recorded
uv run python dev/trigger_dev.py
```

`dev/verifiers_dev.py` (gold splits), `dev/second_verifier_dev.py`, `dev/judge_noise_dev.py` and
`dev/media_resolution_dev.py` give the remaining dev numbers.

### 6. Human labels

```bash
uv run python human/make_label_sheets.py \
    --sources-file results/full/select/sources100.json --taskb-all \
    --prefill data/mc/full/human --overlap 25 --overlap-bd 10 --overlap-e 10 \
    --out data/mc/full/human100
```

The sheets are static HTML pages. Annotators export one CSV per task into the same folder;
`human/carry_over.py` copies earlier answers into new sheets as hints.

### 7. Tables, figures and numbers (free)

Outputs go to `results/paper/tables/` and `results/paper/figures/` (copies next to the sources in
`results/full/`). The paper's own copies were edited by hand afterwards: the 2026-11-04 system names
(the scripts still print run keys), caption wording, and in `frames_by_k_length.tex` the sign of the
S3/S5 column (the script prints S3 − S5).

| Paper | Script |
|---|---|
| Table 1 (`tab:results`) | `paper/table1_results.py` |
| Figure 1 (contrasts) | `paper/fig1_contrasts.py` |
| Figure 2 (trigger) | `paper/fig2_trigger_sweep.py results/paper/data_trigger_sweep.csv`, `paper/trigger_analysis.py`, `paper/fig2_frontier.py`, `paper/fig2_trigger_cis.py`, then `paper/fig2_trigger.py` |
| Ablations table (`tables/ablations.tex`) | assembled from Table 1, `tab_review_checks`, `tab_w5_long_descriptions`, `gemini_frame_verifier` and the eval reports |
| Finding 1 (same ten candidates) and `tab:w1-pools` | `paper/tab_w1_pools.py` |
| Finding 2 and `tab:reagg` (other aggregations) | `paper/tab_reagg.py` |
| Finding 3 (where the frame gain comes from) | `paper/frame_gain_decomposition.py` |
| Finding 3 and `tab:s0pool` (S5 on S0's pool) | `paper/tab_s0pool.py` |
| Finding 3, `tab:w5` (S4L) | `paper/tab_w5_long_descriptions.py` |
| Finding 4 (length and k slopes) | `paper/length_and_k_trends.py` |
| `tab:results-strict`, `tab:results-subsets` | `paper/results_tables.py --labels strict`, `paper/results_tables.py` |
| `tab:review-frames`, `review-accepted`, `review-per-k` | `paper/tab_review_checks.py --sources-file …` |
| Per-bin costs, near-miss k=1, fact type, acceptance by k, Holm | `paper/robustness_checks.py --sources-file …` |
| Second-verifier Spearman, length collinearity, corpus parts | `paper/extra_checks.py {spearman,collinearity,corpus-part} --sources-file …` |
| Splitter errors | `paper/splitter_errors.py --sources-file … --out results/full/todo_recompute/split_check` |
| Fusion variants | `paper/fusion_variants.py --sources-file …` (reads the embedding cache) |
| Cross-encoder | `paper/cross_encoder.py --from-saved --sources-file …` (the reranker itself runs on a local GPU) |
| Gemini frame verifier | `paper/gemini_frame_verifier.py --stage s5 --s5-system S5_gemini-3.8-flash-medium --sources-file …` |
| `tab:perk`, `tab:frames` | `paper/tab_per_k.py` |
| Description length (`length_*.tex`) | `paper/tab_length.py` |
| `desc_samples.tex` | `paper/tab_desc_samples.py` |
| `tab:descstats` | `paper/tab_descstats.py --sources-file … --out results/full/todo_recompute/descstats` |
| `tab:fact-funnel` | `paper/tab_fact_funnel.py --sources-file …` |
| `tab:fact-judge` | `pipeline/fact_judge.py --set test --stage analyze --sources-file … --out results/full/todo_recompute/fact_judge` |
| `tab:fact-types` | `paper/tab_fact_types.py` (prints the numbers; the table was typed from them) |
| Human labels (`human_*.tex`, Task A–E numbers) | `paper/tab_human.py --human data/mc/full/human100` |
| Corpus (`fig:corpus-lengths`, `tab:corpus-lengths`) | `paper/fig_corpus_lengths.py` |
| MultiVENT check | `pipeline/multivent_rerank.py`, `pipeline/multivent_constraints.py` (paid, routing-pilot data) |

`paper/paper_common.py` is the shared loader for the `paper/` reanalyses; `paper/results_tables.py`
also holds the helpers `table1_results.py` imports.

### Numbers no script reproduces

A few numbers in the paper come from one-off computations that are not in this repo: the
gpt-5.5 vs gpt-6-sol S2 row of ablation (d) and the cross-verifier graded contrasts, the "frame gain
on stated and omitted facts" paragraph, the pre-fill label counts, the 5.2 videos S4@50 and S0
share in their top ten, "0.6 contradicted per stated fact" and the dev fact-judge κ .91.

## Tests

```bash
uv run --group dev pytest
```

The tests are offline and use small fixtures; they need no data folder.
