# Reproduction guide

Everything below runs from a clean clone. Data and checkpoints are not in Git
(`data/README.md` says where they come from). Each command prints its own
gates; **a command whose gates fail must not be followed by the next one.**

## 1. Fixed quantities

| Item | Value |
|---|---|
| Unified corpus | 8,780 (series, caption) pairs from SUSHI-Tiny + TRUCE; split 8:1:1 per class, seed 42 (`dataset.py`) |
| Retrieval test set | 878 caption queries over a pool of 386 test series (one per sample; 738 TRUCE + 140 SUSHI) |
| D1 item set (SUSHI) | 5,540 forced-choice items over 279 held-out series; C4 items re-scored on the 738 human-certified items |
| D1 item set (TRACE, NOAA narratives) | 3,200 generated → 3,178 pre-run certified → N3 census-certified 344 + 344 |
| CLaSP training seeds | 42 / 43 / 44 (three independent checkpoints; every CLaSP result is reported over all three) |
| TRACE evaluation | released checkpoint `retriever_demo.pt`; the authors' random pretrain mask (ratio 0.3) is made deterministic with mask seeds 13 / 14 / 15 |
| ChatTS | `bytedance-research/ChatTS-14B`, Hugging Face revision `1e661101dcfff86dc66f3397336b85f2f1cc5e89` (paper-era weights; `main` was replaced in place on 2025-08-01) |
| Text-embedding baseline | OpenAI `text-embedding-3-large`, responses cached under `.cache/` |
| D2 perturbations | sf_all / sf_half / ex_half shuffles + masking (ratio 0.2, fill 0; on TRACE additional to the 0.3 protocol mask); per-series seeded permutations |
| D3 surrogates | sf_all (read from D2) / i.i.d. resample of the series' own values / matched Gaussian; RAW-level construction, then each model's own preprocessing; whole-pool replacement |
| Equivalence margin | ±0.05 (absolute MRR; also applied to relative degradation and to accuracy points, flagged where reused) |

## 2. Statistical machinery (implemented in the `analyze_*` scripts)

- Confidence intervals: bootstrap over **signals** (clusters), never over items or queries that share a signal; `B` is printed by each script.
- Paired tests: Wilcoxon signed-rank on per-signal accuracy differences (D1) or per-query ranks / reciprocal ranks (D2, D3); sidedness and the Holm family are printed by each script and differ by arm — read the script header before quoting.
- Multiple comparisons: Holm correction within each seed's family.
- Equivalence: TOST at ±0.05 with a 90% bootstrap CI; three outcomes (equivalent / shifted / inconclusive-by-width).
- Every result is replicated over the three seeds of its model; a verdict requires the same direction in all three.
- Noise floor for the frozen baselines: sd/mean over seeds (sample sd, ddof = 1), from `scripts/aggregate_seeds.py`.

## 3. Commands, in order

```
python dataset.py build                       # -> data/processed/pairs.jsonl (8,780 pairs)
python scripts/analyze_sushi_labels.py        # grammar: 7 x 20 = 140, complete product
python scripts/build_component_table.py       # 3 gates must all pass
python scripts/generate_probe1_items.py --splits test val    # 5,540 items, 279 signals

# CLaSP (needs the three checkpoints; retrain on Colab T4, ~15 min each)
python -m models.clasp.train --tag baseline_seed42 --seed 42 --epochs 60 --batch-sushi 8
python -m models.clasp.evaluate --checkpoint results/checkpoints/best_baseline_seed42.pt
python scripts/aggregate_seeds.py --inputs results/experiments/eval_baseline_seed4{2,3,4}.json
python -m models.clasp.eval_table3 --checkpoint results/checkpoints/best_baseline_seed42.pt
python -m models.clasp.eval_table3 --untrained
python -m models.clasp.run_probe1 --checkpoints results/checkpoints/best_baseline_seed4{2,3,4}.pt

# floor baseline (needs OPENAI_API_KEY; ~$0.16, cached thereafter)
python scripts/inspect_serialisation.py       # verify spikes survive BEFORE spending
python -m models.openai_embed.run_probe1 --dry-run
python -m models.openai_embed.run_probe1 --yes
python -m models.openai_embed.run_baseline --dry-run    # strict retrieval baseline (2026-08-09)
python -m models.openai_embed.run_baseline --yes
#   -> results/experiments/baseline_openai_embed.json; ~$0.002 with the probe
#      cache present (SUSHI signal embeddings reused); gates G1-G7, pool must be 386

# analysis, model-agnostic
python scripts/information_availability_control.py
python scripts/analyze_probe1_stats.py
python scripts/analyze_probe1_stats.py --per-item results/experiments/probe1_openai_per_item.jsonl --out results/experiments/probe1_openai_statistics.json
python scripts/audit_item_balance.py --results results/experiments/probe1_clasp_per_item.jsonl
python scripts/audit_item_balance.py --results results/experiments/probe1_openai_per_item.jsonl

# hardening layer (2026-08-02..06); order matters — later gates read earlier outputs
python scripts/per_pair_cross_analysis.py
python scripts/information_availability_control_restricted.py
python scripts/per_pair_cross_analysis.py --features results/analysis/information_availability_279.json --out results/analysis/per_pair_cross_analysis_279.json --fig results/analysis/per_pair_scatter_279.png
python scripts/sample_manual_validation.py
python scripts/audit_c4_clause_specificity.py
python scripts/sample_pinning_spotcheck.py    # sheet only; the census judgments are human work
python scripts/census_c4_reanalysis.py        # requires the filled census CSV

# TRACE task zero (2026-08-07/08; needs the authors' repo cloned as a sibling
# and the dataset zip from the Google-Drive link in their README. Layout note: the
# authors' loader reads dataset/retrieval/<split>/<split>.parquet, not the flat
# layout the zip ships — move test.parquet to dataset/retrieval/test/test.parquet)
python models/trace/read_checkpoint_args.py --checkpoint ../TRACE-Multimodal-TSEncoder/results/model_checkpoints/context_align/retriever_demo.pt
python models/trace/run_authors_demo_eval.py --trace-repo ../TRACE-Multimodal-TSEncoder --split test
#   -> results/experiments/trace_demo_repro_test.json; first run downloads the
#      Nomic encoder (~500 MB) and embeds all texts (~60 min CPU, cached; re-runs ~4 min)

# TRACE substrate decision + narrative item set (2026-08-08)
python models/trace/downsample_survival_gate.py            # pre-registered downsampling gate: FAILED (the committed record)
python models/trace/inspect_noaa_narratives.py --trace-repo ../TRACE-Multimodal-TSEncoder
python models/trace/scan_narrative_phrases.py  --trace-repo ../TRACE-Multimodal-TSEncoder
python models/trace/generate_narrative_items.py --trace-repo ../TRACE-Multimodal-TSEncoder
#   -> v2: data/processed/narrative_probe_items.jsonl (3,200 items, seed 42; N4
#      dropped by pre-committed rule) + generation report + validation sheet
python models/trace/excise_items.py --ids results/analysis/n3_excision_ids.txt
#   -> data/processed/narrative_probe_items_certified.jsonl (3,178 items) —
#      the runner's input — plus the excision record

# TRACE narrative runner (2026-08-09; ~30 min first run: 13 min one-time swap-text
# embedding + ~2.6 min per mask seed; gates G1-G9 must all pass)
python models/trace/run_narrative_probe.py --trace-repo ../TRACE-Multimodal-TSEncoder
python scripts/analyze_probe1_stats.py --per-item results/experiments/trace_narrative_per_item.jsonl --out results/experiments/trace_narrative_statistics.json
python scripts/audit_item_balance.py --items data/processed/narrative_probe_items_certified.jsonl --results results/experiments/trace_narrative_per_item.jsonl
#   ^ --items is REQUIRED for non-SUSHI sets (see Notes below); without it the script
#     audits the 5,540 SUSHI items and prints a plausible, wrong table
python scripts/analyze_narrative_slices.py                 # duration + header-vs-prose
python models/trace/verify_n5_investigation.py             # N5 mechanism record (digit-exact expected values)
python models/trace/verify_n5_investigation_v2.py          # same analysis, rules IMPORTED from v1, writes results/analysis/n5_investigation.json (the committed record for 4.2.2.2; 2026-09-07)
python scripts/make_n3_validation_sheet.py                 # N3 hardening sample, seed 20260808

# Probe 2 (order invariance) — the runs below reproduce the committed records
python scripts/classify_sushi_order_groups.py              # 135/4/1, gates green
python scripts/classify_truce_order_groups.py              # rules v2; the census verdicts are human work (sheets in results/analysis/)
python scripts/apply_truce_certification.py                # -> probe2_truce_groups_certified.json (7089/245/46)
python scripts/census_trace_order_content.py               # -> probe2_trace_order_census.json (2005/1/0)
python scripts/check_pool_duplicates.py                    # TRUCE-synth clusters (D1/D2 evidence)
python scripts/check_pool_neighbours.py
python -m models.clasp.run_probe2 --checkpoints CK42 CK43 CK44 --seeds 42 43 44 \
    --sushi-groups results/analysis/probe2_sushi_groups.json \
    --truce-groups results/analysis/probe2_truce_groups_certified.json
python -m models.clasp.analyze_probe2                      # P2-1/2/3/5 scoring
python -m models.openai_embed.run_probe2 --yes             # v2 with D2-F; ~$0.75
python -m models.openai_embed.analyze_probe2               # P2-4 scoring
python scripts/diagnose_floor_ties.py                      # D2-F evidence
# TRACE arm (2026-08-15; diagnostics FIRST — their missed predictions
# are the arm's foundation; then runner, stats, strata, duration)
python models/trace/diagnose_probe2_setup.py --trace-repo ../TRACE-Multimodal-TSEncoder
python models/trace/diagnose_probe2_data.py  --trace-repo ../TRACE-Multimodal-TSEncoder
python models/trace/run_probe2.py --trace-repo ../TRACE-Multimodal-TSEncoder \
    --census results/analysis/probe2_trace_order_census.json
python scripts/analyze_probe2_trace.py \
    --records results/experiments/probe2_trace_per_query_seed13.jsonl \
              results/experiments/probe2_trace_per_query_seed14.jsonl \
              results/experiments/probe2_trace_per_query_seed15.jsonl \
    --summary results/experiments/probe2_trace_summary.json \
    --out results/experiments/probe2_trace_stats.json
python scripts/analyze_probe2_trace_strata.py \
    --records results/experiments/probe2_trace_per_query_seed1{3,4,5 as above}.jsonl \
    --trace-repo ../TRACE-Multimodal-TSEncoder \
    --out results/analysis/probe2_trace_strata.json
python scripts/analyze_probe2_trace_duration.py \
    --records ...seed13/14/15.jsonl \
    --narrative results/experiments/trace_narrative_per_item.jsonl \
    --out results/analysis/probe2_trace_duration.json
#   ^ strata/duration are NOT optional garnish: the pooled profile
#     ordering is a mixture artifact and only the stratum tables are
#     quotable (see results/README.md)

# Probe 3 — CLaSP arm (runner v2; the two diagnostics reproduce the float32
# constant-guard finding described in Notes below)
python -m models.clasp.run_probe3 \
    --checkpoints CK42 CK43 CK44 --seeds 42 43 44 \
    --sushi-groups results/analysis/probe2_sushi_groups.json \
    --truce-groups results/analysis/probe2_truce_groups_certified.json
python -m models.clasp.diagnose_probe3_g8 --checkpoint CK42 --ckpt-seed 42   # G8 mechanism, stage 1
python -m models.clasp.diagnose_probe3_g8_v2 --checkpoint CK42              # G8 mechanism, stage 2
python -m models.clasp.analyze_probe3 \
    --probe3-records results/experiments/probe3_clasp_per_query_seed4{2,3,4}.jsonl \
    --probe2-records results/experiments/probe2_clasp_per_query_seed4{2,3,4}.jsonl \
    --out results/experiments/probe3_clasp_stats.json
#   ^ JG gate: lossless caption_id join + rank_unperturbed <=1e-9 vs
#     the committed Probe-2 records — HARD STOP if the runs are not
#     comparable. P3-5 group-size gate 60/39.

# Probe 3 — TRACE arm (2026-08-16; diagnostic FIRST — its redraw census
# is HARD-gated in the runner; renorm-always construction)
python models/trace/diagnose_probe3_setup.py \
    --trace-repo ../TRACE-Multimodal-TSEncoder
python models/trace/run_probe3.py \
    --trace-repo ../TRACE-Multimodal-TSEncoder \
    --census results/analysis/probe2_trace_order_census.json
python scripts/analyze_probe3_trace.py \
    --records results/experiments/probe3_trace_per_query_seed13.jsonl \
              results/experiments/probe3_trace_per_query_seed14.jsonl \
              results/experiments/probe3_trace_per_query_seed15.jsonl \
    --probe2-records results/experiments/probe2_trace_per_query_seed13.jsonl \
                     results/experiments/probe2_trace_per_query_seed14.jsonl \
                     results/experiments/probe2_trace_per_query_seed15.jsonl \
    --summary results/experiments/probe3_trace_summary.json \
    --out results/experiments/probe3_trace_stats.json
#   ^ sf_all rung is READ from the committed Probe-2 records (JG gate);
#     strata are the quotable unit; gaussian quoted WITH stratum position

# ChatTS local preparation (all CPU) — order matters
python models/chatts/verify_task_zero.py                    # pinned-revision facts, all hard gates
python models/chatts/build_probe1_mcq_manifest.py --items data/processed/probe1_items.jsonl
python models/chatts/build_probe2_mcq_manifest.py --pairs data/processed/pairs.jsonl \
    --truce-groups results/analysis/probe2_truce_groups_certified.json \
    --sushi-groups results/analysis/probe2_sushi_groups.json
python -m models.chatts.selftest_perturbations --pairs data/processed/pairs.jsonl
python -m models.chatts.selftest_manual_path --pairs data/processed/pairs.jsonl \
    --manifest data/processed/chatts_probe2_mcq.jsonl --checkpoint-meta data/chatts_pinned_meta
python -m models.chatts.build_probe3_mcq_manifest --pairs data/processed/pairs.jsonl \
    --p2-manifest data/processed/chatts_probe2_mcq.jsonl
# GPU session: follow docs/chatts_gpu_runbook.md verbatim (smoke first, HARD)
```

### CLaSP training (the only step that needs a GPU besides ChatTS)

The three CLaSP checkpoints were trained on a Google Colab T4 (~15 min each)
with exactly the `models.clasp.train` command above, seeds 42/43/44, and the
resulting `results/checkpoints/best_baseline_seed4{2,3,4}.pt` copied back to
the laptop. Everything downstream is CPU.

### ChatTS (rented A100)

Follow `docs/chatts_gpu_runbook.md` verbatim: it pins the pod environment,
the manifests to upload, the smoke stage (a hard prerequisite: splice
arithmetic, weight-file bytes, manual-encoding recheck, letter-token pin,
determinism), and the run order. The session logs in `results/logs/chatts_gpu_session/`
show every gate as it printed on the pod. Analysis runs off-pod:

```
python scripts/analyze_chatts_probes.py
python scripts/analyze_chatts_probe3_contrasts.py
python scripts/regrade_chatts_c4_census.py        # C4 on the 738 certified items
```

## 4. Notes — things that bit once and are now gates

- **Encodings.** All readers pass `encoding="utf-8"` explicitly; the narrative
  runner has a mojibake canary gate. On a Windows machine with a Chinese
  locale, Excel re-saves CSV sheets as GBK — the certification scripts try
  utf-8 / gbk / cp1252, print which one they used, and join on caption text
  with a hard fail for any unmatched row.
- **`audit_item_balance.py` needs `--items` for non-SUSHI item sets.** Without
  it the script silently audits the 5,540 SUSHI items and prints a plausible,
  wrong table.
- **`dataset.znorm` must not be "fixed".** Its constant-series guard
  (sd < 1e-8) is float64-calibrated; on the native float32 pipeline the
  constant SUSHI series (std 2.384e-7) misses the guard and is embedded as a
  ±1 row. Every frozen baseline depends on that. The D3 runners therefore
  keep float32 end-to-end; `models/clasp/diagnose_probe3_g8*.py` reproduce
  the discovery.
- **TRACE published artifacts do not run as published** (demo reads a field
  that does not exist; constructor demands an unreleased Stage-1 checkpoint;
  stored `model_name` is not implemented; parquet layout differs from the
  README). `models/trace/run_authors_demo_eval.py` documents the repair path;
  all adapters reuse it.
- **Identity controls.** The `clean; constant` SUSHI series is an identity
  control for every shuffle/surrogate condition: its embedding must be
  bitwise unchanged (its rank may legitimately move when the rest of the
  pool is replaced — the gate is on the vector, not the rank).
- **Pooled TRACE D2 numbers are a mixture.** The perturbation ordering
  inverts between the V=168 and V=180 length strata; only the stratum tables
  (`results/analysis/probe2_trace_strata.json`) are quotable, plus sf_all.
