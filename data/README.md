# Data

Nothing in this directory is tracked by Git except this file. Rebuild it as
follows; every count below is a gate that `dataset.py build` and the
validation scripts print and check.

## Sources (third-party; keep their licences)

| Dataset | Used as | Source | Local path |
|---|---|---|---|
| **SUSHI-Tiny** — 1,400 synthetic series (length 2,048) with template captions; 7 fluctuation × 20 (15 trend + 5 periodic) = 140 label classes | D1 component grammar and items; D2/D3 SUSHI substrate | GitHub release by Kawaguchi, Dohi & Ito (2024) `<CONFIRM: repository URL>`. The bare name "SUSHI" in the literature refers to the 140K "Base" set, which is not released; this project uses Tiny throughout | `data/SUSHI_tiny/` |
| **TRUCE** — 1,900 crowd-authored stock-series captions + 560 synthetic series (12 points each) | D2/D3 TRUCE substrate | Jhamtani & Berg-Kirkpatrick (2021), arXiv 2110.01839 `<CONFIRM: repository URL>` | `data/TRUCE/` |
| **TRACE NOAA weather set** — 2,006 test rows, 7 channels, 186-step frame, template descriptions | TRACE D1 (narrative items), D2, D3 | Google-Drive link in the TRACE README (`Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder`); raw data on Hugging Face `catherpker/TRACE-TimeseriesRAG-Dataset` | inside the sibling TRACE clone at `../TRACE-Multimodal-TSEncoder/dataset/retrieval/test/test.parquet` (note the nested layout the loader expects) |

## Processing steps

```
python dataset.py build                      # -> data/processed/pairs.jsonl   (8,780 pairs; split 8:1:1 per class, seed 42)
python scripts/dataset_validation/inspect_sushi.py / inspect_truce.py        # sanity figures -> results/dataset_validation/
python scripts/generate_probe1_items.py --splits test val                    # -> data/processed/probe1_items.jsonl (5,540 items, 279 signals)
python models/trace/generate_narrative_items.py --trace-repo ../TRACE-Multimodal-TSEncoder
python models/trace/excise_items.py --ids results/analysis/n3_excision_ids.txt
                                             # -> data/processed/narrative_probe_items_certified.jsonl (3,178 items)
```

Details of the split logic, z-normalisation and the expected per-source counts
are in `dataset.py` and `docs/dataset_validation.md`.

## What is generated here but never committed

- `data/processed/pairs.jsonl`, `probe1_items.jsonl`, `narrative_probe_items*.jsonl`
- ChatTS MCQ manifests `data/processed/chatts_probe{1,2,3}_mcq.jsonl` (11,080 / 1,756 / 3,912 rows) — built by `models/chatts/build_probe*_mcq_manifest.py`; their gate reports are committed as `results/analysis/chatts_probe*_mcq_report.json`
- `data/chatts_pinned_meta/` — the small non-weight files of the pinned ChatTS revision

## Small shareable sample

`results/dataset_validation/sushi_samples.png` and `truce_samples.png` show
example series and captions from each source. No raw rows are redistributed here.
