# Diagnostic evaluation of time-series–text alignment models

Code, item sets, and result records for an MSc thesis that asks one question:
when a time-series–text alignment model reports high cross-modal retrieval
accuracy, how much of that performance survives controlled diagnostics that
remove specific information from the time series?

Three diagnostics are applied to four models — a reimplemented CLaSP, the
released TRACE checkpoint, ChatTS-14B, and a text-embedding baseline
(`text-embedding-3-large`) that serves as a negative control ("floor"):

| Diagnostic | Question it asks | How |
|---|---|---|
| **D1 — component swap** | Does the model read each described component of the series (trend direction, trend family, periodic waveform, fluctuation type, signal regime)? | Forced choice between the true caption and a caption with exactly one component altered, vs. a random-distractor control |
| **D2 — order invariance** | Does matching depend on the order of the values? | Shuffle / half-swap / mask the series, measure retrieval-rank degradation, split captions into order-dependent vs. order-invariant groups (difference-in-differences) |
| **D3 — statistical-information sufficiency** | Is the value distribution alone enough? | Replace the series by surrogates that keep progressively less information (exact multiset → i.i.d. resample of its own values → matched Gaussian), measure the retained retrieval |

**Naming note.** The thesis text says *diagnostic* (D1/D2/D3); the code and
result files say *probe* (`probe1`/`probe2`/`probe3`). They are the same thing.
The thesis term "five-field statistical summary" is the repo condition `five_number`.

## Layout

```
dataset.py            unified corpus builder (SUSHI-Tiny + TRUCE -> data/processed/pairs.jsonl)
models/clasp/         CLaSP reimplementation: train, evaluate, and the three diagnostic runners
models/trace/         TRACE adapters: authors' checkpoint reproduction, narrative item set, runners
models/chatts/        ChatTS: manifest builders, perturbations, manual encoding path, GPU runner
models/openai_embed/  text-embedding baseline runners (negative control)
scripts/              item generation, caption grouping, censuses, statistics, audits
docs/                 reproduction guide, model spec, judging protocol, GPU runbook
results/experiments/  canonical result records (per-item / per-query + statistics)
results/analysis/     certified groupings, censuses, diagnostics, human judgment sheets
results/logs/         GPU session logs (all gates printed)
data/                 datasets and processed files — not in Git; see data/README.md
```

The layout is the one the code runs from: `python -m models.clasp.train ...`
and `from dataset import ...` assume `models/`, `scripts/` and `dataset.py`
sit at the repository root. There is no `configs/` directory — every setting
is a command-line flag or a constant printed by the runner at start-up.

## Install

```
python 3.11
pip install -r requirements.txt
```

`requirements.txt` pins the laptop environment the CPU-side results were
produced with. The ChatTS GPU session used a different pinned environment;
`docs/chatts_gpu_runbook.md` records it.

## Run

`docs/reproduction.md` lists every command in order, what each one must
print before the next may run, and which result file it writes. The runners
are written to fail loudly: each prints a numbered set of gates (population
counts, frozen-baseline digit-exact reproduction, permutation validity,
identity controls, pairing integrity) and stops on the first failure.

## Results

`results/README.md` maps each result file to the diagnostic, model, and thesis
section it supports, and states which files are *inputs* the pipeline cannot
regenerate (human judgment sheets).

## Third-party assets

- SUSHI-Tiny (Kawaguchi, Dohi & Ito, 2024) and TRUCE (Jhamtani &
  Berg-Kirkpatrick, 2021) — see `data/README.md` for sources and licences.
- TRACE code and checkpoint: `Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder`
  (NeurIPS 2025), cloned as a sibling directory; its NOAA dataset from the link in its README.
- ChatTS-14B: `bytedance-research/ChatTS-14B` on Hugging Face, pinned to
  revision `1e661101dcfff86dc66f3397336b85f2f1cc5e89` (see `docs/reproduction.md`).
- CLaSP is reimplemented from the paper (arXiv 2411.08397); no authors' code was used.
  `docs/REIMPLEMENTATION_SPEC.md` and `docs/clasp_reimplementation_validation.md`
  record every choice and the fidelity check.
