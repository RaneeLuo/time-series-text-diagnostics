# CLaSP Reimplementation Spec
**Source of truth:** CLaSP paper v3 (arXiv:2411.08397, fetched from source 2026-07-23). Left column = verbatim paper facts. Right column = our documented choices where the paper is silent. Every "OUR CHOICE" line goes into the thesis methods chapter as a stated assumption.

## 1. Architecture (paper-specified)

| Component | Paper says |
|---|---|
| Signal encoder f_s | Informer, **trained from scratch** |
| Text encoder f_t | **T5-Small** (HuggingFace `google-t5/t5-small`) |
| Projections | Learnable linear L_s, L_t → common space dim d |
| Similarity | C = τ · (E_t · E_s^T), τ = temperature |
| Loss | L = 0.5·(ℓ_t(C) + ℓ_s(C)), symmetric cross-entropy over in-batch pairs (CLIP-style) |
| Retrieval | cosine similarity in common space, either direction (text→signal, signal→text) |

## 2. Training setup (paper-specified)

- **One joint model** trained on TRUCE-train + SUSHI-train together (Table III rows are query sources at test time, not separate models).
- Splits: 8:1:1 for both datasets (matches our validated data: TRUCE stock 1520/190/190, synthetic 448/56/56; SUSHI 1120/140/140 at 8:1:1 of 1400).
- TRUCE series length 12; SUSHI length 2048.
- TRUCE has 3 captions per series → each (series, caption) pair is a training example.

## 3. Paper-silent → OUR CHOICES (defaults; revisit only with reason, and document any change)

| Question | OUR CHOICE | Rationale |
|---|---|---|
| Common dim d | 512 | CLIP convention; small data → don't go larger |
| τ | learnable, init log(1/0.07) | CLIP standard |
| Informer config | encoder-only stack, 3 layers, d_model 512, 8 heads, no distilling | small data; encoder-only suffices for embedding |
| Series pooling | mean-pool encoder outputs (masked) | simplest; document |
| Text pooling | mean-pool T5 encoder last hidden states (masked) | T5 has no CLS; standard |
| T5 fine-tuned or frozen? | **fine-tune** (paper: encoders "jointly trained") | paper §III.C says both encoders jointly trained with projections |
| Length handling 12 vs 2048 | zero-pad to batch max + attention mask | Informer/Transformer native masking; document |
| Per-series normalization | z-normalize each series (train & eval) | mixed scales across datasets; **note Probe-3 interaction: z-norm removes mean/std → CLaSP's summary-stats probe operates on shape-level stats (min/max/range after norm, length)** |
| Optimizer | AdamW, lr 1e-4 (projections/Informer), 1e-5 (T5), weight decay 0.01 | standard for mixed scratch/pretrained |
| Batch size | per-dataset batches: TRUCE 64, SUSHI 8 (raise --batch-sushi on GPU) | mixed-length batches pad 12→2048 (~170× wasted compute) and 64 SUSHI×8 heads×2048²×4B = 8.6 GB attention alloc (CPU OOM); within-dataset in-batch negatives are also harder negatives; amended 2026-07-24 |
| Epochs | up to 100, early stop on val loss (patience 10) | small data |
| Seed | 42, and report over 3 seeds for the baseline | reproducibility; feeds §6.6 statistics |
| Informer fidelity | standard Transformer encoder (exact attention, no distilling) | ProbSparse/distilling are efficiency approximations of full attention; exact attention tractable at L=2048; at least as expressive; amended 2026-07-23 |

## 4. Evaluation harnesses (build BOTH)

**A. Paper-protocol soft mAP@10** (for baseline comparability with Table III):
- Query = caption from test set; retrieve top-10 signals from test pool.
- A retrieved signal is "correct" if cosine-sim(SBERT(query), SBERT(its own caption)) > ts.
- Sentence-BERT (`all-MiniLM-L6-v2` unless email reply specifies), ts=0.5 → compare against 0.458 (TRUCE queries) / 0.982 (SUSHI queries) / 0.954 (both).
- Expectation: **ballpark + pattern** (SUSHI ≫ TRUCE), not exact match — the paper's own numbers swing to 1.000 under DistilBERT, so the metric is protocol-sensitive.

**B. Strict pair-level retrieval** (for probes): Recall@1/5/10, MRR of the ground-truth signal for each caption query, over a fixed candidate pool. This is the probe-facing metric; all probe degradations Δ are computed on this.

## 5. Baseline gate (Phase-1 exit criterion for CLaSP)

1. Training converges (val loss plateaus, no collapse: check embedding-space rank / similarity histograms).
2. Soft-mAP@10 pattern reproduces: SUSHI-queries high (>0.8), TRUCE-queries lower, joint in between.
3. Strict Recall@k/MRR computed and frozen with config + seed + checkpoint → this tuple is **the baseline** every probe measures against.
4. Record everything in `results/experiments/baseline_clasp.json` + a one-page baseline report.

**GATE MET 2026-07-27.** Seeds 42/43/44 trained; early stopping at epochs 21/26/24; best val loss 3.203/3.187/3.254. Frozen baseline (mean ± SD over seeds, strict pair-level, pool 386): **R@1 0.049 ± 0.006 · R@5 0.221 ± 0.010 · R@10 0.331 ± 0.012 · MRR 0.141 ± 0.008**. Soft mAP@10 (SBERT ts=0.5): TRUCE 0.448 ± 0.020 (paper 0.458, inside range), SUSHI 0.853 ± 0.016. Untrained control and four-protocol fidelity study complete.

## 6. Known open items

- Author reply may supply code/checkpoint/hparams → if so, prefer theirs, document deltas.
- SUSHI version used in paper (email Q3) — determines expectation calibration only.
- SBERT variant used in their eval is unspecified → ours documented as choice.
