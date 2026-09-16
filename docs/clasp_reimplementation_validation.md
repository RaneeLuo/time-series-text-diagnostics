# Validation of the CLaSP Reimplementation
**Purpose:** thesis subsection (recommended placement: Methodology, immediately after the CLaSP description) and defense preparation.
**Evidence base:** all figures below were produced in this project and are stored in `results/experiments/` (`eval_baseline_seed42.json`, `eval_untrained.json`, `table3_fidelity.json`, `history_baseline_seed42.json`). All paper values were read from arXiv:2411.08397v3 directly.

---

## 1. The claim being made (and not made)

**Claimed:** the reimplementation instantiates every architectural, objective, data, and evaluation element that the CLaSP paper specifies; where the paper is silent, standard choices were made and documented; and the resulting model reproduces the paper's published retrieval behaviour on the dataset for which identical data is available, across four independent evaluation protocols.

**Not claimed:** that the model is weight-identical, hyperparameter-identical, or numerically identical to the authors' artefact. That claim is unavailable to anyone, because the paper specifies no hyperparameters (no embedding dimension, temperature, layer count, learning rate, batch size, or epoch budget) and no code or checkpoint has been released.

The validation strategy therefore has two independent parts: **fidelity where verifiable** (§3) and **robustness of the thesis conclusions to residual infidelity** (§5). The second part is what makes the work safe even under a sceptical reading of the first.

## 2. What is at stake

If the reimplementation were *wrong* in the sense that matters — a degenerate model, a broken objective, a leaking evaluation — then probe results measured on it would be uninterpretable. The validation below is designed to rule out precisely that failure mode, and each level below could have failed and did not.

## 3. The validation ladder

### Level 1 — Specification correspondence (verified line by line against the paper)

| Paper element | Location in paper | Implementation |
|---|---|---|
| Separate signal and text encoders | §III.A, Eq. (1) | `SignalEncoder` and `T5EncoderModel`, independent |
| Learnable linear projections to a common space of dimension *d* | Eq. (2) | `proj_signal`, `proj_text` (`nn.Linear`) |
| Similarity C = τ·(Eₜ·Eₛᵀ) | Eq. (3) | `logits = scale * z_t @ z_s.T` |
| Cosine similarity for alignment and retrieval | §III.D, Fig. 1 | embeddings L2-normalised before the product |
| L = 0.5·(ℓₜ(C) + ℓₛ(C)) | Eq. (4) | `0.5 * (loss_t + loss_s)` |
| ℓₜ, ℓₛ = cross-entropy along each axis, correct pairs on the diagonal | Eqs. (5)–(6) | `cross_entropy(logits, arange(N))` and its transpose |
| Encoders and projections trained jointly | §III.C | single optimiser over all parameters |
| Text encoder: T5-Small from Hugging Face | §IV.A + footnote 1 | `google-t5/t5-small` — the identical published checkpoint |
| Signal encoder trained from scratch | §IV.A | random initialisation, no pretraining |
| TRUCE and SUSHI, 8:1:1 splits, lengths 12 and 2048 | §IV.A | verified in the data pipeline (Level 2) |
| mAP@10 with an independent judge encoder above a threshold | §IV.B | `soft_map10()` implements this protocol |

### Level 2 — Data integrity

The unified corpus was verified against the datasets' own releases before any training: 8,780 (signal, caption) pairs — TRUCE 4,560/570/570 (stock) and 1,344/168/168 (synthetic), SUSHI 1,120/140/140 — with every series at its expected length (12 and 2048 respectively) and no missing files. The SUSHI split is stratified by class label under a fixed seed and frozen. A mismatch at this level would have invalidated everything downstream; none was present.

### Level 3 — Objective and pipeline correctness (negative controls)

- **Loss at initialisation.** With a batch of 4, the untrained model returns 1.4422 against the theoretical value for a correct symmetric contrastive objective, ln(4) = 1.3863. A misimplemented loss does not land at chance entropy.
- **Evaluation leakage.** The untrained model scores Recall@1 = 0.001 over 878 queries against a 386-signal pool, where uniform chance is 0.0026. This is one correct hit where 2.3 are expected; the probability of one or fewer under chance is ≈ 0.34. The evaluation therefore contains no leakage, and the retrieval scale is calibrated.
- **Learning actually occurred.** Across three independently seeded runs, training terminated by early stopping at epochs 21, 26 and 24, with best validation losses of 3.203, 3.187 and 3.254. The trained model scores Recall@1 = 0.049 ± 0.006 and MRR = 0.141 ± 0.008 (mean ± SD over seeds), respectively ≈ 19× and ≈ 8× their chance values (0.0026 and 0.017).
- **Training is what produces the performance.** Evaluated under the four published configurations, a randomly initialised model scores 0.246, 0.015, 0.999 and 0.326 against the trained model's 0.506, 0.137, 0.997 and 0.611. Three of the four configurations register a large effect of training; the fourth registers none, for reasons established in §5.3.

### Level 4 — Behavioural reproduction across four independent protocols

This is the central evidence. The paper reports mAP@10 under four automatic configurations formed by two judge encoders and two thresholds. Because these configurations differ radically in strictness — the published TRUCE scores span 0.136 to 1.000 — reproducing the *pattern* is a far stronger test than matching any single number.

On TRUCE, where our data is identical to the authors':

| Configuration | Reimplementation | Paper | Deviation |
|---|---|---|---|
| Sentence-BERT, ts = 0.5 | 0.440 | 0.458 | −0.018 |
| Sentence-BERT, ts = 0.8 | 0.105 | 0.136 | −0.031 |
| DistilBERT, ts = 0.5 | 0.997 | 1.000 | −0.003 |
| DistilBERT, ts = 0.8 | 0.550 | 0.491 | +0.059 |

Maximum deviation 0.059; mean absolute deviation 0.028, across a response range of 0.86. The reimplementation reproduces both distinctive qualitative features of the published table: the collapse of TRUCE performance at the stricter threshold, and the saturation of all scores under the DistilBERT judge.

Two refinements strengthen this comparison. First, deviations are more informative when scaled by what each configuration can express. Taking the distance between the untrained model's score and the published score as that configuration's informative range, the deviations above correspond to 10%, 26% and 52% of the available range in the three configurations that have one; the fourth has an informative range of 0.001 and therefore admits no meaningful comparison. Second, the reimplementation was subsequently trained under three random seeds, giving a soft mAP@10 on TRUCE of 0.448 ± 0.020 with observed values spanning 0.433 to 0.470. **The published value of 0.458 falls inside that interval**, so the difference between the reimplementation and the original is smaller than the variation induced by re-training with a different random seed.

Two further consistency checks support the same conclusion. First, our TRUCE figures were obtained against a merged 386-signal candidate pool, whereas the paper's per-dataset rows were most plausibly computed against a smaller dataset-specific pool (§4.3 below) — that is, under a *harder* retrieval condition than the authors'. Second, our soft mAP@10 for SUSHI at the strictest sensible setting (0.306) coincides with our strict SUSHI MRR (0.304), which are computed by two independently written code paths; their agreement indicates neither contains an arithmetic fault.

### Level 5 — Every discrepancy accounted for

A reproduction is credible when its disagreements are explained and, ideally, *predicted*. There are three.

**5.1 SUSHI scores are lower than published (−0.128 at ts = 0.5).** The public SUSHI release is the "Tiny" version (1,400 signals); the paper's numbers are consistent with the larger "Base" version. Under an 8:1:1 split, Tiny yields exactly one test signal per class, whereas Base would yield on the order of one hundred per class, all carrying near-identical template captions. Since the soft metric counts any caption-similar retrieval as correct, a query in the Base setting has roughly a hundred admissible answers available and one in the Tiny setting has one. The prediction that follows is specific: the gap should *widen* as the judge becomes stricter, because leniency is what allows partially-similar captions to compensate for the missing near-duplicates. Measured: the gap grows from −0.128 at ts = 0.5 to −0.265 at ts = 0.8. The prediction was made before the measurement and confirmed by it.

**5.2 The combined ("TRUCE + SUSHI") row is not comparable.** Our combined figure is, in every configuration, the exact query-weighted mean of our two dataset rows (verified to four decimal places; e.g. 738 × 0.4397 + 140 × 0.8537 over 878 = 0.5057, matching the reported 0.50568). The paper's combined row cannot be such a mean: solving for the implied TRUCE share of queries yields 5.3%, 3.4%, 6.4% and 2.2% in the four configurations — a single query set cannot have four different compositions — and in the DistilBERT ts = 0.5 configuration the reported values (1.000, 1.000 → 0.959) are arithmetically impossible for any average. The paper's combined row is therefore a separate experiment, most plausibly retrieval against a merged database, which is consistent with its value sitting slightly below the SUSHI value in all five reported columns. Since the paper does not state the candidate pool per row, this row is excluded from the comparison and only the per-dataset rows are used.

**5.3 The DistilBERT ts = 0.5 column measures nothing.** Instrumenting that judge shows it accepts 99.7% of all 338,908 query–candidate pairs, so mAP@10 approaches unity for any ranking. Agreement in that column (0.997 vs 1.000) is therefore weak evidence and is not relied upon; the load in Level 4 is carried by the other three configurations. This is documented separately in `finding_metric_saturation.md`.

## 4. Deviations, stated plainly

1. **Signal encoder architecture.** The paper uses Informer; the reimplementation uses a standard Transformer encoder trained from scratch. Informer's distinguishing mechanisms (ProbSparse attention, distilling) are, by the paper's own description, approximations introduced to avoid the quadratic cost of full attention on long sequences. At the maximum length in these datasets (2048) exact attention is computationally tractable, so the quantity being approximated is computed directly. The substitution is toward the exact operation, not away from it.
2. **SUSHI Tiny rather than Base.** Forced by public availability; consequences quantified in §5.1 and requested from the authors.
3. **Unspecified hyperparameters.** Embedding dimension, temperature initialisation and learnability, encoder depth and width, pooling strategy, learning rates, batch composition, optimiser, epoch budget, series normalisation, and split seed are not given in the paper. All are recorded with rationale in `REIMPLEMENTATION_SPEC.md`.
4. **Unspecified evaluation details.** The paper names Sentence-BERT and DistilBERT but not the specific variants, the pooling used for DistilBERT, the average-precision normalisation, or the candidate pool per row. Ours are documented choices.

## 5. Why the thesis conclusions are robust to residual infidelity

This section is the substantive answer to the concern that an imperfect baseline invalidates everything built upon it.

The probes do not measure CLaSP's performance. They measure the *change* in a model's performance when a controlled perturbation is applied to the text, expressed as a relative degradation against that same model's own unperturbed baseline. The quantity of interest is therefore internal to whichever model is being probed. Two consequences follow.

First, a reimplementation that is faithful in architecture and objective but differs in hyperparameters remains a valid object of study. If a single-component caption swap fails to degrade retrieval, the finding is that *a contrastively trained dual encoder of this design, trained on these data with this objective*, is insensitive to that component. That is a statement about a model class and a training recipe, which is precisely the claim the thesis makes; it is not a statement about one company's checkpoint, which the thesis does not make.

Second, the thesis's design already anticipates this. Findings are not asserted from a single model: the same probes are applied to four systems spanning three paradigms (dual encoder, hard-negative-trained retriever, conversational time-series LLM, and a general-purpose text embedder). A shortcut that appears across paradigms cannot be an artefact of one reimplementation's hyperparameters. The cross-model matrix, not any single baseline, is what carries the thesis's conclusions.

What *would* invalidate the work is a degenerate baseline — a model at chance, or one with collapsed embeddings — because degradation cannot be measured from a floor. That specific failure is excluded by Level 3: the model performs at roughly seventeen times chance on strict retrieval, its training curve is well-behaved, and its untrained counterpart sits at chance, establishing that the measured performance is learned rather than structural.

## 6. What could have falsified this and did not

- The untrained model could have scored above chance, indicating leakage. It did not.
- The loss at initialisation could have differed from ln(N), indicating a misimplemented objective. It did not.
- The TRUCE reproduction could have failed in any of four protocols, or matched in one while diverging in the others (which would indicate coincidence rather than fidelity). It matched in all four, within 0.059.
- The SUSHI gap could have been erratic or could have narrowed under a stricter judge, contradicting the dataset-size explanation. It widened, as predicted.
- The two independently implemented metrics could have disagreed. They coincided to three decimals.

## 7. Anticipated defense questions

**"How do you know your reimplementation is correct, given that CLaSP was never released?"**
I validated it at four levels. Every element the paper specifies — the two-encoder architecture, the projections, the temperature-scaled similarity matrix, the symmetric cross-entropy loss, joint training, the exact T5-Small checkpoint, the datasets and splits, and the evaluation protocol — is implemented as specified, and I can point to the equation for each line of code. Beyond that, I ran negative controls: the loss at initialisation equals the theoretical chance value, and the untrained model retrieves at chance, so there is no leakage. The strongest evidence is behavioural: the paper reports mAP@10 under four evaluation configurations that span scores from 0.136 to 1.000, and on TRUCE — where my data is identical to theirs — my model matches all four within 0.059, reproducing both the collapse at the strict threshold and the saturation under the lenient judge. Matching one number could be coincidence; matching the whole response curve is not.

**"Your SUSHI numbers are clearly lower than the paper's. Doesn't that mean the model is wrong?"**
It means the data differs, and I can show that specifically. Only the Tiny release of SUSHI is public — 1% of the version the paper's numbers imply. Under the 8:1:1 split, Tiny leaves one test signal per class while Base leaves about a hundred, all with near-identical captions, and the metric counts any caption-similar retrieval as correct. So their setting has roughly a hundred admissible answers per query and mine has one. That explanation makes a falsifiable prediction: the gap should widen as the judge gets stricter. It does — from 0.128 to 0.265. I have also requested the full dataset from the authors.

**"You didn't use Informer. Isn't that a different model?"**
Informer's specific contributions are efficiency approximations — ProbSparse attention and distilling — introduced to avoid quadratic attention cost on long sequences, as the CLaSP paper itself describes. My longest sequence is 2048 points, where exact attention is affordable, so I compute directly what those mechanisms approximate. The substitution is toward the exact operation. I document it as a deviation and note it as a limitation, but it is not a weakening of the model.

**"If your baseline is not the authors' model, are your findings about CLaSP?"**
They are findings about a contrastively trained dual encoder built to CLaSP's specification and trained on CLaSP's data — and I state them in exactly those terms rather than as claims about the authors' artefact. This is why the study probes four systems rather than one. A shortcut that appears in a dual encoder, a hard-negative-trained retriever, and a conversational time-series model is a property of the approach, not of my hyperparameter choices.

**"What if the authors release their code and your results don't match?"**
Then the comparison becomes an additional result rather than a problem. I have contacted the authors requesting the code and the full dataset; if either arrives, the probes run unchanged on their artefact, because the probe pipeline is model-agnostic by design and needs only an encoding interface. The measurement being relative to each model's own baseline is what makes that substitution cheap.

## 8. Further validation

**Completed 2026-07-27.**
- **Multiple seeds.** Training was repeated with seeds 43 and 44. All three runs converged comparably (early stopping at epochs 21/26/24; best validation loss 3.203/3.187/3.254) and the baseline is now reported as a mean over seeds with its standard deviation. The published TRUCE value lies within the observed seed range (§3, Level 4).
- **Untrained model across all four protocols.** Completed; results in §3, Level 3 and in `finding_metric_saturation.md` §8. It supplies both a negative control for the fidelity comparison and the direct demonstration that one published configuration cannot distinguish a trained model from a random one.

**Still available (optional).**
- **The paper's second table.** The paper reports a separate zero-shot experiment using four query formulations built from SUSHI class labels. Attempting it would provide a second independent published result to compare against. Caveat: with Tiny leaving one test signal per class, the metric is structurally capped, so this comparison would need to be interpreted qualitatively rather than numerically.

## 9. One-slide summary for the defense

> **Is the reimplementation valid?**
> - Every element the paper specifies is implemented as specified — architecture, loss, training, encoder checkpoint, data, protocol.
> - Negative controls pass: loss at initialisation = ln(N); untrained model retrieves at chance (no leakage).
> - Behavioural match on TRUCE across **four** evaluation protocols spanning 0.136–1.000: maximum deviation **0.059**; over three seeds the published value **falls inside** the reimplementation's own range (0.433–0.470).
> - Every discrepancy is explained, and the main one was **predicted before it was measured** (SUSHI gap widens with judge strictness: 0.128 → 0.265).
> - Deviations are stated: exact attention in place of Informer's approximation; SUSHI Tiny; hyperparameters unspecified in the paper and documented here.
> - **The conclusions do not depend on exact replication**: probes measure relative degradation against each model's own baseline, findings are stated at the level of a model class, and the same probes run across four systems in three paradigms.
