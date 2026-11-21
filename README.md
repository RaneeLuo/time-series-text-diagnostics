# Diagnosing Information Dependence in Time-Series–Text Alignment

**Ranyi Luo, Yuanyuan Deng, Songgaojun Deng**  
Eindhoven University of Technology · FMTS Workshop at NeurIPS 2026 · Poster

[Key findings](#key-findings) · [Quickstart](#quickstart) · [Reproduction guide](docs/reproduction.md) · [Result files](results/README.md)

A model can retrieve the right time series even when it struggles to distinguish two descriptions of that series. What information supports a correct match? This repository accompanies our workshop paper and the underlying MSc thesis. It contains the code, human validation records, and saved results for three diagnostics of time-series–text alignment.

In one reproduced CLaSP evaluation, a trained retriever scored **0.997** and a randomly initialised retriever scored **0.999**. Under this judge setting, both received nearly perfect scores. The diagnostics below test what changes when we alter the description, reorder the series, or reduce the information available in the signal.

## Key findings

| Diagnostic | What we tested | What we found |
| --- | --- | --- |
| **D1 · Component sensitivity** | With the series fixed, can the model distinguish its caption from one with a targeted semantic change? A random distractor provides the reference. | On SUSHI, our CLaSP reimplementation chose the correct caption on **60.3%** of 738 human-validated fluctuation-type swaps (95% CI 56.7–64.1%). [Validation and scores](results/analysis/c4_census_reanalysis.json) |
| **D2 · Temporal-order dependence** | Does shuffling hurt more when a caption describes temporal order than when it remains true after reordering? | Shuffling hurt all three models. For CLaSP on TRUCE, the difference between caption groups had a confidence interval above zero in **two of three** training seeds. NOAA had no order-invariant captions for this comparison. [CLaSP results](results/experiments/probe2_clasp_stats.json) · [NOAA caption groups](results/analysis/probe2_trace_order_census.json) |
| **D3 · Statistical sufficiency** | How much performance survives shuffling, resampling each series from its own values, or replacing it with Gaussian values matched to its mean and variance? | For CLaSP on SUSHI, retrieval was better after resampling than after Gaussian replacement. On TRUCE, the remaining performance was consistent with a length cue: all correct candidates were short TRUCE series in a pool that also contained longer SUSHI series. [CLaSP results](results/experiments/probe3_clasp_stats.json) |

## Models and datasets

| Model | Evaluation | Data |
| --- | --- | --- |
| [CLaSP](models/clasp/) | Our retriever reimplementation, trained with three independent seeds. [Implementation checks](docs/clasp_reimplementation_validation.md) | SUSHI-Tiny and TRUCE |
| [TRACE](models/trace/) | Authors’ released retriever checkpoint, evaluated with three inference mask seeds | NOAA weather series and descriptions |
| [ChatTS-14B](models/chatts/) | Fixed checkpoint, evaluated with two answer choices in both orders | SUSHI-Tiny and TRUCE |

The [text-embedding negative control](models/openai_embed/) uses `text-embedding-3-large` on numerical values written as text. Retrievers are scored by mean reciprocal rank (MRR) in retrieval experiments. ChatTS is scored by two-choice accuracy. The code calls D1, D2, and D3 `probe1`, `probe2`, and `probe3`. The ChatTS summary condition is called `five_number`.

## Quickstart

**Inspect a published result.** With Git and Python installed, run these commands to inspect the two result records from the opening example. This only reads the saved results and requires no model weights, datasets, or GPU.

```sh
git clone https://github.com/RaneeLuo/time-series-text-diagnostics.git
cd time-series-text-diagnostics
python -m json.tool results/experiments/table3_fidelity.json
python -m json.tool results/experiments/table3_fidelity_untrained.json
```

In each file, look under `ours` → `distilbert_ts0.5` → `all`. The trained score rounds to **0.997** and the untrained score to **0.999**.

**Reproduce the experiments.** From the repository root, install the CPU dependencies in a Python 3.11 environment with `python -m pip install -r requirements.txt`. Obtain the data and checkpoints for the model you want to run. The [data guide](data/README.md) gives the expected paths, and the [reproduction guide](docs/reproduction.md) gives the commands, seeds, and checks for each stage. ChatTS uses a separate [GPU environment and runbook](docs/chatts_gpu_runbook.md).

## Data and result records

- **Data sources.** [SUSHI-Tiny](https://zenodo.org/records/13882998) provides synthetic signals with template captions. [TRUCE](https://github.com/harsh19/TRUCE) pairs short series with human-written captions. The [TRACE repository](https://github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder) links the NOAA data and released checkpoint.
- **Saved model outputs.** Browse the [experiment records](results/experiments/) for per-item answers, per-query retrieval results, and statistical summaries. The [results index](results/README.md) maps filenames to experiments.
- **Human validation.** The [analysis directory](results/analysis/) contains judgement sheets, caption groupings, and the records used to determine which items were eligible for analysis. The judgement sheets are inputs to reproduction.
