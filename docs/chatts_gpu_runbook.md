# ChatTS GPU session runbook

One sitting, pure execution. Everything below was prepared and gated on
CPU (handoff §4.9). Registered expectations are marked ⟦E⟧ — score every
one; misses are recorded, not explained away.

## 0. Pod requirements
- 1× A100 40GB (or 80GB), CUDA 12.x image, ≥ 80 GB disk, Python 3.11.
- Budget: download + install ~30-45 min; smoke ~10 min; full runs
  ⟦E-time⟧ 1-3 h projected (the smoke stage prints its own projection —
  score my band against it); upload minutes. Total ~3-5 h, ~€15-40.

## 1. Environment (era-pinned stack — the full triple, not one library)
```
python -m venv venv && source venv/bin/activate
pip install "transformers==4.46.2" "torch==2.4.1" "numpy==1.26.4" \
            "huggingface_hub" "accelerate" "sentencepiece"
```
⟦E-env⟧ imports succeed; `python -c "import transformers, torch, numpy;
print(transformers.__version__, torch.__version__, numpy.__version__)"`
prints 4.46.2 / 2.4.1 / 1.26.4.

## 2. Code + data onto the pod
```
git clone <ANON_REPO_URL> && cd Thesis-probes
```
Upload from the laptop (data/ is gitignored — these travel by scp or the
pod's upload UI; all small):
- data/processed/pairs.jsonl
- data/processed/chatts_probe1_mcq.jsonl
- data/processed/chatts_probe2_mcq.jsonl
- data/processed/chatts_probe3_mcq.jsonl

## 3. Pinned checkpoint download (paper-era revision, ~28 GiB)
```
huggingface-cli download bytedance-research/ChatTS-14B \
  --revision 1e661101dcfff86dc66f3397336b85f2f1cc5e89 \
  --local-dir ckpt_paper
```
Integrity gate BEFORE anything runs:
```
python - <<'EOF'
from pathlib import Path
total = sum(p.stat().st_size for p in Path('ckpt_paper').glob('pytorch_model-*.bin'))
print(total, 'OK' if total == 29_749_997_568 else 'FAIL — REDOWNLOAD')
EOF
```
⟦E-bytes⟧ exactly 29,749,997,568 across 6 shards.

## 4. Smoke stage — HARD prerequisite
```
python -m models.chatts.run_probes_gpu --smoke \
  --checkpoint ckpt_paper --pairs data/processed/pairs.jsonl \
  --p1-manifest data/processed/chatts_probe1_mcq.jsonl \
  --p2-manifest data/processed/chatts_probe2_mcq.jsonl \
  --p3-manifest data/processed/chatts_probe3_mcq.jsonl
```
Gates that must all be green (each ⟦E⟧):
- GR0 weight bytes; GR-cuda; GR1/GR2 manifest shas + populations
  (11,080 / 1,756 / 3,912).
- GR3 letter tokens: 'A'/'B' resolve to single-token ids (ids printed —
  pin them in the session notes; first observation).
- GR6 manual-path equivalence re-proven in the pod env, 50/50.
- GR4 splice arithmetic: ⟦E-splice⟧ SUSHI rows +126, TRUCE rows −1
  (patches replace the two <ts> tokens: 128−2 and 1−2). The one gate
  only the GPU could ever check.
- GR5 determinism: repeated questions give identical letters (bitwise
  logit equality NOT required — rule pre-named).
- Projected total time printed. ⟦E-time⟧ inside 1-3 h.
STOP AND PASTE OUTPUT TO CLAUDE IF ANY GATE FAILS. Do not improvise on
the pod — the meter is running but a wrong number costs more.

## 5. Full runs (each resumable; safe to re-invoke after interruption)
```
python -m models.chatts.run_probes_gpu --probe 1 --checkpoint ckpt_paper \
  --pairs data/processed/pairs.jsonl --p1-manifest data/processed/chatts_probe1_mcq.jsonl \
  --p2-manifest data/processed/chatts_probe2_mcq.jsonl --p3-manifest data/processed/chatts_probe3_mcq.jsonl
# then --probe 2, then --probe 3 (same flags)
```
Per probe, the agreement check runs automatically at the end:
⟦E-agree⟧ logit-vs-greedy ≥ 0.95 (pre-named fallback: below 0.95 →
generation becomes the primary readout for that probe; nothing is
"fixed" on the pod — record and continue).

## 6. Bring home (then terminate the pod)
- results/experiments/chatts_probe{1,2,3}_responses.jsonl
- the full terminal log of every command (the gate lines are the record)

## 7. Known risks and their pre-named responses
- Model-side incompat despite the era pin → paste the traceback; do not
  upgrade/downgrade libraries ad hoc on the meter.
- A smoke splice delta other than +126/−1 → the patch arithmetic
  differs from the source read; STOP, paste, we re-derive.
- OOM on SUSHI rows (unlikely at 128 patches + ~100 text tokens on
  40GB) → paste; the fallback is fp16 attention slicing, decided
  off-pod.

Scoring of all ⟦E⟧ marks happens in the analysis chat, not on the pod.
