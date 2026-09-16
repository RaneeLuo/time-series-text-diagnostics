"""
Build the ChatTS Probe-1 MCQ manifest (CPU only, deterministic, no RNG).

Reads the committed Probe-1 items (SUSHI substrate) and writes two MCQ
rows per item: one with the correct caption as option A, one as option B.
This removes answer-position bias by construction. The final prompt text
is baked here, hash-stamped, and never rebuilt elsewhere — the GPU runner
pairs each row's prompt with the raw series via sample_id.

The series itself is NOT copied into the manifest; <ts><ts/> stays a
placeholder that the ChatTS processor fills at run time.

Gates GM1-GM4 and GM7 are HARD STOPS; GM5-GM6 are reports.

Place at: models/chatts/build_probe1_mcq_manifest.py
Run from repo root:
  python models/chatts/build_probe1_mcq_manifest.py --items data/processed/probe1_items.jsonl
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

# Exact prompt template. NEVER edit without a new version note: the GPU
# runner gates on TEMPLATE_SHA below.
SYSTEM_MSG = "You are a helpful assistant."
USER_TEMPLATE = (
    "Here is a time series: <ts><ts/>.\n"
    "Which description matches this time series?\n"
    "(A) {option_a}\n"
    "(B) {option_b}\n"
    "Answer with exactly one letter, A or B."
)
FULL_TEMPLATE = (
    "<|im_start|>system\n" + SYSTEM_MSG + "<|im_end|>\n"
    "<|im_start|>user\n" + USER_TEMPLATE + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
TEMPLATE_SHA = hashlib.sha256(FULL_TEMPLATE.encode("utf-8")).hexdigest()[:12]

EXPECTED_TOTAL_ITEMS = 5540
EXPECTED_PER_CONDITION = 2770
EXPECTED_SIGNALS = 279

FAILURES = []


def gate(name, ok, detail, hard=True):
    status = "PASS" if ok else ("FAIL (HARD)" if hard else "MISS (report)")
    print(f"[{name}] {status} — {detail}")
    if not ok and hard:
        FAILURES.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True,
                    help="path to probe1_items.jsonl (required on purpose — see error #11)")
    ap.add_argument("--out", default="data/processed/chatts_probe1_mcq.jsonl")
    ap.add_argument("--report", default="results/analysis/chatts_probe1_mcq_report.json")
    args = ap.parse_args()

    print("=== ChatTS Probe-1 MCQ manifest builder ===")
    print(f"items    : {args.items}")
    print(f"out      : {args.out}")
    print(f"template : sha256[:12] = {TEMPLATE_SHA}")
    print("determinism: no RNG anywhere in this script (both orders enumerated).")

    items = []
    with open(args.items, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    # ---- GM1: population ----
    n = len(items)
    sigs = {it["sample_id"] for it in items}
    splits = Counter(it["split"] for it in items)
    gate("GM1-count", n == EXPECTED_TOTAL_ITEMS,
         f"items read = {n} (expected {EXPECTED_TOTAL_ITEMS}); splits = {dict(splits)}")
    gate("GM1-signals", len(sigs) == EXPECTED_SIGNALS,
         f"unique signals = {len(sigs)} (expected {EXPECTED_SIGNALS})")
    bad_split = set(splits) - {"test", "val"}
    gate("GM1-splits", not bad_split,
         f"unexpected split labels: {bad_split if bad_split else 'none'}")

    # ---- GM2: condition balance + component table ----
    cond = Counter(it["condition"] for it in items)
    comp_cond = Counter((it["component"], it["condition"]) for it in items)
    print("[GM2-table] items per component x condition:")
    comps = sorted({it["component"] for it in items})
    for c in comps:
        print(f"    {c}: swap={comp_cond.get((c, 'swap'), 0)}  "
              f"random={comp_cond.get((c, 'random'), 0)}")
    gate("GM2-balance",
         cond.get("swap") == EXPECTED_PER_CONDITION
         and cond.get("random") == EXPECTED_PER_CONDITION,
         f"swap={cond.get('swap')} random={cond.get('random')} "
         f"(expected {EXPECTED_PER_CONDITION}/{EXPECTED_PER_CONDITION})")

    # ---- GM3: caption safety ----
    forbidden = ["<ts>", "<ts/>", "(A)", "(B)"]
    problems = Counter()
    for it in items:
        for cap_name in ("caption_correct", "caption_distractor"):
            cap = it[cap_name]
            if not cap or not cap.strip():
                problems[f"empty:{cap_name}"] += 1
            for tok in forbidden:
                if tok in cap:
                    problems[f"forbidden '{tok}' in {cap_name}"] += 1
        if it["caption_correct"] == it["caption_distractor"]:
            problems["identical captions"] += 1
    gate("GM3-captions", not problems,
         f"caption problems: {dict(problems) if problems else 'none'}")

    # ---- GM5 (report): non-ASCII census (mojibake canary) ----
    non_ascii = Counter()
    for it in items:
        for cap_name in ("caption_correct", "caption_distractor"):
            for ch in it[cap_name]:
                if ord(ch) > 127:
                    non_ascii[ch] += 1
    gate("GM5-ascii", sum(non_ascii.values()) == 0,
         f"non-ASCII characters in captions: "
         f"{dict(non_ascii) if non_ascii else 'none (expected none for SUSHI)'}",
         hard=False)

    # ---- build the manifest: 2 rows per item ----
    rows = []
    for it in items:
        for order in ("corrA", "corrB"):
            if order == "corrA":
                option_a, option_b, correct_letter = (
                    it["caption_correct"], it["caption_distractor"], "A")
            else:
                option_a, option_b, correct_letter = (
                    it["caption_distractor"], it["caption_correct"], "B")
            prompt = FULL_TEMPLATE.format(option_a=option_a, option_b=option_b)
            rows.append({
                "mcq_id": f"{it['item_id']}|{order}",
                "item_id": it["item_id"],
                "sample_id": it["sample_id"],
                "split": it["split"],
                "component": it["component"],
                "condition": it["condition"],
                "order": order,
                "option_a": option_a,
                "option_b": option_b,
                "correct_letter": correct_letter,
                "template_sha": TEMPLATE_SHA,
                "prompt": prompt,
            })

    # ---- GM4: manifest structure ----
    n_rows = len(rows)
    order_c = Counter(r["order"] for r in rows)
    letter_c = Counter(r["correct_letter"] for r in rows)
    ids = Counter(r["mcq_id"] for r in rows)
    dup = [k for k, v in ids.items() if v > 1]
    gate("GM4-rows",
         n_rows == 2 * n and order_c.get("corrA") == n and order_c.get("corrB") == n,
         f"rows={n_rows} (expected {2*n}); corrA={order_c.get('corrA')} "
         f"corrB={order_c.get('corrB')} (expected {n}/{n})")
    gate("GM4-letters",
         letter_c.get("A") == n and letter_c.get("B") == n,
         f"correct letter A={letter_c.get('A')} B={letter_c.get('B')} "
         f"(expected exact 50/50 = {n}/{n})")
    gate("GM4-unique", not dup, f"duplicate mcq_ids: {len(dup)} (expected 0)")

    # ---- GM6 (report): caption length by condition ----
    def words(s):
        return len(s.split())
    for c in ("swap", "random"):
        corr = [words(it["caption_correct"]) for it in items if it["condition"] == c]
        dist = [words(it["caption_distractor"]) for it in items if it["condition"] == c]
        print(f"[GM6-report] {c}: mean words correct={sum(corr)/len(corr):.2f} "
              f"distractor={sum(dist)/len(dist):.2f} "
              f"(full length-vs-margin audit stays with audit_item_balance.py)")

    # ---- GM7: template hash sanity ----
    gate("GM7-template",
         "<ts><ts/>" in FULL_TEMPLATE and "{option_a}" in USER_TEMPLATE
         and "{option_b}" in USER_TEMPLATE,
         f"template contains one series placeholder and both option slots; "
         f"sha256[:12]={TEMPLATE_SHA}")

    # ---- verdict + write ----
    print("=" * 50)
    if FAILURES:
        print(f"HARD STOP — failed gates: {FAILURES}. Nothing written.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rep_path = Path(args.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump({
            "items_in": n,
            "rows_out": n_rows,
            "unique_signals": len(sigs),
            "splits": dict(splits),
            "per_component_condition": {f"{k[0]}|{k[1]}": v
                                        for k, v in sorted(comp_cond.items())},
            "template_sha": TEMPLATE_SHA,
            "template_full": FULL_TEMPLATE,
            "non_ascii": dict(non_ascii),
        }, f, indent=2, ensure_ascii=False)

    print(f"ALL HARD GATES GREEN. Wrote {n_rows} rows -> {out_path}")
    print(f"Report -> {rep_path}")
    print(f"Template sha256[:12] = {TEMPLATE_SHA}  (the GPU runner must gate on this)")


if __name__ == "__main__":
    main()
