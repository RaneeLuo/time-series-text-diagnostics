# Judging rules — `q_pins_falsely` (pinning spot-check)

**The question, unpacked.** For each row, the true signal has fluctuation
`swap_from`; the replacement clause (`clause_replaced_to`) claims `swap_to`.
Write **y** only if BOTH halves hold:

- **(a) It genuinely asserts it.** The clause states, affirmatively and about
  this signal, that the `swap_to` fluctuation is present.
- **(b) The assertion is false.** Given the signal's true fluctuation is
  `swap_from`, the claim as worded is wrong — not merely different, but not a
  fair description of what's actually there.

Write **n** if either half fails, and say which in `notes` ("a: ..." or
"b: ..."). Convention: plain-language reading, same standard as the main
validation — judge what the words say to a competent reader, not what the
dataset's class vocabulary intends.

**Judge only the swapped clause.** The rest of the caption is unchanged from
the correct caption and is supposed to remain true — that is by design, not a
flaw. (Verified mechanically for all items.)

## When to slow down

Every clause in this pile contains the naming keywords — that's how it was
selected — so most rows are a fast **y**. The keywords can still mislead in
three ways; these are exactly what the spot-check exists to catch:

1. **Negation / absence.** "No negative spikes appear" contains the keywords
   but asserts the opposite → fails (a) → **n**.
2. **Hedging so soft it stops being a claim.** "may occasionally show small
   negative spikes" — if you judge it too weak to assert presence → fails
   (a) → **n**. Ordinary confident phrasing ("negative spikes are displayed",
   "the signal exhibits...") passes (a).
3. **Claim true anyway — the subset trap.** Watch rows where `swap_from` is
   `positive-and-negative spike` and `swap_to` is `negative spike` (or
   `positive spike`): a signal with both kinds of spikes really does contain
   negative spikes, so "negative spikes are displayed" is literally TRUE →
   fails (b) → **n**, note "b: subset — claim true of pn-spike signal".
   The reverse direction is fine: claiming *both* polarities about a
   single-polarity signal is false. Similar care if a `noisy` claim is made
   about a signal whose true fluctuation could pass as noise — but `smooth`,
   `step`, and spike signals are not "noisy" in SUSHI's sense, so this
   should be rare.

## Mechanics

- Work strictly top-down in `reading_order`; stop only at a batch boundary,
  and only for time — never because of the results so far.
- Blank = unread. Every read row gets y or n; every n gets a note.
- If you hesitate more than ~20 seconds, make the strict call (n) and write
  the hesitation into `notes` — hesitations are data.
- Optional but useful: put the date in `notes` of the first row of each
  batch, so batches read on different days are identifiable.

**Criterion (pre-registered):** cumulative pass rate ≥ 95% at each completed
batch boundary. Registered expectation: ~1% failures or fewer. Several `n`s
sharing one mechanism → stop at the boundary and report; that pattern matters
more than the count.
*(Batch-1 outcome: 84/100 — criterion failed; mechanisms identified and
codified below. Batches 2+ continue under rules R1–R4 for consistency.)*

## Decision rules fixed after batch 1 (codifying the standard batch 1 used)

- **R1 — subset (uniform in batch 1, 7/7):** `swap_from` is
  `positive-and-negative spike` and `swap_to` is `positive spike` or
  `negative spike` → **q3 = n** ("b: subset — claim true"). Exception only if
  the clause asserts exclusivity ("only positive spikes"), which batch 1
  never saw.
- **R2 — noise vs spikes (the batch-1 pervasiveness line, 12/12 consistent):**
  `swap_to` = `noisy`, `swap_from` = any spike type → **q3 = y only if** the
  clause asserts pervasive/continuous noise — "throughout", "permeates",
  "fills", "covers the signal" — which spikes cannot satisfy, making the
  claim genuinely false. Bare magnitude ("significant/considerable noise") or
  spike-compatible frequency words ("intermittent", "sporadic", "frequent")
  → **q3 = n**.
- **R3 — bare existential (row 15):** a clause asserting the class with no
  magnitude or pervasiveness at all ("Noise is exhibited") is compatible with
  the minor-noise baseline every SUSHI signal has → **q2 = n**.
- **R4 — truncation defect (rows 56, 82):** any clause containing the
  malformed "Large part," (truncated "For the most part") → **q1 = n**.

These rules were fixed at the batch-1 boundary, before batch 2 was read, and
apply unchanged to all remaining batches. If a new mechanism appears, fail
the item with a note and flag it at the next boundary rather than inventing a
rule mid-batch.
