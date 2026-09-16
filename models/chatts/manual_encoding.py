"""
Manual encoding path for ChatTS: rebuilds the stock processor's assembly
(prefix splice + tokenization + series tensor) by hand, so that the
prefix TEXT and the series TENSOR can be decoupled — needed for the
A/B/C sub-probes (donor/frozen prefixes) and the PJ prefix-jitter
control. With no overrides it must reproduce the stock processor's
output token-for-token and bit-for-bit (selftest_manual_path.py gates
this on every real manifest row before the path is trusted).

Deliberate engineering choice: the ARITHMETIC (series -> prefix numbers
and series -> tensor) is imported from the checkpoint's own downloaded
processing_qwen2_ts.py at the pinned revision — never re-implemented.
Only the assembly is hand-built here.

Batching: always encode one prompt at a time (batch of 1) so padding
can never differ between paths. The GPU runner follows the same rule
for encoding.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

TS_PLACEHOLDER = "<ts><ts/>"


def load_checkpoint_sp_encoding(checkpoint_meta_dir: str):
    """Import sp_encoding from the pinned checkpoint's own processing file."""
    path = Path(checkpoint_meta_dir) / "processing_qwen2_ts.py"
    spec = importlib.util.spec_from_file_location("chatts_processing_pinned", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sp_encoding


def prefix_text_of(series, sp_encoding) -> str:
    """The bare prefix string (without the <ts><ts/> tail) for a series,
    computed by the checkpoint's own function."""
    _, prompt, _ = sp_encoding(np.asarray(series, dtype=np.float64), eots_token=True)
    assert prompt.endswith(TS_PLACEHOLDER), f"unexpected sp prompt tail: {prompt[-20:]!r}"
    return prompt[:-len(TS_PLACEHOLDER)]


def manual_encode(prompt: str, entries: list, tokenizer, sp_encoding,
                  padding=True, padding_side="left"):
    """
    prompt: text containing one <ts><ts/> placeholder per entry.
    entries: list of dicts, one per placeholder, each with:
        'tensor_source' : raw series (list/ndarray) -> becomes the tensor,
                          scaled by ITS OWN mean/scale (checkpoint math)
        'prefix_source' : optional raw series -> prefix computed from it
                          (defaults to tensor_source = stock behaviour)
        'prefix_text'   : optional explicit prefix string (wins over
                          prefix_source; must include its own brackets)
    Returns dict(input_ids, attention_mask, timeseries) matching the
    stock processor's single-prompt output format.
    """
    segments = prompt.split(TS_PLACEHOLDER)
    assert len(entries) == len(segments) - 1, (
        f"{len(segments)-1} placeholders vs {len(entries)} entries")

    reconstructed = segments[0]
    encoded_arrays = []
    for i, e in enumerate(entries):
        tensor_src = np.asarray(e["tensor_source"], dtype=np.float64)
        encoded_ts, _, _ = sp_encoding(tensor_src, eots_token=True)

        if e.get("prefix_text") is not None:
            prefix = e["prefix_text"]
        elif e.get("prefix_source") is not None:
            prefix = prefix_text_of(e["prefix_source"], sp_encoding)
        else:
            prefix = prefix_text_of(tensor_src, sp_encoding)

        reconstructed += prefix + TS_PLACEHOLDER + segments[i + 1]
        encoded_arrays.append(encoded_ts[None, ...])

    tok_out = tokenizer([reconstructed], padding=padding,
                        padding_side=padding_side, return_tensors="pt")

    timeseries = None
    if encoded_arrays:
        max_len = max(a.shape[1] for a in encoded_arrays)
        padded = [np.pad(a, ((0, 0), (0, max_len - a.shape[1]), (0, 0)),
                         mode="constant", constant_values=0.0)
                  for a in encoded_arrays]
        timeseries = torch.from_numpy(np.concatenate(padded, axis=0)).half()

    return {"input_ids": tok_out["input_ids"],
            "attention_mask": tok_out["attention_mask"],
            "timeseries": timeseries,
            "reconstructed_prompt": reconstructed}
