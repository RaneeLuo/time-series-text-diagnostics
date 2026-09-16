from __future__ import annotations

import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DATA_ROOT = Path("data/TRUCE/processed_data")
# OUTPUT_PATH = Path("results/truce_samples.png")
OUTPUT_PATH = Path("results/dataset_validation/truce_samples.png")

EXPECTED_LENGTH = 12

DATASETS = {
    "synthetic": {
        "train": "pilot13finaltrain.json",
        "val": "pilot13finalval.json",
        "test": "pilot13finaltest.json",
        "expected_total": 560,
    },
    "stock": {
        "train": "pilot16btrain.json",
        "val": "pilot16bval.json",
        "test": "pilot16btest.json",
        "expected_total": 1900,
    },
}


def load_json(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing TRUCE file: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a dictionary in {path}, found {type(data).__name__}")

    return data


def normalize_annotations(raw_annotations: Any) -> list[str]:
    """Convert both TRUCE annotation formats into list[str]."""
    if not isinstance(raw_annotations, list):
        raise TypeError(
            f"Expected annotations to be a list, found "
            f"{type(raw_annotations).__name__}"
        )

    captions: list[str] = []

    for annotation in raw_annotations:
        if isinstance(annotation, str):
            caption = annotation

        elif (
            isinstance(annotation, list)
            and len(annotation) == 1
            and isinstance(annotation[0], str)
        ):
            caption = annotation[0]

        else:
            raise ValueError(f"Unexpected annotation format: {annotation!r}")

        caption = caption.strip()

        if not caption:
            raise ValueError("Empty caption found")

        captions.append(caption)

    return captions


def validate_dataset() -> dict[str, dict[str, list[dict[str, Any]]]]:
    loaded: dict[str, dict[str, list[dict[str, Any]]]] = {}

    all_ids: set[str] = set()
    duplicate_ids: list[str] = []

    for dataset_name, config in DATASETS.items():
        loaded[dataset_name] = {}

        total_instances = 0
        caption_counts = Counter()
        series_lengths = Counter()
        invalid_records: list[str] = []

        print("=" * 72)
        print(f"Dataset: {dataset_name}")

        for split_name in ("train", "val", "test"):
            path = DATA_ROOT / config[split_name]
            raw_data = load_json(path)

            records: list[dict[str, Any]] = []

            for key, record in raw_data.items():
                try:
                    if not isinstance(record, dict):
                        raise TypeError("Record is not a dictionary")

                    record_id = str(record["id"])
                    series = record["series"]
                    captions = normalize_annotations(record["annotations"])

                    if record_id != key:
                        raise ValueError(
                            f"Dictionary key {key!r} does not match "
                            f"record id {record_id!r}"
                        )

                    if not isinstance(series, list):
                        raise TypeError("Series is not stored as a list")

                    if len(series) != EXPECTED_LENGTH:
                        raise ValueError(
                            f"Expected length {EXPECTED_LENGTH}, "
                            f"found {len(series)}"
                        )

                    if len(captions) != 3:
                        raise ValueError(
                            f"Expected 3 captions, found {len(captions)}"
                        )

                    if record_id in all_ids:
                        duplicate_ids.append(record_id)

                    all_ids.add(record_id)
                    series_lengths[len(series)] += 1
                    caption_counts[len(captions)] += 1

                    records.append(
                        {
                            "id": record_id,
                            "series": series,
                            "captions": captions,
                            "meta": record.get("meta", {}),
                        }
                    )

                except Exception as error:
                    invalid_records.append(f"{key}: {error}")

            loaded[dataset_name][split_name] = records
            total_instances += len(records)

            print(f"{split_name:>5}: {len(records):4d}")

        print(f"total: {total_instances:4d}")
        print(f"series lengths: {dict(series_lengths)}")
        print(f"captions per instance: {dict(caption_counts)}")
        print(f"invalid records: {len(invalid_records)}")

        expected_total = int(config["expected_total"])

        if total_instances != expected_total:
            raise ValueError(
                f"{dataset_name}: expected {expected_total} instances, "
                f"found {total_instances}"
            )

        if invalid_records:
            print("\nFirst invalid records:")
            for message in invalid_records[:10]:
                print(f"- {message}")
            raise SystemExit(1)

    if duplicate_ids:
        raise ValueError(
            f"Duplicate IDs found across splits: {duplicate_ids[:10]}"
        )

    print("=" * 72)
    print(f"Unique IDs across both datasets: {len(all_ids)}")
    print("TRUCE validation successful.")

    return loaded


def plot_samples(
    datasets: dict[str, dict[str, list[dict[str, Any]]]]
) -> None:
    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 8),
        constrained_layout=True,
    )

    for row_index, dataset_name in enumerate(("synthetic", "stock")):
        for column_index, split_name in enumerate(("train", "val", "test")):
            axis = axes[row_index, column_index]
            sample = datasets[dataset_name][split_name][0]

            axis.plot(range(EXPECTED_LENGTH), sample["series"], marker="o")

            caption = textwrap.fill(sample["captions"][0], width=38)
            axis.set_title(
                f"{dataset_name} — {split_name}\n"
                f"{sample['id']}\n"
                f"{caption}",
                fontsize=9,
            )

            axis.set_xlabel("Time index")
            axis.set_ylabel("Value")
            axis.set_xticks(range(EXPECTED_LENGTH))
            axis.grid(alpha=0.3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)

    print(f"Sample plot saved to: {OUTPUT_PATH}")


def main() -> None:
    datasets = validate_dataset()
    plot_samples(datasets)


if __name__ == "__main__":
    main()