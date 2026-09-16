from collections import Counter
from pathlib import Path
import re

import numpy as np


DATA_ROOT = Path("data/SUSHI_tiny")
SIGNAL_NAME_PATTERN = re.compile(r"^\d{7}$")


def main() -> None:
    all_csv_files = sorted(DATA_ROOT.rglob("*.csv"))

    # Signal files have names such as 0000004.csv.
    signal_files = [
        path
        for path in all_csv_files
        if SIGNAL_NAME_PATTERN.fullmatch(path.stem)
    ]

    # Other CSV files contain metadata/captions and should not be loaded as signals.
    metadata_files = [
        path
        for path in all_csv_files
        if not SIGNAL_NAME_PATTERN.fullmatch(path.stem)
    ]

    invalid_files = []
    shapes = Counter()

    for path in signal_files:
        try:
            signal = np.loadtxt(path, delimiter=",")
        except Exception as error:
            invalid_files.append((path, f"Could not load: {error}"))
            continue

        shapes[signal.shape] += 1

        if signal.size != 2048:
            invalid_files.append(
                (path, f"Expected 2048 values, found {signal.size}")
            )

    print(f"All CSV files: {len(all_csv_files)}")
    print(f"Signal CSV files: {len(signal_files)}")
    print(f"Metadata/caption CSV files: {len(metadata_files)}")
    print(f"Signal shapes: {dict(shapes)}")
    print(f"Invalid signal files: {len(invalid_files)}")

    if metadata_files:
        print("\nMetadata/caption CSV files:")
        for path in metadata_files:
            print(f"- {path}")

    if invalid_files:
        print("\nInvalid signal files:")
        for path, error in invalid_files:
            print(f"- {path}: {error}")
        raise SystemExit(1)

    if len(signal_files) != 1400:
        raise SystemExit(
            f"Expected approximately 1400 signal files, found {len(signal_files)}"
        )

    print("\nSUSHI Tiny CSV validation successful.")


if __name__ == "__main__":
    main()