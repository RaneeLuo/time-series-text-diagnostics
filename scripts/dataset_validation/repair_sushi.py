from pathlib import Path

import numpy as np


DATA_ROOT = Path("data/SUSHI_tiny")

TARGETS = [
    Path("p_spike/02/0000004.csv"),
    Path("step/11/0000000.csv"),
]


def repair_npy(csv_path: Path) -> None:
    npy_path = csv_path.with_suffix(".npy")

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing source CSV: {csv_path}")

    # Use a valid neighbouring file to recover the intended shape and dtype.
    reference_candidates = sorted(
        path
        for path in csv_path.parent.glob("*.npy")
        if path != npy_path
    )

    if not reference_candidates:
        raise FileNotFoundError(
            f"No valid neighbouring NPY file found in {csv_path.parent}"
        )

    reference = np.load(reference_candidates[0], allow_pickle=False)
    values = np.loadtxt(csv_path, delimiter=",")

    if values.size != reference.size:
        raise ValueError(
            f"{csv_path}: CSV contains {values.size} values, "
            f"but neighbouring signals contain {reference.size}"
        )

    values = values.reshape(reference.shape).astype(reference.dtype, copy=False)
    np.save(npy_path, values, allow_pickle=False)

    repaired = np.load(npy_path, allow_pickle=False)

    print(
        f"Repaired {npy_path}: "
        f"shape={repaired.shape}, dtype={repaired.dtype}"
    )


def main() -> None:
    for relative_csv_path in TARGETS:
        repair_npy(DATA_ROOT / relative_csv_path)

    print("SUSHI numerical files repaired successfully.")


if __name__ == "__main__":
    main()