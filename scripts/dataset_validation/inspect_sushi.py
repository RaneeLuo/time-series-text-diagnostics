from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = Path("data/SUSHI_tiny")
METADATA_PATH = DATA_ROOT / "generated_files_list.csv"
# OUTPUT_PATH = Path("results/sushi_samples.png")
OUTPUT_PATH = Path("results/dataset_validation/sushi_samples.png")
EXPECTED_LENGTH = 2048


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)

    required_columns = {"File path", "Caption", "Class"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing metadata columns: {sorted(missing_columns)}")

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    if df["File path"].duplicated().any():
        duplicates = df.loc[df["File path"].duplicated(), "File path"].tolist()
        raise ValueError(f"Duplicate file paths found: {duplicates[:10]}")

    if df["Caption"].isna().any() or df["Caption"].str.strip().eq("").any():
        raise ValueError("Missing or empty captions found.")

    if df["Class"].isna().any() or df["Class"].str.strip().eq("").any():
        raise ValueError("Missing or empty class labels found.")

    missing_files = []
    invalid_lengths = []

    for relative_path in df["File path"]:
        signal_path = DATA_ROOT / Path(relative_path)

        if not signal_path.exists():
            missing_files.append(str(signal_path))
            continue

        signal = np.loadtxt(signal_path, delimiter=",")

        if signal.size != EXPECTED_LENGTH:
            invalid_lengths.append((str(signal_path), signal.size))

    print(f"Metadata rows: {len(df)}")
    print(f"Unique signal paths: {df['File path'].nunique()}")
    print(f"Unique captions: {df['Caption'].nunique()}")
    print(f"Unique class labels: {df['Class'].nunique()}")
    print(f"Missing signal files: {len(missing_files)}")
    print(f"Signals with incorrect length: {len(invalid_lengths)}")

    if missing_files:
        print("\nMissing files:")
        for path in missing_files[:10]:
            print(f"- {path}")
        raise SystemExit(1)

    if invalid_lengths:
        print("\nIncorrect signal lengths:")
        for path, length in invalid_lengths[:10]:
            print(f"- {path}: {length}")
        raise SystemExit(1)

    class_counts = df["Class"].value_counts()

    print("\nSamples per class:")
    print(class_counts.describe().to_string())

    top_level_categories = (
        df["File path"]
        .map(lambda path: Path(path).parts[0])
        .value_counts()
        .sort_index()
    )

    print("\nTop-level signal categories:")
    print(top_level_categories.to_string())

    print("\nSUSHI validation successful.")


def plot_samples(df: pd.DataFrame) -> None:
    df = df.copy()
    df["Category"] = df["File path"].map(lambda path: Path(path).parts[0])

    categories = sorted(df["Category"].unique())

    figure, axes = plt.subplots(
        nrows=len(categories),
        ncols=1,
        figsize=(12, 2.5 * len(categories)),
        constrained_layout=True,
    )

    if len(categories) == 1:
        axes = [axes]

    for axis, category in zip(axes, categories):
        row = df.loc[df["Category"] == category].iloc[0]

        signal_path = DATA_ROOT / Path(row["File path"])
        signal = np.loadtxt(signal_path, delimiter=",").reshape(-1)

        axis.plot(signal)
        axis.set_title(
            f"{category} | {row['Class']}\n"
            f"{row['Caption']}"
        )
        axis.set_xlabel("Time index")
        axis.set_ylabel("Value")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)

    print(f"Sample plot saved to: {OUTPUT_PATH}")


def main() -> None:
    metadata = load_metadata()
    validate_dataset(metadata)
    plot_samples(metadata)


if __name__ == "__main__":
    main()