from pathlib import Path
import json
import pprint


DATA_ROOT = Path("data/TRUCE/processed_data")


def show_sample(path: Path) -> None:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print("=" * 80)
    print(f"File: {path.name}")
    print(f"Top-level type: {type(data).__name__}")
    print(f"Top-level length: {len(data)}")

    if isinstance(data, list):
        if not data:
            print("The file is empty.")
            return

        sample = data[0]

    elif isinstance(data, dict):
        if not data:
            print("The file is empty.")
            return

        first_key = next(iter(data))
        print(f"First top-level key: {first_key!r}")
        sample = data[first_key]

    else:
        print("Unexpected JSON structure.")
        return

    print(f"Sample type: {type(sample).__name__}")

    if isinstance(sample, dict):
        print(f"Sample keys: {list(sample.keys())}")

    print("\nFirst sample:")
    pprint.pp(sample, width=120, sort_dicts=False)


def main() -> None:
    json_files = sorted(DATA_ROOT.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {DATA_ROOT}")

    print(f"JSON files found: {len(json_files)}")

    for path in json_files:
        show_sample(path)


if __name__ == "__main__":
    main()