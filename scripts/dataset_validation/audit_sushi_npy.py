from pathlib import Path

import numpy as np


root = Path("data/SUSHI_tiny")
files = sorted(root.rglob("*.npy"))

valid = []
incompatible = []

for path in files:
    try:
        array = np.load(path, allow_pickle=False)
        valid.append((path, array.shape, str(array.dtype)))
    except Exception as error:
        incompatible.append((path, str(error)))

print(f"Total NPY files: {len(files)}")
print(f"Loadable on this computer: {len(valid)}")
print(f"Incompatible or damaged: {len(incompatible)}")

for path, error in incompatible:
    print(f"- {path}: {error}")