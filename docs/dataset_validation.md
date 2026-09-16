# Dataset Validation

This document records how the datasets used in the thesis were obtained, inspected, repaired where necessary, and validated before experimentation.

## TRUCE

### Data location

```text
data/TRUCE/processed_data/
```

### Import issue

TRUCE was imported successfully without modifying the source data.

During schema inspection, one representation difference was identified:

- Synthetic annotations are stored as nested one-element lists.
- Stock annotations are stored as plain strings.

The validation code normalizes both formats to the same internal caption representation.

### Validation

```powershell
python scripts\dataset_validation\inspect_truce.py
```

The script verifies split sizes, time-series length, captions per instance, invalid records, uniqueness of IDs, and successful numerical loading.

### Results

#### Synthetic

| Split | Instances |
|---|---:|
| Train | 448 |
| Validation | 56 |
| Test | 56 |
| Total | 560 |

- All series have length 12.
- Every instance has exactly 3 captions.
- Invalid records: 0.

#### Stock

| Split | Instances |
|---|---:|
| Train | 1,520 |
| Validation | 190 |
| Test | 190 |
| Total | 1,900 |

- All series have length 12.
- Every instance has exactly 3 captions.
- Invalid records: 0.

Across both datasets, all 2,460 IDs are unique.

### Outcome

TRUCE was imported and validated successfully.

Validation figure:

```text
results/dataset_validation/truce_samples.png
```

---

## SUSHI Tiny

### Data location

```text
data/SUSHI_tiny/
```

The dataset contains seven top-level categories:

```text
clean/
n_spike/
noisy/
p_spike/
pn_spike/
smooth/
step/
```

Metadata file:

```text
data/SUSHI_tiny/generated_files_list.csv
```

### Problems encountered

The SUSHI GitHub repository contains documentation, but the dataset itself must be downloaded separately from the official Zenodo release.

Two technical issues were encountered on Windows.

#### 1. Archive extraction failures

PowerShell `Expand-Archive` and Windows `tar.exe` did not extract every archive member successfully.

A custom extraction script was created:

```powershell
python scripts\dataset_validation\extract_sushi.py
```

The affected archive members were:

```text
p_spike/02/0000004.npy
p_spike/02/0000004.png
step/11/0000000.npy
step/11/0000000.png
```

#### 2. NumPy compatibility on Windows

The supplied `.npy` files use an extended-precision floating-point representation that is not reliably supported by the Windows NumPy environment used for this project.

### Diagnosis

The following scripts were used:

```text
scripts/dataset_validation/audit_sushi_npy.py
scripts/dataset_validation/validate_sushi_csv.py
scripts/dataset_validation/repair_sushi.py
scripts/dataset_validation/inspect_sushi.py
```

The audit showed that the CSV signal files were complete and readable. The corresponding CSV files for the damaged archive members were intact.

### Resolution

The CSV files were selected as the canonical SUSHI representation on Windows.

The two missing `.npy` signal files were reconstructed from their corresponding CSV files for completeness, but the thesis implementation will load SUSHI signals from CSV.

### Validation

```powershell
python scripts\dataset_validation\validate_sushi_csv.py
python scripts\dataset_validation\inspect_sushi.py
```

The checks include metadata rows, unique signal paths, missing files, signal length, caption count, class count, samples per class, category distribution, and sample visualization.

### Results

| Property | Result |
|---|---:|
| Signal-caption pairs | 1,400 |
| Unique signal paths | 1,400 |
| Unique captions | 1,390 |
| Classes | 140 |
| Samples per class | 10 |
| Signal length | 2,048 |
| Missing signal files | 0 |
| Signals with incorrect length | 0 |

Each top-level category contains 200 instances.

### Outcome

SUSHI Tiny was imported and validated successfully using the CSV representation.

Validation figure:

```text
results/dataset_validation/sushi_samples.png
```

---

## Reproducibility decision

The raw datasets remain under `data/` and are excluded from Git.

The repository tracks extraction and repair scripts, schema inspection scripts, validation scripts, validation figures, and documentation of data-handling decisions.
