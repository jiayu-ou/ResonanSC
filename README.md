# ResonanSC

ResonanSC is a Python workflow for aligning single-cell RNA and ATAC batches through iterative pseudobulk correlation, in-batch merging, cross-batch alignment, and marker/mapping optimization.

## Repository Layout

```text
ResonanSC/
├── configs/                  # YAML configuration files
├── notebooks/                # Reproducible analysis notebooks
├── src/ResonanSC/            # Core Python package
├── run_resonansc.py          # End-to-end RNA workflow entry point
├── run_train_round.py        # Script version of the iterative training notebook
├── plot_round_summary_*.py   # Plotting utilities for round summaries
└── TRAIN_ROUND.md            # Detailed notes for scripted training rounds
```

Large datasets, intermediate files, model checkpoints, and generated figures are intentionally excluded from git. Place local inputs under `data/` and write run results under `outputs/`.

## Installation

Create an environment with Python 3.10 or newer, then install the package in editable mode:

```bash
pip install -e .
```

Core dependencies are declared in `pyproject.toml`. GPU training uses PyTorch; install the CUDA build that matches your system when needed.

## Input Data

The default single-RNA workflow expects an AnnData file at:

```text
data/raw/data.h5ad
```

The iterative multimodal training workflow expects the initialization checkpoint and processed h5ad inputs configured in:

```text
configs/train_round.yaml
```

Update the paths in the YAML files to match the location of the review dataset.

## Usage

Run the end-to-end RNA workflow:

```bash
python run_resonansc.py --config configs/default.yaml
```

Run one multimodal training round:

```bash
python run_train_round.py --config configs/train_round.yaml
```

Run selected stages:

```bash
python run_train_round.py \
  --config configs/train_round.yaml \
  --stages learn_P learn_mapping \
  --epochs-P 100 \
  --epochs-mapping 200 \
  --no-merge-align
```

See `TRAIN_ROUND.md` for resume options and multi-round training examples.

## Outputs

By default, outputs are written below `outputs/`, including:

- learned checkpoints (`*.pt`)
- annotated AnnData files (`*.h5ad`)
- summary JSON files
- UMAP and correlation figures
- marker tables

These files are ignored by git so that review clones stay lightweight.

## Notebooks

The notebooks in `notebooks/` are kept without execution outputs. They document initialization, training, marker-expression inspection, and marker-regulation analysis.
