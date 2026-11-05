# ResonanSC

ResonanSC is a Python workflow for aligning single-cell RNA and ATAC populations across batches and modalities. It initializes clusters from RNA PCA and ATAC peak LSI, merges and matches clusters using pseudobulk similarity, resolves ambiguous ATAC-only labels, and initializes and optimizes cell assignments (`P`), peak-to-gene mapping (`W`), and gene markers (`M`). An optional dual-modality embedding uses the optimized mapping.

This repository provides the source code, configuration templates, tests, and a **small real-data example** derived from paired 10x Multiome PBMC 10k cells. The example runs alignment, `P/W/M` initialization, and one CPU optimization epoch each for `W` and `M`. Its short training schedule is a functionality check, not a biological performance evaluation.

## Quick start: bundled PBMC minimal test

Requirements: Python 3.10 or newer and enough memory for the installed scientific Python and PyTorch dependencies. The quick start uses a CPU; no external data download, GTF download, GPU, or pretrained model is needed.

```bash
git clone https://github.com/jiayu-ou/ResonanSC.git
cd ResonanSC
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/run_pbmc_minimal.py
```

The final line prints `PASS` only after checking all 800 input cells, creating an initialization checkpoint, and finding the optimized `M` checkpoint. Outputs are written to a new `pbmc_minimal_run/` directory:

| File | Contents |
| --- | --- |
| `config.yaml` | Exact alignment configuration |
| `alignment/cell_labels.csv` | One aligned label per input cell |
| `alignment/hierarchical_alignment_checkpoint.pt` | Alignment state |
| `alignment/prepared/init_for_training.pt` | Initialized `P/W/M` model |
| `optimization.yaml` | One-epoch CPU optimization configuration |
| `optimization/train_M_1.pt` | Final minimal-test checkpoint |

Supply `--output PATH` to choose a different **new** directory. ResonanSC refuses to overwrite an existing alignment or optimization output. The test normally takes longer on the first run while dependencies initialize.

The bundled matrices have 400 paired raw barcodes represented as 400 RNA and 400 ATAC rows, 500 shared expression genes, and 2,000 raw-count peaks. A matching GRCh38 GENCODE v43 gene-only GTF subset is included for mapping initialization. The sampling seed is 2026; only input matrices determine the selected genes and peaks. Cell-type annotations are excluded. See [data/pbmc_minimal/README.md](data/pbmc_minimal/README.md) for the data recipe. To regenerate the files from the prepared full PBMC matrices:

```bash
python data/pbmc_minimal/make_subset.py /path/to/resonansc_integrated \
  --gtf /path/to/gencode.v43.annotation.gtf.gz
```

The source directory must contain `expression.h5ad` and `peak_counts.h5ad`. Regeneration is optional because the subset is committed to this repository.

## Algorithm overview

The workflow consists of four stages. The first and third stages are exercised by the bundled example. The second is used when alignment leaves ATAC-only clusters. The fourth is an optional GPU embedding workflow.

1. **Hierarchical alignment.** RNA PCA and ATAC peak LSI initialize Leiden clusters within a batch or condition. Small or highly similar clusters are merged. Same-modality clusters across batches are matched with pairwise Hungarian alignment and rescue rules. RNA and ATAC clusters are then matched in a shared gene-activity space, using significant RNA one-versus-rest genes for cross-modality scoring. The output assigns every cell a final cluster and records similarity and matching audits.
2. **Pairwise label resolution.** For ATAC-only clusters, compare normalized log1p gene-activity profiles with RNA cluster profiles using Pearson correlation. If the two best candidates differ by at most the specified margin, rerank them using the largest absolute RNA rank-sum features. RNA labels remain fixed. This step is exploratory label resolution; its rank-sum features are not donor-level differential-expression results.
3. **`P/W/M` initialization and optimization.** Convert aligned labels to initial cell-assignment logits `P`. Build a peak-to-gene mapping prior `W` from a matching genome annotation and initialize marker logits `M` from one-versus-rest marker information. The optimizer can update `P`, `W`, and `M` in configured stages. The supplied `optimization_WM20.yaml` template holds `P` fixed and updates `W` and `M` for 20 epochs each. The minimal test reduces those two stages to one epoch each.
4. **Optional W-LSI embedding.** Map ATAC signals into gene space using a trusted optimized `W`, calculate pooled LSI, and train RNA and ATAC encoders with parent-label supervision and an MMD objective. This stage requires a compatible NVIDIA GPU and `scvi-tools==1.3.3`. It is not part of the CPU minimal test.

## Installation

```bash
python -m pip install -e .
python -m pip install -e '.[embedding]'  # only for optional GPU embedding
```

Core dependencies are listed in [pyproject.toml](pyproject.toml). Optimization can use a PyTorch-compatible CPU or GPU. Load PyTorch checkpoints only from trusted sources because deserialization can execute code.

## Input contract

| Input | Required content |
| --- | --- |
| `expression_h5ad` | AnnData with cells × the same selected genes for RNA expression and ATAC gene activity. Cell and gene names must be unique. `obs['batch']` identifies `rna_*` and `atac_*` batches. The number of columns must equal `alignment.n_top_genes`. |
| `peak_h5ad` | AnnData containing ATAC cells × peaks with finite, nonnegative **integer raw counts** and matching ATAC barcodes. Gene activity or TF-IDF values cannot replace peak counts. |
| `rna_pca` and `rna_cell_names` | RNA PCA array and a barcode list in exactly the same row order. |
| `atac_lsi` and `atac_cell_names` | LSI of the **raw peak matrix** and a barcode list in exactly the same row order. The algorithm uses up to the first 30 columns. |
| `gtf` | Needed for `P/W/M` preparation. Its genome build must match the peak build, and the selected genes must be represented. |
| `condition` and `alignment.condition_groups` | Needed only in condition mode. Each batch must have one condition, and the configuration must specify exactly two condition groups with RNA and ATAC membership. |

The LSI used for initial ATAC alignment is distinct from the later W-mapped LSI used for embedding. The included PBMC runner computes both RNA PCA and peak LSI from the bundled subset and preserves barcode order.

## 1. Align batches or conditions

Edit [examples/latest_hierarchy.yaml](examples/latest_hierarchy.yaml) for your paths and thresholds. Its condition settings are a study-specific template, not universal defaults. For the default batch mode:

```python
import numpy as np
import resonansc as rs

rna_pca = np.load("rna_pca.npy")
rna_cells = np.load("rna_cell_names.npy", allow_pickle=False)
atac_lsi = np.load("atac_peak_lsi.npy")
atac_cells = np.load("atac_cell_names.npy", allow_pickle=False)

run_dir = rs.align(
    "examples/latest_hierarchy.yaml",
    rna_pca=rna_pca, rna_cell_names=rna_cells,
    atac_lsi=atac_lsi, atac_cell_names=atac_cells,
)
```

Batch mode starts with within-batch Leiden clusters and modality-specific merges. Clusters below 50 cells are merged to the most similar cluster; clusters with 50–99 cells are merged only when the relevant similarity threshold is met. It then aligns all batches within each modality and aligns the two modalities.

For the study's two-condition hierarchy, pass `mode="condition"` to `rs.align` and explicitly configure `alignment.condition_groups`. This mode performs within-condition premerge, donor alignment, within-condition postmerge, cross-condition alignment, and cross-modality alignment. It does not perform batch mode's small-cluster merge. Verify condition annotations for each RNA and ATAC batch before running.

The output directory must be new. `cell_labels.csv`, `hierarchical_alignment_checkpoint.pt`, step-level files, and Hungarian/rescue audit tables are saved there. No input cells are dropped.

## 2. Resolve ATAC-only labels when needed

Use an AnnData object with RNA rows containing raw counts and ATAC rows containing nonnegative `W`-mapped gene activity in the **same genes**. Its `obs` must include `modality` and `final_alignment`. The `W` source should be reported: an existing trusted optimized checkpoint or an explicitly described initialization.

```python
import anndata as ad
import resonansc as rs

adata = ad.read_h5ad("rna_atac_common_gene_activity.h5ad")
labels, decisions, selected_genes = rs.resolve_labels(
    adata, margin=0.02, n_genes=100
)
labels.index.name = "cell_barcode"
labels["final_alignment"] = labels["pairwise_label"]
labels.to_csv("pairwise_labels.csv")
labels.reset_index().to_csv("pairwise_labels_for_embedding.csv", index=False)
decisions.to_csv("pairwise_decisions.csv", index=False)
selected_genes.to_csv("pairwise_selected_genes.csv", index=False)
```

Cells are normalized to a total of 10,000 and log1p transformed before cluster-profile correlation. The resolver maps each ATAC-only cluster as a whole; it does not change RNA labels. It has no minimum-correlation safeguard. If RNA-only clusters exist, set `allow_rna_only=True` to use all RNA clusters as candidates. Any RNA-only class that remains unmatched by ATAC does not gain class-specific cross-modality supervision.

## 3. Prepare and optimize P/W/M

Add `input.gtf`, `input.peak_genome_build`, and `input.gtf_genome_build` to the alignment YAML. Use the same genome build for both inputs.

```python
import resonansc as rs

init_checkpoint = rs.prepare_from_alignment(
    "examples/latest_hierarchy.yaml",
    alignment_dir=run_dir,
    labels=labels, label_key="pairwise_label",
)
```

Omit `labels` to use the original alignment labels. Preparation creates `prepared/init_for_training.pt`, per-batch training data, aligned gene activity, and audit information. Initial assignment logits are 3/-5 for assigned/unassigned classes; `W` starts from a distance-based peak-gene prior; `M` starts from one-versus-rest marker estimates. `rs.prepare_optimization(...)` can also replace labels and rebuild `M` while retaining an existing trusted `W` initialization.

Edit [examples/optimization_WM20.yaml](examples/optimization_WM20.yaml) with the checkpoint, training data, output path, and device, then run:

```python
rs.optimize("examples/optimization_WM20.yaml")
```

The template updates `W` and `M` for 20 epochs each with `P` fixed. It does not establish that this schedule is optimal for another dataset. The output contains stage checkpoints, losses, and labels. Use a new output directory.

The model's saved weighted RNA pseudobulks can be read with `rs.model_pseudobulk(checkpoint, annotation)`. `annotation` must be indexed by unique cell barcode and contain at least `batch` and `final_alignment`. These arrays are model-weighted signals over the model's selected genes, **not** summed raw counts for count-based differential-expression tools.

## 4. Optional dual-modality embedding

Install `.[embedding]` and supply a trusted optimized `W` checkpoint, RNA counts, labels, and reviewed parent-label files.

```python
import anndata as ad
import numpy as np
import resonansc as rs

rs.prepare_embedding(
    "rna_counts.h5ad", "pairwise_labels_for_embedding.csv",
    "output/optimization_WM20/train_M_1.pt",
    "output/hierarchy/prepared/training_multidata",
    "output/embedding_input.h5ad",
)
adata = ad.read_h5ad("output/embedding_input.h5ad")
lsi, atac_cells, _ = rs.atac_w_lsi(adata)
np.save("output/atac_lsi.npy", lsi)
np.save("output/atac_cells.npy", atac_cells)
rs.train_embedding(
    input_path="output/embedding_input.h5ad",
    rna_map="rna_cluster_map.json",
    parents="atac_parent_sets.csv",
    atac_lsi_values="output/atac_lsi.npy",
    atac_lsi_names="output/atac_cells.npy",
    output="output/embedding",
    weight=200, latent=16, hidden=256,
)
```

`rna_cluster_map.json` maps integer RNA cluster IDs to parent labels. `atac_parent_sets.csv` has columns `atac_batch,atac_cluster,parent_candidates`, with multiple candidates separated by `|`. These parent labels are supplied and reviewed by the user. The default schedule pretrains each modality for 100 epochs and jointly trains for 50 epochs. It uses all cells in pretraining and balances known single-label classes during joint training. Cells with only ambiguous or unknown parent sets are not sampled by that single-label sampler. W-LSI splits positive and negative components into nonnegative features for an NB decoder; this is an approximation rather than a strict count model. Missing shared labels or empty groups cause an error.

Outputs include `model_RNA.pt`, `model_ATAC.pt`, `X_latent.npy`, and `embedding_results.h5ad` with `obsm["X_embedding"]`. UMAP is not run automatically. `rs.infer_embedding(...)` requires the same genes, known batches, and ATAC LSI in training barcode order.

## Tests, scope, and provenance

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_hierarchy.py
```

The unit tests cover input contracts, cell accounting, sparse rank-sum behavior, label preservation, pseudobulk export, optimizer initialization, and LSI. The synthetic smoke test runs alignment, preparation, and one W/M epoch. Set `RESONANSC_TEST_GPU=1` to enable GPU embedding train/infer tests on suitable hardware.

The PBMC minimal test checks execution and cell accounting on real data; it is not a biological accuracy benchmark. Historical comparisons against saved study results are recorded in [PROVENANCE.md](PROVENANCE.md), together with source-script origins. Upstream redistribution terms have not been confirmed, so this repository does not assert a license for those components.

For a manuscript's source-code availability statement, link directly to [the public repository](https://github.com/jiayu-ou/ResonanSC) and identify the bundled PBMC minimal test.
