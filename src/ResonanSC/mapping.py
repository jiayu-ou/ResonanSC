from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence
import copy

import numpy as np
import pandas as pd
import torch


def _parse_attributes(value: str) -> Dict[str, str]:
    attrs = {}
    for field in value.strip().strip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        key, _, raw = field.partition(" ")
        attrs[key] = raw.strip().strip('"')
    return attrs


def read_gtf_genes(
    gtf_file,
    gene_names: Sequence[str],
    feature_type: str = "gene",
) -> pd.DataFrame:
    """Read coordinates for genes in the requested output gene space."""
    wanted = set(map(str, gene_names))
    rows = []

    with Path(gtf_file).open() as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != feature_type:
                continue

            attrs = _parse_attributes(fields[8])
            gene_name = attrs.get("gene_name") or attrs.get("gene_id")
            if gene_name not in wanted:
                continue

            start = int(fields[3]) - 1
            end = int(fields[4])
            strand = fields[6]
            tss = start if strand == "+" else end - 1
            rows.append((fields[0], start, end, strand, tss, gene_name))

    genes = pd.DataFrame(
        rows,
        columns=["chrom", "start", "end", "strand", "tss", "gene_name"],
    )
    genes = genes.drop_duplicates("gene_name").set_index("gene_name")

    missing = [g for g in gene_names if str(g) not in genes.index]
    if missing:
        print(f"[mapping] genes absent from GTF: {len(missing)}")
    return genes


def parse_peak_names(peak_names: Iterable[str]) -> pd.DataFrame:
    """Parse peak names formatted as chr:start-end or chr_start_end."""
    rows = []
    for peak_idx, raw_name in enumerate(map(str, peak_names)):
        normalized = raw_name.replace(":", "-").replace("_", "-")
        parts = normalized.rsplit("-", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Cannot parse peak {raw_name!r}; expected chr:start-end."
            )
        chrom, start, end = parts
        rows.append((peak_idx, raw_name, chrom, int(start), int(end)))
    return pd.DataFrame(
        rows, columns=["peak_idx", "peak_name", "chrom", "start", "end"]
    )


def build_peak_gene_mapping_init(
    atacdata,
    gene_names: Sequence[str],
    gtf_file,
    batch_key: str,
    max_distance: int = 250_000,
    distance_scale: float = 50_000.0,
    promoter_upstream: int = 2_000,
) -> Dict[int, Dict[str, object]]:
    """Build sparse, ArchR-style peak-gene mapping initializations.

    Candidate edges connect peaks to genes within ``max_distance`` of the gene
    body. Gene-body and promoter overlaps have distance zero; distal weights
    decay as exp(-distance / distance_scale).
    """
    gene_names = [str(x) for x in gene_names]
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    genes = read_gtf_genes(gtf_file, gene_names)
    mapping_init = {}

    items = (
        sorted(atacdata.items())
        if isinstance(atacdata, Mapping)
        else list(enumerate(atacdata))
    )

    for batch_idx, adata in items:
        batch_name = str(adata.obs[batch_key].iloc[0])
        # ``build_peak_gene_mapping_init`` is used in two places:
        # 1. mixed RNA/ATAC multidata, where RNA batches must be skipped;
        # 2. ATAC-only DA-peak dictionaries, where batch names may be sample IDs.
        # Treat non-RNA batches as ATAC so ATAC-only inputs do not require
        # names such as ``atac_1``.
        if "rna" in batch_name.lower():
            continue

        peaks = parse_peak_names(adata.var_names)
        edge_peak = []
        edge_gene = []
        edge_distance = []

        for chrom, chrom_peaks in peaks.groupby("chrom", sort=False):
            chrom_genes = genes[genes["chrom"] == chrom]
            if chrom_genes.empty:
                continue

            chrom_peaks = chrom_peaks.sort_values("start")
            peak_starts = chrom_peaks["start"].to_numpy()
            peak_records = list(chrom_peaks.itertuples())

            for gene in chrom_genes.itertuples():
                left = np.searchsorted(
                    peak_starts, gene.start - max_distance, side="left"
                )
                right = np.searchsorted(
                    peak_starts, gene.end + max_distance, side="right"
                )
                for peak in peak_records[left:right]:
                    if peak.end < gene.start - max_distance:
                        continue

                    promoter_start = (
                        max(0, gene.tss - promoter_upstream)
                        if gene.strand == "+"
                        else gene.tss
                    )
                    promoter_end = (
                        gene.tss
                        if gene.strand == "+"
                        else gene.tss + promoter_upstream
                    )
                    overlaps_body = peak.end > gene.start and peak.start < gene.end
                    overlaps_promoter = (
                        peak.end > promoter_start and peak.start < promoter_end
                    )

                    if overlaps_body or overlaps_promoter:
                        distance = 0
                    elif peak.end <= gene.start:
                        distance = gene.start - peak.end
                    else:
                        distance = peak.start - gene.end

                    if distance <= max_distance:
                        edge_peak.append(peak.peak_idx)
                        edge_gene.append(gene_to_idx[gene.Index])
                        edge_distance.append(distance)

        distance = np.asarray(edge_distance, dtype=np.float32)
        init_weight = np.exp(-distance / float(distance_scale)).astype(np.float32)
        mapping_init[int(batch_idx)] = {
            "batch_name": batch_name,
            "peak_names": adata.var_names.astype(str).tolist(),
            "gene_names": gene_names,
            "peak_idx": torch.as_tensor(edge_peak, dtype=torch.long),
            "gene_idx": torch.as_tensor(edge_gene, dtype=torch.long),
            "distance": torch.from_numpy(distance),
            "init_weight": torch.from_numpy(init_weight),
            "max_distance": int(max_distance),
            "distance_scale": float(distance_scale),
            "promoter_upstream": int(promoter_upstream),
        }
        print(
            f"[mapping] {batch_name}: peaks={adata.n_vars}, "
            f"genes={len(gene_names)}, edges={len(edge_peak)}"
        )

    return mapping_init


def prepare_multimodal_training_data(
    initialized_data,
    atacdata_da,
    batch_key: str,
):
    """Replace initialized ATAC gene-activity batches with DA peak matrices.

    RNA batches are copied unchanged. ATAC batches are matched by their
    ``batch_key`` value and reordered to exactly match the initialized cell
    order, preserving alignment with P_init/P_align.
    """
    init_items = (
        [initialized_data[k] for k in sorted(initialized_data)]
        if isinstance(initialized_data, Mapping)
        else list(initialized_data)
    )
    atac_items = (
        [atacdata_da[k] for k in sorted(atacdata_da)]
        if isinstance(atacdata_da, Mapping)
        else list(atacdata_da)
    )
    atac_by_name = {
        str(ad.obs[batch_key].iloc[0]): ad
        for ad in atac_items
    }

    training_data = {}
    for batch_idx, initialized_ad in enumerate(init_items):
        batch_name = str(initialized_ad.obs[batch_key].iloc[0])
        if "atac" not in batch_name.lower():
            if "rna" not in batch_name.lower():
                raise ValueError(
                    f"Unknown modality name in batch {batch_name!r}."
                )
            training_data[batch_idx] = initialized_ad.copy()
            continue
        if batch_name not in atac_by_name:
            raise ValueError(f"No DA-peak AnnData found for {batch_name}.")

        atac_ad = atac_by_name[batch_name]
        missing = initialized_ad.obs_names.difference(atac_ad.obs_names)
        if len(missing):
            raise ValueError(
                f"{batch_name}: {len(missing)} initialized cells are absent "
                "from atacdata_da."
            )
        training_data[batch_idx] = atac_ad[initialized_ad.obs_names].copy()

    return training_data


def mapping_init_from_weights(
    mapping_init,
    mapping_weights,
    min_weight: float = 1e-6,
):
    """Use learned sparse mapping weights as the next-round initialization.

    Merge/align changes the cell-type slots ``P`` and marker columns ``M``.
    Peak-gene mapping does not have a cell-type axis, so it is not merged by
    cluster labels. The learned edge weights should still be carried forward by
    replacing each ATAC batch's ``init_weight``.
    """
    out = copy.deepcopy(mapping_init)
    for batch_idx, weights in mapping_weights.items():
        batch_idx = int(batch_idx)
        if batch_idx not in out:
            raise ValueError(f"mapping_init is missing batch {batch_idx}.")
        weights = weights.detach().cpu() if hasattr(weights, "detach") else torch.as_tensor(weights)
        weights = weights.float().clamp_min(min_weight)
        expected = out[batch_idx]["init_weight"].numel()
        if weights.numel() != expected:
            raise ValueError(
                f"Batch {batch_idx}: learned mapping length {weights.numel()} "
                f"does not match candidate edge count {expected}."
            )
        out[batch_idx]["init_weight"] = weights.clone()
    return out
