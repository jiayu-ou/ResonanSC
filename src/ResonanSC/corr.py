import math
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch


TensorLike = Union[torch.Tensor, np.ndarray]

# =========================
# default DE parameters
# =========================
DEFAULT_N_TOP = 20
DEFAULT_N_BOTTOM = 0
DEFAULT_PVAL_TH = 0.05
MIN_CELLS_1VALL = 2
MIN_CELLS_PAIR_EACH = 2
MIN_CELLS_PAIR_TOTAL = 5


# =========================
# basic helpers
# =========================
def _to_torch_float(x: TensorLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.tensor(x, dtype=torch.float32)


def _check_2d_tensor(x: torch.Tensor, name: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, but got shape {tuple(x.shape)}.")


def _default_batch_names(n_batches: int) -> List[str]:
    return [f"batch {i}" for i in range(n_batches)]


def _default_subtype_names_list(bulks_list: Sequence[torch.Tensor]) -> List[List[str]]:
    return [[str(i) for i in range(b.shape[1])] for b in bulks_list]


# =========================
# weighted-corr branch
# =========================
def correlation(x, y, w=None, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation with optional weights.

    Parameters
    ----------
    x, y : 1D tensor-like
    w : 1D tensor-like or None
        If None, compute ordinary Pearson correlation.
        Otherwise compute weighted Pearson correlation.

    Returns
    -------
    corr : torch.Tensor
        Scalar correlation.
    """
    x = torch.nan_to_num(
        _to_torch_float(x).flatten(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y = torch.nan_to_num(
        _to_torch_float(y).flatten(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")

    if w is None:
        xm = x - x.mean()
        ym = y - y.mean()
        cov = (xm * ym).mean()
        var_x = (xm * xm).mean()
        var_y = (ym * ym).mean()
        corr = cov / torch.sqrt(torch.clamp(var_x * var_y, min=eps))
        return torch.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

    w = torch.nan_to_num(
        _to_torch_float(w).flatten(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if w.shape != x.shape:
        raise ValueError(f"w must have the same shape as x/y, got {w.shape} vs {x.shape}")

    w = w.clamp_min(0.0)
    w = w / (w.sum() + eps)

    mean_x = (w * x).sum()
    mean_y = (w * y).sum()

    xm = x - mean_x
    ym = y - mean_y

    cov = (w * xm * ym).sum()
    var_x = (w * xm.pow(2)).sum()
    var_y = (w * ym.pow(2)).sum()

    corr = cov / torch.sqrt(torch.clamp(var_x * var_y, min=eps))
    return torch.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def compute_inbatch_corr_weighted(bulks: TensorLike, M: TensorLike) -> np.ndarray:
    """
    Within-batch subtype correlation using marker/weight matrix M.

    Parameters
    ----------
    bulks : [G, K]
    M : [G, K]
    """
    bulks = _to_torch_float(bulks)
    M = _to_torch_float(M)

    _check_2d_tensor(bulks, "bulks")
    _check_2d_tensor(M, "M")

    if bulks.shape != M.shape:
        raise ValueError(f"bulks and M must have same shape, got {bulks.shape} and {M.shape}")

    K = bulks.shape[1]
    corr = np.full((K, K), np.nan, dtype=float)

    for i in range(K):
        for j in range(K):
            w = (M[:, i] + M[:, j]) / 2.0
            x = bulks[:, i]
            y = bulks[:, j]
            corr[i, j] = float(correlation(x, y, w))

    return corr


def compute_crossbatch_corr_weighted(
    bulks_a: TensorLike,
    bulks_b: TensorLike,
    M_a: TensorLike,
    M_b: TensorLike,
    diag_use_mean: bool = True,
) -> np.ndarray:
    """
    Cross-batch subtype correlation using marker/weight matrices.

    Returns
    -------
    corr : [K_b, K_a]
        rows=batch_b, cols=batch_a
    """
    bulks_a = _to_torch_float(bulks_a)
    bulks_b = _to_torch_float(bulks_b)
    M_a = _to_torch_float(M_a)
    M_b = _to_torch_float(M_b)

    _check_2d_tensor(bulks_a, "bulks_a")
    _check_2d_tensor(bulks_b, "bulks_b")
    _check_2d_tensor(M_a, "M_a")
    _check_2d_tensor(M_b, "M_b")

    if bulks_a.shape[0] != bulks_b.shape[0]:
        raise ValueError("bulks_a and bulks_b must have same number of genes")
    if M_a.shape != bulks_a.shape:
        raise ValueError("M_a must match bulks_a shape")
    if M_b.shape != bulks_b.shape:
        raise ValueError("M_b must match bulks_b shape")

    K_a = bulks_a.shape[1]
    K_b = bulks_b.shape[1]
    corr = np.full((K_b, K_a), np.nan, dtype=float)

    m_all = (M_a.mean(dim=1) + M_b.mean(dim=1)) / 2.0

    for i in range(K_b):
        for j in range(K_a):
            x = bulks_b[:, i]
            y = bulks_a[:, j]

            if diag_use_mean and (K_a == K_b) and (i == j):
                w = m_all
            else:
                w = (M_b[:, i] + M_a[:, j]) / 2.0

            corr[i, j] = float(correlation(x, y, w))

    return corr


# =========================
# DE branch
# =========================
def _pick_genes(
    df: Optional[pd.DataFrame],
    n_top: int = DEFAULT_N_TOP,
    n_bottom: int = DEFAULT_N_BOTTOM,
    pval_th: float = DEFAULT_PVAL_TH,
) -> set:
    if df is None or df.shape[0] == 0:
        return set()

    if "pvals_adj" in df.columns:
        df_sig = df[df["pvals_adj"] < pval_th]
        if df_sig.shape[0] >= (n_top + n_bottom):
            df = df_sig

    top = df.sort_values("logfoldchanges", ascending=False).head(n_top)["names"].astype(str).tolist()
    bottom = df.sort_values("logfoldchanges", ascending=True).head(n_bottom)["names"].astype(str).tolist()
    return set(top) | set(bottom)

def de_1vAll_kbulk(
    ad,
    p_onehot,
    min_cells_1vAll: int = MIN_CELLS_1VALL,
    n_top: int = DEFAULT_N_TOP,
    n_bottom: int = DEFAULT_N_BOTTOM,
    pval_th: float = DEFAULT_PVAL_TH,
) -> Dict[int, set]:
    """
    For each subtype k, compute 1-vs-rest DE genes from hard labels argmax(p_onehot).
    """
    labels = torch.argmax(p_onehot, dim=1).detach().cpu().numpy().astype(str)
    ad = ad.copy()
    ad.obs["kbulk"] = pd.Categorical(labels)

    counts = ad.obs["kbulk"].value_counts()
    K = p_onehot.shape[1]
    de = {k: set() for k in range(K)}

    valid_groups = [str(k) for k in range(K) if counts.get(str(k), 0) >= min_cells_1vAll]

    if len(valid_groups) == 0:
        return de

    sc.tl.rank_genes_groups(
        ad,
        groupby="kbulk",
        groups=valid_groups,
        method="wilcoxon",
        use_raw=False,
        reference="rest",
    )

    for k in range(K):
        sk = str(k)
        if sk not in valid_groups:
            de[k] = set()
            continue
        try:
            df = sc.get.rank_genes_groups_df(ad, group=sk)
            de[k] = _pick_genes(df, n_top=n_top, n_bottom=n_bottom, pval_th=pval_th)
        except KeyError:
            de[k] = set()

    return de


def de_pair_1v1_kbulk(
    ad,
    p_onehot,
    min_cells_each: int = MIN_CELLS_PAIR_EACH,
    min_cells_pair: int = MIN_CELLS_PAIR_TOTAL,
    n_top: int = DEFAULT_N_TOP,
    n_bottom: int = DEFAULT_N_BOTTOM,
    pval_th: float = DEFAULT_PVAL_TH,
) -> Dict[tuple, set]:
    """
    Pairwise subtype DE gene union.
    """
    labels = torch.argmax(p_onehot, dim=1).detach().cpu().numpy().astype(str)
    ad = ad.copy()
    ad.obs["kbulk"] = pd.Categorical(labels)

    counts = ad.obs["kbulk"].value_counts()
    K = p_onehot.shape[1]
    de_pair = {}

    for i in range(K):
        for j in range(i + 1, K):
            si, sj = str(i), str(j)
            ni = int(counts.get(si, 0))
            nj = int(counts.get(sj, 0))

            if ni < min_cells_each or nj < min_cells_each or (ni + nj) < min_cells_pair:
                de_pair[(i, j)] = set()
                de_pair[(j, i)] = set()
                continue

            ad_ij = ad[ad.obs["kbulk"].isin([si, sj])].copy()

            try:
                sc.tl.rank_genes_groups(
                    ad_ij,
                    groupby="kbulk",
                    groups=[si],
                    method="wilcoxon",
                    use_raw=False,
                    reference=sj,
                )
                g_i = _pick_genes(
                    sc.get.rank_genes_groups_df(ad_ij, group=si),
                    n_top=n_top,
                    n_bottom=n_bottom,
                    pval_th=pval_th,
                )
            except Exception:
                g_i = set()

            try:
                sc.tl.rank_genes_groups(
                    ad_ij,
                    groupby="kbulk",
                    groups=[sj],
                    method="wilcoxon",
                    use_raw=False,
                    reference=si,
                )
                g_j = _pick_genes(
                    sc.get.rank_genes_groups_df(ad_ij, group=sj),
                    n_top=n_top,
                    n_bottom=n_bottom,
                    pval_th=pval_th,
                )
            except Exception:
                g_j = set()

            genes = g_i | g_j
            de_pair[(i, j)] = genes
            de_pair[(j, i)] = genes

    return de_pair


def compute_inbatch_corr_de(
    ad,
    bulk: TensorLike,
    p_onehot,
    min_genes: int = 5,
    min_cells_1vAll: int = MIN_CELLS_1VALL,
    min_cells_each: int = MIN_CELLS_PAIR_EACH,
    min_cells_pair: int = MIN_CELLS_PAIR_TOTAL,
    n_top: int = DEFAULT_N_TOP,
    n_bottom: int = DEFAULT_N_BOTTOM,
    pval_th: float = DEFAULT_PVAL_TH,
) -> np.ndarray:
    """
    Within-batch corr based on DE genes.

    genes(i,j) = de_pair(i,j) | de_1vAll(i) | de_1vAll(j)
    """
    bulk = _to_torch_float(bulk)
    K = p_onehot.shape[1]

    de_1v = de_1vAll_kbulk(
        ad,
        p_onehot,
        min_cells_1vAll=min_cells_1vAll,
        n_top=n_top,
        n_bottom=n_bottom,
        pval_th=pval_th,
    )
    de_pair = de_pair_1v1_kbulk(
        ad,
        p_onehot,
        min_cells_each=min_cells_each,
        min_cells_pair=min_cells_pair,
        n_top=n_top,
        n_bottom=n_bottom,
        pval_th=pval_th,
    )

    genes = ad.var_names.astype(str)
    gmap = {g: i for i, g in enumerate(genes)}

    corr = np.full((K, K), np.nan, dtype=float)
    for i in range(K):
        for j in range(K):
            gs = de_pair.get((i, j), set()) | de_1v.get(i, set()) | de_1v.get(j, set())
            gs = [g for g in gs if g in gmap]
            if len(gs) < min_genes:
                continue

            idx = torch.tensor([gmap[g] for g in gs], dtype=torch.long)
            x = bulk.index_select(0, idx)[:, i]
            y = bulk.index_select(0, idx)[:, j]
            corr[i, j] = float(correlation(x, y))

    return corr


def compute_crossbatch_corr_de(
    adA,
    adB,
    bulkA: TensorLike,
    bulkB: TensorLike,
    pA,
    pB,
    min_genes: int = 5,
    min_cells_1vAll: int = MIN_CELLS_1VALL,
    n_top: int = DEFAULT_N_TOP,
    n_bottom: int = DEFAULT_N_BOTTOM,
    pval_th: float = DEFAULT_PVAL_TH,
) -> np.ndarray:
    """
    Cross-batch DE correlation.

    genes(row_i, col_j) = de_1vAll_B(i) | de_1vAll_A(j)

    Returns
    -------
    S : [K_B, K_A]
        rows=batchB, cols=batchA

    This orientation matches your original init notebook logic.
    """
    bulkA = _to_torch_float(bulkA)
    bulkB = _to_torch_float(bulkB)

    K_A = pA.shape[1]
    K_B = pB.shape[1]

    deA_1v = de_1vAll_kbulk(
        adA,
        pA,
        min_cells_1vAll=min_cells_1vAll,
        n_top=n_top,
        n_bottom=n_bottom,
        pval_th=pval_th,
    )
    deB_1v = de_1vAll_kbulk(
        adB,
        pB,
        min_cells_1vAll=min_cells_1vAll,
        n_top=n_top,
        n_bottom=n_bottom,
        pval_th=pval_th,
    )

    genesA = adA.var_names.astype(str)
    genesB = adB.var_names.astype(str)
    common = genesA.intersection(genesB)

    mapA = {g: i for i, g in enumerate(genesA)}
    mapB = {g: i for i, g in enumerate(genesB)}

    S = np.full((K_B, K_A), np.nan, dtype=float)
    for i in range(K_B):
        for j in range(K_A):
            gs = deB_1v.get(i, set()) | deA_1v.get(j, set())
            gs = [g for g in gs if g in common]
            if len(gs) < min_genes:
                continue

            idxB = torch.tensor([mapB[g] for g in gs], dtype=torch.long)
            idxA = torch.tensor([mapA[g] for g in gs], dtype=torch.long)
            x = bulkB.index_select(0, idxB)[:, i]
            y = bulkA.index_select(0, idxA)[:, j]
            S[i, j] = float(correlation(x, y))

    return S


# =========================
# plotting helpers
# =========================
def plot_heatmaps_in_grid(
    matrices: Sequence[np.ndarray],
    titles: Sequence[str],
    xlabels_list: Sequence[Sequence[str]],
    ylabels_list: Sequence[Sequence[str]],
    main_title: Optional[str] = None,
    ncols: int = 2,
    figsize_per_panel: tuple = (6, 5),
    cmap: str = "coolwarm",
    annot: bool = True,
    fmt: str = ".2f",
    vmin: float = -1.0,
    vmax: float = 1.0,
    center: float = 0.0,
    square: bool = True,
) -> None:
    n = len(matrices)
    if n == 0:
        return

    if not (len(titles) == len(xlabels_list) == len(ylabels_list) == n):
        raise ValueError("matrices, titles, xlabels_list, ylabels_list must have the same length.")

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        if idx < n:
            mat = np.asarray(matrices[idx])
            sns.heatmap(
                mat,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                square=square,
                linewidths=0.5,
                cbar_kws={"label": "Pearson r", "shrink": 0.6, "aspect": 20},
                xticklabels=list(xlabels_list[idx]),
                yticklabels=list(ylabels_list[idx]),
                ax=ax,
            )
            ax.set_title(titles[idx])
            ax.set_xlabel("Subtypes")
            ax.set_ylabel("Subtypes")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        else:
            ax.axis("off")

    if main_title is not None:
        fig.suptitle(main_title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()


# =========================
# unified high-level plotting
# =========================
def plot_corr_heatmaps(
    mode: str,
    bulks_list,
    batch_names: Optional[Sequence[str]] = None,
    subtype_names_list: Optional[Sequence[Sequence[str]]] = None,
    rnadata=None,
    p_list=None,
    M_list=None,
    figsize_per_panel: tuple = (6, 5),
    cmap: str = "coolwarm",
    annot: bool = True,
    fmt: str = ".2f",
    ncols: int = 2,
    diag_use_mean: bool = True,
    min_genes: int = 5,
    min_cells_1vAll: int = MIN_CELLS_1VALL,
    min_cells_each: int = MIN_CELLS_PAIR_EACH,
    min_cells_pair: int = MIN_CELLS_PAIR_TOTAL,
    n_top: int = DEFAULT_N_TOP,
    n_bottom: int = DEFAULT_N_BOTTOM,
    pval_th: float = DEFAULT_PVAL_TH,
) -> Dict[str, Any]:
    """
    Unified correlation heatmap plotting.

    Parameters
    ----------
    mode : {"weighted", "de"}
        Correlation mode.
    bulks_list : list[TensorLike]
        Each element shape [G, K_b]
    rnadata : required when mode="de"
    p_list : required when mode="de"
    M_list : required when mode="weighted"
    """
    if len(bulks_list) == 0:
        raise ValueError("bulks_list must not be empty")

    bulks_list = [_to_torch_float(b) for b in bulks_list]
    n_batches = len(bulks_list)

    if batch_names is None:
        batch_names = _default_batch_names(n_batches)
    else:
        batch_names = list(batch_names)

    if subtype_names_list is None:
        subtype_names_list = _default_subtype_names_list(bulks_list)
    else:
        subtype_names_list = [list(x) for x in subtype_names_list]

    if len(batch_names) != n_batches:
        raise ValueError("batch_names length must equal len(bulks_list)")
    if len(subtype_names_list) != n_batches:
        raise ValueError("subtype_names_list length must equal len(bulks_list)")

    # mode-specific preparation
    if mode == "weighted":
        if M_list is None:
            raise ValueError("M_list is required when mode='weighted'")
        if len(M_list) != n_batches:
            raise ValueError("M_list length must equal len(bulks_list)")
        M_list = [_to_torch_float(m) for m in M_list]

    elif mode == "de":
        if p_list is None:
            raise ValueError("p_list is required when mode='de'")
        if len(p_list) != n_batches:
            raise ValueError("p_list length must equal len(bulks_list)")

        if rnadata is None:
            raise ValueError("rnadata is required when mode='de'")

        if isinstance(rnadata, dict):
            ad_list = [rnadata[k] for k in sorted(rnadata.keys())]
        else:
            ad_list = list(rnadata)

        if len(ad_list) != n_batches:
            raise ValueError("rnadata batch count must equal len(bulks_list)")
    else:
        raise ValueError("mode must be either 'weighted' or 'de'")

    # in-batch
    inbatch_corrs = []
    in_titles = []
    in_x = []
    in_y = []

    for bi in range(n_batches):
        if mode == "weighted":
            mat = compute_inbatch_corr_weighted(
                bulks=bulks_list[bi],
                M=M_list[bi],
            )
        else:
            mat = compute_inbatch_corr_de(
                ad=ad_list[bi],
                bulk=bulks_list[bi],
                p_onehot=p_list[bi],
                min_genes=min_genes,
                min_cells_1vAll=min_cells_1vAll,
                min_cells_each=min_cells_each,
                min_cells_pair=min_cells_pair,
                n_top=n_top,
                n_bottom=n_bottom,
                pval_th=pval_th,
            )

        inbatch_corrs.append(mat)
        in_titles.append(f"In-batch\n{batch_names[bi]}")
        in_x.append(subtype_names_list[bi])
        in_y.append(subtype_names_list[bi])

    # cross-batch
    crossbatch_corrs = []
    cross_titles = []
    cross_x = []
    cross_y = []

    for bi in range(n_batches - 1):
        if mode == "weighted":
            mat = compute_crossbatch_corr_weighted(
                bulks_a=bulks_list[bi],
                bulks_b=bulks_list[bi + 1],
                M_a=M_list[bi],
                M_b=M_list[bi + 1],
                diag_use_mean=diag_use_mean,
            )
            # compute_* returns rows=next batch, cols=current batch.
            # Align expects rows=current batch, cols=next batch.
            mat = mat.T
            cross_x.append(subtype_names_list[bi + 1])
            cross_y.append(subtype_names_list[bi])

        else:
            mat = compute_crossbatch_corr_de(
                adA=ad_list[bi],
                adB=ad_list[bi + 1],
                bulkA=bulks_list[bi],
                bulkB=bulks_list[bi + 1],
                pA=p_list[bi],
                pB=p_list[bi + 1],
                min_genes=min_genes,
                min_cells_1vAll=min_cells_1vAll,
                n_top=n_top,
                n_bottom=n_bottom,
                pval_th=pval_th,
            )
            # compute_* returns rows=next batch, cols=current batch.
            # Align expects rows=current batch, cols=next batch.
            mat = mat.T
            cross_x.append(subtype_names_list[bi + 1])
            cross_y.append(subtype_names_list[bi])

        crossbatch_corrs.append(mat)
        cross_titles.append(f"Cross-batch\n{batch_names[bi]} vs {batch_names[bi + 1]}")

    # plot
    suffix = " (DE)" if mode == "de" else ""

    plot_heatmaps_in_grid(
        matrices=inbatch_corrs,
        titles=in_titles,
        xlabels_list=in_x,
        ylabels_list=in_y,
        main_title=f"All In-batch Correlation Heatmaps{suffix}",
        ncols=ncols,
        figsize_per_panel=figsize_per_panel,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
    )

    plot_heatmaps_in_grid(
        matrices=crossbatch_corrs,
        titles=cross_titles,
        xlabels_list=cross_x,
        ylabels_list=cross_y,
        main_title=f"All Chain Cross-batch Correlation Heatmaps{suffix}",
        ncols=ncols,
        figsize_per_panel=figsize_per_panel,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
    )

    return {
        "mode": mode,
        "inbatch_corrs": inbatch_corrs,
        "crossbatch_corrs": crossbatch_corrs,
        "batch_names": batch_names,
        "subtype_names_list": subtype_names_list,
        "cross_titles": cross_titles,
    }
