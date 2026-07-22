import numpy as np
import episcanpy as epi


def rank_features_with_min_cells_filter(
    adata,
    type_key,
    omic="ATAC",
    use_raw=False,
    min_cells_per_group=2,
):
    group_counts = adata.obs[type_key].value_counts()
    excluded_groups = group_counts[group_counts < min_cells_per_group]
    valid_groups = group_counts[group_counts >= min_cells_per_group].index.astype(str).tolist()

    if len(excluded_groups) > 0:
        excluded_text = {
            str(group): int(count)
            for group, count in excluded_groups.items()
        }
        print(
            "[rank_features_with_min_cells_filter] Warning: groups below "
            f"min_cells_per_group={min_cells_per_group} will be excluded "
            f"from marker ranking and DA peak union: {excluded_text}"
        )

    if not valid_groups:
        raise ValueError(
            "No groups have enough cells for marker ranking. "
            f"min_cells_per_group={min_cells_per_group}, counts="
            f"{group_counts.astype(int).to_dict()}"
        )

    epi.tl.rank_features(
        adata,
        type_key,
        omic=omic,
        use_raw=use_raw,
        groups=valid_groups,
    )

    params = dict(adata.uns["rank_features_groups"].get("params", {}))
    params["min_cells_per_group"] = min_cells_per_group
    params["excluded_groups"] = {
        str(group): int(count)
        for group, count in excluded_groups.items()
    }
    params["included_groups"] = valid_groups
    adata.uns["rank_features_groups"]["params"] = params


def rank_features_with_singleton_fallback(
    adata,
    type_key,
    omic="ATAC",
    use_raw=False,
    min_cells_per_group=2,
):
    return rank_features_with_min_cells_filter(
        adata,
        type_key,
        omic=omic,
        use_raw=use_raw,
        min_cells_per_group=min_cells_per_group,
    )


def peak_analysis(
    atac,
    nb_features=120000,
    min_score=0.515,
    annotation=None,
    type_key=None,
    variable=False,
    min_cells_per_group=2,
):
    atac_da = atac.copy()

    if variable:
        epi.pl.variability_features(
            atac_da,
            log=None,
            min_score=min_score,
            nb_features=nb_features,
            save="variability_features_plot_bonemarrow_peakmatrix.png",
        )

        epi.pp.select_var_feature(
            atac_da,
            min_score=min_score,
            nb_features=nb_features,
        )

    rank_features_with_min_cells_filter(
        atac_da,
        type_key,
        omic="ATAC",
        use_raw=False,
        min_cells_per_group=min_cells_per_group,
    )

    epi.pl.rank_feat_groups(atac_da)

    return atac_da


def filter_da(atac_pp):
    rank_dict = atac_pp.uns["rank_features_groups"]
    names = rank_dict["names"]

    da_peaks = []
    for group_peaks in names:
        da_peaks.extend([peak for peak in list(group_peaks) if peak is not None])

    da_peaks = np.unique(da_peaks)
    atac_da_filtered = atac_pp[:, np.isin(atac_pp.var_names, da_peaks)].copy()
    atac_da_filtered.uns["rank_features_groups"] = {
        "names": [da_peaks],
        "scores": [np.ones_like(da_peaks)],
    }

    return atac_da_filtered
