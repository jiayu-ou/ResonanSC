import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import yaml
import scanpy as sc
import torch
import matplotlib.pyplot as plt

from ResonanSC.io import save_M, load_yaml_config, save_yaml_config, parse_args, prepare_output_dir, save_run_summary
from ResonanSC.bulk import get_batch_names, get_bulk, get_subtype_onehot
from ResonanSC.corr import plot_corr_heatmaps
from ResonanSC.merge_align import (
    run_inbatch_merge,
    run_cross_batch_align,
    build_M_align
)
from ResonanSC.model import LearnPseudoMaker


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    input_path = (ROOT / cfg["input"]["adata_path"]).resolve()
    output_dir = prepare_output_dir((ROOT / cfg["output"]["output_dir"]).resolve())

    save_yaml_config(cfg, output_dir)

    device_cfg = cfg["runtime"]["device"]
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    torch.manual_seed(cfg["runtime"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["runtime"]["seed"])

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Output directory: {output_dir}")

    adata = sc.read_h5ad(input_path)
    adata.var_names_make_unique()

    batch_key = cfg["keys"]["batch_key"]

    rnadata = {}
    batch_list = adata.obs[batch_key].unique()

    sc.pp.normalize_total(adata)
    adata.layers["norm"] = adata.X.copy()
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=cfg["preprocess"]["n_top_genes"],
        batch_key=batch_key
    )
    hvg = adata.var_names[adata.var.highly_variable]
    adata = adata[:, hvg].copy()

    sc.pp.pca(adata, n_comps=cfg["preprocess"]["n_pcs"])
    sc.pp.neighbors(adata, n_neighbors=cfg["preprocess"]["n_neighbors"])
    sc.tl.umap(adata)

    for i, b in enumerate(batch_list):
        ad_i = adata[adata.obs[batch_key] == b].copy()
        rnadata[i] = ad_i

    for i in range(len(rnadata)):
        sc.tl.leiden(
            rnadata[i],
            resolution=cfg["preprocess"]["leiden_resolution"],
            key_added=cfg["preprocess"]["leiden_key"]
        )

    adata_all = sc.concat(rnadata.values(), axis=0)
    sc.pl.umap(
        adata_all,
        color=["batch", "subtype", "leiden"],
        show=False
    )
    plt.savefig(output_dir / cfg["output"]["umap_filename_1"], dpi=300, bbox_inches="tight")
    onehot_list, subtypes = get_subtype_onehot(
        rnadata,
        subtype_key=cfg["preprocess"]["leiden_key"]
    )
    p_init = [torch.from_numpy(oh).float() for oh in onehot_list]

    batch_names = get_batch_names(rnadata, col=batch_key)
    bulks_list = get_bulk(rnadata, p_init, device=device)

    corr_init = plot_corr_heatmaps(
        mode=cfg["stage1_init_corr"]["mode"],
        rnadata=rnadata,
        p_list=p_init,
        bulks_list=bulks_list,
        batch_names=batch_names,
        figsize_per_panel=tuple(cfg["plot"]["figsize_per_panel"]),
        cmap=cfg["plot"]["cmap"],
        annot=cfg["plot"]["annot"],
        fmt=cfg["plot"]["fmt"],
        ncols=cfg["plot"]["ncols"],
        min_genes=cfg["stage1_init_corr"]["min_genes"],
    )
    in_corr_init = corr_init["inbatch_corrs"]

    thresholds = [cfg["stage1_inbatch_merge"]["threshold"]] * len(p_init)
    out_merge = run_inbatch_merge(
        inbatch_corrs=in_corr_init,
        p_list=p_init,
        thresholds=thresholds,
        M=None,
        device=cfg["stage1_inbatch_merge"]["device"]
    )
    p_merge = out_merge["p_merge"]

    bulks_list_merge = get_bulk(rnadata, p_merge, device=device)

    corr_merge = plot_corr_heatmaps(
        mode=cfg["stage1_cross_corr"]["mode"],
        rnadata=rnadata,
        p_list=p_merge,
        bulks_list=bulks_list_merge,
        batch_names=batch_names,
        figsize_per_panel=tuple(cfg["plot"]["figsize_per_panel"]),
        cmap=cfg["plot"]["cmap"],
        annot=cfg["plot"]["annot"],
        fmt=cfg["plot"]["fmt"],
        ncols=cfg["plot"]["ncols"],
        min_genes=cfg["stage1_cross_corr"]["min_genes"],
    )
    cross_corr_merge = corr_merge["crossbatch_corrs"]

    out = run_cross_batch_align(
        p_merge=p_merge,
        batch_names=batch_names,
        cross_corrs=cross_corr_merge,
        cap=cfg["stage1_cross_align"]["cap"],
        null_score=cfg["stage1_cross_align"]["null_score"],
        device=cfg["stage1_cross_align"]["device"],
    )

    p_align = out["p_align"]
    align_masks = out["align_masks"]


    for i in range(len(rnadata)):
        hard = torch.argmax(p_merge[i], dim=1).detach().cpu().numpy()
        rnadata[i].obs["p_merge"] = hard.astype(str)
        hard = torch.argmax(p_align[i], dim=1).detach().cpu().numpy()
        rnadata[i].obs["p_align"] = hard.astype(str)

    adata_all = sc.concat(rnadata.values(), axis=0)
    sc.pl.umap(
        adata_all,
        color=["batch", "subtype", "p_merge","p_align"],
        show=False
        )
    plt.savefig(output_dir / cfg["output"]["umap_filename_2"], dpi=300, bbox_inches="tight")

    P_init = p_align
    M_init = None

    margin = cfg["training"]["margin"]

    train_out = LearnPseudoMaker(
        rnadata.copy(),
        epochs=cfg["training"]["learn_P"]["epochs"],
        lr=cfg["training"]["learn_P"]["lr"],
        M_init=M_init,
        P_init=P_init,
        align_masks=align_masks,
        margin=margin["learn_P"],
        lambdas=cfg["training"]["learn_P"]["lambdas"],
        stage="learn_P",
        one_hot_start=cfg["training"]["one_hot_start"],
        present_eps=cfg["training"]["present_eps"],
    )

    torch.save(train_out, output_dir / cfg["output"]["model_1P_filename"])

    train_out = LearnPseudoMaker(
        rnadata.copy(),
        epochs=cfg["training"]["learn_M"]["epochs"],
        lr=cfg["training"]["learn_M"]["lr"],
        M_init=M_init,
        P_init=P_init,
        align_masks=align_masks,
        margin=margin["learn_M"],
        lambdas=cfg["training"]["learn_M"]["lambdas"],
        stage="learn_M",
        resume_path=output_dir / cfg["output"]["model_1P_filename"],
        one_hot_start=cfg["training"]["one_hot_start"],
        present_eps=cfg["training"]["present_eps"],
    )

    torch.save(train_out, output_dir / cfg["output"]["model_2M_filename"])
    probs_soft = train_out["probs_soft"]
    M = train_out["M"]
    bulks_list = train_out["bulks_list"]

    corr_heatmap = plot_corr_heatmaps(
        mode=cfg["stage2_weighted_corr"]["mode"],
        bulks_list=bulks_list,
        M_list=[M for _ in bulks_list],
        batch_names=batch_names,
        figsize_per_panel=tuple(cfg["plot"]["figsize_per_panel"]),
        diag_use_mean=cfg["stage2_weighted_corr"]["diag_use_mean"],
    )
    in_corrs = corr_heatmap["inbatch_corrs"]

    thresholds = [cfg["stage2_inbatch_merge"]["threshold"]] * len(probs_soft)
    out_merge = run_inbatch_merge(
        inbatch_corrs=in_corrs,
        p_list=probs_soft,
        thresholds=thresholds,
        M=M,
        device=cfg["stage2_inbatch_merge"]["device"]
    )
    p_merge = out_merge["p_merge"]
    M_merge = out_merge["M_merge"]

    bulks_list_merge = get_bulk(rnadata, p_merge, device=device)

    corr_merge = plot_corr_heatmaps(
        mode=cfg["stage2_cross_corr"]["mode"],
        bulks_list=bulks_list_merge,
        M_list=M_merge,
        batch_names=batch_names,
        figsize_per_panel=tuple(cfg["plot"]["figsize_per_panel"]),
        diag_use_mean=cfg["stage2_cross_corr"]["diag_use_mean"],
    )
    cross_corr_merge = corr_merge["crossbatch_corrs"]

    out_align = run_cross_batch_align(
        p_merge=p_merge,
        batch_names=batch_names,
        cross_corrs=cross_corr_merge,
        cap=cfg["stage2_cross_align"]["cap"],
        null_score=cfg["stage2_cross_align"]["null_score"],
        device=cfg["stage2_cross_align"]["device"],
    )

    p_align = out_align["p_align"]
    align_masks = out_align["align_masks"]

    M_align = build_M_align(
        M_list=M_merge,
        label_map=out_align["label_map"],
        offsets=out_align["offsets"],
        device=device,
        eps=cfg["final"]["eps"],
        normalize=cfg["final"]["normalize"],
    )

    final_label_key = cfg["output"]["final_label_key"]
    for i in range(len(rnadata)):
        hard = torch.argmax(p_align[i], dim=1).detach().cpu().numpy()
        rnadata[i].obs[final_label_key] = hard.astype(str)

    adata_all = sc.concat(rnadata.values(), axis=0)

    umap_color = cfg["output"]["umap_color"]
    sc.pl.umap(
        adata_all,
        color=umap_color,
        show=False
    )
    plt.savefig(output_dir / cfg["output"]["umap_filename"], dpi=300, bbox_inches="tight")
    plt.close()

    adata_all.write(output_dir / cfg["output"]["adata_filename"])
    save_M(
        M_align,
        adata_all,
        out_csv=str(output_dir / cfg["output"]["marker_filename"]),
        col_prefix=cfg["output"]["marker_col_prefix"]
    )

    summary = {
        "device": device,
        "batch_names": list(batch_names),
        "n_batches": len(batch_names),
        "n_cells_total": int(adata_all.n_obs),
        "n_genes_total": int(adata_all.n_vars),
        "final_label_key": final_label_key,
        "output_dir": str(output_dir),
        "adata_file": str(output_dir / cfg["output"]["adata_filename"]),
        "marker_file": str(output_dir / cfg["output"]["marker_filename"]),
        "umap_file": str(output_dir / cfg["output"]["umap_filename"]),
    }
    save_run_summary(summary, output_dir)

    print("[INFO] Finished successfully.")
    print(f"[INFO] Saved AnnData to: {output_dir / cfg['output']['adata_filename']}")
    print(f"[INFO] Saved markers to: {output_dir / cfg['output']['marker_filename']}")
    print(f"[INFO] Saved UMAP to: {output_dir / cfg['output']['umap_filename']}")


if __name__ == "__main__":
    main()
