import argparse
import ast
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


STAGE_KEYS = {
    "learn_P": "P",
    "learn_mapping": "mapping",
    "learn_M": "M",
}


def import_runtime_deps():
    global np, sc, torch, plt
    global get_batch_names, get_multimodal_bulk, plot_corr_heatmaps
    global mapping_init_from_weights, prepare_multimodal_training_data
    global build_M_align, run_cross_batch_align, run_inbatch_merge
    global LearnPseudoMaker

    import matplotlib

    matplotlib.use("Agg")

    import numpy as np
    import scanpy as sc
    import torch
    import matplotlib.pyplot as plt

    from ResonanSC.bulk import get_batch_names, get_multimodal_bulk
    from ResonanSC.corr import plot_corr_heatmaps
    from ResonanSC.mapping import (
        mapping_init_from_weights,
        prepare_multimodal_training_data,
    )
    from ResonanSC.merge_align import (
        build_M_align,
        run_cross_batch_align,
        run_inbatch_merge,
    )
    from ResonanSC.model import LearnPseudoMaker


def resolve_path(value):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_config(path):
    with Path(path).open() as handle:
        text = handle.read()
    try:
        import yaml

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return parse_simple_yaml(text)


def parse_simple_scalar(value):
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_simple_yaml(text):
    """Parse the small YAML subset used by configs/train_round.yaml."""
    root = {}
    stack = [(-1, root)]

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            node = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = parse_simple_scalar(value)

    return root


def h5ad_index_key(path):
    return int(path.name.split("_", 1)[0])


def split_by_batch(adata, batch_key):
    out = {}
    for i, batch in enumerate(adata.obs[batch_key].unique()):
        out[i] = adata[adata.obs[batch_key] == batch].copy()
    return out


def load_training_data(cfg, ckpt, batch_key):
    # Prefer the directory recorded alongside the checkpoint, but allow an
    # explicit config path to recover when the dataset/checkpoint was moved.
    training_data_dirs = []
    for value in (
        ckpt.get("training_data_dir"),
        cfg["input"].get("training_data_dir"),
    ):
        path = resolve_path(value)
        if path is not None and path not in training_data_dirs:
            training_data_dirs.append(path)

    training_data_dir = training_data_dirs[0] if training_data_dirs else None
    for candidate_dir in training_data_dirs:
        training_files = sorted(candidate_dir.glob("*.h5ad"), key=h5ad_index_key)
        if training_files:
            multidata = {
                i: sc.read_h5ad(path) for i, path in enumerate(training_files)
            }
            return multidata, candidate_dir

    init_adata_path = resolve_path(cfg["input"].get("init_adata"))
    atac_da_dir = resolve_path(ckpt.get("atac_da_dir", cfg["input"].get("atac_da_dir")))
    if init_adata_path is None or not init_adata_path.exists():
        raise FileNotFoundError(
            f"No training h5ad files were found in {training_data_dirs} and "
            f"input.init_adata is missing: {init_adata_path}"
        )
    if atac_da_dir is None:
        raise FileNotFoundError(
            "No training h5ad files were found and input.atac_da_dir is missing."
        )

    init_adata = sc.read_h5ad(init_adata_path)
    initialized_data = split_by_batch(init_adata, batch_key)
    atac_files = sorted(atac_da_dir.glob("*.h5ad"), key=h5ad_index_key)
    if not atac_files:
        raise FileNotFoundError(
            f"No training h5ad files found in {training_data_dir} or {atac_da_dir}"
        )
    atacdata_da = {i: sc.read_h5ad(path) for i, path in enumerate(atac_files)}
    multidata = prepare_multimodal_training_data(
        initialized_data=initialized_data,
        atacdata_da=atacdata_da,
        batch_key=batch_key,
    )
    return multidata, training_data_dir


def load_training_state(cfg):
    init_path = resolve_path(cfg["input"]["init_checkpoint"])
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)

    batch_key = ckpt.get("batch_key", cfg["keys"].get("batch_key", "batch"))
    P_init = ckpt.get("P_align", ckpt.get("P_init"))
    # Prefer the continuous marker probabilities when available.  The hard
    # binary M is still saved for final output, but using it as the next-round
    # initializer can erase progress when M_prob has moved toward sparsity but
    # has not crossed the hard threshold yet.
    M_init = ckpt.get(
        "M_prob_align",
        ckpt.get("M_prob", ckpt.get("M_align", ckpt.get("M", None))),
    )
    if P_init is None:
        raise ValueError(f"{init_path} must contain P_align or P_init.")

    multidata, training_data_dir = load_training_data(cfg, ckpt, batch_key)
    batch_names = get_batch_names(multidata, col=batch_key)

    state = {
        "init_path": init_path,
        "ckpt": ckpt,
        "multidata": multidata,
        "training_data_dir": training_data_dir,
        "P_init": P_init,
        "M_init": M_init,
        "align_masks": ckpt["align_masks"],
        "mapping_init": ckpt["mapping_init"],
        "mapping_max_distance": ckpt.get("dap_deg_window", ckpt.get("mapping_max_distance", 250_000)),
        "gene_names": ckpt["gene_names"],
        "batch_key": batch_key,
        "batch_names": batch_names,
    }
    validate_training_state_shapes(state)
    return state


def tensor_shape(x):
    if hasattr(x, "shape"):
        return tuple(x.shape)
    return tuple(np.asarray(x).shape)


def validate_training_state_shapes(state):
    multidata = state["multidata"]
    P_init = state["P_init"]
    batch_names = state["batch_names"]

    if len(P_init) != len(multidata):
        raise ValueError(
            "Checkpoint/data mismatch: "
            f"P_init has {len(P_init)} batches but training data has {len(multidata)}. "
            f"init_checkpoint={state['init_path']} training_data_dir={state['training_data_dir']}"
        )

    rows = []
    bad = []
    for i in range(len(multidata)):
        p_shape = tensor_shape(P_init[i])
        n_obs = multidata[i].n_obs
        rows.append(
            f"  [{i}] {batch_names[i]}: adata.n_obs={n_obs}, P_init.shape={p_shape}"
        )
        if len(p_shape) < 2 or int(p_shape[0]) != int(n_obs):
            bad.append(i)

    print("[INFO] Training data / P_init shape check:")
    print("\n".join(rows))

    if bad:
        details = "\n".join(rows)
        raise ValueError(
            "Checkpoint/data mismatch: each P_init[i] must have one row per "
            "cell in multidata[i].\n"
            f"init_checkpoint={state['init_path']}\n"
            f"training_data_dir={state['training_data_dir']}\n"
            f"{details}\n"
            "Use the training_data_dir generated with the same init checkpoint, "
            "or start from the matching merge_round_*.pt."
        )

    M_init = state.get("M_init")
    if M_init is not None:
        m_shape = tensor_shape(M_init)
        n_genes = len(state["gene_names"])
        if len(m_shape) != 2 or int(m_shape[0]) != int(n_genes):
            raise ValueError(
                "Checkpoint/data mismatch: M_init first dimension must match "
                f"gene_names. M_init.shape={m_shape}, n_genes={n_genes}"
            )


def stage_output_path(output_dir, prefix, stage, round_id):
    suffix = STAGE_KEYS[stage]
    return output_dir / f"{prefix}_{suffix}_{round_id}.pt"


def output_name(cfg, key, default):
    name = cfg["output"].get(key, default)
    if name is None:
        return None
    return str(name).format(round=cfg["output"].get("round", 1))


def figures_dir(cfg, output_dir):
    path = output_dir / cfg["output"].get("figures_dir", "figures")
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_resume_path(resume_spec, stage_paths, output_dir):
    if resume_spec is None or str(resume_spec).lower() in {"none", ""}:
        return None
    if resume_spec in stage_paths:
        return stage_paths[resume_spec]
    return resolve_path(resume_spec)


def build_stage_checkpoint(out, state, stage):
    """Attach the input context required to reuse a stage checkpoint."""
    checkpoint = dict(out)
    mapping_init = checkpoint.get("mapping_init", state["mapping_init"])
    if "mapping_weights" in checkpoint:
        mapping_init = mapping_init_from_weights(
            mapping_init,
            checkpoint["mapping_weights"],
        )
    checkpoint.update(
        {
            # Before an optional merge/align pass, the learned global slots are
            # already in the input alignment space.
            "P_align": checkpoint["probs_soft"],
            "M_align": checkpoint["M"],
            "M_prob_align": checkpoint.get("M_prob", checkpoint["M"]),
            "align_masks": state["align_masks"],
            "mapping_init_before_training": checkpoint.get(
                "mapping_init", state["mapping_init"]
            ),
            "mapping_init": mapping_init,
            "gene_names": state["gene_names"],
            "batch_key": state["batch_key"],
            "batch_names": list(state["batch_names"]),
            "training_data_dir": str(state["training_data_dir"]),
            "mapping_max_distance": state["mapping_max_distance"],
            "source_init_checkpoint": str(state["init_path"]),
            "training_stage": stage,
        }
    )
    return checkpoint


def save_resumable_checkpoint(out, state, stage, out_path):
    """Save a stage checkpoint only after verifying next-round context."""
    checkpoint = build_stage_checkpoint(out, state, stage)
    required = {
        "P_align",
        "M_align",
        "align_masks",
        "mapping_init",
        "gene_names",
        "batch_key",
        "batch_names",
        "training_data_dir",
    }
    missing = sorted(key for key in required if checkpoint.get(key) is None)
    if missing:
        raise ValueError(
            f"Refusing to save incomplete checkpoint {out_path}; "
            f"missing fields required for continued training: {missing}"
        )
    torch.save(checkpoint, out_path)


def train_stages(cfg, state, output_dir):
    train_cfg = cfg["training"]
    common_cfg = train_cfg["common"]
    runtime_cfg = cfg.get("runtime", {})
    stages = train_cfg.get("stages", ["learn_P", "learn_mapping", "learn_M"])
    prefix = cfg["output"].get("prefix", "train")
    round_id = cfg["output"].get("round", 1)

    common_args = dict(
        rnadata=state["multidata"],
        P_init=state["P_init"],
        M_init=state["M_init"],
        align_masks=state["align_masks"],
        batch_keys=state["batch_names"],
        batch_key=state["batch_key"],
        gene_names=state["gene_names"],
        mapping_init=state["mapping_init"],
        mapping_max_distance=state["mapping_max_distance"],
        one_hot_start=common_cfg.get("one_hot_start", False),
        present_eps=common_cfg.get("present_eps", 0.1),
        rna_chunk_size=common_cfg.get("rna_chunk_size", 512),
        cellbulk_chunk_size=common_cfg.get("cellbulk_chunk_size", 512),
        cache_rna_on_device=common_cfg.get("cache_rna_on_device", False),
        marker_init_mode=common_cfg.get("marker_init_mode", "continuous"),
        marker_init_eps=common_cfg.get("marker_init_eps", 0.05),
        verbose_every=common_cfg.get("verbose_every", 10),
        multi_gpu=runtime_cfg.get("multi_gpu", False),
        devices=runtime_cfg.get("devices"),
    )

    outputs = {}
    stage_paths = {}
    for stage in stages:
        if stage not in STAGE_KEYS:
            raise ValueError(f"Unsupported stage: {stage}")

        resume_spec = train_cfg.get("resume", {}).get(stage)
        resume_path = parse_resume_path(resume_spec, stage_paths, output_dir)
        stage_cfg = train_cfg[stage]
        kwargs = dict(
            **common_args,
            epochs=stage_cfg["epochs"],
            lr=stage_cfg["lr"],
            margin=train_cfg["margin"][stage],
            lambdas=stage_cfg["lambdas"],
            stage=stage,
            resume_path=resume_path,
        )
        if stage == "learn_mapping":
            kwargs["mapping_lambdas"] = stage_cfg.get("mapping_lambdas", [1.0, 1.0])
        if stage == "learn_M":
            kwargs["marker_sparsity"] = stage_cfg.get("marker_sparsity", 1.0)

        print(f"[INFO] Running {stage}; resume_path={resume_path}")
        out = LearnPseudoMaker(**kwargs)
        out_path = stage_output_path(output_dir, prefix, stage, round_id)
        save_resumable_checkpoint(out, state, stage, out_path)
        print(f"[INFO] Saved resumable {stage} checkpoint: {out_path}")
        outputs[stage] = out
        stage_paths[stage] = out_path
        if stage == "learn_P":
            save_p_umap(cfg, state, out, output_dir)

    last_stage = stages[-1]
    return outputs[last_stage], stage_paths


def annotate_predictions(multidata, adata_all, p_list, key):
    hard_all = []
    for i in range(len(multidata)):
        hard = torch.argmax(p_list[i], dim=1).detach().cpu().numpy()
        multidata[i].obs[key] = hard.astype(str)
        hard_all.append(hard)
    adata_all.obs[key] = np.concatenate(hard_all).astype(str)


def build_annotated_adata(cfg, state):
    init_adata_path = resolve_path(cfg["input"].get("init_adata"))
    if init_adata_path is not None and init_adata_path.exists():
        adata = sc.read_h5ad(init_adata_path)
        expected_obs = []
        for ad in state["multidata"].values():
            expected_obs.extend(ad.obs_names.astype(str).tolist())
        if adata.obs_names.astype(str).tolist() == expected_obs:
            return adata
        print(
            "[WARN] input.init_adata cell order does not match multidata; "
            "falling back to concatenated training data."
        )
    return sc.concat(state["multidata"].values(), axis=0)


def requested_colors(adata, colors):
    return [c for c in colors if c in adata.obs]


def save_scanpy_umap(adata, colors, path, dpi=300, **kwargs):
    if "X_umap" not in adata.obsm:
        print(f"[WARN] Skip {path.name}: AnnData has no obsm['X_umap'].")
        return None
    colors = requested_colors(adata, colors)
    if not colors:
        print(f"[WARN] Skip {path.name}: none of the requested obs colors exist.")
        return None
    sc.pl.umap(adata, color=colors, show=False, **kwargs)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"[INFO] Saved figure: {path}")
    return path


def save_p_umap(cfg, state, train_out, output_dir):
    if not cfg.get("plots", {}).get("save", True):
        return []
    fig_dir = figures_dir(cfg, output_dir)
    dpi = cfg.get("plots", {}).get("dpi", 300)
    adata = build_annotated_adata(cfg, state)
    annotate_predictions(state["multidata"], adata, train_out["probs_soft"], "pred_learn_P")
    colors = cfg.get("plots", {}).get(
        "umap_color",
        [state["batch_key"], cfg.get("keys", {}).get("type_key", "cell_type"), "pred_learn_P"],
    )
    path = fig_dir / output_name(cfg, "umap_after_P", "umap_after_learn_P_round_{round}.png")
    return save_scanpy_umap(adata, colors, path, dpi=dpi, wspace=0.4)


def marker_matrix_numpy(M):
    if hasattr(M, "detach"):
        return M.detach().cpu().numpy()
    return np.asarray(M)


def save_marker_histogram(cfg, M, output_dir):
    if not cfg.get("plots", {}).get("save", True):
        return None
    M_np = marker_matrix_numpy(M)
    n_cols = M_np.shape[1]
    n_plot_cols = min(5, max(1, n_cols))
    n_rows = int(np.ceil(n_cols / n_plot_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_plot_cols,
        figsize=(4 * n_plot_cols, 3 * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()
    for i in range(n_cols):
        col = M_np[:, i]
        axes[i].hist(col, bins=50)
        axes[i].set_title(f"Column {i}: {(col > 0).sum()} >0")
        axes[i].set_xlabel("M value")
        axes[i].set_ylabel("Count")
    for ax in axes[n_cols:]:
        ax.axis("off")
    fig.tight_layout()
    path = figures_dir(cfg, output_dir) / output_name(
        cfg, "m_marker_hist", "M_marker_hist_round_{round}.png"
    )
    fig.savefig(path, dpi=cfg.get("plots", {}).get("dpi", 300), bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {path}")
    return path


def save_mscore_umap(cfg, state, train_out, output_dir):
    if not cfg.get("plots", {}).get("save", True):
        return None
    M_np = marker_matrix_numpy(train_out["M"])
    adata = build_annotated_adata(cfg, state)
    annotate_predictions(state["multidata"], adata, train_out["probs_soft"], "pred")

    gene_names = list(map(str, state["gene_names"]))
    if adata.var_names.astype(str).tolist() != gene_names:
        print(
            "[WARN] Skip M score UMAP: annotated AnnData var_names do not match "
            "checkpoint gene_names."
        )
        return None

    from scipy import sparse

    X = adata.X
    if sparse.issparse(X):
        X_score = X @ M_np
    else:
        X_score = np.asarray(X) @ M_np
    adata.obsm["X_mscore"] = np.asarray(X_score)

    score_names = []
    for ct in range(adata.obsm["X_mscore"].shape[1]):
        cname = f"mscore_{ct}"
        adata.obs[cname] = adata.obsm["X_mscore"][:, ct]
        score_names.append(cname)

    base_colors = cfg.get("plots", {}).get(
        "m_umap_color",
        [state["batch_key"], cfg.get("keys", {}).get("type_key", "cell_type"), "pred"],
    )
    save_scanpy_umap(
        adata,
        base_colors,
        figures_dir(cfg, output_dir) / output_name(
            cfg, "m_pred_umap", "M_pred_umap_round_{round}.png"
        ),
        dpi=cfg.get("plots", {}).get("dpi", 300),
        wspace=0.4,
    )

    X_m = adata.obsm["X_mscore"]
    vmin = float(np.percentile(X_m, 1))
    vmax = float(np.percentile(X_m, 99))
    path = figures_dir(cfg, output_dir) / output_name(
        cfg, "m_mscore_umap", "M_mscore_umap_round_{round}.png"
    )
    save_scanpy_umap(
        adata,
        score_names,
        path,
        dpi=cfg.get("plots", {}).get("dpi", 300),
        ncols=cfg.get("plots", {}).get("mscore_ncols", 5),
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        wspace=0.4,
    )
    return path


def save_m_figures(cfg, state, train_out, output_dir):
    if "M" not in train_out:
        print("[WARN] Skip M figures: train_out has no M.")
        return []
    paths = []
    for path in (
        save_marker_histogram(cfg, train_out["M"], output_dir),
        save_mscore_umap(cfg, state, train_out, output_dir),
    ):
        if path is not None:
            paths.append(path)
    return paths


def run_merge_align(cfg, state, train_out, output_dir):
    merge_cfg = cfg["merge_align"]
    batch_names = state["batch_names"]
    multidata = state["multidata"]
    batch_key = state["batch_key"]

    probs_soft = train_out["probs_soft"]
    M = train_out["M"]
    M_for_merge = train_out.get("M_prob", M)
    bulks_list = train_out["bulks_list"]

    corr_heatmap = plot_corr_heatmaps(
        mode="weighted",
        bulks_list=bulks_list,
        M_list=[M_for_merge for _ in bulks_list],
        batch_names=batch_names,
        diag_use_mean=True,
    )
    plt.close("all")

    thresholds = [merge_cfg["inbatch_threshold"]] * len(probs_soft)
    out_merge = run_inbatch_merge(
        inbatch_corrs=corr_heatmap["inbatch_corrs"],
        p_list=probs_soft,
        thresholds=thresholds,
        M=M_for_merge,
        device=merge_cfg.get("device", "cpu"),
    )
    p_merge = out_merge["p_merge"]
    M_merge = out_merge["M_merge"]

    bulks_list_merge = get_multimodal_bulk(
        multidata,
        p_merge,
        train_out=train_out,
        batch_key=batch_key,
    )

    corr_merge = plot_corr_heatmaps(
        mode="weighted",
        bulks_list=bulks_list_merge,
        M_list=M_merge,
        batch_names=batch_names,
        diag_use_mean=True,
    )
    plt.close("all")

    out_align = run_cross_batch_align(
        p_merge=p_merge,
        batch_names=batch_names,
        cross_corrs=corr_merge["crossbatch_corrs"],
        cap=merge_cfg.get("cap", 3),
        null_score=merge_cfg.get("null_score", 0.3),
        device=merge_cfg.get("device", "cpu"),
    )
    p_align = out_align["p_align"]
    align_masks = out_align["align_masks"]

    bulks_list_align = get_multimodal_bulk(
        multidata,
        p_align,
        train_out=train_out,
        batch_key=batch_key,
    )
    M_align = build_M_align(
        M_list=M_merge,
        label_map=out_align["label_map"],
        offsets=out_align["offsets"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        eps=merge_cfg.get("eps", 1e-8),
        normalize=merge_cfg.get("normalize_M", True),
    )
    M_prob_align = build_M_align(
        M_list=M_merge,
        label_map=out_align["label_map"],
        offsets=out_align["offsets"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        eps=merge_cfg.get("eps", 1e-8),
        normalize=False,
    )

    corr_align = plot_corr_heatmaps(
        mode="weighted",
        bulks_list=bulks_list_align,
        M_list=[M_align for _ in bulks_list_align],
        batch_names=batch_names,
        diag_use_mean=True,
    )
    plt.close("all")

    mapping_init_align = mapping_init_from_weights(
        train_out["mapping_init"],
        train_out["mapping_weights"],
    )

    merge_path = output_dir / output_name(cfg, "merge_checkpoint", "merge.pt")
    torch.save(
        {
            "P_merge": p_merge,
            "P_align": p_align,
            "M_merge": M_merge,
            "M_align": M_align.detach().cpu(),
            "M_prob_align": M_prob_align.detach().cpu(),
            "align_masks": align_masks,
            "mapping_init": mapping_init_align,
            "gene_names": state["gene_names"],
            "batch_key": batch_key,
            "batch_names": list(batch_names),
            "training_data_dir": str(state["training_data_dir"]),
            "source_train_checkpoint": str(state["init_path"]),
            "inbatch_threshold": merge_cfg["inbatch_threshold"],
            "cross_align": {
                "cap": merge_cfg.get("cap", 3),
                "null_score": merge_cfg.get("null_score", 0.3),
            },
            "corr_merge_inbatch": corr_merge["inbatch_corrs"],
            "corr_merge_crossbatch": corr_merge["crossbatch_corrs"],
            "corr_align_inbatch": corr_align["inbatch_corrs"],
            "corr_align_crossbatch": corr_align["crossbatch_corrs"],
        },
        merge_path,
    )
    print(f"[INFO] Saved merge/align checkpoint: {merge_path}")
    return merge_path, p_merge, p_align


def write_outputs(cfg, state, train_out, output_dir, stage_paths, merge_path, p_merge, p_align):
    adata_all = build_annotated_adata(cfg, state)
    annotate_predictions(state["multidata"], adata_all, train_out["probs_soft"], "pred")
    if p_merge is not None:
        annotate_predictions(state["multidata"], adata_all, p_merge, "pred_merge")
    if p_align is not None:
        annotate_predictions(state["multidata"], adata_all, p_align, "pred_align")

    annotated_name = output_name(cfg, "annotated_h5ad", None)
    annotated_path = None
    if annotated_name:
        annotated_path = output_dir / annotated_name
        adata_all.write(annotated_path)
        print(f"[INFO] Saved annotated AnnData: {annotated_path}")

    last_stage_path = next(reversed(stage_paths.values()), None)
    next_round_init_path = merge_path if merge_path is not None else last_stage_path
    summary = {
        "init_checkpoint": str(state["init_path"]),
        "training_data_dir": str(state["training_data_dir"]),
        "batch_key": state["batch_key"],
        "batch_names": list(map(str, state["batch_names"])),
        "stage_checkpoints": {k: str(v) for k, v in stage_paths.items()},
        "merge_checkpoint": str(merge_path) if merge_path is not None else None,
        "annotated_h5ad": str(annotated_path) if annotated_path is not None else None,
        "next_round_init_path": (
            str(next_round_init_path) if next_round_init_path is not None else None
        ),
    }
    summary_path = output_dir / output_name(cfg, "summary_json", "training_summary.json")
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[INFO] Saved summary: {summary_path}")


def apply_overrides(cfg, args):
    if args.init_checkpoint is not None:
        cfg["input"]["init_checkpoint"] = args.init_checkpoint
    if args.output_dir is not None:
        cfg["output"]["output_dir"] = args.output_dir
    if args.round is not None:
        cfg["output"]["round"] = args.round
    if args.stages:
        cfg["training"]["stages"] = args.stages
    for stage, epochs in (
        ("learn_P", args.epochs_P),
        ("learn_mapping", args.epochs_mapping),
        ("learn_M", args.epochs_M),
    ):
        if epochs is not None:
            cfg["training"][stage]["epochs"] = epochs
    for stage, lr in (
        ("learn_P", args.lr_P),
        ("learn_mapping", args.lr_mapping),
        ("learn_M", args.lr_M),
    ):
        if lr is not None:
            cfg["training"][stage]["lr"] = lr
    if args.no_merge_align:
        cfg["merge_align"]["enabled"] = False
    if args.multi_gpu:
        cfg.setdefault("runtime", {})["multi_gpu"] = True
    if args.devices is not None:
        cfg.setdefault("runtime", {})["devices"] = args.devices
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the ResonanSC 02_train notebook workflow as a script."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_round.yaml"))
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--round", type=int)
    parser.add_argument("--stages", nargs="+", choices=sorted(STAGE_KEYS))
    parser.add_argument("--epochs-P", type=int)
    parser.add_argument("--epochs-mapping", type=int)
    parser.add_argument("--epochs-M", type=int)
    parser.add_argument("--lr-P", type=float)
    parser.add_argument("--lr-mapping", type=float)
    parser.add_argument("--lr-M", type=float)
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Distribute per-batch bulk computations across visible CUDA devices.",
    )
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated logical CUDA device ids to use after "
            "CUDA_VISIBLE_DEVICES has been applied, e.g. '0,1,2'."
        ),
    )
    parser.add_argument("--no-merge-align", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    cuda_devices = cfg.get("runtime", {}).get("cuda_visible_devices")
    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    import_runtime_deps()

    seed = cfg.get("runtime", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        visible = torch.cuda.device_count()
        print(f"[INFO] CUDA visible devices: {visible}")
        for idx in range(visible):
            free, total = torch.cuda.mem_get_info(idx)
            print(
                f"[INFO] cuda:{idx} {torch.cuda.get_device_name(idx)} "
                f"free={free / 1024**3:.2f}GiB total={total / 1024**3:.2f}GiB"
            )
        if cfg.get("runtime", {}).get("multi_gpu", False):
            print(
                "[INFO] Multi-GPU batch distribution enabled; "
                "device ids are logical after CUDA_VISIBLE_DEVICES."
            )
    else:
        print("[INFO] CUDA unavailable; using CPU.")

    output_dir = resolve_path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state = load_training_state(cfg)
    print(f"[INFO] Loaded init checkpoint: {state['init_path']}")
    print(f"[INFO] Batches: {list(state['batch_names'])}")
    for i, adata in state["multidata"].items():
        print(f"[INFO] multidata[{i}] {state['batch_names'][i]} shape={adata.shape}")

    train_out, stage_paths = train_stages(cfg, state, output_dir)
    save_m_figures(cfg, state, train_out, output_dir)

    merge_path = None
    p_merge = None
    p_align = None
    if cfg.get("merge_align", {}).get("enabled", True):
        merge_path, p_merge, p_align = run_merge_align(cfg, state, train_out, output_dir)

    write_outputs(
        cfg=cfg,
        state=state,
        train_out=train_out,
        output_dir=output_dir,
        stage_paths=stage_paths,
        merge_path=merge_path,
        p_merge=p_merge,
        p_align=p_align,
    )

    print("[INFO] Finished successfully.")
    if merge_path is not None:
        print(f"[INFO] Next round init_path: {merge_path}")


if __name__ == "__main__":
    main()
