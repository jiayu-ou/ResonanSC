"""Portable YAML configuration. Paths are relative to the YAML file."""
from copy import deepcopy
from pathlib import Path

import yaml

DEFAULTS = {
    "input": {"expression_h5ad": None, "peak_h5ad": None, "gtf": None,
              "peak_genome_build": None, "gtf_genome_build": None, "batch_key": "batch",
              "batches": None},
    "output_dir": "outputs/run",
    "runtime": {"seed": 42, "gpu": "0", "neighbors_backend": "gpu", "gpu_python": None,
                "multi_gpu": False, "devices": None},
    "alignment": {"n_top_genes": 2000, "n_pcs": 30, "n_lsi": 30, "n_neighbors": 15,
                  "leiden_resolution": 1.5, "min_peak_cells": 20,
                  "rna_leiden_resolution": .9, "atac_leiden_resolution": 1.0,
                  "cross_condition_atac_min_corr": .5,
                  "condition_key": "condition", "condition_groups": None,
                  "rna_first_threshold": .8, "atac_first_threshold": .6,
                  "min_within": .7, "min_cross": .3, "max_top_gap": .1, "max_profile_rmse": .24,
                  "atac_min_within": .5, "atac_min_cross": .05, "atac_max_top_gap": .3,
                  "atac_max_profile_rmse": .24, "cross_min_within": .7, "cross_min_cross": .12,
                  "cross_max_top_gap": .02, "cross_max_profile_rmse": .017,
                  "max_merge_rounds": 4, "defer_within_merge_across_batches": False,
                  "deferred_min_match_margin": .03,
                  "rna_align_cap": 1, "atac_align_cap": 1,
                  "condition_align_min_score": .95,
                  "condition_align_mutual_best": True,
                  "condition_align_reciprocal_top_k": 2},
    "preparation": {"min_peak_cells": 20, "max_distance": 250000,
                    "distance_scale": 50000, "marker_top_n": 100, "marker_min_cells": 10},
    "plotting": {"rna_cell_type_key": None, "atac_cell_type_key": None,
                 "condition_key": None},
    "training": {"stages": ["learn_P", "learn_mapping", "learn_M"],
        "resume": {"learn_P": None, "learn_mapping": "learn_P", "learn_M": "learn_mapping"},
        "common": {"one_hot_start": False, "present_eps": .1, "rna_chunk_size": 512,
                   "cellbulk_chunk_size": 512, "cache_rna_on_device": False,
                   "marker_init_mode": "continuous", "marker_init_eps": .05, "verbose_every": 10},
        "margin": {"learn_P": [.5, 2., 10.], "learn_mapping": [.3, 1.5, 3.], "learn_M": [.3, 1.5, 3.]},
        "learn_P": {"epochs": 100, "lr": .01, "lambdas": [10., 10., 1.]},
        "learn_mapping": {"epochs": 200, "lr": .01, "lambdas": [1., 1., 0.], "mapping_lambdas": [1., 1.]},
        "learn_M": {"epochs": 30, "lr": .01, "lambdas": [1., 1., 0.], "marker_sparsity": 1.}},
}


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path):
    path = Path(path).resolve()
    supplied = yaml.safe_load(path.read_text()) or {}
    cfg = merge(DEFAULTS, supplied)
    for key in ("expression_h5ad", "peak_h5ad", "gtf"):
        value = cfg["input"].get(key)
        if value:
            value = Path(value).expanduser()
            cfg["input"][key] = str((path.parent / value).resolve())
    cfg["output_dir"] = str((path.parent / Path(cfg["output_dir"]).expanduser()).resolve())
    if cfg["runtime"].get("gpu_python"):
        value = Path(cfg["runtime"]["gpu_python"]).expanduser()
        # Resolving the executable symlink destroys Python venv discovery.
        cfg["runtime"]["gpu_python"] = str((path.parent / value).absolute())
    if cfg["runtime"]["neighbors_backend"] not in ("gpu", "cpu"):
        raise ValueError("runtime.neighbors_backend must be gpu or cpu")
    stages = cfg["training"]["stages"]
    if not stages or len(stages) != len(set(stages)) or any(s not in ("learn_P", "learn_mapping", "learn_M") for s in stages):
        raise ValueError("training.stages must be nonempty, unique supported optimization stages")
    canonical = [s for s in ('learn_P', 'learn_mapping', 'learn_M') if s in stages]
    if stages != canonical:
        raise ValueError('Stages must follow learn_P -> learn_mapping -> learn_M order')
    overrides = supplied.get('training', {}).get('resume', {})
    for i, stage in enumerate(stages):
        resume = overrides.get(stage, stages[i-1] if i else None)
        if resume and resume not in stages[:i]:
            if resume in ('learn_P','learn_mapping','learn_M'):
                raise ValueError(f'{stage}: resume stage is not an earlier selected stage')
            resume = str((path.parent / Path(resume).expanduser()).resolve())
            if not Path(resume).is_file():
                raise ValueError(f'{stage}: resume checkpoint does not exist: {resume}')
        cfg['training']['resume'][stage] = resume
    for stage in stages:
        if cfg["training"][stage]["epochs"] < 1:
            raise ValueError(f"{stage}: epochs must be positive")
    for key in ('n_top_genes','n_pcs','n_lsi','n_neighbors','max_merge_rounds'):
        if cfg['alignment'][key] < 1:
            raise ValueError(f'alignment.{key} must be positive')
    for modality in ('rna', 'atac'):
        cap = cfg['alignment'][f'{modality}_align_cap']
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError(f'alignment.{modality}_align_cap must be a positive integer')
    score = cfg['alignment']['condition_align_min_score']
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not (-1 <= score <= 1):
        raise ValueError('alignment.condition_align_min_score must be between -1 and 1')
    if not isinstance(cfg['alignment']['condition_align_mutual_best'], bool):
        raise ValueError('alignment.condition_align_mutual_best must be boolean')
    top_k = cfg['alignment']['condition_align_reciprocal_top_k']
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError('alignment.condition_align_reciprocal_top_k must be a positive integer')
    cfg["_config_path"] = str(path)
    return cfg
