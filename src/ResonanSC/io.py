from pathlib import Path
import torch
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Any
import json
import argparse
import yaml


def load_h5ad(path: str):
    return sc.read_h5ad(path)

def save_torch(obj: Any, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)

def load_torch(path: str, map_location="cpu") -> Any:
    return torch.load(path, map_location=map_location, weights_only=False)

def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)

def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_config(cfg, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ResonanSC pipeline with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file"
    )
    return parser.parse_args()


def prepare_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_run_summary(summary_dict, output_dir):
    output_dir = Path(output_dir)
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)

    
def save_M(M, adata, out_csv="M_with_gene_info.csv", col_prefix="type"):
    if isinstance(M, torch.Tensor):
        M_np = M.detach().cpu().numpy()
    else:
        M_np = np.asarray(M)

    assert M_np.ndim == 2, "M must be 2D"
    assert M_np.shape[0] == adata.n_vars, (
        f"M rows {M_np.shape[0]} must equal adata.n_vars {adata.n_vars}"
    )

    n_genes, n_types = M_np.shape
    value_cols = [f"{col_prefix}_{i}" for i in range(n_types)]

    df_M = pd.DataFrame(M_np, columns=value_cols)
    df_M.insert(0, "gene_index", np.arange(n_genes))
    df_M.insert(0, "gene_name", adata.var_names.astype(str).to_numpy())

    df_M.to_csv(out_csv, index=False)
    return df_M
