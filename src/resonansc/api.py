"""Public Python interface to ResonanSC workflow stages."""
from pathlib import Path
import subprocess,sys


def align(config, *, rna_pca, rna_cell_names, atac_lsi, atac_cell_names, mode="batch"):
    """Align batches by default; set mode="condition" for the sec3 condition hierarchy."""
    from .hierarchical import run_hierarchical_alignment
    if isinstance(config,(str,Path)):
        from .config import load_config
        config=load_config(config)
    return run_hierarchical_alignment(config,rna_pca=rna_pca,rna_cell_names=rna_cell_names,atac_lsi=atac_lsi,atac_cell_names=atac_cell_names,mode=mode)


def resolve_labels(adata, **kwargs):
    """Resolve ATAC-only classes by pairwise RNA DE; does not mutate input."""
    from .post_alignment import resolve_atac_only
    return resolve_atac_only(adata,**kwargs)




def prepare_from_alignment(config, **kwargs):
    from .alignment_prepare import prepare_from_alignment as prepare
    return prepare(config,**kwargs)

def prepare_optimization(**kwargs):
    from .pairwise_prepare import prepare_pairwise_optimization
    return prepare_pairwise_optimization(**kwargs)

def optimize(config_path, *, python=None):
    """Run selected P/W/M optimization stages from an explicit training YAML.

    Uses a fresh subprocess; only trust input checkpoint files. Set CUDA device
    in that YAML. Returns the completed subprocess. A nonzero exit raises.
    """
    path=Path(config_path).expanduser().resolve()
    if not path.is_file():raise FileNotFoundError(path)
    import yaml
    config=yaml.safe_load(path.read_text()) or {}
    output=config.get('output',{}).get('output_dir')
    if not output:raise ValueError('training YAML requires output.output_dir')
    output=Path(output).expanduser()
    if not output.is_absolute():output=path.parent/output
    if output.exists():raise FileExistsError(f'optimization output already exists: {output}')
    return subprocess.run([str(python or sys.executable),'-m','resonansc.training','--config',str(path)],check=True)


def model_pseudobulk(checkpoint, labels, **kwargs):
    from .pseudobulk import export_model_pseudobulk
    return export_model_pseudobulk(checkpoint,labels,**kwargs)


def prepare_embedding(rna_counts, labels, checkpoint, training_dir, output, *, allow_init_weights=False):
    from .embedding.prepare import prepare
    return prepare(rna_counts,labels,checkpoint,training_dir,output,allow_init_weights)


def atac_w_lsi(adata, **kwargs):
    from .embedding.lsi import atac_w_lsi as transform
    return transform(adata,**kwargs)


def train_embedding(*, input_path, rna_map, parents, atac_lsi_values, atac_lsi_names, output, weight=200, latent=16, hidden=256, pretrain_epochs=100, epochs=50, python=None):
    """Train the latest all-cell balanced parent-supervised MMD200 embedding.

    `atac_lsi_values` and `atac_lsi_names` are .npy files from atac_w_lsi.
    Requires scvi-tools 1.3.3 and CUDA. Output directory must be new.
    """
    cmd=[str(python or sys.executable),'-m','resonansc.embedding.balanced_train','--input',str(input_path),'--rna-map',str(rna_map),'--parents',str(parents),'--atac-lsi-values',str(atac_lsi_values),'--atac-lsi-names',str(atac_lsi_names),'--output',str(output),'--weight',str(weight),'--latent',str(latent),'--hidden',str(hidden),'--pretrain-epochs',str(pretrain_epochs),'--epochs',str(epochs)]
    return subprocess.run(cmd,check=True)


def infer_embedding(input_path, **kwargs):
    """Infer the latest dual RNA/W-LSI embedding from trusted saved models."""
    from .embedding.infer_lsi import infer_embedding as infer
    return infer(input_path,**kwargs)
