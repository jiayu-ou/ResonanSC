"""Input contracts shared by alignment and optimization."""
from pathlib import Path
import re
import anndata as ad
import numpy as np
from scipy import sparse


def rows(matrix, indices):
    indices = np.asarray(indices)
    order = np.argsort(indices)
    value = matrix[indices[order], :]
    return sparse.csr_matrix(value)[np.argsort(order), :]


def require_counts(matrix):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.round(values), atol=1e-6, rtol=0):
        raise ValueError('ATAC X must contain finite nonnegative integer peak counts, not TF-IDF/activity')


def validate(cfg, require_gtf=False):
    inp = cfg['input']
    for key in ('expression_h5ad', 'peak_h5ad') + (('gtf',) if require_gtf else ()):
        if not inp.get(key) or not Path(inp[key]).is_file():
            raise ValueError(f'input.{key}: file does not exist')
    if require_gtf:
        a, b = inp.get('peak_genome_build'), inp.get('gtf_genome_build')
        if not a or not b or a != b:
            raise ValueError('Declare identical peak_genome_build and gtf_genome_build; liftover is not automatic')
    source = ad.read_h5ad(inp['expression_h5ad'], backed='r')
    peak = ad.read_h5ad(inp['peak_h5ad'], backed='r')
    try:
        for x in (source, peak):
            if not x.obs_names.is_unique or not x.var_names.is_unique:
                raise ValueError('Cell and feature names must be unique')
        key = inp['batch_key']
        if key not in source.obs:
            raise ValueError(f'Missing obs[{key!r}]')
        observed = source.obs[key].astype(str)
        names = inp.get('batches') or list(observed.unique())
        if not names or len(names) != len(set(names)) or not set(names) <= set(observed):
            raise ValueError('Invalid requested batches')
        rna = [n for n in names if n.startswith('rna_')]
        atac = [n for n in names if n.startswith('atac_')]
        if not rna or not atac or len(rna) + len(atac) != len(names):
            raise ValueError('Require RNA and ATAC batches named rna_* and atac_*')
        names = rna + atac
        cells = source.obs_names[observed.isin(atac)]
        if (peak.obs_names.get_indexer(cells) < 0).any():
            raise ValueError('ATAC cells missing from peak counts')
        return {'batches': names, 'n_rna': len(rna), 'n_atac': len(atac),
                'n_cells': int(observed.isin(names).sum()), 'n_genes_available': source.n_vars,
                'n_peak_features': peak.n_vars}
    finally:
        source.file.close()
        peak.file.close()


def resolve_condition_groups(cfg, info=None):
    """Resolve configured modality-specific condition names to whole batches."""
    info = info or validate(cfg)
    specs = cfg['alignment'].get('condition_groups')
    if not isinstance(specs, dict) or not specs:
        raise ValueError('alignment.condition_groups must be a nonempty mapping')
    key = cfg['alignment'].get('condition_key', 'condition')
    source = ad.read_h5ad(cfg['input']['expression_h5ad'], backed='r')
    try:
        if key not in source.obs:
            raise ValueError(f'Condition column {key!r} is missing from expression obs')
        batch_values = source.obs[cfg['input']['batch_key']].astype(str)
        condition_values = source.obs[key].astype('string')
        per_batch = {}
        for batch in info['batches']:
            values = condition_values[batch_values == batch]
            unique = values.dropna().unique().tolist()
            if values.isna().any() or len(unique) != 1:
                raise ValueError(f'{batch}: expected exactly one nonmissing {key} value; got {unique}')
            per_batch[batch] = str(unique[0])
    finally:
        source.file.close()
    groups = {}
    assigned = set()
    for name, spec in specs.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Invalid condition group name {name!r}; use letters, digits, _ or -')
        if not isinstance(spec, dict) or set(spec) != {'rna', 'atac'}:
            raise ValueError(f'{name}: specify exactly rna and atac condition lists')
        selected = []
        for modality in ('rna', 'atac'):
            conditions = spec[modality]
            if not isinstance(conditions, list) or not conditions or len(set(conditions)) != len(conditions):
                raise ValueError(f'{name}.{modality}: provide a nonempty list of unique conditions')
            matches = [batch for batch in info['batches']
                       if batch.startswith(modality + '_') and per_batch[batch] in conditions]
            if not matches:
                raise ValueError(f'{name}.{modality}: no batches match {conditions}')
            selected.extend(matches)
        overlap = assigned.intersection(selected)
        if overlap:
            raise ValueError(f'{name}: batches assigned to more than one condition group: {sorted(overlap)}')
        assigned.update(selected)
        groups[name] = selected
    missing = set(info['batches']) - assigned
    if missing:
        raise ValueError(f'Condition groups do not cover configured batches: {sorted(missing)}')
    return groups, per_batch
