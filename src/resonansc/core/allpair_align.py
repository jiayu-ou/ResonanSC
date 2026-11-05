"""Conservative all-pair matching of batch-local clusters.

The algorithm only sees cluster similarity matrices and batch membership.
"""

import numpy as np
import torch


def allpair_align(ps, similarities, min_edge=0.65, min_complete=0.65, top_k=2):
    sizes = [p.shape[1] for p in ps]
    offsets = np.cumsum([0] + sizes)
    total = int(offsets[-1])
    batch = np.concatenate([np.repeat(i, n) for i, n in enumerate(sizes)])
    parent = list(range(total))
    members = {i: [i] for i in range(total)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def score(a, b):
        i, j = int(batch[a]), int(batch[b])
        if i == j:
            return -1.0
        if i > j:
            a, b, i, j = b, a, j, i
        return float(similarities[(i, j)][a - offsets[i], b - offsets[j]])

    edges = []
    for (i, j), matrix in similarities.items():
        value = np.nan_to_num(np.asarray(matrix), nan=-1.0)
        if value.shape != (sizes[i], sizes[j]) or i >= j:
            raise ValueError(f'Invalid all-pair matrix {(i, j)}: {value.shape}')
        row_rank = np.argsort(-value, axis=1)[:, :top_k]
        col_rank = np.argsort(-value, axis=0)[:top_k, :]
        for a in range(value.shape[0]):
            for b in row_rank[a]:
                s = float(value[a, b])
                if s >= min_edge and a in col_rank[:, b]:
                    edges.append((s, int(offsets[i] + a), int(offsets[j] + b)))
    edges.sort(reverse=True)
    accepted = []
    for s, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        ma, mb = members[ra], members[rb]
        if set(batch[ma]) & set(batch[mb]):
            continue
        if min(score(x, y) for x in ma for y in mb) < min_complete:
            continue
        parent[rb] = ra
        members[ra] = ma + mb
        del members[rb]
        accepted.append((a, b, s))
    components = sorted(members.values(), key=min)
    label_map = np.empty(total, dtype=int)
    for k, component in enumerate(components):
        label_map[component] = k
    aligned = []
    for i, p in enumerate(ps):
        y = torch.zeros((p.shape[0], len(components)), dtype=p.dtype)
        for local in range(p.shape[1]):
            y[:, label_map[offsets[i] + local]] += p[:, local]
        aligned.append(y)
    return dict(p_align=aligned, label_map=label_map, components=components,
                offsets=offsets, accepted_edges=accepted)


def rna_de_similarities(batches, ps):
    """Calculate all batch-pair DE correlations, reusing each batch's DE."""
    from .bulk import get_bulk
    from .corr import de_1vAll_kbulk, compute_crossbatch_corr_de

    bulks = get_bulk(batches, ps)
    de = [de_1vAll_kbulk(x, p) for x, p in zip(batches, ps)]
    similarities = {}
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            matrix = compute_crossbatch_corr_de(
                batches[i], batches[j], bulks[i], bulks[j], ps[i], ps[j],
                deA_1v=de[i], deB_1v=de[j]).T
            similarities[(i, j)] = np.nan_to_num(matrix, nan=-1.0)
    return similarities
