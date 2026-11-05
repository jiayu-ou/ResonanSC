"""One-to-one Hungarian matching of every donor pair, with conflict-safe union."""
import numpy as np
import pandas as pd
import torch
from .alignment_algorithms import pearson_matrix
from .core.merge_align import align_clusters_hungarian


def align_all_donors(profiles, assignments, batch_names, *, min_score=None, null_score=-0.5):
    """Align every pair of donors; never combine two classes from one donor.

    `profiles[i]` is clusters × features and `assignments[i]` is cells ×
    clusters; return aligned assignments and an audit of all Hungarian edges.
    The historical sec3 run used min_score=None (no explicit score cutoff).
    """
    if not profiles or len(profiles)!=len(assignments) or len(profiles)!=len(batch_names):
        raise ValueError('profiles, assignments and batch_names must be equally sized and nonempty')
    if len(set(map(str,batch_names)))!=len(batch_names):
        raise ValueError('batch names must be unique')
    arrays=[np.asarray(x,dtype=float) for x in profiles]
    if any(x.ndim!=2 or not np.isfinite(x).all() for x in arrays):raise ValueError('profiles must be finite matrices')
    if len({x.shape[1] for x in arrays})!=1:raise ValueError('all profiles must have equal feature dimension')
    ps=[torch.as_tensor(p) for p in assignments]
    if any(p.ndim!=2 or p.shape[1]!=a.shape[0] or not torch.isfinite(p).all() or (p<0).any() for p,a in zip(ps,arrays)):
        raise ValueError('assignments must be finite nonnegative cells × clusters')
    offsets=np.cumsum([0]+[p.shape[1] for p in ps]);parent=list(range(int(offsets[-1])));owners=[{i} for i,p in enumerate(ps) for _ in range(p.shape[1])]
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    edges=[]
    for a in range(len(ps)):
        for b in range(a+1,len(ps)):
            c=pearson_matrix(arrays[a],arrays[b]);mapping,_=align_clusters_hungarian(c,cap=1,null_score=null_score)
            for i,js in mapping.items():
                for j in js:edges.append((float(c[i,j]),a,int(i),b,int(j)))
    audit=[]
    for score,a,i,b,j in sorted(edges,reverse=True):
        x=find(int(offsets[a]+i));y=find(int(offsets[b]+j))
        if not np.isfinite(score) or (min_score is not None and score<min_score):status='below_threshold'
        elif x==y:status='already_connected'
        elif owners[x]&owners[y]:status='same_batch_conflict'
        else:parent[y]=x;owners[x]|=owners[y];status='accepted'
        audit.append(dict(batch_a=str(batch_names[a]),cluster_a=i,batch_b=str(batch_names[b]),cluster_b=j,correlation=score,status=status))
    roots=[find(i) for i in range(int(offsets[-1]))];levels={r:i for i,r in enumerate(sorted(set(roots)))};out=[]
    for a,p in enumerate(ps):
        q=p.new_zeros((len(p),len(levels)))
        for j in range(p.shape[1]):q[:,levels[roots[int(offsets[a]+j)]]]+=p[:,j]
        if not torch.allclose(q.sum(1),p.sum(1)):raise RuntimeError('assignment mass changed')
        out.append(q)
    return out,pd.DataFrame(audit,columns=['batch_a','cluster_a','batch_b','cluster_b','correlation','status'])
