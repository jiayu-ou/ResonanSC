import numpy as np,pandas as pd,torch
from resonansc.core.merge_align import run_cross_batch_align

def hungarian_then_rescue(ps,cross,within,names,match_min,within_min,cross_min,max_gap,max_rmse,out_prefix):
    """Match one-to-one first, then attach only unmatched classes to matched groups."""
    cross=np.asarray(cross,dtype=float);assert cross.shape==(ps[0].shape[1],ps[1].shape[1]);assert len(within)==2
    initial=run_cross_batch_align(ps,names,[cross],cap=1,null_score=-.5,min_score=match_min)
    byrow,_=initial['edges_list'][0];pairs=sorted((int(i),int(j)) for i,js in byrow.items() for j in js)
    pd.DataFrame([{'left_cluster':i,'right_cluster':j,'corr':float(cross[i,j])} for i,j in pairs],columns=['left_cluster','right_cluster','corr']).to_csv(str(out_prefix)+'_initial_hungarian_pairs.csv',index=False)
    assigned=[{},{}]
    for group,(i,j) in enumerate(pairs):assigned[0][i]=group;assigned[1][j]=group
    initial_sets=[set(x) for x in assigned]; audits=[];rescue_rows=[]
    for side in (0,1):
      profiles=cross if side==0 else cross.T;score_within=np.asarray(within[side],dtype=float)
      for i in range(profiles.shape[0]):
       if i in initial_sets[side]:continue
       candidates=[]
       for anchor in sorted(initial_sets[side]):
        x,y=profiles[i],profiles[anchor];bi=int(np.argmax(x));bj=int(np.argmax(y));sim=float(score_within[i,anchor]);gap=float(abs(x[bi]-y[bj]));rmse=float(np.sqrt(np.mean((x-y)**2)));mc=float(min(x[bi],y[bj]));same=bi==bj
        ok=bool(np.isfinite(sim) and sim>=within_min and same and mc>=cross_min and gap<=max_gap and rmse<=max_rmse)
        row={'side':side,'unmatched_cluster':i,'anchor_cluster':anchor,'matched_group':assigned[side][anchor],'within_corr':sim,'same_best_opposite':same,'best_cross_min':mc,'top_score_gap':gap,'profile_rmse':rmse,'eligible':ok,'selected':False};audits.append(row)
        if ok:candidates.append(row)
       if candidates:
        best=sorted(candidates,key=lambda a:(-a['within_corr'],a['profile_rmse'],a['top_score_gap']))[0];best['selected']=True;assigned[side][i]=best['matched_group'];rescue_rows.append(best)
    for side in (0,1):
      for i in range(ps[side].shape[1]):
       if i not in assigned[side]:assigned[side][i]=max([*assigned[0].values(),*assigned[1].values()],default=-1)+1
    ng=max([*assigned[0].values(),*assigned[1].values()],default=-1)+1; result=[]
    for side,p in enumerate(ps):
      z=torch.zeros((p.shape[0],ng),dtype=p.dtype)
      for orig,dest in assigned[side].items():z[:,dest]+=p[:,orig]
      assert torch.allclose(z.sum(1),p.sum(1));result.append(z)
    pd.DataFrame(audits,columns=['side','unmatched_cluster','anchor_cluster','matched_group','within_corr','same_best_opposite','best_cross_min','top_score_gap','profile_rmse','eligible','selected']).to_csv(str(out_prefix)+'_unmatched_rescue_audit.csv',index=False)
    print('HUNGARIAN_THEN_RESCUE',names,'initial_pairs',len(pairs),'rescued',len(rescue_rows),'output_groups',ng,flush=True)
    return result,pairs,rescue_rows
