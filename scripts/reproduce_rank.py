"""d=8 finite-law compression and rank-two design transfer.

All optimization comparisons are exact over a shared finite candidate family,
not claims of global optimization on the Grassmann manifold.
"""
from pathlib import Path
from itertools import combinations_with_replacement
import json
import math
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/rank_revision'; OUT.mkdir(parents=True,exist_ok=True)
SEED=20260920
law_rng,lp_rng,pool_rng,check_rng=[np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(4)]
u=law_rng.normal(size=(600,8))*np.sqrt([8,5,3,2,1,1,1,1])
u/=np.linalg.norm(u,axis=1)[:,None]
w=np.exp(.6*u[:,0]+.4*u[:,1]*u[:,2]); w/=w.sum()

def coverage(q,points=u): return np.sum((points@q)**2,axis=1)
def info(q,points,weights,gamma): return float(weights@np.log1p(gamma*coverage(q,points))/2)
def refine(q,points,weights,gamma):
    q=q.copy()
    for iteration in range(600):
        v=points@q; c=np.sum(v*v,axis=1)
        g=gamma*points.T@((weights/(1+gamma*c))[:,None]*v)
        g-=q@(q.T@g)
        norm2=float(np.sum(g*g))
        if norm2<1e-18: break
        old=info(q,points,weights,gamma); step=2.
        for _ in range(35):
            new=np.linalg.qr(q+step*g)[0]
            if info(new,points,weights,gamma)>=old+1e-4*step*norm2: break
            step*=.5
        else: break
        q=new
    return q

surrogates=[]
for r in [1,2,3]:
    bound=math.comb(8+2*r-1,2*r)
    if bound>=len(u):
        ids=np.arange(len(u)); weights=w.copy(); residual=0.
        method='original law retained: support bound exceeds original size'
    else:
        powers=list(combinations_with_replacement(range(8),2*r))
        features=np.array([np.prod(u[:,ids],axis=1) for ids in powers])
        features=np.vstack([np.ones(len(u)),features])
        target=features@w
        scale=np.maximum(np.linalg.norm(features,axis=1),1e-15)
        result=linprog(lp_rng.normal(size=len(u)),A_eq=features/scale[:,None],
                       b_eq=target/scale,bounds=(0,None),method='highs-ds',
                       options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        assert result.success,result.message
        ids=np.flatnonzero(result.x>1e-9)
        weights=np.linalg.lstsq(features[:,ids]/scale[:,None],target/scale,rcond=None)[0]
        residual=float(np.max(abs(features[:,ids]@weights-target)))
        assert np.min(weights)>0 and len(ids)<=bound and residual<1e-10
        method='positive moment linear program with support weight refinement'
    surrogates.append(dict(order=r,bound=bound,ids=ids,weights=weights,residual=residual,method=method))
    print('Order',r,'nodes',len(ids),'residual',residual,flush=True)

_,v=np.linalg.eigh(u.T@(w[:,None]*u)); p0=v[:,-2:]
random_pool=[np.linalg.qr(pool_rng.normal(size=(8,2)))[0] for _ in range(512)]
rows=[]; pool_arrays={}
for gamma in [1.,10.]:
    pool=[p0]+random_pool
    starts=[p0]+random_pool[:3]
    for law in [dict(ids=np.arange(len(u)),weights=w)]+surrogates[:2]:
        pool += [refine(q,u[law['ids']],law['weights'],gamma) for q in starts]
    pool=np.array(pool); pool_arrays['gamma_'+str(int(gamma))]=pool
    c=np.array([coverage(q) for q in pool])
    true=np.log1p(gamma*c)@w/2
    for law in surrogates:
        proxy=np.log1p(gamma*c[:,law['ids']])@law['weights']/2
        winner=int(np.argmax(proxy)); actual=float(true[winner])
        error=float(np.max(abs(true-proxy)))
        regret=float(np.max(true)-actual)
        q=gamma/(gamma+2); r=law['order']
        bound=min(gamma**(r+1)/(2*(r+1)),q**(r+1)/((r+1)*(1-q)))
        assert regret<=2*error+1e-13
        assert error<=bound+1e-10
        rows.append(dict(order=r,nodes=len(law['ids']),gamma=gamma,candidates=len(pool),
                         candidate_information_error=error,candidate_transfer_regret=regret,
                         best_original_information=float(np.max(true)),transferred_information=actual,
                         exact_moment_reference_bound=bound,moment_residual=law['residual']))
df=pd.DataFrame(rows); df.to_csv(OUT/'transfer.csv',index=False)
max_coverage_residual=0.
for k in [2,4]:
    for _ in range(128):
        q=np.linalg.qr(check_rng.normal(size=(8,k)))[0]; c=coverage(q)
        for law in surrogates:
            for n in range(1,law['order']+1):
                residual=abs(w@(c**n)-law['weights']@(c[law['ids']]**n))
                max_coverage_residual=max(max_coverage_residual,float(residual))
assert max_coverage_residual<1e-10
np.savez_compressed(OUT/'law_and_candidates.npz',directions=u,weights=w,**pool_arrays,
                    **{f'indices_r{x["order"]}':x['ids'] for x in surrogates},
                    **{f'weights_r{x["order"]}':x['weights'] for x in surrogates})
metrics=dict(seed=SEED,d=8,K=2,original_nodes=600,orders=[1,2,3],numpy=np.__version__,
             scipy=scipy.__version__,status='passed',max_coverage_moment_residual=max_coverage_residual,
             max_monomial_residual=max(x['residual'] for x in surrogates),
             scope='Exact transfer regret over a common finite candidate family; no global projector certificate',
             surrogates=[{k:v for k,v in x.items() if k not in ['ids','weights']} for x in surrogates],
             results=rows)
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
plt.rcParams.update({'font.size':9,'legend.fontsize':8,'pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(7,2.65))
for gamma,marker in [(1.,'o'),(10.,'s')]:
    f=df[df.gamma==gamma]
    for axis,col in zip(ax,['candidate_information_error','candidate_transfer_regret']):
        axis.semilogy(f.order,np.maximum(f[col],1e-16),marker+'-',label=rf'$\gamma={gamma:g}$')
        axis.set_xticks([1,2,3],['1\n36 nodes','2\n330 nodes','3\n600 nodes'])
        axis.set_xlabel('Matching order r / retained directions')
        axis.spines[['top','right']].set_visible(False); axis.legend()
ax[0].set_ylabel('Candidate information error (nats)')
ax[1].set_ylabel('Candidate transfer regret (nats)')
fig.tight_layout(); fig.savefig(ROOT/'paper/figures/fig_rank_transfer.pdf',bbox_inches='tight'); plt.close(fig)
def sci(x):
    a,b=f'{x:.2e}'.split('e'); return rf'{a}\times10^{{{int(b)}}}'
selected=df[(df.order==2)&(df.gamma==10)].iloc[0]
values={'RankError':sci(selected.candidate_information_error),'RankRegret':sci(selected.candidate_transfer_regret),
        'RankResidual':sci(max_coverage_residual),'RankCandidates':str(int(selected.candidates))}
(ROOT/'paper/rank_results.tex').write_text('% Generated by reproduce_rank.py\n'+''.join(
    rf'\newcommand{{\{k}}}{{{v}}}'+'\n' for k,v in values.items()))
print(df.to_string(index=False)); print('Coverage moment residual:',max_coverage_residual)
