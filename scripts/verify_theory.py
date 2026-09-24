"""Independent numerical diagnostics of the revised analytic statements."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import root
from scipy.integrate import quad
from scipy.special import roots_jacobi

out=Path(__file__).resolve().parents[1]/'validation'
rng=np.random.default_rng(671923)
u=rng.normal(size=(9,3))*np.array([2.,1.,.5])
u/=np.linalg.norm(u,axis=1)[:,None]
p=rng.dirichlet(np.ones(9))
e,v=np.linalg.eigh(u.T@(p[:,None]*u))
e=e[::-1]; v=v[:,::-1]; z=u@v
x=z[:,0]; y=z[:,1:]
A=np.sum(p[:,None]*x[:,None]**3*y,axis=0)
prediction=-A/(e[0]-e[1:])

def gradient(t,g):
    vector=np.r_[1,t]
    s=z@vector
    den=vector@vector
    c=s*s/den
    dc=2*s[:,None]*y/den-2*s[:,None]**2*t/den**2
    return np.sum(p[:,None]*dc/(1+g*c[:,None]),axis=0)/2

rows=[]
for g in [.01,.003,.001,.0003]:
    sol=root(lambda t:gradient(t,g),g*prediction,tol=1e-10)
    assert np.linalg.norm(gradient(sol.x,g))<1e-11
    err=np.linalg.norm(sol.x/g-prediction)
    rows.append(dict(gamma=g,scaled_tilt_error=float(err)))
assert rows[-1]['scaled_tilt_error']<rows[0]['scaled_tilt_error']/20

# High-SNR Beta penalty, checked by endpoint-aware integration.
high=[]
for d,k in [(3,1),(8,2),(8,4)]:
    from scipy.special import beta,digamma
    a,b=k/2,(d-k)/2
    expected=(digamma(d/2)-digamma(k/2))/2
    g=1e7
    loss=quad(lambda t:np.log((1+g)/(1+g*t*t))*t**(2*a-1)*(1-t*t)**(b-1)/beta(a,b),
              0,1,points=[1/np.sqrt(g)],epsabs=1e-10)[0]
    assert abs(loss-expected)<.002
    high.append(dict(d=d,K=k,loss=loss,limit=expected))

# Exact moment-matched quadratures provide laws for arbitrary matching order.
checks=0
for order in range(1,6):
    nodes,weights=roots_jacobi(order,0,0)
    c=(nodes+1)/2; weights/=weights.sum()
    r=2*order-1
    for g in [.01,.3,1,10,100]:
        reference=quad(lambda c:.5*np.log1p(g*c),0,1,epsabs=1e-13)[0]
        estimate=.5*np.dot(weights,np.log1p(g*c))
        q=g/(g+2)
        bound=min(g**(r+1)/(2*(r+1)),q**(r+1)/((r+1)*(1-q)))
        assert abs(estimate-reference)<=bound+1e-13
        checks+=1

result=dict(tilt=rows,high_snr=high,matched_moment_bound_cases=checks,status='passed')
(out/'theory_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
