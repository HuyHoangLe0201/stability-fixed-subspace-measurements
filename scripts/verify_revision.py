"""Independent diagnostics; numerical checks do not replace proofs."""
from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(7192026)
cases=0
for d in [2,3,5,8]:
    u=rng.normal(size=(53,d)); u/=np.linalg.norm(u,axis=1)[:,None]
    w=rng.dirichlet(np.ones(len(u)))
    val,vec=np.linalg.eigh(u.T@(w[:,None]*u))
    for k in range(1,d):
        p0=vec[:,-k:]@vec[:,-k:].T
        gap=val[-k]-val[-k-1]
        c0=np.einsum('ni,ij,nj->n',u,p0,u)
        for trial in range(30):
            q,_=np.linalg.qr(rng.normal(size=(d,k)))
            p=q@q.T
            distance=np.linalg.norm(p-p0)
            c=np.einsum('ni,ij,nj->n',u,p,u)
            assert np.dot(w,c0-c)>=gap*distance**2/2-1e-13
            assert np.linalg.norm(p-p0,2)<=distance/np.sqrt(2)+1e-13
            for gamma in [.001,.1,1,10]:
                gain=np.dot(w,np.log1p(gamma*(c-c0)/(1+gamma*c0)))/2
                beta=gamma/(2*(1+gamma))
                envelope=-gamma*gap*distance**2/4+gamma*beta*distance/np.sqrt(2)
                assert gain<=envelope+1e-13
                cases+=1

# Independent checks of the Gaussian projective-Wasserstein and joint constants.
# The displayed coupling cost upper-bounds W_1^proj, so these are conservative tests.
transport_cases=0
for d in [2,4,8]:
    u=rng.normal(size=(71,d)); u/=np.linalg.norm(u,axis=1)[:,None]
    v=u+.2*rng.normal(size=u.shape); v/=np.linalg.norm(v,axis=1)[:,None]
    weights=rng.dirichlet(np.ones(len(u)))
    projective_pair_cost=np.minimum(np.linalg.norm(u-v,axis=1),
                                    np.linalg.norm(u+v,axis=1))
    coupling_cost=float(weights@projective_pair_cost)
    for k in range(1,d):
        for _ in range(20):
            q,_=np.linalg.qr(rng.normal(size=(d,k)))
            p=q@q.T
            cu=np.einsum('ni,ij,nj->n',u,p,u)
            cv=np.einsum('ni,ij,nj->n',v,p,v)
            for gamma,gamma_prime in [(.1,.1),(.1,1.),(1.,10.),(10.,.2)]:
                imu=float(weights@np.log1p(gamma*cu)/2)
                inu_same=float(weights@np.log1p(gamma*cv)/2)
                inu_prime=float(weights@np.log1p(gamma_prime*cv)/2)
                assert abs(imu-inu_same)<=gamma*coupling_cost/2+1e-13
                joint_bound=(abs(gamma-gamma_prime)/2
                             +min(gamma,gamma_prime)*coupling_cost/2)
                assert abs(imu-inu_prime)<=joint_bound+1e-13
                transport_cases+=1
    # Antipodal redistribution has zero projective distance and identical coverage.
    antipodal_q,_=np.linalg.qr(rng.normal(size=(d,1)))
    antipodal_p=antipodal_q@antipodal_q.T
    assert np.max(np.abs(np.einsum('ni,ij,nj->n',u,antipodal_p,u)
                         -np.einsum('ni,ij,nj->n',-u,antipodal_p,-u)))<1e-14
    assert np.max(np.minimum(np.linalg.norm(u-(-u),axis=1),
                             np.linalg.norm(u+(-u),axis=1)))<1e-14
saved=json.loads((root/'experiments/core_revision/compressed_laws.json').read_text())
alpha=np.array(saved['original_angles']); w=np.array(saved['original_weights'])
max_error=0.
for law in saved['surrogates']:
    ids=law['indices']; v=np.array(law['weights']); r=law['order']
    for theta in rng.uniform(0,np.pi,100):
        c=np.cos(theta-alpha)**2
        for n in range(1,r+1):
            err=abs(w@(c**n)-v@(c[ids]**n))
            max_error=max(max_error,err)
            assert err<1e-12
result=dict(status='passed',seed=7192026,multirank_quadratic_envelope_cases=cases,
            projective_transport_perturbation_cases=transport_cases,
            independent_coverage_moment_max_residual=max_error)
(root/'validation/revision_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
