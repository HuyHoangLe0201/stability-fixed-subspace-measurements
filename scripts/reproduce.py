"""Reproduce revised-paper results; no legacy CSV is used as an input."""
from pathlib import Path
import json
import platform
import numpy as np
import scipy
from scipy.special import roots_jacobi, roots_hermitenorm
from scipy.stats import kstest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'reproduced'
FIG = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 20260918
rng = np.random.default_rng(SEED)
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'legend.fontsize': 8,
                     'figure.figsize': (4.4, 3.0), 'savefig.bbox': 'tight',
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})

def beta_rule(a, b, order=256):
    x, w = roots_jacobi(order, b-1, a-1)
    return (x+1)/2, w/w.sum()

def haar(d, k, g, order=256):
    c, w = beta_rule(k/2, (d-k)/2, order)
    return np.dot(w, np.log1p(g*c))/2

def sphere(n, d, complex_=False):
    x = rng.normal(size=(n, d))
    if complex_:
        x = x + 1j*rng.normal(size=(n, d))
    return x/np.linalg.norm(x, axis=1)[:, None]

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name)
    plt.close()

def main():
    metrics = {}
    rows = []
    for complex_ in [False, True]:
        for d, k in [(3, 1), (8, 2), (8, 4), (16, 3)]:
            u = sphere(60000, d, complex_)
            c = np.sum(abs(u[:, :k])**2, axis=1)
            scale = 1 if complex_ else 0.5
            ks = kstest(c, 'beta', args=(scale*k, scale*(d-k))).statistic
            rows.append(dict(complex=complex_, d=d, K=k, N=len(c), ks=ks,
                             empirical_mean=c.mean(), expected_mean=k/d))
    pd.DataFrame(rows).to_csv(OUT/'haar.csv', index=False)
    for field in [False, True]:
        metrics['ks_complex' if field else 'ks_real'] = max(r['ks'] for r in rows if r['complex']==field)

    # ACG: train once, freeze candidate projectors, evaluate independently.
    spectrum = np.array([16, 8, 4, 2, 1, 1, 1, 1.])
    def acg():
        x = rng.normal(size=(40000, 8))*np.sqrt(spectrum)
        return x/np.linalg.norm(x, axis=1)[:, None]
    train, test = acg(), acg()
    ev, vec = np.linalg.eigh(train.T@train/len(train))
    candidates = [vec[:, -2:]]
    candidates += [np.linalg.qr(rng.normal(size=(8,2)))[0] for _ in range(64)]
    ct = np.array([np.sum((train@w)**2, axis=1) for w in candidates])
    ce = np.array([np.sum((test@w)**2, axis=1) for w in candidates])
    a_test = np.linalg.eigvalsh(test.T@test/len(test))[-2:].sum()
    sandwich = []
    for db in [-30, -20, -10, 0, 10, 20]:
        g = 10.**(db/10)
        best = int(np.argmax(np.log1p(g*ct).mean(axis=1)))
        values = np.log1p(g*ce[best])/2
        upper = np.log1p(g*a_test)/2
        assert values.mean() <= upper+1e-12
        sandwich.append(dict(snr_db=db, information=values.mean(),
                             se=values.std(ddof=1)/np.sqrt(len(values)),
                             haar=haar(8,2,g), empirical_converse=upper,
                             selected_candidate=best))
    s = pd.DataFrame(sandwich)
    s.to_csv(OUT/'sandwich.csv', index=False)
    metrics['acg_20db'] = sandwich[-1]
    plt.errorbar(s.snr_db, s.information, yerr=1.96*s.se, fmt='o-', capsize=3, label='Selected projector (test data)')
    plt.plot(s.snr_db, s.haar, '--', label='Exact Haar benchmark')
    plt.plot(s.snr_db, s.empirical_converse, '-.', label='Test-sample covariance bound')
    plt.xlabel('SNR (dB)'); plt.ylabel('Information (nats)'); plt.legend()
    savefig('fig_sandwich.pdf')

    # Cancellation-free convergent series for exact finite-ensemble witnesses.
    # Axes/tetrahedron achieve constant coverage 1/3 and the Jensen optimum.
    # Icosahedral vertex gives coverage 1 (probability 1/6), 1/5 (5/6).
    dbs = np.arange(-35, -9.9, 2.5)
    design = []
    for name in ['Axes / tetrahedron (optimal)', 'Icosahedron (vertex)']:
        power = 2 if name.startswith('Axes') else 3
        for db in dbs:
            g = np.longdouble(10)**(np.longdouble(db)/10)
            difference = np.longdouble(0)
            for n in range(power, 48):
                m = (np.longdouble(1)/3)**n if power==2 else (1+5*(np.longdouble(1)/5)**n)/6
                difference += (-1)**(n+1)*g**n*(m-np.longdouble(1)/(2*n+1))/(2*n)
            bound = float(g**power/(2*power))
            assert 0 < difference <= bound
            design.append(dict(ensemble=name, snr_db=db, information_gap=float(difference), bound=bound))
    ds = pd.DataFrame(design)
    ds.to_csv(OUT/'design.csv', index=False)
    for name, frame in ds.groupby('ensemble', sort=False):
        fit = frame[frame.snr_db <= -20]
        slope = np.polyfit(fit.snr_db/10, np.log10(fit.information_gap), 1)[0]
        metrics['slope_axes' if name.startswith('Axes') else 'slope_icosahedron'] = float(slope)
        plt.loglog(10**(frame.snr_db/10), frame.information_gap, 'o-', ms=3, label=name)
    g = np.logspace(-3.5,-1,100)
    plt.loglog(g, g*g/45, '--', label=r'$\gamma^2/45$')
    plt.loglog(g, 8*g**3/1575, ':', label=r'$8\gamma^3/1575$')
    plt.xlabel(r'SNR $\gamma$'); plt.ylabel('Information above Haar (nats)'); plt.legend()
    savefig('fig_design_hierarchy.pdf')

    # Moment and sharpness checks independent of information-series evaluation.
    phi = (1+np.sqrt(5))/2
    vertices = np.array([(0,a,b*phi) for a in [-1,1] for b in [-1,1]]+
                        [(a,b*phi,0) for a in [-1,1] for b in [-1,1]]+
                        [(b*phi,0,a) for a in [-1,1] for b in [-1,1]])
    vertices /= np.linalg.norm(vertices,axis=1)[:,None]
    directions = sphere(1000,3)
    coverage = (vertices@directions.T)**2
    moment_error = max(np.max(abs(coverage.mean(axis=0)-1/3)),
                       np.max(abs((coverage**2).mean(axis=0)-1/5)))
    assert moment_error < 1e-14
    metrics['icosahedron_moment_error'] = float(moment_error)
    # Covariance-envelope construction: all sign choices have identical coverage.
    import itertools
    lam = np.array([.5,.3,.15,.05])
    signs = np.array(list(itertools.product([-1,1], repeat=4)))
    support = signs*np.sqrt(lam)
    assert np.max(abs(support.T@support/16-np.diag(lam))) < 1e-14
    assert np.max(abs(np.sum(support[:,:2]**2,axis=1)-.8)) < 1e-14

    # Gauss-Hermite expectation of log(2)-log(1+exp(-2s-2sqrt(s)Z)).
    def bpsk_integral(g, order):
        c,wc = beta_rule(1,3,256)
        z,wz = roots_hermitenorm(order)
        wz /= np.sqrt(2*np.pi)
        snr = g*c[:,None]
        integrand = np.log(2)-np.logaddexp(0, -2*snr-2*np.sqrt(snr)*z)
        return float(wc@integrand@wz)
    bpsk = []
    for db in [-30,-20,-10,0,10,20]:
        g = 10.**(db/10)
        val = bpsk_integral(g,512)
        err = abs(val-bpsk_integral(g,1024))
        gaussian = haar(8,2,g)
        assert -1e-12 <= val <= min(np.log(2),gaussian)+1e-10
        bpsk.append(dict(snr_db=db,bpsk=val,gaussian=gaussian,ratio=val/gaussian,quadrature_difference=err))
    b = pd.DataFrame(bpsk)
    b.to_csv(OUT/'bpsk.csv',index=False)
    metrics['bpsk_ratio_20db'] = bpsk[-1]['ratio']
    metrics['bpsk_quadrature_difference'] = max(r['quadrature_difference'] for r in bpsk)
    plt.plot(b.snr_db,b.bpsk,'o-',label='BPSK'); plt.plot(b.snr_db,b.gaussian,'s-',label='Gaussian')
    plt.axhline(np.log(2),color='gray',ls=':',label=r'$\log 2$')
    plt.xlabel('SNR (dB)'); plt.ylabel('Information (nats)'); plt.legend()
    savefig('fig_bpsk.pdf')

    from scipy.stats import beta
    c = np.linspace(.0001,.9999,400)
    plt.plot(c,beta.pdf(c,1,3),label=r'Real: $\mathrm{Beta}(1,3)$')
    plt.plot(c,beta.pdf(c,2,6),'--',label=r'Complex: $\mathrm{Beta}(2,6)$')
    plt.xlabel('Coverage'); plt.ylabel('Density'); plt.legend(title=r'$d=8,\ K=2$')
    savefig('fig_beta_reference.pdf')
    metrics.update(seed=SEED,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    def sci(x):
        a,e = f'{x:.2e}'.split('e')
        return rf'{a}\times10^{{{int(e)}}}'
    defs = {'KSReal':sci(metrics['ks_real']),'KSComplex':sci(metrics['ks_complex']),
            'SlopeAxes':f"{metrics['slope_axes']:.3f}", 'SlopeIcosa':f"{metrics['slope_icosahedron']:.3f}",
            'ACGInfo':f"{s.information.iloc[-1]:.4f}", 'ACGSE':f"{s.se.iloc[-1]:.4f}",
            'ACGBound':f"{s.empirical_converse.iloc[-1]:.4f}", 'HaarTwenty':f"{s.haar.iloc[-1]:.4f}",
            'BPSKRatio':f"{metrics['bpsk_ratio_20db']:.4f}",
            'BPSKError':sci(metrics['bpsk_quadrature_difference']), 'MomentError':sci(moment_error)}
    (ROOT/'paper'/'results.tex').write_text('% Generated by scripts/reproduce.py\n'+''.join(
        rf'\newcommand{{\{k}}}{{{v}}}'+'\n' for k,v in defs.items()))
    print(json.dumps(metrics,indent=2))

if __name__ == '__main__':
    main()
