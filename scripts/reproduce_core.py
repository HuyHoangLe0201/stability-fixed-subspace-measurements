"""Deterministic-law experiments for the 19 September revision.

One-dimensional angular searches are numerical, not formal certificates.
Finite-law expectations are evaluated exactly as weighted sums. No legacy
CSV is consumed. Each randomized LP objective has an independent RNG stream.
"""
from pathlib import Path
import json
import platform
import sys
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq, linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'core_revision'
FIG = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 20260919
streams = np.random.SeedSequence(SEED).spawn(7)
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10,
                     'legend.fontsize': 7.5, 'figure.figsize': (4.4, 3.1),
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})

def info(theta, gamma, angles, weights):
    c = np.cos(np.asarray(theta)[..., None] - angles)**2
    return np.sum(weights * np.log1p(gamma*c), axis=-1)/2

def derivative(theta, gamma, angles, weights):
    d = np.asarray(theta)[..., None]-angles
    return -np.sum(weights*np.sin(2*d)/(1+gamma*np.cos(d)**2), axis=-1)/2

def maximize(gamma, angles, weights, grid_n=8192):
    grid = np.linspace(0, np.pi, grid_n+1)
    deriv = derivative(grid, gamma, angles, weights)
    roots = [0.0]
    for k in np.flatnonzero(deriv[:-1]*deriv[1:] < 0):
        roots.append(brentq(lambda x: derivative(x, gamma, angles, weights),
                            grid[k], grid[k+1], xtol=1e-14))
    roots.extend(grid[np.abs(deriv)<1e-14])
    candidates = np.asarray(roots)
    values = info(candidates, gamma, angles, weights)
    j = int(np.argmax(values))
    return float(candidates[j]), float(values[j])

def stable_gain(theta, baseline, gamma, angles, weights):
    c = np.cos(theta-angles)**2
    b = np.cos(baseline-angles)**2
    return float(np.dot(weights, np.log1p(gamma*(c-b)/(1+gamma*b)))/2)

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name)
    plt.close()

def main():
    metrics = {'seed': SEED, 'python': sys.version, 'platform': platform.platform(),
               'numpy': np.__version__, 'scipy': scipy.__version__,
               'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}
    angles = np.array([0., .6, 1.3, 2.2])
    weights = np.array([.45, .25, .20, .10])
    u = np.column_stack((np.cos(angles), np.sin(angles)))
    eigenvalues, eigenvectors = np.linalg.eigh(u.T@(weights[:,None]*u))
    p0 = eigenvectors[:, -1]
    theta0 = float(np.arctan2(p0[1], p0[0]) % np.pi)
    x = np.cos(angles-theta0)
    y = np.sin(angles-theta0)
    gap = float(eigenvalues[1]-eigenvalues[0])
    contraction = float(np.dot(weights, x**3*y))
    tilt = -contraction/gap
    distance_limit = np.sqrt(2)*abs(tilt)
    regret_limit = contraction**2/(2*gap)
    assert distance_limit > 1e-3
    rows=[]
    search_difference=0.
    for gamma in np.logspace(-4, 0, 21):
        theta, achieved = maximize(gamma, angles, weights)
        theta_check, achieved_check = maximize(gamma, angles, weights, 16384)
        search_difference=max(search_difference, abs(achieved-achieved_check))
        distance = np.sqrt(2)*abs(np.sin(theta-theta0))
        regret = stable_gain(theta, theta0, gamma, angles, weights)
        distance_bound = np.sqrt(2)*gamma/((1+gamma)*gap)
        regret_bound = gamma**3/(8*(1+gamma)**2*gap)
        assert distance <= distance_bound+1e-10
        assert -1e-14 <= regret <= regret_bound+1e-14
        assert abs(derivative(theta,gamma,angles,weights)) < 1e-10
        rows.append(dict(gamma=gamma, theta=theta, information=achieved,
                         distance=distance, distance_over_gamma=distance/gamma,
                         distance_bound=distance_bound, regret=regret,
                         regret_over_gamma_cubed=regret/gamma**3, regret_bound=regret_bound))
    a=pd.DataFrame(rows)
    a.to_csv(OUT/'optimizer.csv',index=False)
    low=a[a.gamma<=.01]
    distance_slope=float(np.polyfit(np.log10(low.gamma),np.log10(low.distance),1)[0])
    regret_slope=float(np.polyfit(np.log10(low.gamma),np.log10(low.regret),1)[0])
    assert abs(a.distance_over_gamma.iloc[0]/distance_limit-1) < .002
    assert abs(a.regret_over_gamma_cubed.iloc[0]/regret_limit-1) < .002
    metrics['optimizer']=dict(angles=angles.tolist(), weights=weights.tolist(),
        theta0=theta0,gap=gap,contraction=contraction,tilt=tilt,
        distance_limit=distance_limit,regret_limit=regret_limit,
        distance_slope=distance_slope,regret_slope=regret_slope,
        doubled_grid_information_difference=search_difference)
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.7))
    axes[0].loglog(a.gamma,a.distance,'o-',ms=3,label='Numerical optimizer')
    axes[0].loglog(a.gamma,distance_limit*a.gamma,'--',label='Predicted leading term')
    axes[0].loglog(a.gamma,a.distance_bound,':',label='Finite-SNR bound')
    axes[0].set(xlabel=r'SNR $\gamma$',ylabel=r'$\|P_\gamma-P_0\|_F$')
    axes[1].loglog(a.gamma,a.regret,'o-',ms=3,label='Covariance-design regret')
    axes[1].loglog(a.gamma,regret_limit*a.gamma**3,'--',label='Predicted cubic term')
    axes[1].loglog(a.gamma,a.regret_bound,':',label='Finite-SNR bound')
    axes[1].set(xlabel=r'SNR $\gamma$',ylabel='Regret (nats)')
    for ax in axes:ax.legend(loc='upper left')
    save('fig_optimizer_stability.pdf')

    # Same covariance, distinct fourth-order geometry and exact maximizers.
    theta_grid=np.linspace(0,np.pi,721)
    fig,axes=plt.subplots(1,2,figsize=(7.,2.7))
    orientation=[]
    for phi in [0.,np.pi/6]:
        law_angles=np.array([phi,phi+np.pi/2])
        c=np.cos(theta_grid[:,None]-law_angles)**2
        m2=(c*c).mean(axis=1)
        information=info(theta_grid,1.,law_angles,np.array([.5,.5]))
        label=rf'$\phi={int(round(phi*180/np.pi))}^\circ$'
        axes[0].plot(theta_grid*180/np.pi,m2,label=label)
        axes[1].plot(theta_grid*180/np.pi,information,label=label)
        chosen=(phi+np.pi/4)%np.pi
        assert abs(info(chosen,1.,law_angles,np.array([.5,.5]))-.5*np.log1p(.5))<1e-14
        orientation.extend(dict(phi=phi,theta=t,m2=m,information=i) for t,m,i in zip(theta_grid,m2,information))
    axes[0].set(xlabel=r'Measurement angle $\theta$ (degrees)',ylabel=r'$M_2(P_\theta)$')
    axes[1].set(xlabel=r'Measurement angle $\theta$ (degrees)',ylabel='Information at SNR = 1 (nats)')
    for ax in axes:ax.legend()
    pd.DataFrame(orientation).to_csv(OUT/'orientation.csv',index=False)
    save('fig_orientation.pdf')

    # Compress an arbitrary positive 41-point law on the projective circle.
    angles=np.pi*np.arange(41)/41
    weights=np.exp(.7*np.cos(2*angles)+.8*np.sin(4*angles)+.3*np.cos(6*angles))
    weights/=weights.sum()
    order_values=[1,2,3,4,6,8]
    measurements=np.linspace(0,np.pi,4096,endpoint=False)
    compressed=[]
    compression_rows=[]
    for r,seed in zip(order_values,streams):
        features=np.vstack([np.ones(len(angles))]+[
            f(2*n*angles) for n in range(1,r+1) for f in [np.cos,np.sin]])
        target=features@weights
        objective=np.random.default_rng(seed).normal(size=len(angles))
        result=linprog(objective,A_eq=features,b_eq=target,bounds=(0,None),method='highs-ds',
                       options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        assert result.success,result.message
        support=np.flatnonzero(result.x>1e-10)
        # Refine the selected weights against the equality constraints, without clipping.
        w,_,_,_=np.linalg.lstsq(features[:,support],target,rcond=None)
        assert np.all(w>0) and len(support)<=2*r+1
        residual=float(np.max(abs(features[:,support]@w-target)))
        assert residual<1e-12
        compressed.append(dict(order=r,indices=support.tolist(),weights=w.tolist(),
                               nodes=len(support),moment_residual=residual))
        for gamma in [.1,1.,10.]:
            true_values=info(measurements,gamma,angles,weights)
            surrogate_values=info(measurements,gamma,angles[support],w)
            sampled_error=float(np.max(abs(true_values-surrogate_values)))
            theta_true,value_true=maximize(gamma,angles,weights)
            theta_sur,value_sur=maximize(gamma,angles[support],w)
            regret=stable_gain(theta_true,theta_sur,gamma,angles,weights)
            q=gamma/(gamma+2)
            error_bound=min(gamma**(r+1)/(2*(r+1)),q**(r+1)/((r+1)*(1-q)))
            # Each cos^(2n) Fourier expansion has total absolute coefficient <=1.
            moment_tolerance_term=residual*sum(gamma**n/(2*n) for n in range(1,r+1))
            conservative_bound=gamma**(r+1)/(r+1)+moment_tolerance_term
            assert sampled_error<=error_bound+1e-11
            assert -1e-12<=regret<=2*error_bound+1e-11
            compression_rows.append(dict(order=r,nodes=len(support),gamma=gamma,
                sampled_information_error=sampled_error,information_bound=error_bound,
                transferred_regret=max(0.,regret),regret_bound=2*error_bound,
                optimal_value_error=abs(value_true-value_sur),moment_residual=residual,
                residual_aware_taylor_bound=conservative_bound))
    df=pd.DataFrame(compression_rows)
    df.to_csv(OUT/'compression.csv',index=False)
    (OUT/'compressed_laws.json').write_text(json.dumps(dict(original_angles=angles.tolist(),
        original_weights=weights.tolist(),surrogates=compressed),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(7.,2.7))
    for gamma,style in [(1.,'o'),(10.,'s')]:
        frame=df[df.gamma==gamma]
        axes[0].semilogy(frame.nodes,np.maximum(frame.sampled_information_error,1e-16),style+'-',
                         label=rf'Grid error, $\gamma={gamma:g}$')
        axes[0].semilogy(frame.nodes,frame.information_bound,'--',alpha=.6,
                         label=rf'Exact-moment bound, $\gamma={gamma:g}$')
        axes[1].semilogy(frame.nodes,np.maximum(frame.transferred_regret,1e-16),style+'-',
                         label=rf'Transfer regret, $\gamma={gamma:g}$')
    axes[0].set(xlabel='Retained directions (original: 41)',ylabel='Information error (nats)')
    axes[1].set(xlabel='Retained directions (original: 41)',ylabel='Achieved regret (nats)')
    for ax in axes:ax.legend()
    save('fig_surrogate.pdf')
    metrics['compression']=dict(original_nodes=41,orders=order_values,
        max_moment_residual=max(x['moment_residual'] for x in compressed),
        grid_size=len(measurements),rows=compression_rows)

    # Finite-sample check for Corollary 5 on the same 41-direction law.
    # This is a scaling diagnostic, not a proof of the high-probability bound.
    empirical_rng=np.random.default_rng(streams[-1])
    sample_sizes=np.array([100,400,1600,6400])
    empirical_trials=200
    empirical_gamma=1.
    kernel=np.log1p(empirical_gamma*np.cos(measurements[:,None]-angles)**2)/2
    true_curve=kernel@weights
    empirical_rows=[]
    empirical_summary=[]
    for sample_size in sample_sizes:
        counts=empirical_rng.multinomial(int(sample_size),weights,size=empirical_trials)
        curves=(counts/sample_size)@kernel.T
        errors=np.max(np.abs(curves-true_curve),axis=1)
        for trial,error in enumerate(errors):
            empirical_rows.append(dict(sample_size=int(sample_size),trial=trial,
                                       grid_information_error=float(error)))
        empirical_summary.append(dict(sample_size=int(sample_size),
            median_error=float(np.median(errors)),
            percentile90_error=float(np.quantile(errors,.9))))
    empirical_slope=float(np.polyfit(np.log(sample_sizes),
        np.log([row['median_error'] for row in empirical_summary]),1)[0])
    assert -.75 < empirical_slope < -.25
    pd.DataFrame(empirical_rows).to_csv(OUT/'empirical_learning.csv',index=False)
    metrics['finite_sample']=dict(gamma=empirical_gamma,trials=empirical_trials,
                                  grid_size=len(measurements),
                                  median_loglog_slope=empirical_slope,
                                  summary=empirical_summary)
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    def sci(v):
        mantissa,exponent=f'{v:.2e}'.split('e')
        return rf'{mantissa}\times10^{{{int(exponent)}}}'
    chosen=df[(df.order==4)&(df.gamma==1)].iloc[0]
    macros={'CoreGap':f'{gap:.5f}','CoreAngle':f'{theta0:.5f}',
            'CoreA':f'{contraction:.6f}','CoreTilt':f'{tilt:.6f}',
            'CoreDistanceLimit':f'{distance_limit:.6f}','CoreRegretLimit':sci(regret_limit),
            'CoreDistanceSlope':f'{distance_slope:.3f}','CoreRegretSlope':f'{regret_slope:.3f}',
            'CoreMomentResidual':sci(metrics['compression']['max_moment_residual']),
            'CoreCompressionError':sci(chosen.sampled_information_error),
            'CoreCompressionRegret':sci(chosen.transferred_regret),
            'CoreEmpiricalErrorA':sci(empirical_summary[0]['median_error']),
            'CoreEmpiricalErrorB':sci(empirical_summary[1]['median_error']),
            'CoreEmpiricalErrorC':sci(empirical_summary[2]['median_error']),
            'CoreEmpiricalErrorD':sci(empirical_summary[3]['median_error']),
            'CoreEmpiricalSlope':f'{empirical_slope:.3f}'}
    (ROOT/'paper'/'core_results.tex').write_text('% Generated by scripts/reproduce_core.py\n'+''.join(
        rf'\newcommand{{\{name}}}{{{value}}}'+'\n' for name,value in macros.items()))
    print(json.dumps({'optimizer':metrics['optimizer'],'compression_r4_snr1':chosen.to_dict()},indent=2))

if __name__=='__main__':
    main()
