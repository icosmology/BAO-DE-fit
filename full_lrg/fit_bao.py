#!/usr/bin/env python3
"""Joint BAO inference from measured xi, dxi/dz and d2xi/dz2.
No published distance measurements, earlier fits or fabricated error bars enter.
Pre-reconstruction. Gaussian likelihood with explicitly regularized jackknife
covariance. Results are conditional on the stated finite redshift basis.
"""
import argparse,json,time,multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular,cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.stats import chi2
import emcee
from bao_model import Lightcone,prepare_template,distance_history,DZ,ZP
from reduce_counts import joint_cov

ENGINE=None

class Likelihood:
    def __init__(self,lc,data,cov,shrink=.2):
        self.lc=lc;self.y=data;self.cov=(1-shrink)*cov+shrink*np.diag(np.diag(cov));self.chol=np.linalg.cholesky(self.cov)
        self.yw=solve_triangular(self.chol,data,lower=True)
        self.ng=6 if lc.quadratic else 4
        self.bounds=[(-.2,.2),(-.2,.2),(-.35,.35),(-.35,.35)]+([(-.8,.8),(-.8,.8)] if lc.quadratic else [])+[(2.,7.),(4.,15.)]
        self.priors=np.r_[np.full(9,30.),np.full(lc.bb.shape[1],50.)]
    def calc(self,p,return_model=False):
        if any(not(lo<=x<=hi) for x,(lo,hi) in zip(p,self.bounds)):return (np.inf,None,None) if return_model else np.inf
        M=self.lc.matrix(p);A=solve_triangular(self.chol,M,lower=True)
        normal=A.T@A+np.diag(1/self.priors**2);v=A.T@self.yw
        try:
            fac=cho_factor(normal,lower=True);c=cho_solve(fac,v)
        except np.linalg.LinAlgError:return (np.inf,None,None) if return_model else np.inf
        residual=self.yw-A@c
        prior=(p[self.ng]-4.5)**2+(p[self.ng+1]-9.)**2/4.
        chis=float(residual@residual+np.sum((c/self.priors)**2)+prior)
        logdet=2*np.log(np.diag(fac[0])).sum()
        val=.5*(chis+logdet)
        return (val,M@c,c) if return_model else val


def logprob(p):
    value=ENGINE.calc(p)
    return -value if np.isfinite(value) else -np.inf


def optimize(engine,seed=4629,tries=5):
    rng=np.random.default_rng(seed);ng=engine.ng
    initial=np.r_[np.zeros(ng),4.5,9.]
    best=None
    for trial in range(tries):
        p=initial.copy()
        if trial:
            p[:ng]=rng.normal(0,.015,ng);p[ng:]+=rng.normal(0,.2,2)
        res=minimize(engine.calc,p,bounds=engine.bounds,method='L-BFGS-B',options={'maxiter':1200,'ftol':1e-9,'gtol':1e-5,'maxls':30})
        print('optimization',trial,res.fun,res.success,res.x.tolist(),flush=True)
        if best is None or res.fun<best.fun:best=res
    return best


def summary_parameters(samples,names):
    qq=np.percentile(samples,[2.5,16,50,84,97.5],axis=0)
    return {name:{'p2p5':float(qq[0,i]),'p16':float(qq[1,i]),'median':float(qq[2,i]),'p84':float(qq[3,i]),'p97p5':float(qq[4,i])} for i,name in enumerate(names)}


def run_fit(m,t,out,quadratic,steps=3000):
    global ENGINE
    name='quadratic' if quadratic else 'linear';orders=3 if quadratic else 2
    lc=Lightcone(m,t,orders=orders,quadratic=quadratic)
    ds=lc.select(m['multipoles']);cov=joint_cov(ds,int(m['njack_ngc']),int(m['njack_sgc']))
    ENGINE=Likelihood(lc,ds[0],cov,.2)
    start=time.time();best=optimize(ENGINE)
    names=['ln_alpha_perp_p','ln_alpha_parallel_p','dln_alpha_perp_du','dln_alpha_parallel_du']+(['d2ln_alpha_perp_du2','d2ln_alpha_parallel_du2'] if quadratic else [])+['Sigma_perp','Sigma_parallel']
    dim=len(best.x);rng=np.random.default_rng(64832+int(quadratic));walkers=4*dim
    widths=np.r_[np.full(ENGINE.ng,.002),.05,.08]
    init=best.x+widths*rng.normal(size=(walkers,dim))
    for j,(lo,hi) in enumerate(ENGINE.bounds):init[:,j]=np.clip(init[:,j],lo+1e-5,hi-1e-5)
    with mp.get_context('fork').Pool(4) as pool:
        sampler=emcee.EnsembleSampler(walkers,dim,logprob,pool=pool)
        sampler.run_mcmc(init,steps,progress=True)
    chain=sampler.get_chain();burn=steps//3;samples=sampler.get_chain(discard=burn,thin=5,flat=True)
    try:tau=sampler.get_autocorr_time(tol=0)
    except Exception:tau=np.full(dim,np.nan)
    namesdist=['DM_over_rd','DH_over_rd','dDM_over_rd_dz','dDH_over_rd_dz','d2DM_over_rd_dz2','d2DH_over_rd_dz2']
    deriv=np.array([distance_history(p,np.array([ZP]),float(t['rd']),quadratic)[:,0] for p in samples])
    np.savez_compressed(out/f'{name}_posterior.npz',samples=samples,chain=chain,logprob=sampler.get_log_prob(),parameter_names=names,derived=deriv,derived_names=namesdist,best_fit=best.x,covariance_raw=cov,covariance_used=ENGINE.cov)
    val,model,coeff=ENGINE.calc(best.x,True)
    measured_err=np.sqrt(np.diag(cov));fit_chi=float((ds[0]-model)@np.linalg.solve(ENGINE.cov,ds[0]-model))
    report={'basis':name,'derivative_channels':orders,'redshift_pivot':ZP,'u_definition':'(z-0.75)/0.35','data_dimension':len(ds[0]),'raw_covariance_rank':int(np.linalg.matrix_rank(cov)),'covariance_shrinkage':.2,'linear_nuisance_parameters':len(coeff),'nonlinear_parameters':dim,'conditional_best_fit_chi2_data':fit_chi,'best_fit':best.x.tolist(),'parameters':summary_parameters(samples,names),'distances_and_derivatives':summary_parameters(deriv,namesdist),'mean_acceptance_fraction':float(np.mean(sampler.acceptance_fraction)),'autocorrelation_time_estimate':tau.tolist(),'steps_per_walker':steps,'burn_in':burn,'effective_sample_estimate':float(walkers*(steps-burn)/np.nanmax(tau)),'optimizer_success':bool(best.success),'seconds':time.time()-start}
    # How much posterior probability is close to each hard parameter bound?
    report['boundary_fractions']={n:float(np.mean((samples[:,i]<lo+.02*(hi-lo))|(samples[:,i]>hi-.02*(hi-lo)))) for i,(n,(lo,hi)) in enumerate(zip(names,ENGINE.bounds))}
    # Covariance regularization sensitivity, holding model and priors unchanged.
    robust=[]
    for shrink in [0.,.1,.4]:
        test=Likelihood(lc,ds[0],cov,shrink)
        fit=minimize(test.calc,best.x,bounds=test.bounds,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-8})
        derived=distance_history(fit.x,np.array([ZP]),float(t['rd']),quadratic)[:,0]
        robust.append({'covariance_shrinkage':shrink,'optimizer_success':bool(fit.success),'parameters':fit.x.tolist(),'distances':derived.tolist()})
    report['regularization_sensitivity']=robust
    # Change damping prior centres by their quoted widths as a template sensitivity diagnostic.
    report['amplitude_prior_sigma']=30.;report['broadband_coefficient_prior_sigma']=50.
    report['limitations']=['Pre-reconstruction, no shifted randoms or mocks used.','The empirical jackknife covariance is regularized; no claim of an official DESI covariance or calibrated discovery significance.','Smooth finite-dimensional redshift model; distance derivatives are not model-independent point estimates.','Actual RR redshift windows and separation-bin averaging are included, but finite angular/redshift discretization and estimator integral constraints need further mock validation.','First-/second-derivative clustering includes bias, growth, RSD, damping and selection evolution.','Gaussian nuisance amplitude priors allow flexible angular/redshift evolution; no growth parameter inference is made.']
    (out/f'{name}_fit_summary.json').write_text(json.dumps(report,indent=2))
    plot_fit(lc,ds[0],measured_err,model,out,name)
    plot_distances(samples,deriv,t,out,name,quadratic)
    print('FIT_REPORT',json.dumps(report),flush=True)
    return report


def plot_fit(lc,y,err,model,out,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    yy=y.reshape(lc.orders,2,-1);ee=err.reshape(lc.orders,2,-1);mm=model.reshape(lc.orders,2,-1)
    labels=[r'$s^2\xi_\ell$',r'$s^2\langle\partial_z\xi_\ell\rangle_{W_1}$',r'$s^2\langle\partial_z^2\xi_\ell\rangle_{W_2}$']
    rows=[]
    for order in range(lc.orders):
        for li,ell in enumerate([0,2]):
            fig,ax=plt.subplots(figsize=(8.4,5.3))
            ax.errorbar(lc.s,yy[order,li],yerr=ee[order,li],fmt='o',capsize=3,label='Measured; 120-region jackknife')
            ax.plot(lc.s,mm[order,li],label='Joint CAMB-based light-cone BAO fit')
            ax.axhline(0,linewidth=.8,linestyle='--')
            ax.set(xlabel=r'$s\ [h^{-1}{\rm Mpc}]$',ylabel=labels[order],title=f'Full DESI DR1 LRG: derivative order {order}, ell={ell}')
            ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/f'{name}_fit_order{order}_ell{ell}.png',dpi=200);fig.savefig(out/f'{name}_fit_order{order}_ell{ell}.svg');plt.close(fig)
            for i,s in enumerate(lc.s):rows.append([order,ell,s,yy[order,li,i],ee[order,li,i],mm[order,li,i]])
    np.savetxt(out/f'{name}_fit_curves.csv',rows,delimiter=',',comments='',header='order,ell,s_Mpc_h,s2_measurement,s2_jackknife_sigma,s2_model')


def plot_distances(samples,deriv,t,out,name,quadratic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z=np.linspace(.4,1.1,151)
    use=samples[::max(1,len(samples)//1500)]
    histories=np.array([distance_history(p,z,float(t['rd']),quadratic) for p in use])
    fid=distance_history(np.zeros(8),z,float(t['rd']),True)
    labels=[r'$D_M/r_d$',r'$D_H/r_d$',r'$d(D_M/r_d)/dz$',r'$d(D_H/r_d)/dz$',r'$d^2(D_M/r_d)/dz^2$',r'$d^2(D_H/r_d)/dz^2$']
    names=['DM','DH','dDM_dz','dDH_dz','d2DM_dz2','d2DH_dz2']
    rows=[]
    for j in range(6 if quadratic else 4):
        q=np.percentile(histories[:,j],[2.5,16,50,84,97.5],axis=0)
        fig,ax=plt.subplots(figsize=(8.4,5.3))
        ax.fill_between(z,q[0],q[4],alpha=.16,label='95% marginal interval')
        ax.fill_between(z,q[1],q[3],alpha=.32,label='68% marginal interval')
        ax.plot(z,q[2],label='NGC+SGC catalogue fit')
        ax.plot(z,fid[j],linestyle='--',label='Fiducial flat LCDM reference')
        ax.axvline(ZP,linestyle=':',linewidth=1)
        ax.set(xlabel='Redshift z',ylabel=labels[j],title=f'Full LRG: {name} redshift model; pre-reconstruction')
        ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/f'{name}_{names[j]}.png',dpi=200);fig.savefig(out/f'{name}_{names[j]}.svg');plt.close(fig)
        for i,zz in enumerate(z):rows.append([j,zz,*q[:,i]])
    np.savetxt(out/f'{name}_distance_bands.csv',rows,delimiter=',',comments='',header='quantity_index,z,p2p5,p16,median,p84,p97p5')
    fig,ax=plt.subplots(figsize=(7,6))
    H,xe,ye=np.histogram2d(deriv[:,2],deriv[:,3],bins=45)
    flat=np.sort(H.ravel())[::-1];cs=np.cumsum(flat)/flat.sum()
    levels=sorted(set(float(flat[min(len(flat)-1,np.searchsorted(cs,p))]) for p in [.95,.68]))
    if len(levels)>1:ax.contour((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2,H.T,levels=levels)
    ax.scatter(np.median(deriv[:,2]),np.median(deriv[:,3]),marker='o',label='Posterior median')
    ref=distance_history(np.zeros(8),np.array([ZP]),float(t['rd']),True)[:,0]
    ax.scatter(ref[2],ref[3],marker='x',label='Fiducial flat LCDM')
    ax.set(xlabel=labels[2],ylabel=labels[3],title=f'Joint derivative constraint, z=0.75 ({name} model)')
    ax.legend();fig.tight_layout();fig.savefig(out/f'{name}_joint_derivatives.png',dpi=200);fig.savefig(out/f'{name}_joint_derivatives.svg');plt.close(fig)
    difference=deriv[:,2]-deriv[:,1]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(difference,bins=55,density=True,histtype='step',label='Joint fit: anchors and slopes both free')
    ax.axvline(0,linestyle='--',label='Flat FLRW identity')
    ax.set(xlabel=r'$d(D_M/r_d)/dz-D_H/r_d$ at $z=0.75$',ylabel='Marginal probability density',title='Model-conditional geometry consistency')
    ax.legend();fig.tight_layout();fig.savefig(out/f'{name}_flat_geometry.png',dpi=200);plt.close(fig)
    q=np.percentile(difference,[2.5,16,50,84,97.5])
    (out/f'{name}_flat_geometry.json').write_text(json.dumps({'quantiles':q.tolist(),'meaning':'Same fitted latent functions and pivot; conditional on finite redshift model, not a model-free matched-kernel null test.'},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/measurements_rebin2.npz');p.add_argument('--output',default='results');p.add_argument('--steps',type=int,default=3000);a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    with np.load(a.input) as f:m={k:f[k] for k in f.files}
    t=prepare_template(out/'camb_bao_template.npz')
    for quadratic in [False,True]:run_fit(m,t,out,quadratic,a.steps)
    (out/'ANALYSIS_SCOPE.md').write_text('Full DR1 v1.5 NGC+SGC data and complete random_0. No object thinning. Pre-reconstruction clustering, exact moment-based first and second redshift derivative estimators, and a joint physical-template BAO fit with free distance anchors. Jackknife covariance, actual RR redshift windows, finite redshift basis, broad Gaussian amplitude priors, and damping priors are used. This is not an official DESI analysis or a mock-validated publication likelihood. See fit summaries for convergence, basis/covariance sensitivity and any prior-boundary effects. Raw clustering measurements are distinct from model-conditional distance derivatives.\n')

if __name__=='__main__':main()
