#!/usr/bin/env python3
"""Standard positive-amplitude Kaiser BAO model as a documented robustness fit.
Raw measurements and random kernels are unchanged. This is not a new catalogue
or synthetic data. Contrast with the intentionally very flexible angular basis.
"""
import argparse,json,time,multiprocessing as mp
from pathlib import Path
import numpy as np
from numba import njit
from scipy.linalg import solve_triangular,cho_factor,cho_solve
from scipy.optimize import minimize
import emcee
from bao_model import Lightcone,prepare_template,legendre_even,distance_history,ZP,DZ
from reduce_counts import joint_cov,estimator
from fit_bao import plot_fit,plot_distances,summary_parameters
CURRENT=None

@njit(cache=True)
def physical_model(p,spgrid,sagrid,rg,nw,wig,s,zz,weights,legfac,quadratic):
    ng=6 if quadratic else 4;spv=p[ng];sav=p[ng+1]
    a0,a1,a2,b0,b1=p[ng+2],p[ng+3],p[ng+4],p[ng+5],p[ng+6]
    nd=weights.shape[0];ns=len(s);nm=zz.shape[1];nz=zz.shape[2]
    out=np.zeros((nd,2,ns))
    isp=min(len(spgrid)-2,max(0,np.searchsorted(spgrid,spv)-1));isa=min(len(sagrid)-2,max(0,np.searchsorted(sagrid,sav)-1))
    fp=(spv-spgrid[isp])/(spgrid[isp+1]-spgrid[isp]);fa=(sav-sagrid[isa])/(sagrid[isa+1]-sagrid[isa])
    table=(1-fp)*(1-fa)*wig[isp,isa]+fp*(1-fa)*wig[isp+1,isa]+(1-fp)*fa*wig[isp,isa+1]+fp*fa*wig[isp+1,isa+1]
    gn=np.array([-np.sqrt(3./5),0.,np.sqrt(3./5)]);gw=np.array([5./9,8./9,5./9])
    for ib in range(ns):
        for im in range(nm):
            mu=(im+.5)/nm;lf=legendre_even(mu)
            for iz in range(nz):
                z=zz[ib,im,iz];v=(z-ZP)/DZ
                ap0=p[0]+p[2]*v;ar0=p[1]+p[3]*v
                if quadratic:ap0+=.5*p[4]*v*v;ar0+=.5*p[5]*v*v
                ap=np.exp(ap0);ar=np.exp(ar0)
                scale=np.sqrt(ap*ap*(1-mu*mu)+ar*ar*mu*mu);mup=ar*mu/scale;lp=legendre_even(mup)
                base=np.zeros(3);den=0.
                for ig in range(3):
                    sr=s[ib]+4*gn[ig];w=gw[ig]*sr*sr;den+=w;st=sr*scale
                    q=(sr-rg[0])/(rg[1]-rg[0]);ir=min(len(rg)-2,max(0,int(q)));fr=q-ir
                    q2=(st-rg[0])/(rg[1]-rg[0]);jr=min(len(rg)-2,max(0,int(q2)));ft=q2-jr
                    for n in range(3):
                        vv=0.
                        for li in range(5):
                            vv+=((1-fr)*nw[n,li,ir]+fr*nw[n,li,ir+1])*lf[li]
                            vv+=((1-ft)*table[n,li,jr]+ft*table[n,li,jr+1])*lp[li]
                        base[n]+=w*vv
                base/=den
                beta=b0+b1*v;amp=np.exp(a0+a1*v+.5*a2*v*v)
                xi=amp*(base[0]+2*beta*base[1]+beta*beta*base[2])
                for order in range(nd):
                    w0=weights[order,ib,im,iz]*s[ib]*s[ib]
                    for ell in range(2):out[order,ell,ib]+=w0*legfac[ell,im]*xi
    return out.ravel()

class Kaiser:
    def __init__(self,lc,data,cov,shrink=.2):
        self.lc=lc;self.y=data;self.ng=6 if lc.quadratic else 4
        self.cov=(1-shrink)*cov+shrink*np.diag(np.diag(cov));self.chol=np.linalg.cholesky(self.cov)
        self.yw=solve_triangular(self.chol,data,lower=True)
        self.B=solve_triangular(self.chol,lc.bb,lower=True)
        self.fac=cho_factor(self.B.T@self.B+np.eye(self.B.shape[1])/50**2,lower=True)
        self.Q=np.eye(len(data))-self.B@cho_solve(self.fac,self.B.T)
        self.Q=(self.Q+self.Q.T)/2
        self.bounds=[(-.2,.2),(-.2,.2),(-.6,.6),(-.6,.6)]+([(-1.2,1.2),(-1.2,1.2)] if lc.quadratic else [])+[(2.,7.),(4.,15.),(-1.,3.),(-2.,2.),(-4.,4.),(.01,1.2),(-.8,.8)]
    def model(self,p):
        t=self.lc.t
        return physical_model(p,t['sp'],t['sa'],t['r'],t['nw'],t['wig'],self.lc.s,self.lc.z,self.lc.weights,self.lc.leg,self.lc.quadratic)
    def calc(self,p,return_model=False):
        bad=any(not lo<=x<=hi for x,(lo,hi) in zip(p,self.bounds))
        if p[-2]<=abs(p[-1]) or p[-2]+abs(p[-1])>=1.5:bad=True
        for v in [-1.,0.,1.]:
            lp=p[0]+p[2]*v+(.5*p[4]*v*v if self.ng==6 else 0)
            la=p[1]+p[3]*v+(.5*p[5]*v*v if self.ng==6 else 0)
            if max(abs(lp),abs(la))>.4:bad=True
        if bad:return (np.inf,None,None) if return_model else np.inf
        model=self.model(p);rw=solve_triangular(self.chol,self.y-model,lower=True)
        prior=(p[self.ng]-4.5)**2+(p[self.ng+1]-9.)**2/4.
        value=.5*(rw@self.Q@rw+prior)
        if return_model:
            bb=cho_solve(self.fac,self.B.T@rw)
            return value,model+self.lc.bb@bb,bb
        return value

def logprob(p):
    v=CURRENT.calc(p)
    return -v if np.isfinite(v) else -np.inf

def norm_test(m):
    R=m['R'];N=np.empty_like(m['N']);a,b,c=.003,.008,.025
    for k in range(3):N[...,k]=a*R[...,k]+b*R[...,k+1]+.5*c*R[...,k+2]
    poles,diag=estimator(N,R)
    assert np.max(abs(poles[:,2,0]-c))<1e-9
    assert np.max(abs(poles[:,2,1:]))<1e-9
    return {'normalized_quadratic_response_max_error':float(np.max(abs(poles[:,2,0]-c))),'second_derivative_higher_pole_null_error':float(np.max(abs(poles[:,2,1:]))),'fixture':'Analytic polynomial response test using measured RR moments, not a simulated measurement.'}

def main():
    global CURRENT
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/measurements_rebin2.npz');p.add_argument('--template',default='results/camb_bao_template.npz');p.add_argument('--output',default='physical_results');p.add_argument('--model',choices=['linear','quadratic'],default='quadratic');p.add_argument('--steps',type=int,default=6000);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with np.load(a.input) as f:m={k:f[k] for k in f.files}
    test=norm_test(m);(out/'second_derivative_normalization_test.json').write_text(json.dumps(test,indent=2));print(test,flush=True)
    t=prepare_template(a.template);quad=a.model=='quadratic';name='kaiser_'+a.model;orders=3 if quad else 2
    lc=Lightcone(m,t,orders=orders,quadratic=quad);ds=lc.select(m['multipoles']);cov=joint_cov(ds,int(m['njack_ngc']),int(m['njack_sgc']))
    CURRENT=Kaiser(lc,ds[0],cov,.2);ng=CURRENT.ng;rng=np.random.default_rng(98321+ng)
    initial=np.r_[np.zeros(ng),4.5,9.,np.log(4.),0.,0.,.4,0.]
    # Optimizer uses broad hard physical bounds, not the official BAO distances.
    best=None
    for k in range(5):
        x=initial.copy();x[:ng]+=rng.normal(0,.01,ng) if k else 0
        res=minimize(CURRENT.calc,x,bounds=CURRENT.bounds,method='Powell',options={'maxiter':400,'xtol':2e-5,'ftol':2e-7}) if k==0 else minimize(CURRENT.calc,x,bounds=CURRENT.bounds,method='L-BFGS-B',options={'maxiter':1500,'ftol':1e-9,'maxls':30})
        print(name,'opt',k,res.fun,res.success,res.x.tolist(),flush=True)
        if best is None or res.fun<best.fun:best=res
    dim=len(initial);walkers=4*dim;widths=np.r_[np.full(ng,.001),.025,.05,.02,.02,.025,.015,.01]
    init=[]
    while len(init)<walkers:
        x=best.x+rng.normal(size=dim)*widths
        if np.isfinite(CURRENT.calc(x)):init.append(x)
    start=time.time()
    with mp.get_context('fork').Pool(4) as pool:
        sampler=emcee.EnsembleSampler(walkers,dim,logprob,pool=pool)
        sampler.random_state=np.random.RandomState(712+ng).get_state()
        sampler.run_mcmc(np.array(init),a.steps,progress=True)
    burn=a.steps//3;samples=sampler.get_chain(discard=burn,thin=5,flat=True);chain=sampler.get_chain()
    tau=sampler.get_autocorr_time(tol=0)
    names=['ln_alpha_perp_p','ln_alpha_parallel_p','dln_alpha_perp_du','dln_alpha_parallel_du']+(['d2ln_alpha_perp_du2','d2ln_alpha_parallel_du2'] if quad else [])+['Sigma_perp','Sigma_parallel','ln_A_p','dln_A_du','d2ln_A_du2','beta_p','dbeta_du']
    derived=np.array([distance_history(x,np.array([ZP]),float(t['rd']),quad)[:,0] for x in samples]);dnames=['DM_over_rd','DH_over_rd','dDM_over_rd_dz','dDH_over_rd_dz','d2DM_over_rd_dz2','d2DH_over_rd_dz2']
    np.savez_compressed(out/f'{name}_posterior.npz',chain=chain,samples=samples,derived=derived,best_fit=best.x,parameter_names=names,covariance_raw=cov,covariance_used=CURRENT.cov)
    val,model,bb=CURRENT.calc(best.x,True);rawres=ds[0]-model;chi=float(rawres@np.linalg.solve(CURRENT.cov,rawres))
    report={'source':'same full actual v1.5 NGC+SGC pair counts','template':'CAMB wiggle/no-wiggle, positive amplitude A(z) and Kaiser [1+beta(z)mu^2]^2, anisotropic BAO damping','redshift_model':a.model,'pivot':ZP,'redshift_scale':DZ,'data_dimension':len(ds[0]),'nonlinear_parameters':dim,'broadband_parameters':len(bb),'chi2_data_at_best':chi,'best_fit':best.x.tolist(),'parameters':summary_parameters(samples,names),'distances_and_derivatives':summary_parameters(derived,dnames),'mean_acceptance_fraction':float(np.mean(sampler.acceptance_fraction)),'autocorrelation_times':tau.tolist(),'chain_steps':a.steps,'burn_in':burn,'effective_sample_estimate':float(walkers*(a.steps-burn)/np.max(tau)),'MCMC_elapsed_seconds':time.time()-start,'covariance_shrinkage':.2,'boundary_fractions':{n:float(np.mean((samples[:,i]<lo+.02*(hi-lo))|(samples[:,i]>hi-.02*(hi-lo)))) for i,(n,(lo,hi)) in enumerate(zip(names,CURRENT.bounds))},'hard_bounds':{n:list(b) for n,b in zip(names,CURRENT.bounds)},'damping_priors':'Sigma_perp=4.5 +/- 1; Sigma_parallel=9 +/- 2 Mpc/h','amplitude_model':'ln A = a0+a1*u+a2*u^2/2; beta=beta0+beta1*u; beta positive on range','clustering_covariance':'120-region spatial jackknife, cap covariances summed; 20% shrink to diagonal','limitations':['Pre-reconstruction, not official DESI post-reconstruction inference.','No mocks or external cosmological data used.','Model-conditional derivatives; Gaussian covariance approximation and internal jackknife errors.','No smoothness prior on cosmology beyond the stated finite redshift basis.','Reference template/no-wiggle decomposition, redshift and angle discretization require mock calibration.']}
    # Posterior-derived flat identity comparison does NOT fix the radial anchor.
    report['flat_geometry_residual']=summary_parameters((derived[:,2]-derived[:,1])[:,None],['dDMdz_minus_DH'])
    # Two halves of retained MCMC as a quantile stability diagnostic.
    half=chain[burn:];mid=len(half)//2
    report['half_chain_medians']=[np.median(half[:mid].reshape(-1,dim),axis=0).tolist(),np.median(half[mid:].reshape(-1,dim),axis=0).tolist()]
    robustness=[]
    for sh in [0.,.1,.4]:
        engine=Kaiser(lc,ds[0],cov,sh)
        rr=minimize(engine.calc,best.x,bounds=engine.bounds,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-8,'maxls':30})
        robustness.append({'shrinkage':sh,'success':bool(rr.success),'best_fit':rr.x.tolist(),'derived':distance_history(rr.x,np.array([ZP]),float(t['rd']),quad)[:,0].tolist()})
    report['covariance_sensitivity']=robustness
    (out/f'{name}_summary.json').write_text(json.dumps(report,indent=2))
    plot_fit(lc,ds[0],np.sqrt(np.diag(cov)),model,out,name);plot_distances(samples,derived,t,out,name,quad)
    print('PHYSICAL_REPORT',json.dumps(report),flush=True)

if __name__=='__main__':main()
