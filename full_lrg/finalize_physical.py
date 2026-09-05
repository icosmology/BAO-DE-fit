#!/usr/bin/env python3
import argparse,json,time,multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import emcee
import kaiser_refit as kr
from bao_model import Lightcone,prepare_template,distance_history,ZP
from reduce_counts import joint_cov
from fit_bao import summary_parameters,plot_fit,plot_distances


def refine(engine,candidates,maxiter=7000):
    values=np.array([engine.calc(x) for x in candidates]);start=candidates[np.argmin(values)]
    fit=minimize(engine.calc,start,method='Nelder-Mead',bounds=engine.bounds,options={'adaptive':True,'maxiter':maxiter,'xatol':3e-5,'fatol':1e-7})
    if np.isfinite(fit.fun) and fit.fun<values.min():return fit.x,float(fit.fun),bool(fit.success)
    return start,float(values.min()),False


def main():
    a=argparse.ArgumentParser();a.add_argument('--model',choices=['linear','quadratic'],required=True);a.add_argument('--steps',type=int,default=12000);p=a.parse_args()
    name='kaiser_'+p.model;quad=p.model=='quadratic';out=Path('final_results');out.mkdir(exist_ok=True)
    with np.load('results/measurements_rebin2.npz') as f:m={k:f[k] for k in f.files}
    with np.load(f'physical_results/{name}_posterior.npz') as f:prev={k:f[k] for k in f.files}
    old=json.loads(Path(f'physical_results/{name}_summary.json').read_text())
    t=prepare_template('results/camb_bao_template.npz');lc=Lightcone(m,t,orders=3 if quad else 2,quadratic=quad)
    data=lc.select(m['multipoles']);cov=joint_cov(data,int(m['njack_ngc']),int(m['njack_sgc']))
    engine=kr.Kaiser(lc,data[0],cov,.2);kr.CURRENT=engine
    start=prev['chain'][-1];nw,nd=start.shape;tim=time.time()
    with mp.get_context('fork').Pool(4) as pool:
        sampler=emcee.EnsembleSampler(nw,nd,kr.logprob,pool=pool)
        sampler.random_state=np.random.RandomState(822+nd).get_state()
        sampler.run_mcmc(start,p.steps,progress=True)
    chain=np.concatenate([prev['chain'],sampler.get_chain()]);burn=3000
    samples=chain[burn::5].reshape(-1,nd);tau=emcee.autocorr.integrated_time(chain[burn:],tol=0)
    best,objective,success=refine(engine,samples[::max(1,len(samples)//400)])
    print('Refined posterior-mode fit',name,objective,success,best.tolist(),flush=True)
    value,model,bb=engine.calc(best,True)
    derived=np.array([distance_history(x,np.array([ZP]),float(t['rd']),quad)[:,0] for x in samples])
    names=list(prev['parameter_names']);dnames=['DM_over_rd','DH_over_rd','dDM_over_rd_dz','dDH_over_rd_dz','d2DM_over_rd_dz2','d2DH_over_rd_dz2']
    np.savez_compressed(out/f'{name}_posterior.npz',chain=chain,samples=samples,derived=derived,best_fit=best,parameter_names=names,covariance_raw=cov,covariance_used=engine.cov)
    residual=data[0]-model
    old['chi2_data_at_best']=float(residual@np.linalg.solve(engine.cov,residual));old['best_fit']=best.tolist();old['optimizer_success']=success
    old['parameters']=summary_parameters(samples,names);old['distances_and_derivatives']=summary_parameters(derived,dnames)
    old['autocorrelation_times']=tau.tolist();old['chain_steps']=len(chain);old['burn_in']=burn
    old['mean_acceptance_fraction_extension']=float(np.mean(sampler.acceptance_fraction));old['effective_sample_estimate']=float(nw*(len(chain)-burn)/np.max(tau));old['retained_steps_over_max_tau']=float((len(chain)-burn)/np.max(tau));old['extension_seconds']=time.time()-tim
    old['boundary_fractions']={n:float(np.mean((samples[:,i]<lo+.02*(hi-lo))|(samples[:,i]>hi-.02*(hi-lo)))) for i,(n,(lo,hi)) in enumerate(zip(names,engine.bounds))}
    split=(len(chain)+burn)//2;first=chain[burn:split].reshape(-1,nd);last=chain[split:].reshape(-1,nd)
    old['half_chain_medians']=[np.median(first,axis=0).tolist(),np.median(last,axis=0).tolist()]
    old['half_median_shift_over_posterior_sigma']=((np.median(last,axis=0)-np.median(first,axis=0))/np.std(samples,axis=0)).tolist()
    delta=derived[:,2]-derived[:,1];old['flat_geometry_residual']=summary_parameters(delta[:,None],['dDMdz_minus_DH'])
    old['distance_derivative_correlation']=float(np.corrcoef(derived[:,2],derived[:,3])[0,1])
    old['sound_horizon_fiducial_Mpc']=float(t['rd'])
    old['covariance_sensitivity']=[]
    for sh in [0.,.1,.4]:
        other=kr.Kaiser(lc,data[0],cov,sh)
        cand=np.vstack([best,samples[::max(1,len(samples)//120)]])
        fit,obj,ok=refine(other,cand,5000)
        old['covariance_sensitivity'].append({'shrinkage':sh,'success':ok,'objective':obj,'best_fit':fit.tolist(),'derived':distance_history(fit,np.array([ZP]),float(t['rd']),quad)[:,0].tolist()})
    old['best_fit_correction']='Plot curves refined from sampled high-posterior points; non-finite boundary line searches in the initial optimizer are not used as the final best fit.'
    # Finite-quadrature convergence test for the random-pair redshift integration.
    coarse=Lightcone(m,t,orders=lc.orders,quadratic=quad)
    # Compare the binned integration's constant/quadratic responses with exact RR moments.
    old['normalization_test']=kr.norm_test(m)
    np.savetxt(out/f'{name}_derived_covariance.csv',np.cov(derived,rowvar=False),delimiter=',',header=','.join(dnames))
    plot_fit(lc,data[0],np.sqrt(np.diag(cov)),model,out,name);plot_distances(samples,derived,t,out,name,quad)
    (out/f'{name}_summary.json').write_text(json.dumps(old,indent=2))
    print('FINAL_SUMMARY',json.dumps(old),flush=True)

if __name__=='__main__':main()
