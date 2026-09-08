#!/usr/bin/env python3
"""Fit actual QSO weighted clustering. No published BAO points or toy data enter.
The catalogue derivatives are window averages; distance/shape curves are
conditional on the stated finite-dimensional redshift model and nuisance priors.
"""
import argparse,json,time,sys,multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter
import emcee
sys.path.insert(0,str(Path('qso_runtime').resolve()))
from bao_model import Lightcone,prepare_template,ZP,DZ,OM,HH
from kaiser_refit import Kaiser,norm_test
from reduce_counts import joint_cov
from fit_bao import summary_parameters,plot_fit
ENGINE=None

def logprob(p):
    val=ENGINE.calc(p)
    return -val if np.isfinite(val) else -np.inf

def grouped_lightcone(lc,factor=5):
    # Preserve each integrated derivative weight. Evaluate the smooth template
    # at the RR-weighted mean of five 0.01-redshift cells. Validate against all
    # 130 original cells at the optimum and posterior diagnostic draws.
    nz=lc.z.shape[-1]
    assert nz%factor==0
    h=lc.weights[0]
    sh=(*h.shape[:-1],nz//factor,factor)
    hsum=h.reshape(sh).sum(-1)
    zw=(h*lc.z).reshape(sh).sum(-1)
    z=lc.z.reshape(sh).mean(-1)
    np.divide(zw,hsum,out=z,where=hsum>0)
    lc.weights=np.ascontiguousarray(lc.weights.reshape(*lc.weights.shape[:-1],nz//factor,factor).sum(-1))
    lc.z=np.ascontiguousarray(z)
    return lc

def distance_arrays(samples,z,rd,quad):
    p=np.atleast_2d(samples);z=np.atleast_1d(z);v=(z-ZP)/DZ
    zg=np.linspace(0,max(2.2,float(np.max(z))),20001)
    yg=299792.458/(100*HH*rd)/np.sqrt(OM*(1+zg)**3+1-OM)
    xg=cumulative_trapezoid(yg,zg,initial=0)
    xf=np.interp(z,zg,xg);yf=np.interp(z,zg,yg);ez=OM*(1+z)**3+1-OM
    L=-1.5*OM*(1+z)**2/ez
    L1=-3*OM*(1+z)/ez+4.5*OM**2*(1+z)**4/ez**2
    gp=(p[:,2,None]+(p[:,4,None]*v if quad else 0))/DZ
    gh=(p[:,3,None]+(p[:,5,None]*v if quad else 0))/DZ
    cp=p[:,4,None]/DZ**2 if quad else 0
    ch=p[:,5,None]/DZ**2 if quad else 0
    ap=np.exp(p[:,0,None]+p[:,2,None]*v+(.5*p[:,4,None]*v*v if quad else 0))
    ah=np.exp(p[:,1,None]+p[:,3,None]*v+(.5*p[:,5,None]*v*v if quad else 0))
    X=ap*xf;Y=ah*yf
    X1=ap*(yf+xf*gp);Y1=Y*(L+gh)
    X2=ap*(yf*L+2*yf*gp+xf*(gp*gp+cp))
    Y2=Y*((L+gh)**2+L1+ch)
    return np.stack([X,Y,X1,Y1,X2,Y2],axis=1)

def shape_arrays(samples,z,rd,quad):
    d=distance_arrays(samples,z,rd,quad);dp=distance_arrays(samples,[ZP],rd,quad)
    Y,Y1,Y2=d[:,1],d[:,3],d[:,5]
    y,yp=dp[:,1,0],dp[:,3,0]
    az=1/(1+np.asarray(z));ap=1/(1+ZP)
    F=az[None,:]**3/Y**2;Fp=ap**3/y**2
    L=Y1/Y;Lp=Y2/Y-L*L
    B=3+2*(1+np.asarray(z))*L
    Bx=-2*(1+np.asarray(z))*L-2*(1+np.asarray(z))**2*Lp
    Fx=F*B;Fxx=F*(B*B+Bx)
    Fxp=Fp*(3+2*(1+ZP)*yp/y)
    with np.errstate(divide='ignore',invalid='ignore'):
        S0=(az/ap)[None,:]**3-3*(F-Fp[:,None])/Fxp[:,None]
        S1=(ap/az)[None,:]**3*Fx/Fxp[:,None]
        S2=-Fxx/(3*Fx)
    return np.stack([S0,S1,S2],axis=1),B,Fxp

def draw_geometric_prior(n,ng,bounds,rng):
    out=[]
    while sum(len(a) for a in out)<n:
        p=np.column_stack([rng.uniform(lo,hi,n) for lo,hi in bounds[:ng]])
        u=np.linspace((.8-ZP)/DZ,(2.1-ZP)/DZ,41)
        l1=p[:,0,None]+p[:,2,None]*u
        l2=p[:,1,None]+p[:,3,None]*u
        if ng==6:l1+=.5*p[:,4,None]*u*u;l2+=.5*p[:,5,None]*u*u
        out.append(p[(np.max(abs(l1),axis=1)<=.4)&(np.max(abs(l2),axis=1)<=.4)])
    return np.concatenate(out)[:n]

def contour_plot(values,labels,out,title,reference=None):
    import matplotlib.pyplot as plt
    lim=np.percentile(values,[.5,99.5],axis=0)
    H,xe,ye=np.histogram2d(values[:,0],values[:,1],bins=55,range=[lim[:,0],lim[:,1]])
    H=gaussian_filter(H,1.0);sort=np.sort(H.ravel())[::-1];cs=np.cumsum(sort)/sort.sum()
    levels=sorted(set(float(sort[min(len(sort)-1,np.searchsorted(cs,f))]) for f in [.95,.68]))
    fig,ax=plt.subplots(figsize=(7.6,6.2))
    if len(levels)>1:ax.contour((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2,H.T,levels=levels)
    ax.plot(*np.median(values,axis=0),'o',label='Posterior median')
    if reference is not None:ax.plot(*reference,'x',markersize=9,label='Fiducial LCDM reference')
    ax.set(xlabel=labels[0],ylabel=labels[1],title=title)
    ax.legend();fig.tight_layout();fig.savefig(out,dpi=180);fig.savefig(Path(out).with_suffix('.svg'));plt.close(fig)

def diagnostics_plots(samples,derived,chain,rd,quad,bounds,out,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(38429)
    z=np.unique(np.r_[np.linspace(.8,2.1,151),ZP]);use=samples[::max(1,len(samples)//4000)]
    hist=distance_arrays(use,z,rd,quad);fid=distance_arrays(np.zeros((1,6)),z,rd,True)[0]
    labels=[r'$D_M/r_d$',r'$D_H/r_d$',r'$d(D_M/r_d)/dz$',r'$d(D_H/r_d)/dz$',r'$d^2(D_M/r_d)/dz^2$',r'$d^2(D_H/r_d)/dz^2$']
    names=['DM','DH','dDM_dz','dDH_dz','d2DM_dz2','d2DH_dz2'];rows=[]
    for j,label in enumerate(labels):
        q=np.percentile(hist[:,j],[2.5,16,50,84,97.5],axis=0)
        fig,ax=plt.subplots(figsize=(8.4,5.3))
        ax.fill_between(z,q[0],q[4],alpha=.15,label='95% pointwise posterior interval')
        ax.fill_between(z,q[1],q[3],alpha=.3,label='68% pointwise posterior interval')
        ax.plot(z,q[2],label='QSO catalogue-conditioned posterior')
        ax.plot(z,fid[j],'--',label='Fiducial flat LCDM')
        ax.axvline(ZP,linestyle=':',linewidth=.8)
        ax.set(xlabel='Redshift z',ylabel=label,title=f'DR1 QSO: {name}; model-conditional inference')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'{name}_{names[j]}.png',dpi=180);fig.savefig(out/f'{name}_{names[j]}.svg');plt.close(fig)
        for i,zz in enumerate(z):rows.append([j,zz,*q[:,i]])
    np.savetxt(out/f'{name}_distance_bands.csv',rows,delimiter=',',header='quantity_index,z,p2p5,p16,median,p84,p97p5',comments='')
    ref=distance_arrays(np.zeros((1,6)),[ZP],rd,True)[0,:,0]
    for ii,jj,key in [(0,1,'anchors'),(2,3,'first_derivatives'),(4,5,'second_derivatives')]:
        contour_plot(derived[:,[ii,jj]],[labels[ii],labels[jj]],out/f'{name}_joint_{key}.png',f'DR1 QSO: 68/95% posterior contours at z={ZP}',ref[[ii,jj]])
    diff=derived[:,2]-derived[:,1]
    fig,ax=plt.subplots(figsize=(8.4,5.0));ax.hist(diff,bins=65,density=True,histtype='step')
    ax.axvline(0,linestyle='--',label='Flat-FLRW identity');ax.legend()
    ax.set(xlabel=r"$d(D_M/r_d)/dz-D_H/r_d$",ylabel='Posterior density',title=f'QSO geometry check at z={ZP}; {name}')
    fig.tight_layout();fig.savefig(out/f'{name}_flat_geometry.png',dpi=180);plt.close(fig)
    # All shape draws retained: no positivity cut on F_x and no tail clipping.
    prior=draw_geometric_prior(10000,6 if quad else 4,bounds,rng)
    shapes,B,Fxp=shape_arrays(use,z,rd,quad)
    ps,PB,PFxp=shape_arrays(prior,z,rd,quad)
    shape_report={'normalization':'S0p and S1p normalized at z_p=1.49, NOT at z=0; S2 has no anchor normalization',
                  'conditional_on':name,'posterior_fraction_Fx_crosses_zero':float(np.mean((B.min(1)<0)&(B.max(1)>0))),
                  'prior_fraction_Fx_crosses_zero':float(np.mean((PB.min(1)<0)&(PB.max(1)>0))),
                  'posterior_fraction_positive_Fx_at_pivot':float(np.mean(Fxp>0)),
                  'warning':'Ratios can have poles. Quantiles do not define Gaussian uncertainties; finite means/variances are not asserted. The pivot pinch S0p=S1p=1 is imposed normalization, not observational precision.'}
    srows=[]
    for j,label in enumerate([r'$S_{0,p}$',r'$S_{1,p}$',r'$S_2$']):
        q=np.nanpercentile(shapes[:,j],[2.5,16,50,84,97.5],axis=0);pq=np.nanpercentile(ps[:,j],[16,50,84],axis=0)
        fig,ax=plt.subplots(figsize=(8.4,5.4))
        ax.fill_between(z,q[0],q[4],alpha=.13,label='95% posterior quantiles')
        ax.fill_between(z,q[1],q[3],alpha=.3,label='68% posterior quantiles')
        ax.plot(z,q[2],label='Posterior median')
        ax.plot(z,pq[0],':',label='Geometrical prior: 16th/84th percentiles')
        ax.plot(z,pq[2],':')
        ax.axhline(-1 if j==2 else 1,linestyle='--',label='LCDM shape reference')
        ax.set_yscale('symlog',linthresh=2)
        ax.set(xlabel='Redshift z',ylabel=label,title=f'QSO shape diagnostic ({name}): ratio-sensitive, model-conditional')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'{name}_S{j}.png',dpi=180);fig.savefig(out/f'{name}_S{j}.svg');plt.close(fig)
        for i,zz in enumerate(z):srows.append([j,zz,*q[:,i],*pq[:,i]])
        ip=int(np.argmin(abs(z-ZP)))
        shape_report[f'S{j}_at_pivot_quantiles']=q[:,ip].tolist()
    np.savetxt(out/f'{name}_shape_bands.csv',srows,delimiter=',',header='shape_index,z,post_p2p5,post_p16,post_median,post_p84,post_p97p5,prior_p16,prior_median,prior_p84',comments='')
    (out/f'{name}_shape_audit.json').write_text(json.dumps(shape_report,indent=2))
    # One trace figure for AP geometry, averaged over walkers; no figure panels.
    fig,ax=plt.subplots(figsize=(8.4,5.3))
    for i,lab in enumerate(['ln alpha_perp','ln alpha_parallel','transverse slope in u','radial slope in u']):
        ax.plot(chain[:,:,i].mean(axis=1),linewidth=.8,label=lab)
    ax.set(xlabel='MCMC step',ylabel='Ensemble-mean parameter',title=f'QSO chain diagnostic: {name}')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'{name}_chain_trace.png',dpi=180);plt.close(fig)
    return shape_report

def main():
    global ENGINE
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='qso_measurements/measurements_rebin2.npz');ap.add_argument('--output',default='qso_fit');ap.add_argument('--model',choices=['linear','quadratic'],default='quadratic');ap.add_argument('--steps',type=int,default=7000);ap.add_argument('--seed',type=int,default=94168);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with np.load(a.input) as ff:m={k:ff[k] for k in ff.files}
    check=norm_test(m);(out/'normalization_test.json').write_text(json.dumps(check,indent=2))
    t=prepare_template(out/'camb_qso_template.npz');quad=a.model=='quadratic';orders=3 if quad else 2;name=a.model
    full_lc=Lightcone(m,t,orders=orders,quadratic=quad)
    lc=grouped_lightcone(Lightcone(m,t,orders=orders,quadratic=quad),5)
    data=lc.select(m['multipoles']);cov=joint_cov(data,int(m['njack_ngc']),int(m['njack_sgc']))
    ENGINE=Kaiser(lc,data[0],cov,.2);ng=ENGINE.ng;rng=np.random.default_rng(a.seed)
    initial=np.r_[np.zeros(ng),3.,8.,np.log(9.),0.,0.,.3,0.]
    ENGINE.calc(initial)
    start=time.time()
    best=None
    for trial in range(4):
        p=initial.copy()
        if trial:p[:ng]+=rng.normal(0,.025,ng)
        method='Powell' if trial in [0,2] else 'L-BFGS-B'
        opts={'maxiter':300,'xtol':3e-5,'ftol':2e-7} if method=='Powell' else {'maxiter':1000,'ftol':1e-9,'maxls':35}
        res=minimize(ENGINE.calc,p,bounds=ENGINE.bounds,method=method,options=opts)
        print('OPTIMIZER',trial,method,float(res.fun),res.success,res.x.tolist(),flush=True)
        if best is None or res.fun<best.fun:best=res
    ndim=len(initial);nw=4*ndim;width=np.r_[np.full(ng,.004),.05,.1,.03,.04,.04,.02,.02]
    chains=[];taus=[];accept=[];samplers=[]
    for ensemble in range(2):
        positions=[]
        while len(positions)<nw:
            p=best.x+rng.normal(size=ndim)*width
            if np.isfinite(ENGINE.calc(p)):positions.append(p)
        with mp.get_context('fork').Pool(4) as pool:
            sampler=emcee.EnsembleSampler(nw,ndim,logprob,pool=pool)
            sampler.random_state=np.random.RandomState(a.seed+100*ensemble).get_state()
            sampler.run_mcmc(np.array(positions),a.steps,progress=True)
        chains.append(sampler.get_chain());taus.append(sampler.get_autocorr_time(tol=0));accept.append(float(np.mean(sampler.acceptance_fraction)));samplers.append(sampler)
        print('ENSEMBLE',ensemble,'tau',taus[-1].tolist(),'accept',accept[-1],flush=True)
    burn=a.steps//3
    ss=[c[burn::5].reshape(-1,ndim) for c in chains];samples=np.concatenate(ss)
    chain=np.concatenate(chains,axis=1)
    logs=np.concatenate([s.get_log_prob(discard=burn,thin=5,flat=True) for s in samplers]);bm=samples[np.argmax(logs)]
    candidate=minimize(ENGINE.calc,bm,bounds=ENGINE.bounds,method='L-BFGS-B',options={'maxiter':1200,'ftol':1e-10,'maxls':40})
    if candidate.fun<best.fun:best=candidate
    derived=distance_arrays(samples,[ZP],float(t['rd']),quad)[:,:,0]
    names=['ln_alpha_perp_p','ln_alpha_parallel_p','dln_alpha_perp_du','dln_alpha_parallel_du']+(['d2ln_alpha_perp_du2','d2ln_alpha_parallel_du2'] if quad else [])+['Sigma_perp_effective','Sigma_parallel_effective','ln_A_p','dln_A_du','d2ln_A_du2','beta_p','dbeta_du']
    dnames=['DM_over_rd','DH_over_rd','dDM_over_rd_dz','dDH_over_rd_dz','d2DM_over_rd_dz2','d2DH_over_rd_dz2']
    np.savez_compressed(out/f'{name}_posterior.npz',samples=samples,derived=derived,chain=chain,logprob=logs,best_fit=best.x,parameter_names=names,derived_names=dnames,rd=t['rd'],covariance_raw=cov,covariance_used=ENGINE.cov)
    value,model,bb=ENGINE.calc(best.x,True);resid=data[0]-model;chi=float(resid@np.linalg.solve(ENGINE.cov,resid))
    errors=np.sqrt(np.diag(cov))
    fine_engine=Kaiser(full_lc,data[0],cov,.2)
    qerr=[]
    for p in np.vstack([best.x,samples[rng.choice(len(samples),min(20,len(samples)),replace=False)]]):
        qerr.append(float(np.max(abs(ENGINE.model(p)-fine_engine.model(p))/errors)))
    robust=[]
    for sh in [.1,.4]:
        e=Kaiser(lc,data[0],cov,sh);r=minimize(e.calc,best.x,bounds=e.bounds,method='L-BFGS-B',options={'maxiter':600,'ftol':1e-8,'maxls':30})
        robust.append({'diagonal_shrinkage':sh,'optimizer_success':bool(r.success),'objective':float(r.fun),'derived_at_pivot':distance_arrays(r.x,[ZP],float(t['rd']),quad)[0,:,0].tolist()})
    dV=(ZP*derived[:,0]**2*derived[:,1])**(1/3)
    dp=(2*derived[:,2]/derived[:,0]+derived[:,3]/derived[:,1])/3
    q=-1-(1+ZP)*derived[:,3]/derived[:,1]
    flat=derived[:,2]-derived[:,1]
    halfshifts=[]
    for c in chains:
        retained=c[burn:];mid=len(retained)//2
        shift=(np.median(retained[:mid].reshape(-1,ndim),axis=0)-np.median(retained[mid:].reshape(-1,ndim),axis=0))/samples.std(0)
        halfshifts.append(shift.tolist())
    ens_shift=(np.median(ss[0],axis=0)-np.median(ss[1],axis=0))/samples.std(0)
    report={'dataset':'actual DESI DR1 LSS v1.5 QSO NGC+SGC, complete selected data and random_0; no thinning',
            'redshift_range':[.8,2.1],'pivot':ZP,'redshift_scale':DZ,'model':name,'derivative_orders':list(range(orders)),
            'data_dimension':len(data[0]),'raw_covariance_rank':int(np.linalg.matrix_rank(cov)),'diagonal_shrinkage':.2,
            'parameter_names':names,'hard_bounds':dict(zip(names,[list(b) for b in ENGINE.bounds])),
            'damping_priors':'effective Sigma_perp=3 +/- 1, Sigma_parallel=8 +/- 3 Mpc/h; includes radial smearing approximately, not a separate redshift-error likelihood',
            'broadband_prior':'Gaussian coefficient sigma=50 on each column normalized with factor20; coefficients analytically marginalized',
            'parameters':summary_parameters(samples,names),'distances_and_derivatives':summary_parameters(derived,dnames),
            'kinematics':summary_parameters(np.column_stack([dV,dp,q,flat]),['DV_over_rd','isotropic_fractional_distance_slope','q_pivot','dDMdz_minus_DH']),
            'best_fit':best.x.tolist(),'chi2_data_at_conditional_best':chi,'nonlinear_parameters':ndim,'broadband_parameters':len(bb),
            'independent_ensembles':2,'walkers_per_ensemble':nw,'steps_per_ensemble':a.steps,'burn_in':burn,'acceptance_fractions':accept,
            'autocorrelation_estimates':[x.tolist() for x in taus],
            'retained_steps_over_max_tau':[float((a.steps-burn)/max(t)) for t in taus],
            'effective_sample_estimate':float(sum(nw*(a.steps-burn)/max(t) for t in taus)),
            'half_chain_shift_in_posterior_sigma':halfshifts,'ensemble_median_difference_in_posterior_sigma':ens_shift.tolist(),
            'boundary_fractions':{n:float(np.mean((samples[:,i]<lo+.02*(hi-lo))|(samples[:,i]>hi-.02*(hi-lo)))) for i,(n,(lo,hi)) in enumerate(zip(names,ENGINE.bounds))},
            'redshift_quadrature_max_error_in_raw_diagonal_sigma':max(qerr),'covariance_sensitivity':robust,
            'seconds':time.time()-start,
            'limitations':['Pre-reconstruction; no mock calibration or reconstruction performed.','Same weighted random0 and analysis choices as stated; not the DESI official QSO likelihood.','Distance and shape derivatives conditional on log-AP redshift polynomial and positive Kaiser angular model.','Finite angular binning, Gaussian covariance and shrinkage, radial smearing approximation and integral constraints need mock validation.','Shape ratios can be singular; S0p,S1p use internal pivot, not present-day normalization.','No claim of new information relative to standard DESI without joint mock cross-covariance.']}
    plot_fit(lc,data[0],errors,model,out,name)
    report['shape_audit']=diagnostics_plots(samples,derived,chain,float(t['rd']),quad,ENGINE.bounds,out,name)
    np.savetxt(out/f'{name}_posterior_summary.csv',[[j,*np.percentile(derived[:,j],[2.5,16,50,84,97.5])] for j in range(6)],delimiter=',',header='quantity_index,p2p5,p16,median,p84,p97p5',comments='')
    (out/f'{name}_summary.json').write_text(json.dumps(report,indent=2))
    print('QSO_INFERENCE_SUMMARY',json.dumps(report),flush=True)

if __name__=='__main__':main()
