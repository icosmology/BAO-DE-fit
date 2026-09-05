#!/usr/bin/env python3
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
from reduce import reduce,jk_cov
from fit import make_template,Fit,distances


def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=190);fig.savefig(out/(name+'.svg'));plt.close(fig)

def derivative_quantities(theta,reps,rd,quad,z):
    central=distances(theta,z,rd,quad)
    values=np.array([distances(t,z,rd,quad) for t in reps]);dd=values-values.mean(axis=0)
    err=np.sqrt((len(reps)-1)/len(reps)*np.sum(dd*dd,axis=0))
    return central,err,values

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='counts');p.add_argument('--output',default='results_v15');args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    s,mu,z,poles,cov,err,kernel,leg,audit=reduce(args.input,out)
    ndata=sum(audit['catalogues'][c]['data']['selected_rows'] for c in ['NGC','SGC']);nrandom=sum(audit['catalogues'][c]['random']['selected_rows'] for c in ['NGC','SGC'])
    names=['xi','dxi_dz','d2xi_dz2'];labels=[r'\xi',r'\partial_z\xi',r'\partial_z^2\xi']
    for a,name in enumerate(names):
        for j,ell in enumerate([0,2,4]):
            fig,ax=plt.subplots(figsize=(8.5,5.1));ax.errorbar(s,s*s*poles[0,a,j],yerr=s*s*err[a,j],fmt='o',ms=4,capsize=2,label='NGC + SGC; all galaxies and random_0')
            ax.axhline(0,ls='--',lw=.8);ax.set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$');ax.set_ylabel('$s^2 '+labels[a]+'_{'+str(ell)+'}$')
            ax.set_title('DESI DR1 v1.5 LRG — direct catalogue measurement\n'+('ordinary clustering' if a==0 else ('first redshift derivative' if a==1 else 'unit-normalized second redshift derivative'))+'; 120-region jackknife')
            ax.legend(fontsize=9);save(fig,out,name+f'_ell{ell}')
    # Diagnostic kernels from exact-moment-corrected random histograms.
    isel=int(np.argmin(abs(s-100)));k=kernel[0,:,isel].mean(axis=1);zg=np.linspace(.4,1.1,1401)
    ca=np.array([np.sum(k[a][None,:]*(z[None,:]<=zg[:,None]),axis=1) for a in range(3)])
    w1=-ca[1];w2=np.sum(k[2][None,:]*np.maximum(zg[:,None]-z[None,:],0),axis=1)
    z1=audit['kernel_diagnostics']['first_derivative_z'];z2=audit['kernel_diagnostics']['second_derivative_z']
    fig,ax=plt.subplots(figsize=(8.5,5.1));ax.plot(z,k[0]/.01,label='ordinary RR redshift density');ax.plot(zg,w1,label=f'first derivative kernel, mean z={z1:.3f}');ax.plot(zg,w2,label=f'second derivative kernel, mean z={z2:.3f}')
    ax.set_xlabel('Pair-midpoint redshift');ax.set_ylabel('Normalized kernel');ax.set_title('Measured redshift resolution near the BAO scale');ax.legend(fontsize=9);save(fig,out,'redshift_kernels')
    np.savetxt(out/'redshift_kernels.csv',np.column_stack([zg,w1,w2]),delimiter=',',header='z,W_first,W_second',comments='')
    print('CLUSTERING_PLOTS_READY',flush=True)
    template=make_template(out);rd=template[-1]['rdrag_Mpc']
    primary=Fit(s,mu,z,poles,cov,kernel,leg,template,quadratic=False,shrink=.5)
    result=primary.run(out,'linear_AP_evolution')
    quadratic=Fit(s,mu,z,poles,cov,kernel,leg,template,quadratic=True,shrink=.5)
    quadratic.initial[:4]=result[0][:4];quadratic.initial[-4:]=result[0][-4:]
    result2=quadratic.run(out,'quadratic_AP_evolution')
    report={'status':'full_catalogue_measurement_and_diagnostic_BAO_fits_completed','input_version':'DESI DR1 v1.5','data_used':ndata,'randoms_used':nrandom,'data_fraction':1.0,'random_fraction':1.0,'njack':120,'pivot_redshift':.75,'rdrag_fid_Mpc':rd,'kernel_diagnostics':audit['kernel_diagnostics'],'model_note':'Distance inference is conditional on the explicitly specified smooth AP evolution model; clustering derivative measurements do not require this model.','limitations':['Pre-reconstruction only; not the official reconstructed DESI pipeline.','One complete random realization per sky cap, not all official random realizations.','Spatial jackknife covariance, not a mock-validated covariance.','Full correlation matrix regularized by 50% diagonal shrinkage; parameter errors from refitting all jackknife deletions.','Fiducial CAMB acoustic template and finite AP expansion; damping ratio fixed to 1.6.','Actual (s,mu,z) RR kernel used; no separately calibrated fiber-assignment/RIC/AMR systematic correction.','Pivot-normalized shape ratios can be unstable and are diagnostic, not independent measurements.']}
    ref=np.zeros_like(result2[0]);ref[-4:]=[.4,0,7,0]
    quantities=['DM_over_rd','DH_over_rd','dDM_over_rd_dz','dDH_over_rd_dz','d2DM_over_rd_dz2','d2DH_over_rd_dz2']
    for label,fit,rr,quad in [('linear_AP_evolution',primary,result,False),('quadratic_AP_evolution',quadratic,result2,True)]:
        theta,reps,cp,model,meta=rr;c,e,vals=derivative_quantities(theta,reps,rd,quad,np.array([.75]));d={quantities[k]:{'value':float(c[k,0]),'jackknife_sigma':float(e[k,0])} for k in range(6)}
        delta=vals[:,2,0]-vals[:,1,0];d['flat_FLRW_Xprime_minus_Y']={'value':float(c[2,0]-c[1,0]),'jackknife_sigma':float(np.sqrt((len(delta)-1)/len(delta)*np.sum((delta-delta.mean())**2)))}
        d['jackknife_convergence']=meta['jackknife_successes'];d['boundary_hits']=meta['jackknife_boundary_hits_per_parameter'];d['parameter_names']=fit.names
        d['all_AP_parameters']=theta.tolist();d['geometry_covariance']=cp[:(6 if quad else 4),:(6 if quad else 4)].tolist();report[label]=d
        sf=fit.s;data=fit.obs[0].reshape(3,2,-1);model=model.reshape(3,2,-1);ef=fit.std.reshape(3,2,-1)
        for a in range(3):
            fig,ax=plt.subplots(figsize=(8.5,5.1));ax.errorbar(sf,sf*sf*data[a,0],yerr=sf*sf*ef[a,0],fmt='o',ms=4,capsize=2,label='full NGC+SGC data')
            ax.plot(sf,sf*sf*model[a,0],label='CAMB light-cone BAO model + profiled broadband')
            ax.axhline(0,ls=':',lw=.8);ax.set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$');ax.set_ylabel('$s^2 '+labels[a]+'_0$');ax.set_title(label.replace('_',' ')+' — '+names[a]);ax.legend(fontsize=8);save(fig,out,label+'_'+names[a]+'_fit')
    # Explicit covariance / fit-range stability checks, holding the same data pipeline.
    stability=[]
    for shrink,lo,hi in [(.2,60,150),(.8,60,150),(.5,66,144)]:
        f=Fit(s,mu,z,poles,cov,kernel,leg,template,quadratic=False,shrink=shrink,slo=lo,shi=hi);r=f.fitone(0,result[0],350);v=distances(r.x,np.array([.75]),rd,False)
        stability.append({'shrinkage':shrink,'smin':lo,'smax':hi,'success':bool(r.success),'dDM_over_rd_dz':float(v[2,0]),'dDH_over_rd_dz':float(v[3,0]),'theta':r.x.tolist(),'objective':float(r.fun@r.fun)})
    report['stability_checks']=stability
    # Curves are model-conditional, with errors propagated by full JK refits.
    zplot=np.linspace(.4,1.1,181);cent,errors,vs=derivative_quantities(result2[0],result2[1],rd,True,zplot);fid=distances(ref,zplot,rd,True)
    ylabels=[r'$D_M/r_d$',r'$D_H/r_d$',r'$d(D_M/r_d)/dz$',r'$d(D_H/r_d)/dz$',r'$d^2(D_M/r_d)/dz^2$',r'$d^2(D_H/r_d)/dz^2$']
    for j in [0,1,2,3,4,5]:
        fig,ax=plt.subplots(figsize=(8.5,5.1));ax.fill_between(zplot,cent[j]-errors[j],cent[j]+errors[j],alpha=.25,label='1-sigma jackknife propagation')
        ax.plot(zplot,cent[j],label='quadratic AP fit; anchors free');ax.plot(zplot,fid[j],ls='--',label='fiducial reference');ax.set_xlabel('Redshift');ax.set_ylabel(ylabels[j]);ax.set_title('Full-catalogue BAO fit — model-conditional distance evolution');ax.legend(fontsize=9);save(fig,out,quantities[j])
    for q,label in [(2,'dDM'),(3,'dDH')]:
        fig,ax=plt.subplots(figsize=(8.5,4.4))
        for row,key in enumerate(['linear_AP_evolution','quadratic_AP_evolution']):
            v=report[key][quantities[q]];ax.errorbar(v['value'],row,xerr=v['jackknife_sigma'],fmt='o',capsize=4,label=key.replace('_',' '))
        val=distances(ref,np.array([.75]),rd,True)[q,0];ax.axvline(val,ls='--',label='fiducial reference');ax.set_yticks([0,1],['linear log-AP','quadratic log-AP']);ax.set_xlabel(ylabels[q]);ax.set_title('Derivative robustness at z=0.75; all distance anchors fitted');ax.legend(fontsize=8);save(fig,out,label+'_robustness')
    # Pair of first-distance derivatives: Gaussian contour from parameter JK fits.
    vv=np.array([distances(t,np.array([.75]),rd,True)[[2,3],0] for t in result2[1]])
    cm=(len(vv)-1)/len(vv)*((vv-vv.mean(axis=0)).T@(vv-vv.mean(axis=0)));mean=distances(result2[0],np.array([.75]),rd,True)[[2,3],0];ev,ec=np.linalg.eigh(cm);phi=np.linspace(0,2*np.pi,300);circle=np.array([np.cos(phi),np.sin(phi)])
    fig,ax=plt.subplots(figsize=(7.3,6.0))
    for prob,ls in [(.68,'-'),(.95,'--')]:
        points=mean[:,None]+ec@np.diag(np.sqrt(np.maximum(ev,0)))@(circle*np.sqrt(chi2.ppf(prob,2)));ax.plot(*points,ls=ls,label=f'{int(prob*100)}% Gaussian JK ellipse')
    ax.plot(*mean,marker='o',ls='none',label='quadratic AP fit');ax.plot(*distances(ref,np.array([.75]),rd,True)[[2,3],0],marker='x',ls='none',label='fiducial reference')
    ax.set_xlabel(ylabels[2]);ax.set_ylabel(ylabels[3]);ax.set_title('Full LRG differential BAO — conditional joint estimate');ax.legend(fontsize=8);save(fig,out,'joint_derivative_constraint')
    # Shape diagnostics: use F and its derivatives from the fitted Y history.
    def shapes(t):
        d=distances(t,zplot,rd,True);Y,Yp,Ypp=d[1],d[3],d[5];a=1/(1+zplot);F=a**3/Y**2;eta=Yp/Y;etap=Ypp/Y-eta**2;uF=3+2*(1+zplot)*eta;Fx=F*uF;Fxx=F*(uF*uF-2*(1+zplot)*(eta+(1+zplot)*etap))
        dp=distances(t,np.array([.75]),rd,True);ap=1/1.75;Fp=ap**3/dp[1,0]**2;Fxp=Fp*(3+2*1.75*dp[3,0]/dp[1,0])
        return np.array([-Fxx/(3*Fx),(ap/a)**3*Fx/Fxp,(a/ap)**3-3*(F-Fp)/Fxp]),Fx,Fxp
    sh,Fx,Fxp=shapes(result2[0]);J=np.zeros((3,len(zplot),6));JF=np.zeros((len(zplot),6));Jp=np.zeros(6)
    for j in range(6):
        eps=1e-5;tp=result2[0].copy();tm=tp.copy();tp[j]+=eps;tm[j]-=eps;sp,fp,fpp=shapes(tp);sm,fm,fpm=shapes(tm);J[:,:,j]=(sp-sm)/(2*eps);JF[:,j]=(fp-fm)/(2*eps);Jp[j]=(fpp-fpm)/(2*eps)
    gc=result2[2][:6,:6];sherr=np.sqrt(np.maximum(0,np.einsum('azj,jk,azk->az',J,gc,J)));Ferr=np.sqrt(np.maximum(0,np.einsum('zj,jk,zk->z',JF,gc,JF)));Fperr=float(np.sqrt(Jp@gc@Jp));stable2=np.abs(Fx)>2*Ferr;stable01=abs(Fxp)>2*Fperr
    report['shape_diagnostics']={'present_day_normalization_measured':False,'pivot_redshift':.75,'S1_S0_pivot_denominator_significant_at_2sigma':bool(stable01),'fraction_of_redshift_grid_with_S2_denominator_significant_at_2sigma':float(stable2.mean()),'interpretation':'Only formal delta-method, quadratic-log-AP-model conditional diagnostics. A small denominator or model/boundary dependence invalidates a Gaussian shape-constraint interpretation.'}
    for j,name in enumerate(['S2','S1_pivot','S0_pivot']):
        stable=bool(stable2.mean()>.9) if j==0 else stable01
        fig,ax=plt.subplots(figsize=(8.5,5.1));ax.plot(zplot,sh[j],label='formal quadratic-AP reconstruction');ax.fill_between(zplot,sh[j]-sherr[j],sh[j]+sherr[j],alpha=.25,label='formal local error propagation')
        ax.axhline(-1 if j==0 else 1,ls='--',label='flat LCDM');ax.set_xlabel('Redshift');ax.set_ylabel(name.replace('_',' '));ax.set_title(('MODEL-CONDITIONAL DIAGNOSTIC' if stable else 'NOT A RELIABLE GAUSSIAN CONSTRAINT: denominator poorly determined')+'\n'+name.replace('_',' '));ax.legend(fontsize=8);save(fig,out,name+'_diagnostic')
    allcols=[zplot]
    header=['z']
    for j,n in enumerate(quantities):header.extend([n,n+'_jk_sigma']);allcols.extend([cent[j],errors[j]])
    for j,n in enumerate(['S2','S1_pivot','S0_pivot']):header.extend([n,n+'_formal_sigma']);allcols.extend([sh[j],sherr[j]])
    np.savetxt(out/'distance_and_shape_diagnostics.csv',np.column_stack(allcols),delimiter=',',header=','.join(header),comments='')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    lines=['# Full DESI DR1 v1.5 LRG catalogue analysis','',f'All {ndata:,} selected galaxies and all {nrandom:,} selected random_0 points, NGC+SGC. No thinning.','', '## Completed measurements','Ordinary xi, normalized first and second redshift derivatives, all 120 spatial leave-one-out realizations, full cross-covariance, exact random moments through sixth order, and redshift-resolved random kernels.','', '## BAO inference','Both transverse and radial AP anchors are free. CAMB acoustic template; independent smooth log-AP evolution. Linear and quadratic evolution fits are provided, not silently identified as model-free point derivatives. Errors are from repeated fits to all jackknife deletions.','', '## Limitations']+['- '+t for t in report['limitations']]+['','## Numerical results','```json',json.dumps(report,indent=2),'```']
    (out/'README.md').write_text('\n'.join(lines))
    print('ANALYSIS_COMPLETED',json.dumps(report),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
