#!/usr/bin/env python3
"""Robustness rerun: endpoint-positive damping and RSD evolution, broad limits.
Uses saved actual-catalogue pair statistics, not new simulated data.
"""
from pathlib import Path
import json,argparse,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import spherical_jn,eval_legendre
from scipy.stats import chi2
from fit import Fit,evaluate_table,distances

class WideFit(Fit):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.initial[-4:]=[.4,.4,7.,7.]
        self.lower[-4:]=[.001,.001,0.,0.]
        self.upper[-4:]=[3.,3.,30.,30.]
        self.names[-4:]=['beta_at_z04','beta_at_z11','Sigma_at_z04','Sigma_at_z11']
    def design(self,t,ir):
        a0,h0,a1,h1=t[:4];a2,h2=t[4:6] if self.quadratic else (0.,0.)
        betaL,betaH,sigL,sigH=t[-4:];v=(self.z-.4)/.7
        beta=np.exp((1-v)*np.log(betaL)+v*np.log(betaH));sigma=(1-v)*sigL+v*sigH
        ap=np.exp(a0+a1*self.u+.5*a2*self.u**2);al=np.exp(h0+h1*self.u+.5*h2*self.u**2)
        field=evaluate_table(self.s,self.mu,ap,al,beta,sigma,self.table,self.sg,self.rg[0],self.rg[1]-self.rg[0])
        amp=np.column_stack([self.project(field*self.up[None,None,:]**j,ir) for j in range(3)])
        return np.column_stack([amp,self.B[ir]])

def wide_template(root,out):
    old=np.load(root/'physical_template.npz');meta=json.loads((root/'template_manifest.json').read_text())
    k=old['k'];P=old['p_linear'];NW=old['p_nowiggle'];rg=old['r'];sg=np.arange(0.,31.,1.);m,w=np.polynomial.legendre.leggauss(64)
    dk=k[1]-k[0];kw=np.ones(len(k))*dk;kw[[0,-1]]*=.5;kw*=k*k/(2*np.pi**2)*np.exp(-(k*1.5)**2)
    tab=np.zeros((len(sg),3,3,len(rg)))
    for il,ell in enumerate([0,2,4]):
        hankel=spherical_jn(ell,rg[:,None]*k[None,:])*kw[None,:]*(-1)**(ell//2)
        for j,sig in enumerate(sg):
            damp=np.exp(-.5*k[:,None]**2*sig**2*(1+(1.6**2-1)*m[None,:]**2))
            for power in range(3):
                leg=(2*ell+1)/2*w*eval_legendre(ell,m)*m**(2*power)
                pell=NW*leg.sum()+(P-NW)*(damp@leg);tab[j,power,il]=hankel@pell
    meta['Sigma_perp_grid']=[0,30];meta['nuisance_evolution']='Positive beta exponential between two free endpoints; Sigma linear between two free nonnegative endpoints.'
    np.savez_compressed(out/'expanded_template.npz',r=rg,sigma_grid=sg,xi_basis=tab,k=k,p_linear=P,p_nowiggle=NW,rd=meta['rdrag_Mpc'])
    (out/'template_manifest.json').write_text(json.dumps(meta,indent=2));return tab,sg,rg,meta

def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=190);fig.savefig(out/(name+'.svg'));plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='previous');ap.add_argument('--output',default='bounds_results');args=ap.parse_args();root=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    f=np.load(root/'clustering_and_jackknife.npz');s=f['s'];mu=f['mu'];z=f['z'];poles=f['all_multipoles'];cov=f['covariance'];kernels=f['kernels'];leg=f['legendre_weights']
    template=wide_template(root,out);rd=template[-1]['rdrag_Mpc'];report={'input':'saved full v1.5 actual-catalogue measurements','nuisance_revision':'Remove former artificial Sigma0>=4 and slope limits. Fit independent beta and Sigma endpoints over 0.4<z<1.1.','interpretation':'Fits remain conditional on AP polynomial evolution, acoustic template and regularized jackknife covariance.'}
    records=[];repsout={}
    for quad,label in [(False,'linear_AP_wide_nuisance'),(True,'quadratic_AP_wide_nuisance')]:
        fit=WideFit(s,mu,z,poles,cov,kernels,leg,template,quadratic=quad,shrink=.5)
        old=np.load(root/('fit_quadratic_AP_evolution.npz' if quad else 'fit_linear_AP_evolution.npz'))['theta'];fit.initial[:(6 if quad else 4)]=old[:(6 if quad else 4)]
        t,reps,cp,model,meta=fit.run(out,label);point=distances(t,np.array([.75]),rd,quad)[:,0];vv=np.array([distances(v,np.array([.75]),rd,quad)[:,0] for v in reps]);cv=(len(vv)-1)/len(vv)*((vv-vv.mean(0)).T@(vv-vv.mean(0)));err=np.sqrt(np.maximum(0,np.diag(cv)))
        names=['DM_over_rd','DH_over_rd','dDM_over_rd_dz','dDH_over_rd_dz','d2DM_over_rd_dz2','d2DH_over_rd_dz2'];r={name:{'value':float(point[i]),'jk_sigma':float(err[i])} for i,name in enumerate(names)}
        r['covariance_distances']=cv.tolist();r['best_parameters']=t.tolist();r['parameter_names']=fit.names;r['boundary_hits']=meta['jackknife_boundary_hits_per_parameter'];r['converged_fits']=meta['jackknife_successes'];r['objective']=meta['objective'];r['objective_dof']=meta['objective_dof'];r['amplitude_min']=meta['minimum_profiled_amplitude_across_redshift']
        delta=vv[:,2]-vv[:,1];r['flat_consistency']={'value':float(point[2]-point[1]),'jk_sigma':float(np.sqrt((len(delta)-1)/len(delta)*np.sum((delta-delta.mean())**2)))}
        report[label]=r;repsout[label]=(point,err,vv,cv,t,reps,fit,model)
        for i,name in enumerate(names):records.append([label,name,point[i],err[i]])
        sf=fit.s;observ=fit.obs[0].reshape(3,2,-1);pred=model.reshape(3,2,-1);errs=fit.std.reshape(3,2,-1)
        for order,prefix in enumerate(['xi','dxi','ddxi']):
            fig,ax=plt.subplots(figsize=(8.5,5.1));ax.errorbar(sf,sf**2*observ[order,0],yerr=sf**2*errs[order,0],fmt='o',capsize=2,ms=4,label='All LRG galaxies and random_0, NGC+SGC');ax.plot(sf,sf**2*pred[order,0],label='Revised light-cone fit');ax.axhline(0,ls=':',lw=.8)
            ax.set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$');ax.set_ylabel(['$s^2\\xi_0$','$s^2\\langle\\partial_z\\xi_0\\rangle$','$s^2\\langle\\partial_z^2\\xi_0\\rangle$'][order]);ax.set_title(label.replace('_',' ')+'\n'+prefix+'; 120-region jackknife');ax.legend(fontsize=8);save(fig,out,label+'_'+prefix)
        for ilo,ihi,sh in [(66,144,.5),(60,150,.2),(60,150,.8)]:
            test=WideFit(s,mu,z,poles,cov,kernels,leg,template,quadratic=quad,shrink=sh,slo=ilo,shi=ihi);rr=test.fitone(0,t,350);v=distances(rr.x,np.array([.75]),rd,quad)[:,0]
            r.setdefault('sensitivity',[]).append({'smin':ilo,'smax':ihi,'shrinkage':sh,'converged':bool(rr.success),'quantities':dict(zip(names,v.tolist())),'theta':rr.x.tolist()})
        # Local noiseless recovery checks for all free geometrical coefficients.
        # Existing data are preserved; these tests replace only the whitened mean
        # and validate the forward-model inversion, not the real-data systematics.
        original=fit.y[0].copy();checks=[];linear=fit.profile(t,0)[1]
        for index in range(6 if quad else 4):
            truth=t.copy();truth[index]+=(.01 if index<2 else (.08 if index<4 else .3));fit.y[0]=fit.W@(fit.design(truth,0)@linear)
            rr=fit.fitone(0,truth+np.r_[np.full(6 if quad else 4,.0005),np.zeros(4)],400)
            checks.append({'parameter':fit.names[index],'max_geometry_error':float(np.max(abs(rr.x[:(6 if quad else 4)]-truth[:(6 if quad else 4)]))),'residual_norm':float(np.linalg.norm(rr.fun))})
        fit.y[0]=original;r['noiseless_forward_model_checks']=checks
        print('WIDE_FIT_COMPLETE',label,json.dumps(r),flush=True)
    zplot=np.linspace(.4,1.1,181);zp=np.array([.75]);point,err,vv,cv,t,reps,fit,model=repsout['quadratic_AP_wide_nuisance']
    for j,label in [(2,'dDM_dz'),(3,'dDH_dz')]:
        fig,ax=plt.subplots(figsize=(8.4,4.9))
        for row,key in enumerate(['linear_AP_wide_nuisance','quadratic_AP_wide_nuisance']):
            pv,er,*_=repsout[key];ax.errorbar(pv[j],row,xerr=er[j],fmt='o',capsize=4,label=key.replace('_',' '))
        ref=np.zeros_like(t);ref[-4:]=[.4,.4,7,7];fid=distances(ref,zp,rd,True)[j,0];ax.axvline(fid,ls='--',label='fiducial reference')
        ax.set_yticks([0,1],['Linear log-AP evolution','Quadratic log-AP evolution']);ax.set_xlabel(['','','d(D_M/r_d)/dz','d(D_H/r_d)/dz'][j]);ax.set_title('Full-sample differential BAO: free anchors, wider nuisances');ax.legend(fontsize=8);save(fig,out,label+'_comparison')
    c=cv[np.ix_([2,3],[2,3])];mean=point[[2,3]];ev,ec=np.linalg.eigh(c);phi=np.linspace(0,2*np.pi,400);circle=np.array([np.cos(phi),np.sin(phi)])
    fig,ax=plt.subplots(figsize=(7.3,6.0))
    for prob,ls in [(.68,'-'),(.95,'--')]:
        pp=mean[:,None]+ec@np.diag(np.sqrt(np.maximum(ev,0)))@(np.sqrt(chi2.ppf(prob,2))*circle);ax.plot(*pp,ls=ls,label=f'{int(100*prob)}% Gaussian JK ellipse')
    ax.plot(*mean,marker='o',ls='none',label='quadratic evolution fit');ax.plot(*distances(np.zeros_like(t),zp,rd,True)[[2,3],0],marker='x',ls='none',label='fiducial reference');ax.set_xlabel('d(D_M/r_d)/dz');ax.set_ylabel('d(D_H/r_d)/dz');ax.set_title('Model-conditional differential BAO, z=0.75');ax.legend(fontsize=9);save(fig,out,'joint_derivatives_wide_nuisance')
    np.savetxt(out/'derivative_results.csv',np.asarray(records,dtype=object),fmt='%s',delimiter=',',header='fit,quantity,value,jk_sigma',comments='')
    (out/'robustness_summary.json').write_text(json.dumps(report,indent=2))
    (out/'README.md').write_text('# Nuisance-bound robustness rerun\n\nAll fits use the existing unthinned v1.5 full-catalogue measurements. Neither simulated catalogues nor published distance values supply the measured data.\n\nThe earlier damping/evolution bounds were saturated. This run uses positive beta endpoint values in [0.001,3] and damping endpoint values in [0,30] Mpc/h. It checks fit-range and covariance regularization sensitivity, and provides model-conditional linear/quadratic AP results. Numerical convergence does not constitute mock validation.\n\nSee robustness_summary.json for all boundary, sensitivity, and inversion diagnostics. Do not interpret second derivatives fixed by the linear-AP model as independently measured curvature.\n')
    print('ROBUSTNESS_FINISHED',json.dumps(report),flush=True)
if __name__=='__main__':main()
