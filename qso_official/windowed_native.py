#!/usr/bin/env python3
"""Actual QSO moment likelihood with the unmodified native DESI BAO template.
No distance Taylor series or emulator is used. Independent redshift nodes are
numerical degrees of freedom. The four reported quantities are explicitly
windowed functionals, never advertised as prior-free point derivatives.
"""
import argparse,json,time,hashlib,sys
from pathlib import Path
import numpy as np
if not hasattr(np,'trapezoid'):np.trapezoid=np.trapz
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor,cho_solve
from scipy.special import eval_legendre
from numpy.polynomial.legendre import Legendre,leggauss
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC,NUTS,init_to_value
from numpyro.diagnostics import summary as chain_summary
from cosmoprimo.fiducial import DESI
from desilike.base import jit as calculator_jit
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate,DampedBAOWigglesTracerCorrelationFunctionMultipoles
from desilike.theories.galaxy_clustering.bao import DampedBAOWigglesCorrelationFunctionMultipoles
import desilike.theories.galaxy_clustering.bao as native_bao

ZLO,ZHI,ZREF=.8,2.1,1.49
NATIVE_SHA='4c77845a48fe444b1e8f7f4ce406072071b0ed01'

def interp_matrix(x,nodes):
    eye=np.eye(len(nodes));return np.column_stack([np.interp(x,nodes,eye[:,i]) for i in range(len(nodes))])

def jk_cov(samples,ng,sg):
    a=samples.reshape(len(samples),-1);C=np.zeros((a.shape[1],a.shape[1]))
    for lo,n in [(1,ng),(1+ng,sg)]:
        d=a[lo:lo+n];d=d-d.mean(0);C+=(n-1)/n*d.T@d
    return C

def make_weights(m,mask,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R=m['R'][0,mask];H=m['H'][0,mask]
    mean=m['mean_u'][0,mask];central=m['central'][0,mask];v=central[...,2];c3=central[...,3];norm=m['second_norm'][0,mask]
    h=H[...,0]/R[...,0,None]
    zfallback=np.broadcast_to(np.arange(.805,2.10,.01),h.shape).copy()
    zz=np.divide(H[...,1],H[...,0],out=zfallback,where=H[...,0]>0)
    u=zz-ZREF-mean[...,None]
    q1=u/v[...,None];q2=2*(u*u-v[...,None]-(c3/v)[...,None]*u)/norm[...,None]
    # Recover exact global second moment inside the thin RR redshift cells.
    gap=v-np.sum(h*u*u,axis=-1)
    q2+=2*gap[...,None]/norm[...,None]
    weights=np.stack([h,h*q1,h*q2])
    tests={'q0_normalization_max_error':float(np.max(abs(weights[0].sum(-1)-1))),
           'q1_constant_response_max_error':float(np.max(abs(weights[1].sum(-1)))),
           'q2_constant_response_max_error':float(np.max(abs(weights[2].sum(-1))))}
    # Representative actual RR distribution and moments for a common output
    # window. This aggregation is only for the four reported functionals;
    # every s,mu cell retains its own measured kernel in the likelihood.
    hs=H.sum((0,1));rs=R.sum((0,1));raw=rs/rs[0];mu=raw[1]
    vagg=raw[2]-mu*mu;c3agg=raw[3]-3*mu*raw[2]+2*mu**3
    c4agg=raw[4]-4*mu*raw[3]+6*mu*mu*raw[2]-3*mu**4
    zmean=ZREF+mu;normagg=c4agg-vagg*vagg-c3agg*c3agg/vagg
    # Piecewise-linear density matches H0 and H1 in every 0.01 bin.
    # Its agreement with separately accumulated RR2/RR3 is audited.
    dz=.01;zc=np.arange(.805,2.10,.01);zf=np.linspace(ZLO,ZHI,26001)
    ib=np.minimum(len(zc)-1,np.maximum(0,((zf-ZLO)/dz).astype(int)))
    A=hs[:,0]/(rs[0]*dz);B=12*(hs[:,1]-zc*hs[:,0])/(rs[0]*dz**3)
    K=A[ib]+B[ib]*(zf-zc[ib]);assert K.min()>-1e-8
    K=np.maximum(K,0);K/=np.trapezoid(K,zf)
    qf1=(zf-zmean)/vagg;qf2=2*((zf-zmean)**2-vagg-c3agg/vagg*(zf-zmean))/normagg
    W=-cumulative_trapezoid(K*qf1,zf,initial=0)
    drift=float(W[-1]);W-=drift*(zf-ZLO)/(ZHI-ZLO)
    norm_before=float(np.trapezoid(W,zf));W/=norm_before
    Wprime=np.gradient(W,zf,edge_order=2)
    zeff=float(np.trapezoid(zf*W,zf));width=float(np.sqrt(np.trapezoid((zf-zeff)**2*W,zf)))
    tests.update({'RR_mean_z':float(zmean),'RR_variance_z':float(vagg),'common_window_z_eff':zeff,'common_window_sigma_z':width,
      'histogram_kernel_boundary_drift':drift,'histogram_kernel_integral_before_unit_normalization':norm_before,
      'RR_histogram_second_moment_fractional_error':float(np.trapezoid(K*(zf-zmean)**2,zf)/vagg-1),
      'exact_RR_first_derivative_kernel_mean':float(zmean+c3agg/(2*vagg)),
      'scope':'Raw moment estimators use exact accumulated moments; displayed/common kernels use a 0.01-bin density reconstruction. They are not point-resolution measurements.'})
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'weights_and_common_window.npz',z=zf,K=K,q1=qf1,q2=qf2,W=W,Wprime=Wprime,cell_z=zz,cell_weights=weights,cell_q1=q1,cell_q2=q2,RR_moments=R)
    np.savetxt(out/'redshift_weights.csv',np.column_stack([zf,K,np.ones_like(zf),qf1,qf2,W]),delimiter=',',header='z,K,q0,q1,q2,common_derivative_window_W',comments='')
    (out/'weight_audit.json').write_text(json.dumps(tests,indent=2))
    for q,name,lab in [(qf1,'redshift_weight_first',r'$q_1(z)$'),(qf2,'redshift_weight_second',r'$q_2(z)$')]:
        fig,ax=plt.subplots(figsize=(8.3,5.1));ax.plot(zf,q,label='Measured RR-normalized representative weight');ax.axhline(0,linestyle='--',linewidth=.7)
        ax.axvline(zmean,linestyle=':',label=rf'RR mean $z={zmean:.4f}$');ax.set(xlabel='Pair midpoint redshift',ylabel=lab,title='Actual QSO catalogue redshift weights')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=200);fig.savefig(out/f'{name}.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.3,5.1));ax.plot(zf,K,label='RR pair-redshift kernel K');ax.plot(zf,W,label='Common derivative averaging kernel W')
    ax.axvline(zeff,linestyle=':',label=rf'$z_W={zeff:.4f}$');ax.set(xlabel='Redshift z',ylabel='Normalized kernel density',title='Distance functionals are windowed, not point derivatives')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'common_window.png',dpi=200);fig.savefig(out/'common_window.svg');plt.close(fig)
    return zz,weights,dict(z=zf,W=W,Wprime=Wprime,zeff=zeff,audit=tests)

class NativeMomentModel:
    def __init__(self,m,out,nodes=6,group=5,orders=3,shrink=.2):
        self.nodes=np.linspace(ZLO,ZHI,nodes);self.n=nodes;self.orders=orders
        self.mask=(m['s']>=52)&(m['s']<=148);self.s=m['s'][self.mask];self.cosmo=DESI(engine='class');self.rd=float(self.cosmo.rs_drag)
        self.zz,ww,self.window=make_weights(m,self.mask,out)
        self.ww=ww[:orders]
        # Fixed redshift quadrature; interpolation parameters are free node
        # values, NOT Taylor coefficients about a pivot.
        hz=m['H'][0,self.mask,...,0].sum((0,1));zw=m['H'][0,self.mask,...,1].sum((0,1));zb=np.divide(zw,hz,out=np.arange(.805,2.10,.01).copy(),where=hz>0)
        assert len(zb)%group==0
        self.z=np.sum((hz*zb).reshape(-1,group),axis=-1)/np.sum(hz.reshape(-1,group),axis=-1)
        self.weights=self.ww.reshape(*self.ww.shape[:-1],-1,group).sum(-1)
        self.I=jnp.array(interp_matrix(self.z,self.nodes))
        self.ells=(0,2,4,6,8);nm=self.weights.shape[-2];muc=(np.arange(nm)+.5)/nm;edges=np.linspace(0,1,nm+1)
        outleg=np.array([(2*ell+1)*np.diff(Legendre.basis(ell).integ()(edges)) for ell in (0,2)])
        inleg=np.array([eval_legendre(ell,muc) for ell in self.ells])
        # K[derivative order, observed ell, separation, zquad, true ell]
        self.K=jnp.asarray(np.einsum('osmz,lm,tm->olszt',self.weights,outleg,inleg))
        # RR radial bin integration is numerical, not an empirical BAO shell.
        gg,gw=leggauss(4);self.squad=(self.s[:,None]+4*gg).ravel();rw=gw[None,:]*(self.s[:,None]+4*gg)**2;rw/=rw.sum(axis=1)[:,None]
        self.rw=jnp.asarray(rw);self.nrad=len(gg)
        self.template=BAOPowerSpectrumTemplate(z=ZREF,fiducial=self.cosmo,with_now='peakaverage',apmode='qparqper')
        self.native=DampedBAOWigglesCorrelationFunctionMultipoles(s=self.squad,ells=self.ells,template=self.template,mode='',model='standard')
        initial=dict(qper=1.,qpar=1.,b1=2.4,dbeta=1.,sigmaper=3.5,sigmapar=9.,sigmas=2.)
        raw=np.asarray(self.native(**initial));self.native_jit=calculator_jit(self.native)
        self.native_jit(**initial)
        def local(v):
            return self.native_jit(qper=v[0],qpar=v[1],b1=v[2],dbeta=v[3],sigmaper=v[4],sigmapar=v[5],sigmas=v[6])
        self.batch=jax.jit(jax.vmap(local))
        pv=jnp.array(list(initial.values()));check=np.asarray(self.batch(jnp.tile(pv,(len(self.z),1))))
        self.native_error=float(np.max(abs(check-raw)))
        # DESI native PCS broadband basis, generated by native code calls.
        trtemp=BAOPowerSpectrumTemplate(z=ZREF,fiducial=self.cosmo,with_now='peakaverage',apmode='qparqper')
        tr=DampedBAOWigglesTracerCorrelationFunctionMultipoles(s=self.squad,ells=(0,2),template=trtemp,mode='',model='standard',broadband='pcs2')
        tr(**initial);base=np.asarray(tr(**initial));names=[p.basename for p in tr.all_params if p.basename.startswith(('al','bl')) and not p.fixed]
        bases=[];prec=[]
        for name in names:
            bb=np.asarray(tr(**{**initial,name:1.}))-base
            bb=bb.reshape(2,len(self.s),self.nrad);bb=np.sum(bb*np.asarray(self.rw)[None,:,:],axis=-1)
            bases.append(bb*self.s[None,:]**2);prec.append(1e-8 if name.startswith('al') else 0.)
        bb=np.asarray(bases).reshape(len(names),-1).T
        B=np.zeros((orders*2*len(self.s),orders*len(names)))
        for order in range(orders):B[order*2*len(self.s):(order+1)*2*len(self.s),order*len(names):(order+1)*len(names)]=bb
        self.bbnames=names
        values=m['multipoles'][:,:orders,:2][:,:,:,self.mask]*self.s[None,None,None,:]**2
        self.y=np.asarray(values[0]).ravel();self.Craw=jk_cov(values,int(m['njack_ngc']),int(m['njack_sgc']))
        self.C=(1-shrink)*self.Craw+shrink*np.diag(np.diag(self.Craw));Ci=cho_solve(cho_factor(self.C,lower=True),np.eye(len(self.C)))
        norms=np.sqrt(np.sum(B*(Ci@B),axis=0));assert np.all(norms>0)
        Bs=B/norms;pp=np.tile(prec,orders)/norms**2
        normal=Bs.T@Ci@Bs+np.diag(pp)
        self.projector=jnp.asarray(Ci-Ci@Bs@np.linalg.solve(normal,Bs.T@Ci))
        self.B=B;self.Bs=Bs;self.Ci=Ci;self.normal=normal;self.yj=jnp.asarray(self.y)
        self.predict=jax.jit(self._predict)
        self.posterior_loglik=jax.jit(lambda p:-.5*(self.yj-self._predict(p))@self.projector@(self.yj-self._predict(p)))
        self.raw_cov_rank=int(np.linalg.matrix_rank(self.Craw));self.shrink=shrink;self.group=group
        self.audit={'native_template':'desilike.DampedBAOWigglesCorrelationFunctionMultipoles, model=standard, CLASS, peakaverage','native_source_commit':NATIVE_SHA,
                    'native_source_sha256':hashlib.sha256(Path(native_bao.__file__).read_bytes()).hexdigest(),'native_vmap_max_abs_error':self.native_error,
                    'broadband':'DESI native pcs2 basis per measured derivative channel','broadband_names':names,'data_dimension':len(self.y),'covariance_rank':self.raw_cov_rank,'shrinkage':shrink,
                    'distance_nodes':self.nodes.tolist(),'no_redshift_Taylor_series':True,'geometry_prior':'independent node alpha_iso and alpha_AP each U[0.8,1.2]; no curvature/smoothness penalty',
                    'nuisance_prior':'each node b1 U[0.2,4], dbeta U[0.7,1.3]; global Sigma_perp N[3.5,1], Sigma_parallel N[9,2], sigmas N[2,2], truncated [0,20]',
                    'numerical_scope':'Piecewise-linear interpolation of free AP node values, finite quadrature and finite RR histograms remain numerical/model-resolution choices. This is not a completely assumption-free point-derivative measurement.'}
    def _predict(self,p):
        n=self.n;aiso=self.I@p[:n];aap=self.I@p[n:2*n];b=self.I@p[2*n:3*n];db=self.I@p[3*n:4*n]
        qper=aiso*aap**(-1/3);qpar=aiso*aap**(2/3)
        sig=jnp.broadcast_to(p[-3:],(len(self.z),3))
        pars=jnp.column_stack([qper,qpar,b,db,sig]);corr=self.batch(pars).reshape(len(self.z),len(self.ells),len(self.s),self.nrad)
        corr=jnp.sum(corr*self.rw[None,None,:,:],axis=-1)
        model=jnp.einsum('olszt,zts->ols',self.K,corr)*jnp.asarray(self.s)[None,None,:]**2
        return model.ravel()
    def mean_broadband(self,p):
        model=np.asarray(self.predict(p));coef=np.linalg.solve(self.normal,self.Bs.T@self.Ci@(self.y-model))
        return model+self.Bs@coef
    def targets(self,params):
        p=np.atleast_2d(params);n=self.n;z=self.window['z'];I=interp_matrix(z,self.nodes)
        ai=p[:,:n]@I.T;aa=p[:,n:2*n]@I.T
        Xf=np.asarray(self.cosmo.comoving_angular_distance(z))/(self.cosmo.h*self.rd)
        Yf=299792.458/np.asarray(self.cosmo.hubble_function(z))/self.rd
        X=ai*aa**(-1/3)*Xf;Y=ai*aa**(2/3)*Yf
        W=self.window['W'];Wp=self.window['Wprime']
        return np.column_stack([np.trapezoid(X*W,z,axis=-1),np.trapezoid(Y*W,z,axis=-1),-np.trapezoid(X*Wp,z,axis=-1),-np.trapezoid(Y*Wp,z,axis=-1)])

def sampling_model(engine):
    n=engine.n
    ai=numpyro.sample('alpha_iso_nodes',dist.Uniform(.8,1.2).expand([n]).to_event(1))
    aa=numpyro.sample('alpha_AP_nodes',dist.Uniform(.8,1.2).expand([n]).to_event(1))
    b=numpyro.sample('bias_nodes',dist.Uniform(.2,4.).expand([n]).to_event(1))
    db=numpyro.sample('dbeta_nodes',dist.Uniform(.7,1.3).expand([n]).to_event(1))
    sp=numpyro.sample('sigmaper',dist.TruncatedNormal(3.5,1.,low=0.,high=20.))
    sa=numpyro.sample('sigmapar',dist.TruncatedNormal(9.,2.,low=0.,high=20.))
    ss=numpyro.sample('sigmas',dist.TruncatedNormal(2.,2.,low=0.,high=20.))
    p=jnp.concatenate([ai,aa,b,db,jnp.array([sp,sa,ss])])
    numpyro.factor('observed_weighted_correlation',engine.posterior_loglik(p))

def flatten_samples(s):
    return np.column_stack([np.asarray(s[k]) for k in ['alpha_iso_nodes','alpha_AP_nodes','bias_nodes','dbeta_nodes','sigmaper','sigmapar','sigmas']])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='measured/measurements_rebin2.npz');ap.add_argument('--out',default='native_results');ap.add_argument('--nodes',type=int,default=6);ap.add_argument('--samples',type=int,default=2000);ap.add_argument('--warmup',type=int,default=1500);ap.add_argument('--group',type=int,default=5);ap.add_argument('--orders',type=int,default=3);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with np.load(a.input) as ff:m={k:ff[k] for k in ff.files}
    eng=NativeMomentModel(m,out,a.nodes,a.group,a.orders)
    (out/'native_method.json').write_text(json.dumps(eng.audit,indent=2));print('NATIVE_MODEL',json.dumps(eng.audit),flush=True)
    init={'alpha_iso_nodes':jnp.ones(a.nodes),'alpha_AP_nodes':jnp.ones(a.nodes),'bias_nodes':jnp.full(a.nodes,2.4),'dbeta_nodes':jnp.ones(a.nodes),'sigmaper':3.5,'sigmapar':9.,'sigmas':2.}
    p0=np.r_[np.ones(a.nodes),np.ones(a.nodes),np.full(a.nodes,2.4),np.ones(a.nodes),3.5,9.,2.]
    t0=time.time();val=eng.posterior_loglik(jnp.array(p0));grad=jax.grad(eng.posterior_loglik)(jnp.array(p0));print('INITIAL_LOG_LIKELIHOOD',float(val),'maxgrad',float(jnp.max(abs(grad))),'compile_seconds',time.time()-t0,flush=True)
    kernel=NUTS(lambda:sampling_model(eng),target_accept_prob=.85,max_tree_depth=9,init_strategy=init_to_value(values=init))
    sampler=MCMC(kernel,num_warmup=a.warmup,num_samples=a.samples,num_chains=2,chain_method='sequential',progress_bar=True)
    start=time.time();sampler.run(jax.random.PRNGKey(5817+a.nodes),extra_fields=('diverging','num_steps','accept_prob'))
    ss=sampler.get_samples();chains=sampler.get_samples(group_by_chain=True);params=flatten_samples(ss)
    targets=[]
    for lo in range(0,len(params),250):targets.append(eng.targets(params[lo:lo+250]))
    targets=np.concatenate(targets);names=['DM_over_rd_W','DH_over_rd_W','dDM_over_rd_dz_W','dDH_over_rd_dz_W'];qq=np.percentile(targets,[2.5,16,50,84,97.5],axis=0)
    diag=chain_summary(chains,group_by_chain=True);fields=sampler.get_extra_fields()
    diagnostics={k:{kk:np.asarray(vv).tolist() for kk,vv in v.items()} for k,v in diag.items()}
    pmean=np.mean(params,axis=0);pred=eng.mean_broadband(pmean)
    np.savez_compressed(out/'joint_posterior.npz',samples=params,targets=targets,target_names=np.array(names),target_covariance=np.cov(targets,rowvar=False),nodes=eng.nodes,data=eng.y,covariance=eng.Craw,covariance_used=eng.C,mean_model=pred,s=eng.s)
    np.savetxt(out/'four_quantity_samples.csv',targets,delimiter=',',header=','.join(names),comments='')
    report={'status':'completed_native_template_free_node_inference','four_quantities':{name:dict(zip(['p2p5','p16','median','p84','p97p5'],qq[:,i].tolist())) for i,name in enumerate(names)},
            'z_effective_common_window':eng.window['zeff'],'method':eng.audit,'convergence':diagnostics,'divergences':int(np.sum(fields['diverging'])),'acceptance_mean':float(np.mean(fields['accept_prob'])),'maximum_tree_steps':int(np.max(fields['num_steps'])),'seconds':time.time()-start,
            'important':'No claim that these windowed, node-marginalized quantities are exact point derivatives. Mesh convergence and prior sensitivity are necessary before quoting model-independent constraints.'}
    (out/'fit_summary.json').write_text(json.dumps(report,indent=2));print('FOUR_QUANTITY_REPORT',json.dumps(report),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import corner
    labs=[r'$\langle D_M/r_d\rangle_W$',r'$\langle D_H/r_d\rangle_W$',r'$\langle d(D_M/r_d)/dz\rangle_W$',r'$\langle d(D_H/r_d)/dz\rangle_W$']
    fig=corner.corner(targets,labels=labs,quantiles=[.16,.5,.84],show_titles=True,title_fmt='.2f',levels=[.68,.95],plot_datapoints=False,smooth=1.,bins=40)
    fig.suptitle(f'Native DESI template; {a.nodes} free nodes; common window z={eng.window["zeff"]:.4f}',fontsize=13)
    fig.savefig(out/'four_parameter_triangle.png',dpi=190,bbox_inches='tight');fig.savefig(out/'four_parameter_triangle.svg',bbox_inches='tight');plt.close(fig)
    y=eng.y.reshape(a.orders,2,-1);pp=pred.reshape(a.orders,2,-1);err=np.sqrt(np.diag(eng.Craw)).reshape(a.orders,2,-1)
    for o in range(a.orders):
        for l,ell in enumerate([0,2]):
            fig,ax=plt.subplots(figsize=(8.3,5.1));ax.errorbar(eng.s,y[o,l],yerr=err[o,l],fmt='o',capsize=3,label='Direct catalogue weighted statistic')
            ax.plot(eng.s,pp[o,l],label='Native official-template free-node fit');ax.axhline(0,linestyle='--',linewidth=.6)
            ax.set(xlabel=r'$s\ [h^{-1}{\rm Mpc}]$',ylabel=rf'$s^2\xi_{{{ell}}}^{{[{o}]}}$',title=f'QSO, derivative order {o}, ell={ell}; no distance Taylor series')
            ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'fit_order{o}_ell{ell}.png',dpi=190);plt.close(fig)
    # Geometrical prior-only output: expose changes induced by finite node prior.
    rng=np.random.default_rng(331+a.nodes);pr=np.tile(p0,(5000,1));pr[:,:2*a.nodes]=rng.uniform(.8,1.2,(len(pr),2*a.nodes))
    pv=[]
    for lo in range(0,len(pr),250):pv.append(eng.targets(pr[lo:lo+250]))
    pv=np.concatenate(pv);np.savez_compressed(out/'target_prior_comparison.npz',prior_targets=pv,posterior_targets=targets)
    for j in range(4):
        fig,ax=plt.subplots(figsize=(7.8,4.8));ax.hist(pv[:,j],bins=55,density=True,histtype='step',label='Independent-node geometry prior')
        ax.hist(targets[:,j],bins=55,density=True,histtype='step',label='Catalogue-conditioned posterior');ax.set(xlabel=labs[j],ylabel='Density',title='Data information versus finite-grid prior');ax.legend();fig.tight_layout();fig.savefig(out/f'prior_check_{j}.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
