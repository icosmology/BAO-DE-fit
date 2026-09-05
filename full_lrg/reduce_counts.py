#!/usr/bin/env python3
"""Reduce real pair counts. Derivative order 2 has measured, not assumed, RR4 normalization."""
import argparse,json,math
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.integrate import cumulative_trapezoid

PIVOT=.75

def cap_arrays(root,cap,rebin=1):
    out={}
    for kind in ['DD','DR','RR']:
        files=sorted(Path(root).rglob(f'{cap}_{kind}_part*.npz'))
        if len(files)!=8:raise RuntimeError(f'Expected 8 disjoint pieces for {cap} {kind}, found {len(files)}')
        total=None;hist=None
        for path in files:
            with np.load(path) as f:
                ar=f['moments']
                if total is None:total=ar.copy()
                else:total+=ar
                if kind=='RR':
                    hh=f['hist']
                    if hist is None:hist=hh.copy()
                    else:hist+=hh
                    dn,rn=f['data_norms'],f['random_norms']
        # Saved arrays are [full, removed-region contributions]. Convert to actual deletions.
        total[1:]=total[0]-total[1:]
        if rebin>1:
            total=total.reshape(total.shape[0],total.shape[1]//rebin,rebin,*total.shape[2:]).sum(axis=2)
        out[kind]=total
        if kind=='RR':
            hist[1:]=hist[0]-hist[1:]
            if rebin>1:hist=hist.reshape(hist.shape[0],hist.shape[1]//rebin,rebin,*hist.shape[2:]).sum(axis=2)
            out['hist']=hist;out['dn']=dn;out['rn']=rn
    wd,wd2=out['dn'].T;wr,wr2=out['rn'].T
    cross=(wd*wd-wd2)/(wd*wr)
    ratio=(wd*wd-wd2)/(wr*wr-wr2)
    out['N']=out['DD'][...,:3]-cross[:,None,None,None]*out['DR'][...,:3]+ratio[:,None,None,None]*out['RR'][...,:3]
    out['R']=ratio[:,None,None,None]*out['RR']
    out['H']=ratio[:,None,None,None,None]*out['hist']
    return out


def central_moments(R):
    raw=R/R[...,0,None];m=raw[...,1]
    c=np.zeros_like(raw);c[...,0]=1.
    for n in range(2,7):
        for k in range(n+1):c[...,n]+=math.comb(n,k)*(-m)**(n-k)*raw[...,k]
    return m,c


def estimator(N,R):
    mean,c=central_moments(R);v=c[...,2]
    n0=N[...,0];n1=N[...,1]-mean*n0;n2=N[...,2]-2*mean*N[...,1]+mean**2*n0
    norm=c[...,4]-v*v-c[...,3]**2/v
    if np.any(v<=0) or np.any(norm<=0):raise RuntimeError('Degenerate redshift distribution')
    a0=n0/R[...,0];a1=n1/(v*R[...,0]);a2=2*(n2-v*n0-(c[...,3]/v)*n1)/(norm*R[...,0])
    fields=np.stack([a0,a1,a2],axis=1)
    muedges=np.linspace(0,1,N.shape[-2]+1)
    poles=[]
    for ell in [0,2,4]:
        leg=Legendre.basis(ell).integ();fac=(2*ell+1)*np.diff(leg(muedges))
        poles.append(np.sum(fields*fac[None,None,None,:],axis=-1))
    poles=np.stack(poles,axis=2)
    w1m=c[...,3]/(2*v)
    w1var=c[...,4]/(3*v)-w1m*w1m
    w2m=(c[...,5]-v*c[...,3]-c[...,3]*c[...,4]/v)/(3*norm)
    w2var=(c[...,6]-v*c[...,4]-c[...,3]*c[...,5]/v)/(6*norm)-w2m*w2m
    diag={'pair_zmean':PIVOT+mean,'pair_zvar':v,'W1_zmean':PIVOT+mean+w1m,'W1_zvar':w1var,'W2_zmean':PIVOT+mean+w2m,'W2_zvar':w2var,'mean_u':mean,'central':c,'second_norm':norm}
    return poles,diag


def joint_cov(samples,nngc,nsgc):
    flat=samples.reshape(len(samples),-1);cov=np.zeros((flat.shape[1],flat.shape[1]))
    for start,n in [(1,nngc),(1+nngc,nsgc)]:
        d=flat[start:start+n];d=d-d.mean(axis=0)
        cov+=(n-1)/n*(d.T@d)
    return cov


def combine(root,rebin):
    ng=cap_arrays(root,'NGC',rebin);sg=cap_arrays(root,'SGC',rebin)
    nn,ns=len(ng['N'])-1,len(sg['N'])-1
    ids=[(0,0)]+[(i,0) for i in range(1,nn+1)]+[(0,i) for i in range(1,ns+1)]
    N=np.stack([ng['N'][i]+sg['N'][j] for i,j in ids])
    R=np.stack([ng['R'][i]+sg['R'][j] for i,j in ids])
    # Full and all leave-one RR light-cone histograms are retained for each realization.
    H=np.stack([ng['H'][i]+sg['H'][j] for i,j in ids])
    poles,diag=estimator(N,R);cov=joint_cov(poles,nn,ns)
    s=40+4*rebin*(np.arange(poles.shape[-1])+.5)
    percap=[]
    for cap in [ng,sg]:percap.append(estimator(cap['N'][:1],cap['R'][:1])[0][0])
    return dict(s=s,multipoles=poles,covariance=cov,N=N,R=R,H=H,ngc_multipoles=percap[0],sgc_multipoles=percap[1],njack_ngc=nn,njack_sgc=ns,**diag)


def export_plots(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=result['s'];vals=result['multipoles'][0]
    err=np.sqrt(np.diag(result['covariance'])).reshape(vals.shape)
    labels=[r'$s^2\xi_\ell$',r'$s^2\langle\partial_z\xi_\ell\rangle_{W_1}$',r'$s^2\langle\partial_z^2\xi_\ell\rangle_{W_2}$']
    names=['xi','dxi_dz','d2xi_dz2']
    rows=[]
    for order in range(3):
        for li,ell in enumerate([0,2,4]):
            fig,ax=plt.subplots(figsize=(8.4,5.2))
            ax.errorbar(s,s*s*vals[order,li],yerr=s*s*err[order,li],fmt='o',capsize=2,label='NGC+SGC; all data + full random_0')
            ax.axhline(0,linestyle='--',linewidth=.8)
            ax.set(xlabel=r'$s\ [h^{-1}{\rm Mpc}]$',ylabel=labels[order],title=f'DESI DR1 v1.5 LRG: {names[order]}, ell={ell}')
            ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/f'{names[order]}_ell{ell}.png',dpi=190);fig.savefig(out/f'{names[order]}_ell{ell}.svg');plt.close(fig)
            for k,x in enumerate(s):rows.append([order,ell,x,vals[order,li,k],err[order,li,k]])
    np.savetxt(out/'measurements.csv',rows,delimiter=',',header='redshift_derivative_order,ell,s_Mpc_h,value,jackknife_sigma',comments='')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='partials');p.add_argument('--output',default='results');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    audits=[json.loads(f.read_text()) for f in Path(a.input).rglob('*_audit.json')]
    summary={'source':'actual DESI DR1 v1.5 catalogues','caps':{},'covariance':'sum of independent cap delete-one jackknife covariances','pre_reconstruction':True,'thinning':False}
    for cap in ['NGC','SGC']:
        aa=sorted([x for x in audits if x['cap']==cap],key=lambda x:x['part'])
        assert [x['part'] for x in aa]==list(range(8))
        t={k:aa[0][k] for k in ['selected_data_rows','selected_random_rows','input_data_rows','input_random_rows','njack','sha256']}
        t['pair_counts']={kind:sum(x[kind]['accepted_pairs'] for x in aa) for kind in ['DD','DR','RR']}
        summary['caps'][cap]=t
    for rebin in [1,2]:
        r=combine(a.input,rebin)
        np.savez_compressed(out/f'measurements_rebin{rebin}.npz',**r)
        if rebin==1:export_plots(r,out)
        mask=(r['s']>=52)&(r['s']<=148);rr=r['R'][0,mask,:,0]
        for name in ['pair_zmean','W1_zmean','W2_zmean','W1_zvar','W2_zvar']:
            summary[f'{name}_rebin{rebin}']=float(np.sum(rr*r[name][0,mask])/rr.sum())
        summary[f'covariance_rank_rebin{rebin}']=int(np.linalg.matrix_rank(r['covariance']))
        del r
    summary['status']='complete_raw_clustering_and_two_derivatives'
    (out/'measurement_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
