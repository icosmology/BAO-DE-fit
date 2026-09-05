import json, math
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import Legendre

PIVOT=.75

def jk_cov(a):
    a=np.asarray(a);x=a[1:]-a[1:].mean(axis=0);n=len(x)
    return (n-1)/n*(x.T@x)

def read_cap(root,cap):
    d=np.load(root/f'{cap}_DDDR_DD.npz');c=np.load(root/f'{cap}_DDDR_DR.npz')
    rr=[];hh=[];pairs=0
    for i in range(4):
        q=np.load(root/f'{cap}_RR{i}_RR.npz');rr.append(q['moments']);hh.append(q['hist']);pairs+=int(q['pairs'])
        np.testing.assert_allclose(q['norms'],d['norms'],rtol=1e-13)
    return {'D':d['moments'],'C':c['moments'],'R':sum(rr),'H':sum(hh),'norms':d['norms'],'pairs':[int(d['pairs']),int(c['pairs']),pairs]}

def cap_view(c,region):
    def v(name):return c[name][0] if region<0 else c[name][0]-c[name][region+1]
    norm=c['norms'][0] if region<0 else c['norms'][0]-c['norms'][region+1]
    wd,wd2,wr,wr2=norm;nd=(wd*wd-wd2)/2;nr=(wr*wr-wr2)/2;nc=wd*wr
    ratio=nd/nr
    return v('D')-2*nd/nc*v('C')+ratio*v('R')[...,:3],ratio*v('R'),ratio*v('H')

def reduce(root,out):
    root=Path(root);out=Path(out);caps=[read_cap(root,'NGC'),read_cap(root,'SGC')]
    nj=[len(c['norms'])-1 for c in caps];allN=[];allR=[];allH=[]
    cases=[(-1,-1)]+[(k,-1) for k in range(nj[0])]+[(-1,k) for k in range(nj[1])]
    for ng,sg in cases:
        n1,r1,h1=cap_view(caps[0],ng);n2,r2,h2=cap_view(caps[1],sg)
        allN.append(n1+n2);allR.append(r1+r2);allH.append(h1+h2)
    N,R,H=map(np.asarray,[allN,allR,allH]);R0=R[...,0]
    assert np.all(R0>0)
    raw=R/R0[...,None];c1=raw[...,1]
    central=[]
    for power in range(7):
        v=sum(math.comb(power,k)*(-c1)**(power-k)*raw[...,k] for k in range(power+1));central.append(v)
    m2,m3,m4,m5,m6=central[2:7]
    den=m4-m2*m2-m3*m3/m2
    assert np.all(m2>0) and np.all(den>0)
    xi=N[...,0]/R0
    dxi=(N[...,1]-c1*N[...,0])/(m2*R0)
    ddxi=2*(N[...,2]-(2*c1+m3/m2)*N[...,1]+(c1*c1-m2+c1*m3/m2)*N[...,0])/(den*R0)
    cells=np.stack([xi,dxi,ddxi],axis=1)
    mu_edges=np.linspace(0,1,R0.shape[-1]+1);ells=[0,2,4]
    leg=np.array([(2*l+1)*np.diff(Legendre.basis(l).integ()(mu_edges)) for l in ells])
    poles=np.einsum('rism,lm->rils',cells,leg)
    covariance=jk_cov(poles.reshape(len(poles),-1));error=np.sqrt(np.maximum(0,np.diag(covariance))).reshape(poles.shape[1:])
    zc=np.linspace(.405,1.095,H.shape[-1]);u=zc-PIVOT
    # Correct midpoint histogram quadrature to reproduce the seven exact RR
    # polynomial moments. This is quadrature calibration, not a signal prior.
    phi=np.array([(u/.35)**k for k in range(7)]).T
    h=H/R0[...,None];target=raw/np.array([.35**k for k in range(7)])
    current=np.einsum('rsmz,zk->rsmk',h,phi,optimize=True)
    gram=np.einsum('rsmz,zk,zj->rsmkj',h,phi,phi,optimize=True)
    coef=np.linalg.solve(gram,(target-current)[...,None])[...,0]
    h=h*(1+np.einsum('rsmk,zk->rsmz',coef,phi,optimize=True))
    moment_error=float(np.max(np.abs(np.einsum('rsmz,zk->rsmk',h,phi,optimize=True)-target)))
    uc=u[None,None,None,:]-c1[...,None]
    q0=np.ones_like(uc);q1=uc/m2[...,None]
    q2=2*(uc*uc-m2[...,None]-(m3/m2)[...,None]*uc)/den[...,None]
    kernels=np.stack([h*q0,h*q1,h*q2],axis=1)
    tests={'quadrature_moment_max_error':moment_error,'minimum_quadrature_probability':float(h.min()),'xi_prime_constant_leakage':float(abs(kernels[:,1].sum(axis=-1)).max()),'xi_second_constant_leakage':float(abs(kernels[:,2].sum(axis=-1)).max()),'xi_second_linear_leakage':float(abs(np.einsum('rsmz,z->rsm',kernels[:,2],u)).max())}
    assert moment_error<1e-8
    assert tests['xi_second_constant_leakage']<1e-6 and tests['xi_second_linear_leakage']<1e-6
    if h.min() < -1e-4:raise RuntimeError('Large negative quadrature correction; increase histogram resolution.')
    zm=PIVOT+c1;z1=zm+m3/(2*m2);var1=m4/(3*m2)-(m3/(2*m2))**2
    off2=(m5-m2*m3-m3*m4/m2)/(3*den);z2=zm+off2
    var2=(m6-m2*m4-m3*m5/m2)/(6*den)-off2**2
    s=40+6*(np.arange(R0.shape[1])+.5);mu=.5*(mu_edges[1:]+mu_edges[:-1]);mask=(s>=60)&(s<=150)
    weight=R0[0,mask];weight=weight/weight.sum()
    means={name:float(np.sum(weight*a[0,mask])) for name,a in [('pair_z',zm),('first_derivative_z',z1),('second_derivative_z',z2)]}
    means['pair_sigma_z']=float(np.sqrt(np.sum(weight*m2[0,mask])))
    means['first_derivative_sigma_z']=float(np.sqrt(np.sum(weight*(var1[0,mask]+(z1[0,mask]-means['first_derivative_z'])**2))))
    means['second_derivative_sigma_z']=float(np.sqrt(np.sum(weight*(var2[0,mask]+(z2[0,mask]-means['second_derivative_z'])**2))))
    manifests={}
    for cap in ['NGC','SGC']:
        manifests[cap]=json.loads((root/f'{cap}_DDDR_manifest.json').read_text());manifests[cap]['total_pairs']=caps[0 if cap=='NGC' else 1]['pairs']
    audit={'status':'full_v15_NGC_SGC_all_data_all_random0_counts_completed','njack':sum(nj),'covariance_rank':int(np.linalg.matrix_rank(covariance)),'covariance_dimension':len(covariance),'pre_reconstruction':True,'catalogues':manifests,'kernel_diagnostics':means,'normalization_tests':tests,'note':'No simulation or published distance table used to generate the measurements.'}
    np.savez_compressed(out/'clustering_and_jackknife.npz',s=s,mu=mu,z=zc,all_multipoles=poles,covariance=covariance,error=error,kernels=kernels,RR_moments=R,RR_histogram=H,pair_z=zm,first_z=z1,second_z=z2,first_variance=var1,second_variance=var2,legendre_weights=leg)
    head=['s_Mpc_h'];cols=[s]
    for i,name in enumerate(['xi','dxi_dz','d2xi_dz2']):
        for j,l in enumerate(ells):head.extend([name+f'_ell{l}',name+f'_ell{l}_jk_sigma']);cols.extend([poles[0,i,j],error[i,j]])
    np.savetxt(out/'measured_multipoles.csv',np.column_stack(cols),delimiter=',',header=','.join(head),comments='')
    (out/'count_audit.json').write_text(json.dumps(audit,indent=2))
    print('REDUCTION_COMPLETE',json.dumps(audit),flush=True)
    return s,mu,zc,poles,covariance,error,kernels,leg,audit
