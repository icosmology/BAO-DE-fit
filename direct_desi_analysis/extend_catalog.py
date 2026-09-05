#!/usr/bin/env python3
"""Extend real DESI counts to seven redshift moments and random-pair z histograms."""
import sys, importlib.util, json, math, hashlib, time
from pathlib import Path
import numpy as np
from numba import set_num_threads
ROOT=Path(__file__).resolve().parent
basepath=ROOT/'measure_dxi_jackknife.py'
src=basepath.read_text()
src=src.replace('shards, n_moments, cos_theta_cut):','shards, n_moments, cos_theta_cut, hist_nz):')
src=src.replace('pair_n = np.zeros(shards, np.int64)', 'pair_n = np.zeros(shards, np.int64)\n    hist = np.zeros((shards, njack+1, ns, nmu, hist_nz))')
needle='                            for m in range(n_moments):'
add='''                            if hist_nz > 0:
                                zh = min(hist_nz-1, max(0, int((u+0.35)/0.70*hist_nz)))
                                hist[task, 0, ib, im, zh] += value
                                hist[task, ri+1, ib, im, zh] += value
                                if ri != rj: hist[task, rj+1, ib, im, zh] += value
'''
assert src.count(needle)==1
src=src.replace(needle,add+needle)
src=src.replace('return sums.sum(axis=0), pair_n.sum()', 'return sums.sum(axis=0), pair_n.sum(), hist.sum(axis=0)')
src=src.replace("cfg['shards'], 4,", "cfg['shards'], cfg.get('nmom',7),")
src=src.replace("math.cos(math.radians(cfg['theta_cut'])))", "math.cos(math.radians(cfg['theta_cut'])), cfg.get('hist_nz',0))")
gen=ROOT/'extended_counter.py';gen.write_text(src)
spec=importlib.util.spec_from_file_location('extended_counter',gen); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
set_num_threads(4)

def test():
    cfg=dict(smin=5.,ds=10.,ns=8,nmu=5,njack=4,shards=4,theta_cut=0.,nmom=7,hist_nz=70)
    t=np.arange(100.)
    def fixture(t):
        xyz=np.column_stack([100+35*np.sin(t*1.17),80+33*np.cos(t*.71),110+39*np.sin(t*.37)])
        return np.column_stack([xyz,.8+.2*np.sin(t*.31)**2,-.15+.3*(t%17)/16,t.astype(int)%4,np.sum(xyz**2,axis=1)])
    a,b=fixture(t[:61]),fixture(t+.2); tests=[]
    for name,d,r,auto in [('DD',a,a,True),('DR',a,b,False),('RR',b,b,True)]:
        got,ngot,hist=mod.count(d,r,auto,cfg)
        expected=np.zeros_like(got); nexpect=0
        for i in range(len(d)):
            for j in range(i+1 if auto else 0,len(r)):
                sep=r[j,:3]-d[i,:3];s=np.linalg.norm(sep)
                if not 5<=s<85:continue
                los=d[i,:3]+r[j,:3];mu=abs(sep@los)/(s*np.linalg.norm(los));ib=int((s-5)/10);im=min(4,int(mu*5))
                vals=d[i,3]*r[j,3]*((d[i,4]+r[j,4])/2)**np.arange(7)
                expected[0,ib,im]+=vals
                for k in set([int(d[i,5])+1,int(r[j,5])+1]):expected[k,ib,im]+=vals
                nexpect+=1
        np.testing.assert_allclose(got,expected,rtol=3e-12,atol=3e-10)
        np.testing.assert_allclose(hist.sum(-1),got[...,0],rtol=3e-12,atol=3e-10)
        assert ngot==nexpect
        tests.append(dict(counter=name,pairs=int(ngot),max_abs_error=float(abs(got-expected).max())))
    return tests

def main():
    out=Path('results_second/NGC');out.mkdir(parents=True,exist_ok=True)
    tests=test(); (out/'unit_tests.json').write_text(json.dumps(tests,indent=2))
    cfg=dict(smin=40.,ds=4.,ns=30,nmu=20,njack=40,shards=8,theta_cut=.05,nmom=7,hist_nz=0,pivot=.75,omega_m=.3153,zmin=.4,zmax=1.1)
    paths=[Path('catalogs/LRG_NGC_clustering.dat.fits'),Path('catalogs/LRG_NGC_0_clustering.ran.fits')]
    d,dm=mod.read_fits(paths[0],.4,1.1,1.,73429)
    r,rm=mod.read_fits(paths[1],.4,1.1,.15,73429)
    tree,label=mod.angular_regions(r,40,'NGC');dl,rl=label(d),label(r)
    (out/'regions.json').write_text(json.dumps(tree))
    np.savez_compressed(out/'catalog_redshift_hist.npz',edges=np.linspace(.4,1.1,141),data_weight=np.histogram(d[:,2],np.linspace(.4,1.1,141),weights=d[:,3])[0],random_weight=np.histogram(r[:,2],np.linspace(.4,1.1,141),weights=r[:,3])[0])
    d=mod.to_xyz(d,dl,.3153,.75,1.1);r=mod.to_xyz(r,rl,.3153,.75,1.1)
    wd,wd2=mod.weight_norms(d,40);wr,wr2=mod.weight_norms(r,40)
    np.savez(out/'weight_normalizations.npz',wd=wd,wd2=wd2,wr=wr,wr2=wr2)
    meta=dict(configuration=cfg,data=dm,random=rm,data_fraction=1.,random_fraction=.15,status='running',pre_reconstruction=True,provenance=[])
    for p in paths:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
        meta['provenance'].append(dict(file=p.name,bytes=p.stat().st_size,sha256=h.hexdigest(),url=mod.BASES[0]+p.name))
    for name,a,b,auto in [('DD',d,d,True),('DR',d,r,False),('RR',r,r,True)]:
        cfg['hist_nz']=140 if name=='RR' else 0
        t=time.time(); m,n,h=mod.count(a,b,auto,cfg)
        np.save(out/(name+'_moments.npy'),m)
        if name=='RR':np.savez_compressed(out/'RR_redshift_hist.npz',counts=h,edges=np.linspace(.4,1.1,141))
        meta[name]=dict(pairs=int(n),seconds=time.time()-t)
        print(name,n,time.time()-t,flush=True)
    meta['status']='completed_real_catalog_second_order';(out/'run_manifest.json').write_text(json.dumps(meta,indent=2))
    import camb
    pars=camb.CAMBparams();pars.set_cosmology(H0=67.36,ombh2=.02237,omch2=.12,mnu=.06,omk=0,tau=.0544)
    pars.InitPower.set_params(As=2.1e-9,ns=.9649)
    pars.set_matter_power(redshifts=[0.75],kmax=3.)
    results=camb.get_results(pars); k,z,p=results.get_matter_power_spectrum(minkh=1e-4,maxkh=3.,npoints=4096)
    np.savez(out/'camb_fiducial_template.npz',k_h_Mpc=k,P_Mpc3_h3=p[0],redshift=z,rd_Mpc=results.get_derived_params()['rdrag'])
    import shutil
    shutil.copy2(__file__,out/'extend_catalog.py');shutil.copy2(gen,out/'extended_counter.py');shutil.copy2(basepath,out/'original_counter.py')
    print(json.dumps(meta),flush=True)
if __name__=='__main__':
    if '--self-test' in sys.argv:print(json.dumps(test(),indent=2))
    else:main()
