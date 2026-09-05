#!/usr/bin/env python3
"""DESI DR1 v1.5, no thinning. Distributed exact pair moments and spatial JK.
All auto pairs are unordered. Histograms/moments include every pair touching
one region, so full-minus-touch gives an exact delete-one recount.
"""
import os, json, time, math, hashlib, argparse
from pathlib import Path
import numpy as np
import fitsio
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from numba import njit, prange, set_num_threads

SIZES={'NGC':(143196480,985916160),'SGC':(64272960,520781760)}
CFG={'zmin':0.4,'zmax':1.1,'pivot':0.75,'smin':40.,'ds':6.,'ns':20,'nmu':20,'nz':70,'om':0.3153,'theta':0.05,'threads':4}

def readcat(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):digest.update(b)
    with fitsio.FITS(str(path)) as f:
        n=f[1].get_nrows(); chunks=[]; discarded=0
        for lo in range(0,n,250000):
            t=f[1].read(rows=np.arange(lo,min(lo+250000,n)),columns=['RA','DEC','Z','WEIGHT','WEIGHT_FKP'])
            w=np.asarray(t['WEIGHT'],float)*np.asarray(t['WEIGHT_FKP'],float)
            good=(t['Z']>=CFG['zmin'])&(t['Z']<CFG['zmax'])&np.isfinite(w)&(w>0)&np.isfinite(t['RA'])&np.isfinite(t['DEC'])
            chunks.append(np.column_stack([t['RA'][good],t['DEC'][good],t['Z'][good],w[good]])); discarded+=int((~good).sum())
    a=np.ascontiguousarray(np.concatenate(chunks),dtype='f8')
    return a,{'file':path.name,'bytes':path.stat().st_size,'sha256':digest.hexdigest(),'input_rows':n,'selected_rows':len(a),'selection_excluded':discarded,'thinning_fraction':1.0}

def regions(random,njack,cap):
    def sky(a):
        ra=a[:,0] if cap=='NGC' else (a[:,0]+180)%360-180
        return np.column_stack([np.deg2rad(ra),np.sin(np.deg2rad(a[:,1]))])
    p=sky(random); nextlabel=[0]
    def split(ids,n):
        if n==1:
            lab=nextlabel[0];nextlabel[0]+=1;return {'label':lab}
        axis=int(np.argmax(np.ptp(p[ids],axis=0)))
        ids=ids[np.argsort(p[ids,axis],kind='stable')]
        c=np.cumsum(random[ids,3]); nl=n//2
        k=int(np.searchsorted(c,c[-1]*nl/n))+1;k=min(max(1,k),len(ids)-1)
        edge=float(0.5*(p[ids[k-1],axis]+p[ids[k],axis]))
        return {'axis':axis,'edge':edge,'left':split(ids[:k],nl),'right':split(ids[k:],n-nl)}
    tree=split(np.arange(len(random)),njack)
    def labels(a):
        p2=sky(a);out=np.empty(len(a),dtype='i4')
        def visit(t,ids):
            if 'label' in t:out[ids]=t['label'];return
            m=p2[ids,t['axis']]<=t['edge'];visit(t['left'],ids[m]);visit(t['right'],ids[~m])
        visit(tree,np.arange(len(a)));return out
    return tree,labels

def xyz(a,labels):
    zz=np.linspace(0,1.2,24001);ch=cumulative_trapezoid(2997.92458/np.sqrt(CFG['om']*(1+zz)**3+1-CFG['om']),zz,initial=0)
    r=PchipInterpolator(zz,ch)(a[:,2]);ra=np.deg2rad(a[:,0]);dc=np.deg2rad(a[:,1])
    return np.ascontiguousarray(np.column_stack([r*np.cos(dc)*np.cos(ra),r*np.cos(dc)*np.sin(ra),r*np.sin(dc),a[:,3],a[:,2]-CFG['pivot'],labels,r*r]))

def order_cells(a,origin,dims,cell):
    c=np.floor((a[:,:3]-origin)/cell).astype('i8'); keys=(c[:,0]*dims[1]+c[:,1])*dims[2]+c[:,2]
    order=np.argsort(keys,kind='stable');n=np.bincount(keys,minlength=int(np.prod(dims)));offset=np.r_[0,np.cumsum(n)]
    return np.ascontiguousarray(a[order]),np.ascontiguousarray(c[order]),offset

@njit(parallel=True,cache=True)
def traverse(a,ca,b,off,dims,auto,lo,hi,nj,ns,nmu,nmom,nz,smin,ds,pivot,zmin,zmax,costheta,nt):
    sums=np.zeros((nt,nj+1,ns,nmu,nmom));hist=np.zeros((nt,nj+1,ns,nmu,nz));counts=np.zeros(nt,np.int64)
    rmin2=smin*smin;rmax2=(smin+ns*ds)**2
    for worker in prange(nt):
        for i in range(lo+(hi-lo)*worker//nt,lo+(hi-lo)*(worker+1)//nt):
            ai=a[i]; ri=int(ai[5]);cx,cy,cz=ca[i]
            for ix in range(max(0,cx-1),min(dims[0],cx+2)):
                for iy in range(max(0,cy-1),min(dims[1],cy+2)):
                    for iz in range(max(0,cz-1),min(dims[2],cz+2)):
                        cell=(ix*dims[1]+iy)*dims[2]+iz
                        for j in range(off[cell],off[cell+1]):
                            if auto and j<=i:continue
                            bj=b[j];r2=(ai[0]-bj[0])**2+(ai[1]-bj[1])**2+(ai[2]-bj[2])**2
                            if r2<rmin2 or r2>=rmax2 or r2==0:continue
                            if costheta<1 and (ai[6]+bj[6]-r2)/(2*math.sqrt(ai[6]*bj[6]))>costheta:continue
                            mid2=2*(ai[6]+bj[6])-r2
                            if mid2<=0:continue
                            mu=abs(ai[6]-bj[6])/math.sqrt(r2*mid2)
                            im=min(nmu-1,int(mu*nmu));ib=int((math.sqrt(r2)-smin)/ds)
                            u=(ai[4]+bj[4])/2; val=ai[3]*bj[3];rj=int(bj[5])
                            if nz:
                                izz=min(nz-1,max(0,int((u+pivot-zmin)/(zmax-zmin)*nz)))
                                hist[worker,0,ib,im,izz]+=val;hist[worker,ri+1,ib,im,izz]+=val
                                if ri!=rj:hist[worker,rj+1,ib,im,izz]+=val
                            for m in range(nmom):
                                sums[worker,0,ib,im,m]+=val;sums[worker,ri+1,ib,im,m]+=val
                                if ri!=rj:sums[worker,rj+1,ib,im,m]+=val
                                val*=u
                            counts[worker]+=1
    return sums.sum(axis=0),hist.sum(axis=0),counts.sum()

def count(a,b,auto,nj,nmom,nz,part=0,nparts=1,cfg=None):
    c=CFG if cfg is None else cfg;rmax=c['smin']+c['ns']*c['ds']
    origin=np.minimum(a[:,:3].min(axis=0),b[:,:3].min(axis=0))-0.001
    mx=np.maximum(a[:,:3].max(axis=0),b[:,:3].max(axis=0))+0.001
    dims=np.floor((mx-origin)/rmax).astype('i8')+1
    aa,cc,oa=order_cells(a,origin,dims,rmax)
    if auto:bb,ob=aa,oa
    else:bb,_,ob=order_cells(b,origin,dims,rmax)
    return traverse(aa,cc,bb,ob,dims,auto,len(aa)*part//nparts,len(aa)*(part+1)//nparts,nj,c['ns'],c['nmu'],nmom,nz,c['smin'],c['ds'],c['pivot'],c['zmin'],c['zmax'],math.cos(math.radians(c['theta'])),c['threads'])

def selftest():
    rng=np.random.default_rng(98242);cfg=dict(CFG,smin=2.,ds=10.,ns=8,nmu=5,nz=7,threads=2,theta=0.)
    def fixture(n):
        p=rng.normal(size=(n,3))*35+[150,120,200]
        return np.column_stack([p,rng.uniform(.5,1.4,n),rng.uniform(-.3,.3,n),rng.integers(0,4,n),np.sum(p*p,axis=1)])
    a=fixture(83);b=fixture(99);report=[]
    for name,x,y,auto in [('DD',a,a,True),('DR',a,b,False),('RR',b,b,True)]:
        got,hh,nn=count(x,y,auto,4,7,7,cfg=cfg);expect=np.zeros_like(got);eh=np.zeros_like(hh);ne=0
        for i in range(len(x)):
            for j in range(i+1 if auto else 0,len(y)):
                v=x[i,:3]-y[j,:3];r=np.linalg.norm(v)
                if not 2<=r<82:continue
                mid=x[i,:3]+y[j,:3];mu=abs(v@mid)/(r*np.linalg.norm(mid))
                ib=int((r-2)/10);im=min(4,int(mu*5));u=(x[i,4]+y[j,4])/2;w=x[i,3]*y[j,3]
                ids=set([0,int(x[i,5])+1,int(y[j,5])+1]);iz=min(6,max(0,int((u+.75-.4)/.7*7)))
                for rr in ids:expect[rr,ib,im]+=w*u**np.arange(7);eh[rr,ib,im,iz]+=w
                ne+=1
        assert nn==ne
        np.testing.assert_allclose(got,expect,atol=2e-10,rtol=2e-12);np.testing.assert_allclose(hh,eh,atol=2e-10,rtol=2e-12)
        for k in range(4):
            xa=x[x[:,5]!=k];ya=y[y[:,5]!=k];drop,dh,dn=count(xa,ya,auto,4,7,7,cfg=cfg)
            np.testing.assert_allclose(drop[0],got[0]-got[k+1],atol=2e-10,rtol=2e-12)
        report.append({'counter':name,'pairs':int(nn),'max_abs_error':float(np.max(abs(got-expect)))})
    print('SELF_TEST',json.dumps(report),flush=True)
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cap',choices=['NGC','SGC']);ap.add_argument('--task',default='DDDR');ap.add_argument('--input',default='catalogues');ap.add_argument('--output',default='out');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    set_num_threads(4)
    if args.self_test:selftest();return
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);root=Path(args.input);cap=args.cap;nj=80 if cap=='NGC' else 40
    dp=next(root.rglob('LRG_'+cap+'_clustering.dat.fits'));rp=next(root.rglob('LRG_'+cap+'_0_clustering.ran.fits'))
    assert (dp.stat().st_size,rp.stat().st_size)==SIZES[cap]
    d,dm=readcat(dp);r,rm=readcat(rp);tree,label=regions(r,nj,cap);dl=label(d);rl=label(r)
    norms=np.array([[d[:,3].sum(),np.square(d[:,3]).sum(),r[:,3].sum(),np.square(r[:,3]).sum()]]+[[d[dl==k,3].sum(),np.square(d[dl==k,3]).sum(),r[rl==k,3].sum(),np.square(r[rl==k,3]).sum()] for k in range(nj)])
    meta={'cap':cap,'task':args.task,'configuration':dict(CFG,njack=nj),'data':dm,'random':rm,'pre_reconstruction':True,'tests':'brute_force_and_every_jackknife_recount_passed'}
    print('CATALOGUES',json.dumps(meta),flush=True)
    a=xyz(d,dl);b=xyz(r,rl);del d,r,dl,rl
    tasks=[('DD',a,a,True,3,0,0,1),('DR',a,b,False,3,0,0,1)] if args.task=='DDDR' else [('RR',b,b,True,7,70,int(args.task[2:]),4)]
    for name,c1,c2,auto,nmom,nz,part,npart in tasks:
        t=time.time();mom,hist,npair=count(c1,c2,auto,nj,nmom,nz,part,npart)
        np.savez_compressed(out/f'{cap}_{args.task}_{name}.npz',moments=mom,hist=hist,norms=norms,pairs=npair,cap=cap,task=args.task)
        meta[name]={'pairs':int(npair),'seconds':time.time()-t};print('COUNT_COMPLETE',name,json.dumps(meta[name]),flush=True)
    (out/f'{cap}_{args.task}_manifest.json').write_text(json.dumps(meta,indent=2));(out/f'{cap}_{args.task}_regions.json').write_text(json.dumps(tree))
    print('JOB_COMPLETED',cap,args.task,flush=True)
if __name__=='__main__':main()
