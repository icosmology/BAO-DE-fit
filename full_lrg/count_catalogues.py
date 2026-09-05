#!/usr/bin/env python3
"""Full, unthinned DESI DR1 v1.5 NGC/SGC pair counting.
No synthetic scientific data. Internal deterministic fixtures only test code.
All auto pairs are unordered. Each spatial deletion removes all touching pairs.
"""
import argparse, gc, hashlib, json, math, time
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from numba import njit, prange, set_num_threads

PIVOT=0.75
ZMIN,ZMAX=0.4,1.1
SMIN,DS,NS,NMU,NZ=40.,4.,30,20,70
OM=0.3153
NJK={'NGC':80,'SGC':40}


def read_catalog(path):
    import fitsio
    with fitsio.FITS(str(path)) as f:
        n=f[1].get_nrows(); parts=[]
        for lo in range(0,n,250000):
            t=f[1].read(rows=np.arange(lo,min(n,lo+250000)),columns=['RA','DEC','Z','WEIGHT','WEIGHT_FKP'])
            a=np.column_stack([t['RA'],t['DEC'],t['Z'],t['WEIGHT']*t['WEIGHT_FKP']]).astype('f8')
            ok=np.all(np.isfinite(a),axis=1)&(a[:,2]>=ZMIN)&(a[:,2]<ZMAX)&(a[:,3]>0)
            parts.append(a[ok])
    return np.ascontiguousarray(np.concatenate(parts)),n


def regions(r,n,cap):
    def xy(a):
        ra=a[:,0] if cap=='NGC' else (a[:,0]+180)%360-180
        return np.column_stack([np.deg2rad(ra),np.sin(np.deg2rad(a[:,1]))])
    pos=xy(r); num=[0]
    def split(ids,k):
        if k==1:
            lab=num[0];num[0]+=1;return {'label':lab}
        axis=int(np.argmax(np.ptp(pos[ids],axis=0)))
        ii=ids[np.argsort(pos[ids,axis],kind='stable')]
        cw=np.cumsum(r[ii,3]);left=k//2
        cut=int(np.searchsorted(cw,cw[-1]*left/k))+1
        cut=min(max(cut,1),len(ii)-1)
        val=float((pos[ii[cut-1],axis]+pos[ii[cut],axis])/2)
        return {'axis':axis,'bound':val,'left':split(ii[:cut],left),'right':split(ii[cut:],k-left)}
    tree=split(np.arange(len(r)),n)
    def label(a):
        pp=xy(a);out=np.empty(len(a),dtype='i4')
        def walk(node,ids):
            if 'label' in node:out[ids]=node['label'];return
            sel=pp[ids,node['axis']]<=node['bound']
            walk(node['left'],ids[sel]);walk(node['right'],ids[~sel])
        walk(tree,np.arange(len(a)));return out
    return tree,label


def coordinates(a,labs):
    zz=np.linspace(0,1.2,20001)
    chi=cumulative_trapezoid(2997.92458/np.sqrt(OM*(1+zz)**3+1-OM),zz,initial=0)
    rr=np.interp(a[:,2],zz,chi);ra,dec=np.deg2rad(a[:,0]),np.deg2rad(a[:,1])
    xyz=np.column_stack([rr*np.cos(dec)*np.cos(ra),rr*np.cos(dec)*np.sin(ra),rr*np.sin(dec)])
    return np.ascontiguousarray(np.column_stack([xyz,a[:,3],a[:,2]-PIVOT,labs,rr*rr]))


def cell_sort(a,origin,dims,size):
    cc=np.floor((a[:,:3]-origin)/size).astype(np.int64)
    keys=(cc[:,0]*dims[1]+cc[:,1])*dims[2]+cc[:,2]
    order=np.argsort(keys,kind='stable')
    offs=np.r_[0,np.cumsum(np.bincount(keys,minlength=int(np.prod(dims))))]
    return np.ascontiguousarray(a[order]),np.ascontiguousarray(cc[order]),offs


@njit(parallel=True,cache=True)
def counter(a,ca,b,offs,dims,auto,njk,part,parts,threads,nmom,with_hist,smin,ds,ns,nmu,nz,zmin,zmax,theta):
    sums=np.zeros((threads,njk+1,ns,nmu,nmom))
    if with_hist:hist=np.zeros((threads,njk+1,ns,nmu,nz,2))
    else:hist=np.zeros((threads,1,1,1,1,2))
    number=np.zeros(threads,np.int64)
    hi2=(smin+ds*ns)**2;lo2=smin*smin
    ilo=part*len(a)//parts;ihi=(part+1)*len(a)//parts
    coscut=math.cos(theta*math.pi/180)
    for task in prange(threads):
        first=ilo+task*(ihi-ilo)//threads;last=ilo+(task+1)*(ihi-ilo)//threads
        for i in range(first,last):
            ai=a[i];ri=int(ai[5]);cx,cy,cz=ca[i]
            for ix in range(max(0,cx-1),min(dims[0],cx+2)):
                for iy in range(max(0,cy-1),min(dims[1],cy+2)):
                    for iz in range(max(0,cz-1),min(dims[2],cz+2)):
                        key=(ix*dims[1]+iy)*dims[2]+iz
                        for j in range(offs[key],offs[key+1]):
                            if auto and j<=i:continue
                            bj=b[j]
                            ss=(bj[0]-ai[0])**2+(bj[1]-ai[1])**2+(bj[2]-ai[2])**2
                            if ss<lo2 or ss>=hi2 or ss==0:continue
                            if theta>0:
                                ct=(ai[6]+bj[6]-ss)/(2*math.sqrt(ai[6]*bj[6]))
                                if ct>coscut:continue
                            lm2=2*(ai[6]+bj[6])-ss
                            if lm2<=0:continue
                            mu=abs(bj[6]-ai[6])/math.sqrt(ss*lm2)
                            im=min(nmu-1,int(mu*nmu));ib=int((math.sqrt(ss)-smin)/ds)
                            u=(ai[4]+bj[4])/2;w=ai[3]*bj[3];rj=int(bj[5]);v=w
                            for m in range(nmom):
                                sums[task,0,ib,im,m]+=v
                                sums[task,ri+1,ib,im,m]+=v
                                if ri!=rj:sums[task,rj+1,ib,im,m]+=v
                                v*=u
                            if with_hist:
                                zb=u+PIVOT;izb=min(nz-1,max(0,int((zb-zmin)/(zmax-zmin)*nz)))
                                hist[task,0,ib,im,izb,0]+=w
                                hist[task,0,ib,im,izb,1]+=w*zb
                                hist[task,ri+1,ib,im,izb,0]+=w
                                hist[task,ri+1,ib,im,izb,1]+=w*zb
                                if ri!=rj:
                                    hist[task,rj+1,ib,im,izb,0]+=w
                                    hist[task,rj+1,ib,im,izb,1]+=w*zb
                            number[task]+=1
    return sums.sum(axis=0),hist.sum(axis=0),number.sum()


def do_count(a,b,auto,njk,part=0,parts=1,threads=4,hist=False,test=False):
    smin,ds,ns,nmu,nz=(5.,10.,8,5,7) if test else (SMIN,DS,NS,NMU,NZ)
    size=smin+ds*ns
    origin=np.minimum(a[:,:3].min(axis=0),b[:,:3].min(axis=0))-1.
    high=np.maximum(a[:,:3].max(axis=0),b[:,:3].max(axis=0))+1.
    dims=np.floor((high-origin)/size).astype(np.int64)+1
    aa,cc,oa=cell_sort(a,origin,dims,size)
    if auto:bb,ob=aa,oa
    else:bb,cb,ob=cell_sort(b,origin,dims,size)
    return counter(aa,cc,bb,ob,dims,auto,njk,part,parts,threads,7 if hist else 3,hist,smin,ds,ns,nmu,nz,ZMIN,ZMAX,0. if test else .05)


def self_test():
    t=np.arange(87,dtype=float)
    def fixture(v):
        xyz=np.column_stack([110+29*np.sin(v*.91),90+32*np.cos(v*.37),130+26*np.sin(v*.61)])
        return np.column_stack([xyz,.7+.3*np.cos(v*.2)**2,.25*np.sin(v*.77),v.astype(int)%4,np.sum(xyz*xyz,axis=1)])
    aa,bb=fixture(t[:39]),fixture(t+.3);results=[]
    for name,a,b,auto in [('DD',aa,aa,True),('DR',aa,bb,False),('RR',bb,bb,True)]:
        got,hh,n=do_count(a,b,auto,4,hist=True,test=True)
        want=np.zeros_like(got);h=np.zeros_like(hh);nb=0
        for i in range(len(a)):
            for j in range(i+1 if auto else 0,len(b)):
                sep=np.linalg.norm(b[j,:3]-a[i,:3])
                if not 5<=sep<85:continue
                los=a[i,:3]+b[j,:3]
                mu=abs((b[j,:3]-a[i,:3])@los)/(sep*np.linalg.norm(los))
                ib=int((sep-5)/10);im=min(4,int(mu*5));u=(a[i,4]+b[j,4])/2;w=a[i,3]*b[j,3]
                zb=u+PIVOT;iz=min(6,max(0,int((zb-ZMIN)/(ZMAX-ZMIN)*7)))
                for lab in {0,int(a[i,5])+1,int(b[j,5])+1}:
                    want[lab,ib,im]+=w*u**np.arange(7)
                    h[lab,ib,im,iz]+=[w,w*zb]
                nb+=1
        np.testing.assert_allclose(got,want,rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(hh,h,rtol=1e-11,atol=1e-11)
        assert n==nb
        # Distributed row ranges must exactly partition all unordered/cross pairs.
        g0,h0,n0=do_count(a,b,auto,4,part=0,parts=2,hist=True,test=True)
        g1,h1,n1=do_count(a,b,auto,4,part=1,parts=2,hist=True,test=True)
        np.testing.assert_allclose(g0+g1,got,rtol=1e-11,atol=1e-11)
        assert n0+n1==n
        results.append({'counter':name,'pairs':int(n),'max_abs_error':float(np.max(abs(got-want)))})
    print(json.dumps({'test_status':'PASS','checks':results},indent=2),flush=True)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--cap',choices=['NGC','SGC']);p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=8)
    p.add_argument('--input',default='catalogues');p.add_argument('--output',default='partial');p.add_argument('--self-test',action='store_true');p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();set_num_threads(args.threads)
    if args.self_test:self_test();return
    cap=args.cap;out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    paths=[next(Path(args.input).rglob(f'LRG_{cap}_clustering.dat.fits')),next(Path(args.input).rglob(f'LRG_{cap}_0_clustering.ran.fits'))]
    data,nd=read_catalog(paths[0]);random,nr=read_catalog(paths[1])
    audit={'version':'DR1-v1.5','cap':cap,'input_data_rows':nd,'input_random_rows':nr,'selected_data_rows':len(data),'selected_random_rows':len(random),'data_fraction':1.,'random_fraction':1.,'njack':NJK[cap],'part':args.part,'parts':args.parts,'pre_reconstruction':True,'theta_cut_degrees':.05,'smin':SMIN,'ds':DS,'ns':NS,'nmu':NMU,'nz':NZ,'zmin':ZMIN,'zmax':ZMAX,'pivot':PIVOT,'omega_m':OM,'sha256':{}}
    for path in paths:
        with path.open('rb') as f:audit['sha256'][path.name]=hashlib.file_digest(f,'sha256').hexdigest()
    print(json.dumps(audit,indent=2),flush=True)
    tree,label=regions(random,NJK[cap],cap);dl,rl=label(data),label(random)
    norms=[]
    for cat,labels in [(data,dl),(random,rl)]:
        w=cat[:,3];a=np.array([[w.sum(),(w*w).sum()]])
        reg=np.column_stack([np.bincount(labels,weights=w,minlength=NJK[cap]),np.bincount(labels,weights=w*w,minlength=NJK[cap])])
        norms.append(np.vstack([a,a-reg]))
    d,r=coordinates(data,dl),coordinates(random,rl)
    del data,random,dl,rl;gc.collect()
    for name,a,b,auto in [('DD',d,d,True),('DR',d,r,False),('RR',r,r,True)]:
        start=time.time();print(f'START {cap} part {args.part} {name}',flush=True)
        moments,hist,number=do_count(a,b,auto,NJK[cap],part=args.part,parts=args.parts,threads=args.threads,hist=name=='RR')
        fn=out/f'{cap}_{name}_part{args.part:02d}.npz'
        np.savez_compressed(fn,moments=moments,hist=hist,data_norms=norms[0],random_norms=norms[1])
        audit[name]={'accepted_pairs':int(number),'elapsed_seconds':time.time()-start}
        print(f'END {name}: {number:,} pairs; {time.time()-start:.2f}s; {fn.stat().st_size} bytes',flush=True)
        del moments,hist;gc.collect()
    (out/f'{cap}_part{args.part:02d}_audit.json').write_text(json.dumps(audit,indent=2))
    (out/f'{cap}_part{args.part:02d}_regions.json').write_text(json.dumps(tree))

if __name__=='__main__':main()
