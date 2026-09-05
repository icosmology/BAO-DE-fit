"""CAMB-based, BAO-phase-only light-cone model. No official distance points enter.
The no-wiggle part remains in fiducial coordinates; broadband and redshift
amplitude evolution are nuisance parameters. Distances are independently free.
"""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline, LSQUnivariateSpline
from scipy.special import spherical_jn, eval_legendre
from scipy.integrate import cumulative_trapezoid
from numpy.polynomial.legendre import leggauss,Legendre
from numba import njit

SP=np.array([2.,3.,4.,5.,6.,7.]);SA=np.array([4.,6.,8.,10.,12.,15.])
RG=np.arange(20.,260.01,.25)
OM=.3153;HH=.6736;ZP=.75;DZ=.35


def prepare_template(path):
    path=Path(path)
    if path.exists():return dict(np.load(path))
    import camb
    pars=camb.CAMBparams();obh2=.02237;onuh2=.06/93.14
    pars.set_cosmology(H0=100*HH,ombh2=obh2,omch2=OM*HH**2-obh2-onuh2,mnu=.06,omk=0,tau=.054)
    pars.InitPower.set_params(As=2.1e-9,ns=.9649)
    pars.set_matter_power(redshifts=[ZP],kmax=2.)
    result=camb.get_results(pars);kh,zz,pp=result.get_matter_power_spectrum(minkh=1e-4,maxkh=2.,npoints=6000)
    k=np.linspace(.0002,1.2,3500);pk=np.exp(np.interp(np.log(k),np.log(kh),np.log(pp[0])))
    # Analytic Eisenstein-Hu smooth reference, then spline-correct its slowly varying shape.
    omhh=OM*HH**2;fb=obh2/omhh;sound=44.5*np.log(9.83/omhh)/np.sqrt(1+10*obh2**.75)
    ag=1-.328*np.log(431*omhh)*fb+.38*np.log(22.3*omhh)*fb*fb
    gamma=OM*HH*(ag+(1-ag)/(1+(.43*k*HH*sound)**4));q=k/gamma*(2.7255/2.7)**2
    l0=np.log(2*np.e+1.8*q);c0=14.2+731/(1+62.5*q);tf=l0/(l0+c0*q*q)
    pref=k**.9649*tf*tf;pref*=np.median(pk[(k>.04)&(k<.2)]/pref[(k>.04)&(k<.2)])
    knots=np.arange(.055,1.15,.07)
    smooth=LSQUnivariateSpline(k,np.log(pk/pref),knots,k=3)
    pnw=pref*np.exp(smooth(k));pw=pk-pnw
    kmid=np.gradient(k);x=k[None,:]*RG[:,None]
    hankel=np.array([(-1)**li*spherical_jn(2*li,x)*(k*k*kmid/(2*np.pi**2))[None,:] for li in range(5)])
    mus,mw=leggauss(64);leg=np.array([(4*li+1)/2*eval_legendre(2*li,mus)*mw for li in range(5)])
    powers=np.array([mus**(2*n) for n in range(3)])
    # A smooth high-k convergence factor affects only the non-BAO baseline.
    pnwm=pnw*np.exp(-k*k)
    nspec=np.einsum('lm,nm,k->nlk',leg,powers,pnwm)
    nw=np.einsum('lrk,nlk->nlr',hankel,nspec)
    wig=np.empty((len(SP),len(SA),3,5,len(RG)))
    for i,sp in enumerate(SP):
        for j,sa in enumerate(SA):
            damp=np.exp(-.5*k[:,None]**2*(sp*sp*(1-mus**2)+sa*sa*mus**2))
            spec=np.einsum('lm,nm,km,k->nlk',leg,powers,damp,pw)
            wig[i,j]=np.einsum('lrk,nlk->nlr',hankel,spec)
    rd=float(result.get_derived_params()['rdrag'])
    np.savez_compressed(path,sp=SP,sa=SA,r=RG,nw=nw,wig=wig,k=k,pk=pk,pnw=pnw,rd=rd)
    print('CAMB template built; sound horizon',rd,flush=True)
    return dict(sp=SP,sa=SA,r=RG,nw=nw,wig=wig,k=k,pk=pk,pnw=pnw,rd=np.array(rd))


@njit(cache=True)
def legendre_even(mu):
    out=np.zeros(5);out[0]=1.;p0=1.;p1=mu
    for ell in range(2,9):
        p=((2*ell-1)*mu*p1-(ell-1)*p0)/ell
        if ell%2==0:out[ell//2]=p
        p0=p1;p1=p
    return out


@njit(cache=True)
def evaluate_columns(geom,spv,sav,spgrid,sagrid,rg,nw,wig,s,zz,weights,legfac,quadratic):
    # Output dimensions: [redshift order, observed ell, s, 3 angular x 3 redshift amplitudes].
    nd=weights.shape[0];ns=len(s);nm=zz.shape[1];nz=zz.shape[2]
    out=np.zeros((nd,2,ns,9))
    isp=min(len(spgrid)-2,max(0,np.searchsorted(spgrid,spv)-1));isa=min(len(sagrid)-2,max(0,np.searchsorted(sagrid,sav)-1))
    fp=(spv-spgrid[isp])/(spgrid[isp+1]-spgrid[isp]);fa=(sav-sagrid[isa])/(sagrid[isa+1]-sagrid[isa])
    table=(1-fp)*(1-fa)*wig[isp,isa]+fp*(1-fa)*wig[isp+1,isa]+(1-fp)*fa*wig[isp,isa+1]+fp*fa*wig[isp+1,isa+1]
    gn=np.array([-np.sqrt(3./5),0.,np.sqrt(3./5)]);gw=np.array([5./9,8./9,5./9])
    for ib in range(ns):
        for im in range(nm):
            mu=(im+.5)/nm;lf=legendre_even(mu)
            for iz in range(nz):
                z=zz[ib,im,iz];v=(z-ZP)/DZ
                ap0=geom[0]+geom[2]*v;ar0=geom[1]+geom[3]*v
                if quadratic:ap0+=.5*geom[4]*v*v;ar0+=.5*geom[5]*v*v
                ap=np.exp(ap0);ar=np.exp(ar0)
                scale=np.sqrt(ap*ap*(1-mu*mu)+ar*ar*mu*mu);mup=ar*mu/scale;lp=legendre_even(mup)
                base=np.zeros(3);den=0.
                # Integrate the 8 Mpc/h separation bin instead of a point template.
                for ig in range(3):
                    sr=s[ib]+4*gn[ig];w=gw[ig]*sr*sr;den+=w;st=sr*scale
                    q=(sr-rg[0])/(rg[1]-rg[0]);ir=min(len(rg)-2,max(0,int(q)));fr=q-ir
                    q2=(st-rg[0])/(rg[1]-rg[0]);jr=min(len(rg)-2,max(0,int(q2)));ft=q2-jr
                    for n in range(3):
                        vv=0.
                        for li in range(5):
                            vv+=((1-fr)*nw[n,li,ir]+fr*nw[n,li,ir+1])*lf[li]
                            vv+=((1-ft)*table[n,li,jr]+ft*table[n,li,jr+1])*lp[li]
                        base[n]+=w*vv
                base/=den
                for order in range(nd):
                    w0=weights[order,ib,im,iz]*s[ib]*s[ib]
                    for ell in range(2):
                        w1=w0*legfac[ell,im]
                        for n in range(3):
                            amp=w1*base[n]
                            out[order,ell,ib,3*n]+=amp
                            out[order,ell,ib,3*n+1]+=amp*v
                            out[order,ell,ib,3*n+2]+=amp*.5*v*v
    return out.reshape(nd*2*ns,9)


class Lightcone:
    def __init__(self,measurement,template,orders=2,quadratic=False,realization=0,fitmask=None):
        self.quadratic=quadratic;self.orders=orders;self.t=template
        s=measurement['s'];sel=(s>=52)&(s<=148) if fitmask is None else fitmask
        self.sel=sel;self.s=s[sel];H=measurement['H'][realization,sel];R=measurement['R'][realization,sel]
        self.z=np.divide(H[...,1],H[...,0],out=np.broadcast_to(np.linspace(.405,1.095,70),H[...,0].shape).copy(),where=H[...,0]>0)
        h=H[...,0]/R[...,0,None];mean=measurement['mean_u'][realization,sel];c=measurement['central'][realization,sel];m2=c[...,2];m3=c[...,3];nn=measurement['second_norm'][realization,sel]
        u=self.z-ZP-mean[...,None]
        q1=u/m2[...,None]
        q2=2*(u*u-m2[...,None]-(m3/m2)[...,None]*u)/nn[...,None]
        # Restore the measured within-redshift-bin variance in the quadratic weight.
        # Only xi's interpolation is approximated within the 0.01-wide redshift bins.
        missing=m2-np.sum(h*u*u,axis=-1)
        q2+=2*missing[...,None]/nn[...,None]
        self.weights=np.ascontiguousarray(np.stack([h,h*q1,h*q2])[:orders])
        self.z=np.ascontiguousarray(self.z)
        me=np.linspace(0,1,h.shape[1]+1)
        self.leg=np.array([(2*ell+1)*np.diff(Legendre.basis(ell).integ()(me)) for ell in [0,2]])
        self.bb=np.zeros((orders*2*len(self.s),orders*2*3))
        for order in range(orders):
            for ell in range(2):
                sl=slice((order*2+ell)*len(self.s),(order*2+ell+1)*len(self.s))
                for power in range(3):self.bb[sl,(order*2+ell)*3+power]=((self.s-100)/50)**power*20.
    def matrix(self,p):
        ng=6 if self.quadratic else 4
        model=evaluate_columns(np.ascontiguousarray(p[:ng]),p[ng],p[ng+1],self.t['sp'],self.t['sa'],self.t['r'],self.t['nw'],self.t['wig'],self.s,self.z,self.weights,self.leg,self.quadratic)
        return np.column_stack([model,self.bb])
    def select(self,all_poles):
        return (all_poles[:,:self.orders,:2][:,:,:,self.sel]*self.s[None,None,None,:]**2).reshape(len(all_poles),-1)


def distance_history(p,z,rd,quadratic=False):
    z=np.asarray(z);v=(z-ZP)/DZ
    zz=np.linspace(0,max(1.2,float(np.max(z))),10001)
    yy=299792.458/(100*HH*rd)/np.sqrt(OM*(1+zz)**3+1-OM);xx=cumulative_trapezoid(yy,zz,initial=0)
    Xf=np.interp(z,zz,xx);Yf=np.interp(z,zz,yy)
    gY=-1.5*OM*(1+z)**2/(OM*(1+z)**3+1-OM)
    gYprime=-3*OM*(1+z)/(OM*(1+z)**3+1-OM)+4.5*OM**2*(1+z)**4/(OM*(1+z)**3+1-OM)**2
    gp=(p[2]+(p[4]*v if quadratic else 0))/DZ
    ga=(p[3]+(p[5]*v if quadratic else 0))/DZ
    ap=np.exp(p[0]+p[2]*v+(.5*p[4]*v*v if quadratic else 0))
    aa=np.exp(p[1]+p[3]*v+(.5*p[5]*v*v if quadratic else 0))
    X=ap*Xf;Y=aa*Yf
    X1=ap*(Yf+Xf*gp);Y1=Y*(gY+ga)
    X2=ap*(Yf*gY+2*Yf*gp+Xf*(gp*gp+(p[4]/DZ**2 if quadratic else 0)))
    Y2=Y*((gY+ga)**2+gYprime+(p[5]/DZ**2 if quadratic else 0))
    return np.array([X,Y,X1,Y1,X2,Y2])
