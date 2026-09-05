import math, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.special import spherical_jn, eval_legendre
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular
from numba import njit
from reduce import jk_cov

@njit(cache=True,nogil=True)
def evaluate_table(s,mu,ap,al,beta,sigma,table,sg,r0,dr):
    out=np.zeros((len(s),len(mu),len(ap)))
    for iz in range(len(ap)):
        j=min(len(sg)-2,max(0,int((sigma[iz]-sg[0])/(sg[1]-sg[0]))));t=(sigma[iz]-sg[j])/(sg[j+1]-sg[j])
        for i in range(len(s)):
            for im in range(len(mu)):
                fac=math.sqrt(ap[iz]**2*(1-mu[im]**2)+al[iz]**2*mu[im]**2);r=s[i]*fac;m=al[iz]*mu[im]/fac
                ir=min(table.shape[-1]-2,max(0,int((r-r0)/dr)));tr=(r-r0-ir*dr)/dr
                p0=1.;p2=(3*m*m-1)/2;p4=(35*m**4-30*m*m+3)/8
                val=0.
                for ell in range(3):
                    v=0.
                    for power in range(3):
                        low=(1-tr)*table[j,power,ell,ir]+tr*table[j,power,ell,ir+1]
                        high=(1-tr)*table[j+1,power,ell,ir]+tr*table[j+1,power,ell,ir+1]
                        coeff=1. if power==0 else (2*beta[iz] if power==1 else beta[iz]**2)
                        v+=coeff*((1-t)*low+t*high)
                    val+=v*(p0 if ell==0 else (p2 if ell==1 else p4))
                out[i,im,iz]=val
    return out

def make_template(out):
    import camb
    h=.6736;om=.3153;obh2=.02237;onuh2=.06/93.14
    p=camb.CAMBparams();p.set_cosmology(H0=h*100,ombh2=obh2,omch2=om*h*h-obh2-onuh2,mnu=.06,omk=0,tau=.0544)
    p.InitPower.set_params(As=2.1e-9,ns=.9649);p.set_matter_power(redshifts=[.75],kmax=2.0);p.NonLinear=camb.model.NonLinear_none
    res=camb.get_results(p);kh,zz,pk=res.get_matter_power_spectrum(minkh=1e-4,maxkh=1.5,npoints=5000);pl=pk[0]
    rd=float(res.get_derived_params()['rdrag'])
    # Eisenstein-Hu no-wiggle shape with a broad smooth correction to CAMB.
    fb=obh2/(om*h*h);sh=44.5*math.log(9.83/(om*h*h))/math.sqrt(1+10*obh2**.75)
    ag=1-.328*math.log(431*om*h*h)*fb+.38*math.log(22.3*om*h*h)*fb*fb
    ge=om*h*(ag+(1-ag)/(1+(0.43*kh*sh*h)**4));q=kh*(2.7255/2.7)**2/ge
    l=np.log(2*np.e+1.8*q);c=14.2+731/(1+62.5*q);tnw=l/(l+c*q*q)
    pbase=kh**.9649*tnw**2
    corr=gaussian_filter1d(np.log(pl/pbase),sigma=.25/np.log(kh[1]/kh[0]),mode='nearest');pnw=pbase*np.exp(corr)
    k=np.linspace(.0001,1.2,4096);P=np.exp(np.interp(np.log(k),np.log(kh),np.log(pl)));NW=np.exp(np.interp(np.log(k),np.log(kh),np.log(pnw)))
    rg=np.arange(1.,421.,.25);sg=np.arange(1.,24.,2.);m,w=np.polynomial.legendre.leggauss(64)
    dk=k[1]-k[0];kw=np.ones(len(k))*dk;kw[[0,-1]]*=.5;kw*=k*k/(2*np.pi**2)*np.exp(-(k*1.5)**2)
    table=np.zeros((len(sg),3,3,len(rg)))
    for il,ell in enumerate([0,2,4]):
        hankel=spherical_jn(ell,rg[:,None]*k[None,:])*kw[None,:]*(-1)**(ell//2)
        for isig,sig in enumerate(sg):
            damping=np.exp(-.5*k[:,None]**2*sig**2*(1+(1.6**2-1)*m[None,:]**2))
            for power in range(3):
                leg=(2*ell+1)/2*w*eval_legendre(ell,m)*m**(2*power)
                pell=NW*leg.sum()+(P-NW)*(damping@leg)
                table[isig,power,il]=hankel@pell
    np.savez_compressed(out/'physical_template.npz',r=rg,sigma_grid=sg,xi_basis=table,k=k,p_linear=P,p_nowiggle=NW,rd=rd)
    meta={'CAMB_version':camb.__version__,'Omega_m':om,'h':h,'ombh2':obh2,'mnu_eV':.06,'ns':.9649,'rdrag_Mpc':rd,'power_redshift':.75,'model':'linear CAMB P(k), smooth no-wiggle subtraction, anisotropic Gaussian BAO damping, Kaiser RSD; broadband profiled','fixed_parallel_to_perpendicular_damping_ratio':1.6,'extra_Hankel_high_k_smoothing_Mpc_h':1.5}
    (out/'template_manifest.json').write_text(json.dumps(meta,indent=2));print('TEMPLATE_BUILT',json.dumps(meta),flush=True)
    return table,sg,rg,meta

class Fit:
    def __init__(self,s,mu,z,poles,cov,kernels,leg,template,quadratic=False,shrink=.5,slo=60,shi=150):
        self.smask=(s>=slo)&(s<=shi);self.s=s[self.smask];self.mu=mu;self.z=z;self.u=z-.75;self.up=self.u/.35
        self.leg=leg[:2];self.table,self.sg,self.rg,self.meta=template;self.quadratic=quadratic
        self.kernel=kernels[:,:,self.smask,:,:]
        self.obs=poles[:,:,:2,:][:,:,:,self.smask].reshape(len(poles),-1)
        C=jk_cov(self.obs);std=np.sqrt(np.diag(C));corr=C/np.outer(std,std);corr=(1-shrink)*corr+shrink*np.eye(len(std))
        self.W=solve_triangular(np.linalg.cholesky(corr),np.eye(len(std)),lower=True)/std[None,:]
        self.std=std;self.C=C;self.y=self.obs@self.W.T;self.shrink=shrink;self.n=len(self.obs)-1
        self.B=[];self.BW=[]
        for ir in range(len(self.obs)):
            bb=[]
            for ell in [0,2]:
                for power in range(3):
                    for iz in range(3):
                        field=1e-4*(100/self.s[:,None,None])**power*eval_legendre(ell,mu)[None,:,None]*(self.up[None,None,:]**iz)
                        bb.append(self.project(field,ir))
            bb=np.column_stack(bb);self.B.append(bb);self.BW.append(self.W@bb)
        # Order: log alpha_perp, log alpha_parallel, their redshift slopes,
        # optional log-alpha curvatures, then beta0, beta slope, Sigma0, Sigma slope.
        self.initial=np.array([0.,0.,0.,0.]+([0.,0.] if quadratic else [])+[.4,0.,7.,0.])
        self.lower=np.array([-.20,-.20,-1.,-1.]+([-6.,-6.] if quadratic else [])+[.02,-3.,4.,-5.])
        self.upper=np.array([.20,.20,1.,1.]+([6.,6.] if quadratic else [])+[1.5,3.,18.,5.])
        self.names=['ln_alpha_perp','ln_alpha_parallel','dln_alpha_perp_dz','dln_alpha_parallel_dz']+(['d2ln_alpha_perp_dz2','d2ln_alpha_parallel_dz2'] if quadratic else [])+['beta0','dln_beta_dz','Sigma_perp0','dSigma_perp_dz']
    def project(self,field,ir):
        return np.einsum('ismz,lm,smz->ils',self.kernel[ir],self.leg,field,optimize=True).ravel()
    def design(self,t,ir):
        a0,h0,a1,h1=t[:4];a2,h2=t[4:6] if self.quadratic else (0.,0.)
        beta,bs,sigma,ss=t[-4:]
        ap=np.exp(a0+a1*self.u+.5*a2*self.u**2);al=np.exp(h0+h1*self.u+.5*h2*self.u**2)
        field=evaluate_table(self.s,self.mu,ap,al,beta*np.exp(bs*self.u),sigma+ss*self.u,self.table,self.sg,self.rg[0],self.rg[1]-self.rg[0])
        amp=np.column_stack([self.project(field*self.up[None,None,:]**j,ir) for j in range(3)])
        return np.column_stack([amp,self.B[ir]])
    def profile(self,t,ir):
        M=self.design(t,ir);MW=self.W@M;b=np.linalg.lstsq(MW,self.y[ir],rcond=1e-10)[0]
        return self.y[ir]-MW@b,b,M@b
    def fitone(self,ir,start,max_nfev=240):
        st=np.clip(start,self.lower+1e-6,self.upper-1e-6)
        res=least_squares(lambda t:self.profile(t,ir)[0],st,bounds=(self.lower,self.upper),x_scale='jac',ftol=2e-6,xtol=2e-6,gtol=2e-6,max_nfev=max_nfev)
        return res
    def run(self,out,label):
        starts=[self.initial.copy()]
        for a,h in [(-.035,0),(.035,0),(0,-.035),(0,.035)]:
            t=self.initial.copy();t[0]=a;t[1]=h;starts.append(t)
        full=[self.fitone(0,t,400) for t in starts];best=min(full,key=lambda r:np.sum(r.fun*r.fun));t=best.x
        print('FIT_FULL',label,'theta',t.tolist(),'cost',float(best.fun@best.fun),'success',best.success,flush=True)
        residual,lin,model=self.profile(t,0);rep=np.zeros((self.n,len(t)));success=np.zeros(self.n,dtype=bool)
        def work(i):return i,self.fitone(i+1,t,200)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,r in pool.map(work,range(self.n)):
                rep[i]=r.x;success[i]=r.success
                if i%20==0:print('JK_FIT',label,i,'cost',float(r.fun@r.fun),flush=True)
        cp=(self.n-1)/self.n*((rep-rep.mean(axis=0)).T@(rep-rep.mean(axis=0)))
        bound=(np.abs(rep-self.lower)<1e-3)|(np.abs(rep-self.upper)<1e-3)
        d={'label':label,'parameter_names':self.names,'best_fit':t.tolist(),'jackknife_sigma':np.sqrt(np.diag(cp)).tolist(),'jackknife_covariance':cp.tolist(),'objective':float(residual@residual),'objective_dof':int(len(residual)-len(lin)-len(t)),'objective_note':'regularized full jackknife covariance; not a calibrated chi-square probability','correlation_shrinkage_to_diagonal':self.shrink,'bounds_lower':self.lower.tolist(),'bounds_upper':self.upper.tolist(),'full_fit_success':bool(best.success),'jackknife_successes':int(success.sum()),'jackknife_boundary_hits_per_parameter':bound.sum(axis=0).tolist(),'profiled_amplitude_coefficients':lin[:3].tolist(),'minimum_profiled_amplitude_across_redshift':float(np.min(lin[:3]@np.array([self.up**j for j in range(3)]))),'geometry_model':'ln alpha = a0 + a1(z-.75)'+(' + a2(z-.75)^2/2' if self.quadratic else ''),'absolute_AP_anchors_free':True,'survey_kernel':'actual per-(s,mu) RR redshift histogram matched to exact moments 0..6'}
        np.savez_compressed(out/f'fit_{label}.npz',theta=t,replicates=rep,covariance=cp,linear_coefficients=lin,model=model,residual=residual,names=self.names,s=self.s,observations=self.obs,covariance_data=self.C)
        (out/f'fit_{label}.json').write_text(json.dumps(d,indent=2))
        return t,rep,cp,model,d

def distances(t,z,rd,quad=False):
    om=.3153;h=.6736;u=z-.75;zz=np.linspace(0,max(1.11,float(np.max(z))+.01),10001)
    e=np.sqrt(om*(1+zz)**3+1-om);x=cumulative_trapezoid(299792.458/(100*h*rd*e),zz,initial=0);Xf=np.interp(z,zz,x)
    E2=om*(1+z)**3+1-om;Yf=299792.458/(100*h*rd*np.sqrt(E2))
    eta=-1.5*om*(1+z)**2/E2;etaprime=-3*om*(1+z)/E2+4.5*om**2*(1+z)**4/E2**2
    a0,h0,a1,h1=t[:4];a2,h2=t[4:6] if quad else (0.,0.)
    ap=np.exp(a0+a1*u+.5*a2*u*u);al=np.exp(h0+h1*u+.5*h2*u*u)
    X=ap*Xf;Y=al*Yf;gp=a1+a2*u;gh=h1+h2*u
    Xp=ap*(Yf+Xf*gp);Yp=Y*(eta+gh)
    Xpp=ap*(Yf*eta+2*gp*Yf+(a2+gp*gp)*Xf);Ypp=Y*((eta+gh)**2+etaprime+h2)
    return np.array([X,Y,Xp,Yp,Xpp,Ypp])
