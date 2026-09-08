#!/usr/bin/env python3
import os,json,time,inspect,hashlib,ast,urllib.request,traceback,subprocess
from pathlib import Path
import numpy as np
if not hasattr(np,'trapezoid'):np.trapezoid=np.trapz
OUT=Path('official_audit');OUT.mkdir(exist_ok=True)
import desilike,cosmoprimo
from cosmoprimo.fiducial import DESI
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate,DampedBAOWigglesTracerCorrelationFunctionMultipoles
from desilike.theories.galaxy_clustering.bao import DampedBAOWigglesCorrelationFunctionMultipoles
import desilike.theories.galaxy_clustering.bao as bao
cosmo=DESI(engine='class')
template=BAOPowerSpectrumTemplate(z=1.49,fiducial=cosmo,with_now='peakaverage',apmode='qparqper')
theory=DampedBAOWigglesTracerCorrelationFunctionMultipoles(s=np.arange(50.,151.,4.),ells=(0,2),template=template,mode='',model='standard',broadband='pcs2')
t0=time.time();r=np.asarray(theory());print('THEORY_SHAPE',r.shape,'seconds',time.time()-t0,flush=True)
report={'template_class':str(type(template)),'theory_class':str(type(theory)),'model':'standard','broadband':'pcs2','nowiggle':'peakaverage','cosmology_engine':'class','z':1.49,'native_params':str(theory.all_params),'cosmology':str(cosmo),'rd':float(cosmo.rs_drag),'native_source_sha256':hashlib.sha256(Path(bao.__file__).read_bytes()).hexdigest()}
print('PARAMETERS',[(p.basename,p.value,p.fixed) for p in theory.all_params],flush=True)
pars=dict(qper=1.01,qpar=.99,b1=2.1,dbeta=1.,sigmaper=3.5,sigmapar=9.,sigmas=2.)
raw=np.asarray(theory(**pars));np.savez(OUT/'official_template_example.npz',s=theory.s,corr=raw)
try:
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    from desilike.base import jit
    jt=jit(theory)
    t0=time.time();value=np.asarray(jt(**pars));report['native_jit_abs_error']=float(np.max(abs(value-raw)));report['jit_compile_seconds']=time.time()-t0
    t0=time.time()
    for i in range(20):jt(qper=1.+i*.0001,qpar=.99,b1=2.1,dbeta=1.,sigmaper=3.5,sigmapar=9.,sigmas=2.)
    report['jit_eval_seconds']=(time.time()-t0)/20
    keys=list(pars);pv=jnp.asarray([pars[k] for k in keys])
    def pure(v):return jt(**dict(zip(keys,v)))
    batch=jax.jit(jax.vmap(pure));t0=time.time();batchvalue=np.asarray(batch(jnp.tile(pv,(16,1))))
    report['native_vmap_abs_error']=float(np.max(abs(batchvalue-raw)));report['vmap_compile_seconds']=time.time()-t0
    t0=time.time()
    for i in range(10):np.asarray(batch(jnp.tile(pv,(16,1))))
    report['vmap16_eval_seconds']=(time.time()-t0)/10
    print('JIT_SUCCESS',report,flush=True)
except Exception as e:
    report['jit_error']=repr(e);traceback.print_exc()
url='https://raw.githubusercontent.com/cosmodesi/desilike/a533f2ae958028f81f31d52b0e05672fb0f31b6b/desilike/theories/galaxy_clustering/bao.py'
with urllib.request.urlopen(url) as f:old=f.read().decode()
(OUT/'desilike_DR1_2024_bao.py').write_text(old)
def method(text,cls,name):
    tr=ast.parse(text)
    for x in tr.body:
        if isinstance(x,ast.ClassDef) and x.name==cls:
            for f in x.body:
                if isinstance(f,ast.FunctionDef) and f.name==name:return ast.dump(f,include_attributes=False)
report['DR1_calculate_AST_identical']=(method(old,'DampedBAOWigglesPowerSpectrumMultipoles','calculate')==method(Path(bao.__file__).read_text(),'DampedBAOWigglesPowerSpectrumMultipoles','calculate'))
report['DR1_source_ref']='a533f2ae958028f81f31d52b0e05672fb0f31b6b'
(OUT/'template_audit.json').write_text(json.dumps(report,indent=2));print('AUDIT_JSON',json.dumps(report),flush=True)
base='https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering/'
for sub in ['data_v1.2/likelihood/','data_v1.2/recsym/correlation/','data/likelihood/']:
    try:
        with urllib.request.urlopen(base+sub,timeout=45) as f:txt=f.read().decode()
        (OUT/('listing_'+sub.replace('/','_')+'.html')).write_text(txt)
        import re
        links=[a for a in re.findall(r'href="([^"]+)"',txt) if 'QSO' in a]
        print('QSO_FILES',sub,links,flush=True)
        if sub=='data_v1.2/likelihood/':
            match=[x for x in links if 'correlation-recon-poles_QSO' in x]
            if match:
                import h5py
                dest=OUT/Path(match[0]).name
                subprocess.run(['curl','-fLsS','--retry','3','--connect-timeout','30','--max-time','180','-o',str(dest),base+sub+match[0]],check=True)
                schema=[]
                with h5py.File(dest) as hf:
                    def show(name,obj):schema.append({'name':name,'type':type(obj).__name__,'shape':str(getattr(obj,'shape','')),'attrs':{str(k):str(v)[:1000] for k,v in obj.attrs.items()}})
                    hf.visititems(show)
                (OUT/'official_likelihood_schema.json').write_text(json.dumps(schema,indent=2));print('H5_SCHEMA',json.dumps(schema),flush=True)
    except Exception as e:print('DIRECTORY_FAILED',sub,repr(e),flush=True)
