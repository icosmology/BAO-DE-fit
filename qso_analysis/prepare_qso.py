#!/usr/bin/env python3
"""Prepare a QSO-specific copy of the existing pipeline; never reuse LRG data."""
from pathlib import Path
import hashlib,json,shutil
SRC=Path('full_lrg');DST=Path('qso_runtime');DST.mkdir(exist_ok=True)

def patch(text,old,new,count=None):
    n=text.count(old)
    if n==0 or (count is not None and n!=count):
        raise RuntimeError(f'Patch mismatch {old!r}: {n}')
    return text.replace(old,new)

for source in SRC.glob('*.py'):
    text=source.read_text()
    text=text.replace('LRG','QSO').replace('Full LRG','Full QSO').replace('full_lrg','qso_runtime')
    if source.name=='count_catalogues.py':
        text=patch(text,'PIVOT=0.75','PIVOT=1.49',1)
        text=patch(text,'ZMIN,ZMAX=0.4,1.1','ZMIN,ZMAX=0.8,2.1',1)
        text=patch(text,'SMIN,DS,NS,NMU,NZ=40.,4.,30,20,70','SMIN,DS,NS,NMU,NZ=40.,4.,30,20,130',1)
        text=patch(text,'np.linspace(0,1.2,20001)','np.linspace(0,ZMAX+.1,30001)',1)
    if source.name=='reduce_counts.py':
        text=patch(text,'PIVOT=.75','PIVOT=1.49',1)
    if source.name=='bao_model.py':
        text=patch(text,'SP=np.array([2.,3.,4.,5.,6.,7.]);SA=np.array([4.,6.,8.,10.,12.,15.])','SP=np.array([1.,2.,3.,4.,5.,6.]);SA=np.array([4.,6.,8.,10.,12.,15.,20.])',1)
        text=patch(text,'ZP=.75;DZ=.35','ZP=1.49;DZ=.65',1)
        text=patch(text,'np.linspace(.405,1.095,70)','np.linspace(.805,2.095,130)',1)
    if source.name=='kaiser_refit.py':
        text=patch(text,'[(2.,7.),(4.,15.),(-1.,3.),','[(1.,6.),(4.,20.),(-1.,4.),',1)
        text=patch(text,'prior=(p[self.ng]-4.5)**2+(p[self.ng+1]-9.)**2/4.','prior=(p[self.ng]-3.)**2+(p[self.ng+1]-8.)**2/9.',1)
        text=patch(text,'if p[-2]<=abs(p[-1]) or p[-2]+abs(p[-1])>=1.5:bad=True','if min(p[-2]+p[-1]*(.8-ZP)/DZ,p[-2]+p[-1]*(2.1-ZP)/DZ)<=0 or max(p[-2]+p[-1]*(.8-ZP)/DZ,p[-2]+p[-1]*(2.1-ZP)/DZ)>=1.5:bad=True',1)
        text=patch(text,'for v in [-1.,0.,1.]:','for v in [-1.06153846154,-.5,0.,.5,.93846153846]:',1)
    if source.name=='fit_bao.py':
        text=text.replace('np.linspace(.4,1.1,151)','np.linspace(.8,2.1,151)').replace('z=0.75','z=1.49').replace('(z-0.75)/0.35','(z-1.49)/0.65')
    (DST/source.name).write_text(text)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DST.glob('*.py')}
(DST/'source_checksums.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'prepared':'QSO-specific pipeline','range':[.8,2.1],'pivot':1.49,'redshift_scale':.65,'checksums':manifest},indent=2))
