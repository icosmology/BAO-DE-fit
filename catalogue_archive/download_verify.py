#!/usr/bin/env python3
"""Download an unmodified DESI DR1 v1.5 LRG FITS catalogue and audit it."""
import argparse, datetime, hashlib, json, subprocess
from pathlib import Path
import numpy as np
from astropy.io import fits

p = argparse.ArgumentParser()
p.add_argument('--filename', required=True)
p.add_argument('--bytes', type=int, required=True)
a = p.parse_args()
out = Path('catalogues'); out.mkdir(exist_ok=True)
path = out / a.filename
base = 'https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/'
mirror = 'https://webdav-hdfs.pic.es/data/public/DESI/DR1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/'
used = None
for root in (base, mirror):
    partial = path.with_suffix(path.suffix + '.part')
    partial.unlink(missing_ok=True)
    url = root + a.filename
    command = ['curl', '-fL', '--retry', '5', '--retry-all-errors', '--retry-delay', '3', '--connect-timeout', '30', '--speed-time', '90', '--speed-limit', '1024', '--max-time', '1800', '--output', str(partial), url]
    result = subprocess.run(command)
    if result.returncode == 0 and partial.stat().st_size == a.bytes:
        partial.replace(path); used = url; break
if used is None:
    raise RuntimeError('No successful complete download: '+a.filename)
hashobj = hashlib.sha256()
with path.open('rb') as f:
    for buf in iter(lambda:f.read(8*1024**2), b''):
        hashobj.update(buf)
metadata = dict(filename=a.filename, release='DR1', catalog_version='v1.5', tracer='LRG', sky_region='NGC' if '_NGC_' in a.filename else 'SGC', kind='random_0' if '_0_' in a.filename else 'data', source_url=used, bytes=path.stat().st_size, sha256=hashobj.hexdigest(), downloaded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), galaxy_thinning=False, random_thinning=False, additional_cuts_applied=False, preserves_all_original_columns=True, original_bytes_unmodified=True)
with fits.open(path, memmap=True, checksum=True) as hdus:
    hdus.verify('exception')
    assert len(hdus) >= 2
    tab = hdus[1].data
    names = list(hdus[1].columns.names)
    assert all(c in names for c in ('RA','DEC','Z','WEIGHT','WEIGHT_FKP'))
    metadata['fits_hdus'] = len(hdus)
    metadata['rows'] = len(tab)
    metadata['columns'] = names
    metadata['column_formats'] = {c.name:c.format for c in hdus[1].columns}
    metadata['fits_structural_validation'] = 'passed'
    metadata['fits_internal_checksum_status'] = ['present_and_checked' if 'CHECKSUM' in h.header else 'not_supplied' for h in hdus]
    stats = {}
    valid = 0; selected = 0
    for name in ('RA','DEC','Z','WEIGHT','WEIGHT_FKP'):
        finite_count = 0; minimum = np.inf; maximum = -np.inf
        for lo in range(0, len(tab), 250000):
            vals = np.asarray(tab[name][lo:lo+250000], dtype='f8')
            good = np.isfinite(vals); finite_count += int(good.sum())
            if good.any(): minimum = min(minimum, float(vals[good].min())); maximum = max(maximum,float(vals[good].max()))
        stats[name] = dict(finite_rows=finite_count, min=minimum, max=maximum)
    for lo in range(0,len(tab),250000):
        t = tab[lo:lo+250000]
        z=np.asarray(t['Z'],dtype='f8'); w=np.asarray(t['WEIGHT'],dtype='f8')*np.asarray(t['WEIGHT_FKP'],dtype='f8')
        good=np.isfinite(z)&np.isfinite(w)&(w>0)&np.isfinite(t['RA'])&np.isfinite(t['DEC'])
        valid += int(good.sum()); selected += int((good & (z>=0.4)&(z<1.1)).sum())
    metadata['column_statistics'] = stats
    metadata['valid_coordinate_and_positive_total_weight_rows'] = valid
    metadata['valid_rows_0p4_le_z_lt_1p1_no_cut_applied'] = selected
    (out/(a.filename+'.header.txt')).write_text('\n\n'.join(h.header.tostring(sep='\n',endcard=True,padding=False) for h in hdus))
(out/(a.filename+'.manifest.json')).write_text(json.dumps(metadata,indent=2)+'\n')
(out/(a.filename+'.sha256')).write_text(metadata['sha256']+'  '+a.filename+'\n')
print('VERIFIED_CATALOGUE_JSON_BEGIN', flush=True)
print(json.dumps(metadata,indent=2), flush=True)
print('VERIFIED_CATALOGUE_JSON_END', flush=True)
