#!/usr/bin/env python3
"""Actual DESI LRG pair counts and spatial delete-one jackknife.

No simulated measurement is generated. --self-test only checks the counter
against exhaustive deterministic pair enumeration. Production input is the
public DESI DR1 v1.2 FITS catalogue, downloaded by --download, or local FITS.

Dependencies: numpy scipy numba matplotlib and either fitsio or astropy.
Example:
 python measure_desi_dxi_jackknife.py --download --cap SGC --njack 32 --threads 8

--data-fraction and --random-fraction perform deterministic-seed uniform
thinning for a computationally bounded direct-catalogue demonstration.
Set both to 1 for the full selected galaxy catalogue and random file 0. This is a direct
PRE-reconstruction clustering measurement, not an official DESI BAO result.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time, urllib.request
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from numba import njit, prange, set_num_threads

BASES = ['https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.2/', 'https://webdav-hdfs.pic.es/data/public/DESI/DR1/survey/catalogs/dr1/LSS/iron/LSScats/v1.2/']
EXPECTED_SIZE = {'NGC': (166812480, 1136165760), 'SGC': (74868480, 600108480)}

def fetch(name: str, path: Path, expected: int) -> dict:
    """Download a whole binary FITS, trying both official DESI mirrors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.stat().st_size == expected):
        last_error = None
        for base in BASES:
            url = base + name
            partial = path.with_suffix(path.suffix + '.part')
            try:
                if partial.exists(): partial.unlink()
                request = urllib.request.Request(url, headers={'User-Agent': 'DESI-dxi-direct-analysis/1.0'})
                with urllib.request.urlopen(request, timeout=180) as r, partial.open('wb') as f:
                    while True:
                        chunk = r.read(8 * 1024**2)
                        if not chunk: break
                        f.write(chunk)
                if partial.stat().st_size != expected:
                    raise RuntimeError(f'Truncated/wrong file: {partial.stat().st_size} != {expected}; {url}')
                partial.replace(path)
                used_url = url
                break
            except Exception as exc:
                last_error = exc
                print(f'Download failed from {url}: {exc}', flush=True)
        else:
            raise RuntimeError(f'All DESI mirrors failed for {name}: {last_error}')
    else:
        used_url = 'existing-local-file'
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): digest.update(block)
    return {'url': used_url, 'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}

def read_fits(path: Path, zmin: float, zmax: float, fraction: float, seed: int):
    """Select rows in chunks; WEIGHT already contains catalogue corrections."""
    names = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP']
    try:
        import fitsio
        f = fitsio.FITS(str(path))
        n = f[1].get_nrows()
        def read_chunk(lo, hi): return f[1].read(rows=np.arange(lo, hi), columns=names)
    except ImportError:
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise RuntimeError('Install fitsio or astropy to read the actual FITS inputs.') from exc
        f = fits.open(path, memmap=True)
        n = len(f[1].data)
        def read_chunk(lo, hi): return f[1].data[lo:hi]
    rng = np.random.default_rng(seed)
    chunks, available = [], 0
    for lo in range(0, n, 250000):
        t = read_chunk(lo, min(n, lo + 250000))
        z = np.asarray(t['Z'], dtype='f8')
        w = np.asarray(t['WEIGHT'], dtype='f8') * np.asarray(t['WEIGHT_FKP'], dtype='f8')
        good = (z >= zmin) & (z < zmax) & np.isfinite(z) & np.isfinite(w) & (w > 0)
        good &= np.isfinite(t['RA']) & np.isfinite(t['DEC'])
        available += int(good.sum())
        if fraction < 1: good &= (rng.random(len(t)) < fraction)
        chunks.append(np.column_stack([np.asarray(t['RA'])[good], np.asarray(t['DEC'])[good], z[good], w[good]]))
    f.close()
    a = np.ascontiguousarray(np.concatenate(chunks), dtype='f8')
    if len(a) < 100: raise RuntimeError(f'Too few selected objects in {path}')
    return a, {'input_rows': n, 'selected_before_thinning': available, 'used_rows': len(a)}

def angular_regions(random: np.ndarray, njack: int, cap: str):
    """Equal-random-weight rectangular sky regions, determined without data.

    Split the longer angular dimension recursively at its weighted median.
    Every region extends over the entire redshift interval. The tree need not
    have a power-of-two number of leaves. Regions are approximately equal in
    weighted random mass, a condition behind the conventional JK prefactor.
    """
    def coordinates(a):
        ra = np.deg2rad(a[:, 0] if cap == 'NGC' else (a[:, 0] + 180) % 360 - 180)
        return np.column_stack([ra, np.sin(np.deg2rad(a[:, 1]))])
    xy = coordinates(random)
    leaves = []
    def split(ids, count):
        if count == 1:
            label = len(leaves); leaves.append(ids)
            return {'label': label}
        ranges = np.ptp(xy[ids], axis=0)
        axis = int(np.argmax(ranges))
        order = ids[np.argsort(xy[ids, axis], kind='stable')]
        left_n = count // 2
        cumulative = np.cumsum(random[order, 3])
        k = int(np.searchsorted(cumulative, cumulative[-1] * left_n/count)) + 1
        k = min(max(k, 1), len(order)-1)
        bound = 0.5 * (xy[order[k-1], axis] + xy[order[k], axis])
        return {'axis': axis, 'bound': float(bound),
                'left': split(order[:k], left_n), 'right': split(order[k:], count-left_n)}
    tree = split(np.arange(len(random)), njack)
    def labels(a):
        p = coordinates(a); out = np.empty(len(a), dtype=np.int32)
        def visit(node, ids):
            if 'label' in node: out[ids] = node['label']; return
            mask = p[ids, node['axis']] <= node['bound']
            visit(node['left'], ids[mask]); visit(node['right'], ids[~mask])
        visit(tree, np.arange(len(a)))
        return out
    return tree, labels

def to_xyz(a, labels, om, pivot, zmax):
    """Fixed flat-LCDM analysis coordinates, chi in Mpc/h; no model fit."""
    zz = np.linspace(0, zmax + 0.05, 20001)
    chi = cumulative_trapezoid(2997.92458/np.sqrt(om*(1+zz)**3 + 1-om), zz, initial=0)
    dist = PchipInterpolator(zz, chi)(a[:, 2])
    ra, dec = np.deg2rad(a[:, 0]), np.deg2rad(a[:, 1])
    xyz = np.column_stack([dist*np.cos(dec)*np.cos(ra), dist*np.cos(dec)*np.sin(ra), dist*np.sin(dec)])
    # x,y,z, weight, centred redshift u, JK label, squared radius
    return np.ascontiguousarray(np.column_stack([xyz, a[:, 3], a[:, 2]-pivot, labels, dist**2]))

def sort_cells(a, origin, dims, size):
    c = np.floor((a[:, :3]-origin)/size).astype(np.int64)
    keys = (c[:, 0]*dims[1] + c[:, 1])*dims[2] + c[:, 2]
    order = np.argsort(keys, kind='stable')
    sizes = np.bincount(keys, minlength=int(np.prod(dims)))
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    return np.ascontiguousarray(a[order]), np.ascontiguousarray(c[order]), offsets

@njit(parallel=True, cache=True)
def²È="25‘•±¥µ¥Ñ•Èôœ°œ°¡•…‘•Èôœ°œ¹©½¥¸¡¹…µ•Ì¤°½µµ•¹ÑÌôœœ¤()‘•˜‰ÉÕÑ”¡„°ˆ°…ÕÑ¼°™œ¤è(€€€É•Ì€ô¹À¹é•É½Ì ¡™l¹©…¬t¬Ä°™l¹Ìt°™l¹µÔt°€Ð¤¤ì¹Õµ‰•È€ô€À(€€€™½È¤¥¸É…¹”¡±•¸¡„¤¤è(€€€€€€€™½È¨¥¸É…¹”¡¤¬Ä¥˜…ÕÑ¼•±Í”€À°±•¸¡ˆ¤¤è(€€€€€€€€€€€Ø€ô‰m¨°€èÍtµ…m¤°€èÍtìÌ€ô¹À¹±¥¹…±œ¹¹½É´¡Ø¤(€€€€€€€€€€€¥˜¹½Ð™lÍµ¥¸t€ðôÌ€ð™lÍµ¥¸t­™l‘Ìt©™l¹Ìtè½¹Ñ¥¹Õ”(€€€€€€€€€€€±½Ì€ô…m¤°€èÍt­‰m¨°€èÍt(€€€€€€€€€€€µÔ€ô…‰Ì¡Ù±½Ì¤¼¡Ì©¹À¹±¥¹…±œ¹¹½É´¡±½Ì¤¤(€€€€€€€€€€€¥ˆ€ô¥¹Ð ¡Ìµ™lÍµ¥¸t¤½™l‘Ìt¤ì¥´€ôµ¥¸¡™l¹µÔt´Ä°¥¹Ð¡µÔ©™l¹µÔt¤¤(€€€€€€€€€€€µ½µ•¹ÑÌ€ô…m¤°€Ít©‰m¨°€Ít¨ ¡…m¤°€Ñt­‰m¨°€Ñt¤¼È¤¨©¹À¹…É…¹” Ð¤(€€€€€€€€€€€É•ÍlÀ°¥ˆ°¥µt€¬ôµ½µ•¹ÑÌ(€€€€€€€€€€€™½È¬¥¸Í•Ð¡m¥¹Ð¡…m¤°€Õt¤¬Ä°¥¹Ð¡‰m¨°€Õt¤¬Åt¤èÉ•Ím¬°¥ˆ°¥µt€¬ôµ½µ•¹ÑÌ(€€€€€€€€€€€¹Õµ‰•È€¬ô€Ä(€€€É•ÑÕÉ¸É•Ì°¹Õµ‰•È()‘•˜Í•±™}Ñ•ÍÐ ¤è(€€€€ˆˆ‰•Ñ•Éµ¥¹¥ÍÑ¥Œ™¥áÑÕÉ•Ì°•áÁ±¥¥Ñ±ä9=PÍ¥•¹Ñ¥™¥Œ‘…Ñ„½Á±½ÑÌ¸ˆˆˆ(€€€™œ€ô‘¥Ð¡Íµ¥¸ôÔ¸°‘ÌôÄÀ¸°¹Ìôà°¹µÔôÔ°¹©…¬ôÐ°Í¡…É‘ÌôÐ°Ñ¡•Ñ…}ÕÐôÀ¸¤(€€€Ð€ô¹À¹…É…¹” ÄÐÀ¸¤(€€€‘•˜™¥áÑÕÉ”¡Ð¤è(€€€€€€€áåè€ô¹À¹½±Õµ¹}ÍÑ…¬¡lÄÀÀ¬ÌÔ©¹À¹Í¥¸¡Ð¨Ä¸ÄÜ¤°€àÀ¬ÌÌ©¹À¹½Ì¡Ð¨¸ÜÄ¤°€ÄÄÀ¬Ìä©¹À¹Í¥¸¡Ð¨¸ÌÜ¥t¤(€€€€€€€É•ÑÕÉ¸¹À¹½±Õµ¹}ÍÑ…¬¡máåè°€¸à¬¸È©¹À¹Í¥¸¡Ð¨¸ÌÄ¤¨¨È°€´¸ÄÔ¬¸Ì¨¡Ð”ÄÜ¤¼ÄØ°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€¡Ð¹…ÍÑåÁ”¡¥¹Ð¤”Ð¤°¹À¹ÍÕ´¡áåè¨¨È°…á¥ÌôÄ¥t¤(€€€°È€ô™¥áÑÕÉ”¡ÑlèØÅt¤°™¥áÑÕÉ”¡Ð¬À¸È¤(€€€•ÉÉ½ÉÌ€ômt(€€€™½È¹…µ”°„°ˆ±…ÕÑ¼¥¸l œ±±±QÉÕ”¤° Hœ±±È±…±Í”¤° IHœ±È±Ét±QÉÕ”¥tè(€€€€€€€½Ð°¹½Ð€ô½Õ¹Ð¡„±ˆ±…ÕÑ¼±™œ¤ì•áÁ•Ñ•°¹•áÁ•Ð€ô‰ÉÕÑ”¡„±ˆ±…ÕÑ¼±™œ¤(€€€€€€€¹À¹Ñ•ÍÑ¥¹œ¹…ÍÍ•ÉÑ}…±±±½Í”¡½Ð±•áÁ•Ñ•±ÉÑ½°ôÉ”´ÄÈ±…Ñ½°ôÉ”´ÄÀ¤(€€€€€€€…ÍÍ•ÉÐ¹½Ð€ôô¹•áÁ•Ð(€€€€€€€™½È¬¥¸É…¹”¡™l¹©…¬t¤è(€€€€€€€€€€€…„€ô…m…lè°Õt€„ô­tì‰ˆ€ô‰m‰lè°Õt€„ô­t(€€€€€€€€€€€É•ÉÕ¸°|€ô‰ÉÕÑ”¡…„±‰ˆ±…ÕÑ¼±™œ¤(€€€€€€€€€€€¹À¹Ñ•ÍÑ¥¹œ¹…ÍÍ•ÉÑ}…±±±½Í”¡½ÑlÁtµ½Ñm¬¬Åt±É•ÉÕ¹lÁt±ÉÑ½°ôÉ”´ÄÈ±…Ñ½°ôÉ”´ÄÀ¤(€€€€€€€•ÉÉ½ÉÌ¹…ÁÁ•¹¡ì½Õ¹Ñ•Èœé¹…µ”°Á…¥ÉÌœé¥¹Ð¡¹½Ð¤°µ…á}…‰Í}‘¥™™•É•¹”œé™±½…Ð¡¹À¹µ…à¡…‰Ì¡½Ðµ•áÁ•Ñ•¤¤¥ô¤(€€€€ŒI•‘Í¡¥™Ðµµ½µ•¹Ð•ÍÑ¥µ…Ñ½ÈÉ•ÑÕÉ¹ÌÕ¹¥ÐÍ±½Á”™½Èá¤¡Ô¤õ½¹ÍÑ…¹Ð­Ô¸(€€€ÉÈ°|€ô½Õ¹Ð¡È±È±QÉÕ”±™œ¤(€€€´À±´Ä±´È€ô€¡ÉÉlÀ°¸¸¸±¥t™½È¤¥¸É…¹” Ì¤¤(€€€Ù…±¥€ô´ÀøÀ(€€€µ•…¸€ô´ÅmÙ…±¥‘t½´ÁmÙ…±¥‘tìÙ…É¥…¹”€ô´ÉmÙ…±¥‘t½´ÁmÙ…±¥‘tµµ•…¸¨¨È(€€€½¹ÍÑ…¹Ð°Í±½Á”€ô€¸ÀÈ°€¸ÀÜ(€€€¸À€ô½¹ÍÑ…¹Ð©´ÁmÙ…±¥‘t­Í±½Á”©´ÅmÙ…±¥‘t(€€€¸Ä€ô½¹ÍÑ…¹Ð©´ÅmÙ…±¥‘t­Í±½Á”©´ÉmÙ…±¥‘t(€€€•ÍÑ¥µ…Ñ•}Í±½Á”€ô€¡¸Äµµ•…¸©¸À¤¼¡Ù…É¥…¹”©´ÁmÙ…±¥‘t¤(€€€¹À¹Ñ•ÍÑ¥¹œ¹…ÍÍ•ÉÑ}…±±±½Í”¡•ÍÑ¥µ…Ñ•}Í±½Á”°Í±½Á”°ÉÑ½°ôÅ”´ÄÄ°…Ñ½°ôÅ”´ÄÈ¤(€€€ÁÉ¥¹Ð¡©Í½¸¹‘ÕµÁÌ¡ìÍÑ…ÑÕÌœèAMMœ°Ñ•ÍÑÌœé•ÉÉ½ÉÌ°µ•…¹¥¹œœè½Õ¹Ñ•È…¹©…­­¹¥™”Õ¹¥ÐÑ•ÍÑÌ½¹±äì¹¼M$µ•…ÍÕÉ•µ•¹Ð¸ô±¥¹‘•¹ÐôÈ¤¤()‘•˜µ…¥¸ ¤è(€€€À€ô…ÉÁ…ÉÍ”¹ÉÕµ•¹ÑA…ÉÍ•È¡‘•ÍÉ¥ÁÑ¥½¸õ}}‘½}|°™½Éµ…ÑÑ•É}±…ÍÌõ…ÉÁ…ÉÍ”¹I…Ý•ÍÉ¥ÁÑ¥½¹!•±Á½Éµ…ÑÑ•È¤(€€€À¹…‘‘}…ÉÕµ•¹Ð œ´µÍ•±˜µÑ•ÍÐœ°…Ñ¥½¸ôÍÑ½É•}ÑÉÕ”œ¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µ‘½Ý¹±½…œ±…Ñ¥½¸ôÍÑ½É•}ÑÉÕ”œ¤(€€€À¹…‘‘}…ÉÕµ•¹Ð œ´µ…Àœ±¡½¥•Ìõl9œ°Mt±‘•™…Õ±ÐôMœ¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µÉ½½Ðœ±ÑåÁ”õA…Ñ ±‘•™…Õ±ÐõA…Ñ  ‘•Í¥}‘á¥}…ÑÕ…°œ¤¤(€€€À¹…‘‘}…ÉÕµ•¹Ð œ´µ‘…Ñ„œ±ÑåÁ”õA…Ñ ¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µÉ…¹‘½´œ±ÑåÁ”õA…Ñ ¤(€€€À¹…‘‘}…ÉÕµ•¹Ð œ´µ¹©…¬œ±ÑåÁ”õ¥¹Ð±‘•™…Õ±ÐôÌÈ¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µÑ¡É•…‘Ìœ±ÑåÁ”õ¥¹Ð±‘•™…Õ±ÐôÐ¤(€€€À¹…‘‘}…ÉÕµ•¹Ð œ´µÍµ¥¸œ±ÑåÁ”õ™±½…Ð±‘•™…Õ±ÐôÐÀ¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µÍµ…àœ±ÑåÁ”õ™±½…Ð±‘•™…Õ±ÐôÄØÀ¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µ‘Ìœ±ÑåÁ”õ™±½…Ð±‘•™…Õ±ÐôÔ¤(€€€À¹…‘‘}…ÉÕµ•¹Ð œ´µ¹µÔœ±ÑåÁ”õ¥¹Ð±‘•™…Õ±ÐôÈÀ¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µéµ¥¸œ±ÑåÁ”õ™±½…Ð±‘•™…Õ±Ðô¸Ð¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µéµ…àœ±ÑåÁ”õ™±½…Ð±‘•™…Õ±ÐôÄ¸Ä¤(€€€À¹…‘‘}…ÉÕµ•¹Ð œ´µ½µ•„µ´œ±ÑåÁ”õ™±½…Ð±‘•™…Õ±Ðô¸ÌÄÔÌ¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µÁ¥Ù½Ðœ±ÑåÁ”õ™±½…Ð±‘•™…Õ±Ðô¸ÜÔ¤(€€€À¹…‘‘}…ÉÕµ•¹Ð œ´µ‘…Ñ„µ™É…Ñ¥½¸œ±ÑåÁ”õ™±½…Ð±‘•™…Õ±ÐôÄ¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µÉ…¹‘½´µ™É…Ñ¥½¸œ±ÑåÁ”õ™±½…Ð±‘•™…Õ±ÐôÄ¤ìÀ¹…‘‘}…ÉÕµ•¹Ð œ´µÍ••œ±ÑåÁ”õ¥¹Ð±‘•™…Õ±ÐôÜÌÐÈä¤(€€€À¹…‘‘}…ÉÕµ•¹Ð œ´µÑ¡•Ñ„µÕÐœ±ÑåÁ”õ™±½…Ð±‘•™…Õ±ÐôÀ±¡•±Àô=ÁÑ¥½¹…°µ¥¸…¹Õ±…ÈÁ…¥ÈÍ•Á…É…Ñ¥½¸¥¸‘•É••Ìì‘•™…Õ±Ð¹½¹”¸œ¤(€€€…ÉÌ€ôÀ¹Á…ÉÍ•}…ÉÌ ¤ìÍ•Ñ}¹Õµ}Ñ¡É•…‘Ì¡…ÉÌ¹Ñ¡É•…‘Ì¤(€€€¥˜…ÉÌ¹Í•±™}Ñ•ÍÐèÍ•±™}Ñ•ÍÐ ¤ìÉ•ÑÕÉ¸(€€€¥˜¹½Ð€À€ð…ÉÌ¹‘…Ñ…}™É…Ñ¥½¸€ðô€ÄèÀ¹•ÉÉ½È ‘…Ñ„µ™É…Ñ¥½¸µÕÍÐ‰”¥¸€ À°Åt¸œ¤(€€€¥˜¹½Ð€À€ð…ÉÌ¹É…¹‘½µ}™É…Ñ¥½¸€ðô€ÄèÀ¹•ÉÉ½È É…¹‘½´µ™É…Ñ¥½¸µÕÍÐ‰”¥¸€ À°Åt¸œ¤(€€€¥˜…ÉÌ¹¹©…¬ðÐèÀ¹•ÉÉ½È ¹©…¬µÕÍÐ‰”…Ð±•…ÍÐ€Ð¸œ¤(€€€¹Ì€ôÉ½Õ¹ ¡…ÉÌ¹Íµ…àµ…ÉÌ¹Íµ¥¸¤½…ÉÌ¹‘Ì¤(€€€¥˜¹Ì€ð€Ä½È¹½Ð¹À¹¥Í±½Í”¡…ÉÌ¹Íµ¥¸­…ÉÌ¹‘Ì©¹Ì±…ÉÌ¹Íµ…à¤èÀ¹•ÉÉ½È ÌÉ…¹”µÕÍÐ‰”‘¥Ù¥Í¥‰±”‰ä‘Ì¸œ¤(€€€½ÕÐ€ô…ÉÌ¹É½½Ð½…ÉÌ¹…Àì½ÕÐ¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”±•á¥ÍÑ}½¬õQÉÕ”¤(€€€™œ€ô‘¥Ð¡Ù…ÉÌ¡…ÉÌ¤¤ì™œ¹ÕÁ‘…Ñ”¡¹Ìõ¹Ì±Í¡…É‘ÌôÈ©…ÉÌ¹Ñ¡É•…‘Ì¤(€€€™œ€ôí¬éÍÑÈ¡Ø¤¥˜¥Í¥¹ÍÑ…¹”¡Ø±A…Ñ ¤•±Í”Ø™½È¬±Ø¥¸™œ¹¥Ñ•µÌ ¥ô(€€€¹…µ•Ì€ôm˜1I}í…ÉÌ¹…Áõ}±ÕÍÑ•É¥¹œ¹‘…Ð¹™¥ÑÌœ°˜1I}í…ÉÌ¹…Áõ|Á}±ÕÍÑ•É¥¹œ¹É…¸¹™¥ÑÌt(€€€Á…Ñ¡Ì€ôm…ÉÌ¹‘…Ñ„½È…ÉÌ¹É½½Ð¼…Ñ…±½Ìœ½¹…µ•ÍlÁt°…ÉÌ¹É…¹‘½´½È…ÉÌ‹œ›ÛÝÉØØ][ÙÜÉËÛ˜[Y\ÖÌWWBˆ›Ý™[˜[˜ÙHH×BˆYˆ\™ÜË™ÝÛ›ØY‚ˆ›Üˆ˜[YK]Ú^™H[ˆš\
˜[Y\Ë]ËVPÕQÔÒV‘VØ\™ÜË˜Ø\JN‚ˆš[
‰ÑÝÛ›ØY[™ËÝ™\šYžZ[™ÈÛ˜[Y_IË›\ÚUYJBˆ›Ý™[˜[˜ÙK˜\[™
™]Ú
˜[YK]Ú^™JJBˆYˆ›Ý[
]™^\ÝÊ
H›Üˆ][ˆ]ÊNˆ™\œ›ÜŠ	Ò[œ]’UÈZ\ÜÚ[™Ëˆ\ÙHKYÝÛ›ØYÜˆKY]H[™K\˜[™ÛK‰ÊBˆ]KY]HH™XYÙš]Ê]ÖÌK\™ÜËž›Z[‹\™ÜËž›X^\™ÜË™]WÙœ˜XÝ[Û‹\™ÜËœÙYY
Bˆ˜[™ÛK›Y]HH™XYÙš]Ê]ÖÌWK\™ÜËž›Z[‹\™ÜËž›X^\™ÜËœ˜[™ÛWÙœ˜XÝ[Û‹\™ÜËœÙYY
Bˆš[
‰ÐXÝX[Ù[XÝYØš™XÝÎˆ]O^Û[Š]JN‹NÈ˜[™ÛO^Û[Š˜[™ÛJN‹IË›\ÚUYJBˆ™YKX™[H[™Ý[\—Ü™YÚ[ÛœÊ˜[™ÛK\™ÜË›š˜XÚË\™ÜË˜Ø\
BˆX™[›X™[HX™[
]JKX™[
˜[™ÛJBˆœœØ]™^—ØÛÛ\™\ÜÙY
Ý]ÉÜ™YÚ[Û—ÙYš[š][Û‹›œ‰Ë]WÜ˜OY]VÎ‹K]WÙXÏY]VÎ‹WK]WÜ™YÚ[ÛYX™[ˆ˜[™ÛWÜ™YÚ[Û—ÝÙZYÚ[œ˜š[˜ÛÝ[
›X™[ÙZYÚÏ\˜[™ÛVÎ‹×KZ[›[™ÝX\™ÜË›š˜XÚÊJBˆ
Ý]ÉÜ™YÚ[ÛœËšœÛÛ‰ÊKÜš]WÝ^
œÛÛ‹™[\J™YK[™[LŠJBˆH×Þ^Š]KX™[\™ÜË›ÛYYØWÛK\™ÜËœ]›Ý\™ÜËž›X^
BˆˆH×Þ^Š˜[™ÛK›X™[\™ÜË›ÛYYØWÛK\™ÜËœ]›Ý\™ÜËž›X^
Bˆ[]K˜[™ÛBˆY]HHXÝ
ÛÛ™šYÝ\˜][ÛXÙ™Ë]OYY]K˜[™ÛO\›Y]K›Ý™[˜[˜ÙO\›Ý™[˜[˜ÙKˆÝ]\ÏIØXÝX[ØØ][ÙÝYWÜZ\—ØÛÝ[[™×Ú[—Ü›ÙÜ™\ÜÉË™WÜ™XÛÛœÝXÝ[ÛUYJBˆ
Ý]ÉÜ[—ÛX[šY™\ÝšœÛÛ‰ÊKÜš]WÝ^
œÛÛ‹™[\JY]K[™[LŠJBˆÛÝ[\œÈH×Bˆ›Üˆ˜[YKK‹]]È[ˆÊ	Ñ	ËYJK
	Ñ‰Ë‹˜[ÙJK
	Ô”‰Ë‹—KYJWN‚ˆÝ\][YK[YJ
NÈš[
‰ÐÛÝ[[™ÈÛ˜[Y_K[˜ÛY[™È]™\žH’È[][Û‹‹‹‰Ë›\ÚUYJBˆ[ÛY[ËˆHÛÝ[
K‹]]ËÙ™ÊNÈÛÝ[\œË˜\[™
[ÛY[ÊBˆœœØ]™JÝ]Ù‰ÞÛ˜[Y_WÛ[ÛY[Ë›œIË[ÛY[ÊBˆš[
‰ÞÛ˜[Y_NˆÛŽ‹HXØÙ\YZ\œËÝ[YK[YJ
K\Ý\‹Œ™ŸHÉË›\ÚUYJBˆY]VÛ˜[YWO^ÉÜZ\œÉÎš[
ŠK	ÜÙXÛÛ™ÉÎ[YK[YJ
K\Ý\Bˆ™\Ý[Y\Ý[X]J
˜ÛÝ[\œË‹Ù™ÊBˆœœØ]™^—ØÛÛ\™\ÜÙY
Ý]ÉÛYX\Ý\™[Y[Ø[™Ú˜XÚÚÛšY™K›œ‰Ë
Šœ™\Ý[
BˆÝÜ™\Ý[Ê™\Ý[Ù™ËÝ]
BˆY]VÉÜÝ]\É×OIØXÝX[ØØ][ÙÝYWÛYX\Ý\™[Y[ØÛÛ\]Y	ÂˆY]VÉØÛÝ˜\šX[˜ÙWÛÜ™\‰×OIÞL
ÊKLŠÊKM
ÊKLÙŠÊKL—ÙŠÊKMÙŠÊIÂˆY]VÉÛ›Ý\É×OVÉÓ›ÈSÈš]ÜˆÛZ[HÙˆSËY]›Û][Ûˆ]XÝ[Û‹‰Ëˆ	ÓÛ™HX›XÈ˜[™ÛHš[NÈ›È[œÚ]KYšY[™XÛÛœÝXÝ[Û‹‰Ëˆ‰Ñ]H˜[™ÛK][›š[™Èœ˜XÝ[ÛŽˆØ\™ÜË™]WÙœ˜XÝ[ÛŸK‰Ëˆ	Ò˜XÚÚÛšY™H›Ü›X[^˜][Ûˆ[™”ˆ™YÚY[ÛY[È™KY\Ý[X]Y[ˆXXÚ[][Û‹‰Ëˆ	ÒÙ\›™[˜\šY\ÈÚ]Ù\\˜][Ûˆ[™]NÈ›ÈÚ[™ÛHÚ[Y\š]˜]]™H[\œ™]][Û‹‰Ëˆ	ÐÛÝ˜\šX[˜ÙH˜[šÈ\È][ÜÝš˜XÚËLNÈÈ›Ý[™\HÛ™Ù\ˆ]K]™XÝÜˆÛÝ˜\šX[˜ÙK‰×Bˆ
Ý]ÉÜ[—ÛX[šY™\ÝšœÛÛ‰ÊKÜš]WÝ^
œÛÛ‹™[\JY]K[™[LŠJBˆš[
‰ÐÛÛ\]YXÝX[Y]HYX\Ý\™[Y[ˆÛÝ]IË›\ÚUYJB‚šYˆ×Û˜[YW×ÏOI××ÛXZ[—×ÉÎˆXZ[Š
B