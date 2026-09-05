#!/usr/bin/env python3
"""Build the deliverable from completed real-catalogue result artifacts only."""
import argparse,base64,csv,html,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def interval(d):
    return f"{d['median']:.3f} (+{d['p84']-d['median']:.3f}, -{d['median']-d['p16']:.3f})"


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='base_results');ap.add_argument('--physical',default='final_results');ap.add_argument('--out',default='delivery');a=ap.parse_args()
    base,physical,out=Path(a.base),Path(a.physical),Path(a.out)
    out.mkdir(exist_ok=True)
    for src in [base,physical]:
        for p in src.iterdir():
            if p.is_file():shutil.copy2(p,out/p.name)
    rep=out/'reproduction';rep.mkdir(exist_ok=True)
    for p in Path('full_lrg').glob('*.py'):shutil.copy2(p,rep/p.name)
    scope=json.loads((out/'measurement_summary.json').read_text())
    names=['kaiser_linear','kaiser_quadratic','linear','quadratic']
    labels=['Kaiser model; linear AP evolution','Kaiser model; quadratic AP evolution','Flexible angular amplitudes; linear AP','Flexible angular amplitudes; quadratic AP']
    summaries=[]
    for name in names:
        suffix='_summary.json' if name.startswith('kaiser_') else '_fit_summary.json'
        summaries.append(json.loads((out/(name+suffix)).read_text()))
    obs=['DM_over_rd','DH_over_rd','dDM_over_rd_dz','dDH_over_rd_dz','d2DM_over_rd_dz2','d2DH_over_rd_dz2']
    with (out/'all_model_comparison.csv').open('w') as f:
        wr=csv.writer(f);wr.writerow(['model','quantity','median','p16','p84','p2p5','p97p5'])
        for name,summary in zip(names,summaries):
            for key in obs:
                d=summary['distances_and_derivatives'][key]
                wr.writerow([name,key,d['median'],d['p16'],d['p84'],d['p2p5'],d['p97p5']])
    for key,label,fn in [('dDM_over_rd_dz',r'$d(D_M/r_d)/dz$ at $z=0.75$','model_sensitivity_dDM'),('dDH_over_rd_dz',r'$d(D_H/r_d)/dz$ at $z=0.75$','model_sensitivity_dDH')]:
        fig,ax=plt.subplots(figsize=(10,5.5))
        for row,(lab,summary) in enumerate(zip(labels,summaries)):
            d=summary['distances_and_derivatives'][key];v=d['median']
            ax.errorbar(v,3-row,xerr=np.array([[v-d['p16']],[d['p84']-v]]),fmt='o',capsize=4)
        ax.set_yticks(range(4),labels[::-1]);ax.set_xlabel(label);ax.set_title('Same full catalogue: dependence on the evolution/nuisance model')
        ax.grid(axis='x',alpha=.2);fig.subplots_adjust(left=.40,right=.97,bottom=.15,top=.88)
        fig.savefig(out/(fn+'.png'),dpi=190);fig.savefig(out/(fn+'.svg'));plt.close(fig)
    totald=sum(v['selected_data_rows'] for v in scope['caps'].values());totalr=sum(v['selected_random_rows'] for v in scope['caps'].values())
    delivery={'dataset':'DESI DR1 LSS iron v1.5, NGC+SGC, complete data + random_0 within the analysis redshift selection','galaxies':totald,'randoms':totalr,'no_thinning':True,'redshift_range':[.4,1.1],'pre_reconstruction':True,'jackknife_regions':120,'primary_model':'kaiser_quadratic','primary_fit':summaries[1]['distances_and_derivatives'],'primary_flat_identity_residual':summaries[1]['flat_geometry_residual'],'physical_linear_comparison':summaries[0]['distances_and_derivatives'],'flexible_angular_comparison':summaries[3]['distances_and_derivatives'],'pair_redshift':scope['pair_zmean_rebin1'],'first_derivative_kernel_redshift':scope['W1_zmean_rebin1'],'second_derivative_kernel_redshift':scope['W2_zmean_rebin1'],'first_derivative_kernel_width':float(np.sqrt(scope['W1_zvar_rebin1'])),'second_derivative_kernel_width':float(np.sqrt(scope['W2_zvar_rebin1'])),'primary_derivative_correlation':summaries[1].get('distance_derivative_correlation'),'primary_retained_steps_over_tau':summaries[1].get('retained_steps_over_max_tau'),'primary_half_chain_max_shift_sigma':float(np.max(abs(np.array(summaries[1].get('half_median_shift_over_posterior_sigma',[np.nan]))))),'source_directory':'https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/','counts_workflow_run':33949291138,'initial_inference_run':33949788858,'first_Kaiser_run':33950476799}
    (out/'delivery_summary.json').write_text(json.dumps(delivery,indent=2))
    primary=summaries[1]
    rows='\n'.join('| '+k+' | '+interval(primary['distances_and_derivatives'][k])+' |' for k in obs)
    text=f'''# Full DESI DR1 v1.5 LRG differential clustering and BAO analysis

## Actual inputs and scope
Both complete NGC and SGC galaxy catalogues and matching random_0 files were processed. No galaxy or random thinning was performed. The analysis selection is 0.4 <= z < 1.1 with finite coordinates and positive finite WEIGHT*WEIGHT_FKP. Selected objects: {totald:,} galaxies and {totalr:,} random points. Input checksums and separate-cap counts are in measurement_summary.json.

This is a full-sample **pre-reconstruction** analysis, not an official DESI likelihood and not a mock-validated publication result. All raw measurements are from real catalogue pair counts. No published BAO-distance points, simulated signals or previous illustrative constraints were used as data.

## Raw measurements
The estimator measures xi and normalized first and second redshift derivatives of xi. Exact random-pair moments through order six are retained, including the fourth moment required to normalize the second derivative. The complete random-pair midpoint histogram is retained for the full sample and all deletions. There are 80 NGC and 40 SGC angular jackknife regions; every deletion removes all pairs touching the deleted region and recomputes normalizations. Cap covariances are summed. The 270-component unbinned joint covariance has rank 118 and is not inverted as a 270-dimensional likelihood.

## Joint distance inference
A CAMB-based oscillatory BAO template is Fourier transformed with anisotropic Gaussian damping, AP-remapped, and averaged through the measured RR light-cone kernels and separation bins. The smooth component is kept in fiducial coordinates with broadband freedom. The primary positive-amplitude Kaiser model fits A(z)[1+beta(z)mu^2]^2. Both distance normalizations and their redshift evolution are free. No relation X'=Y is imposed. The alternative flexible angular-amplitude fit deliberately relaxes the Kaiser restriction and exposes nuisance-model dependence.

The pivot is z_p=0.75 and u=(z-z_p)/0.35. Primary model: ln(alpha_perp)=a0+a1*u+a2*u^2/2, with an independent parallel expansion. The comparison model omits the quadratic AP terms. These finite-dimensional choices mean the quoted distance derivatives are **model-conditional point derivatives at the pivot**, not exactly model-independent windowed distance measurements. The raw xi derivatives are finite-window observables.

Primary 68% marginal intervals:

| Quantity | Median (+upper, -lower) |
|---|---|
{rows}

## Covariance and priors
The raw data points carry diagonal errors from the 120-region jackknife covariance. Inference uses the full cross-covariance among orders and multipoles after rebinning to 8 Mpc/h, with C_used=0.8*C_JK+0.2*diag(C_JK). Covariance sensitivity at shrinkage 0, 0.1 and 0.4 is recorded. No independent-mock inverse-covariance correction is claimed for the jackknife. The likelihood is Gaussian and its covariance uncertainty has not been calibrated using mocks.

Damping priors: Sigma_perp=4.5 +/- 1 and Sigma_parallel=9 +/- 2 Mpc/h, truncated to the template grid. The Kaiser amplitude and beta have the parameterization and broad hard bounds listed in each summary. Broad Gaussian priors on polynomial broadband amplitudes are analytically marginalized. All parameter bounds, chain lengths, autocorrelation estimates, split-chain median shifts and regularization tests are saved.

## Scientific interpretation
First-derivative BAO constraints depend on the treatment of intrinsic angular/redshift evolution. The flexible angular-amplitude model gives substantially weaker radial constraints than the Kaiser model. The second derivatives remain broad and covariance/model dependent; they are not evidence of curvature evolution or a robust shape-function determination. The old fixed-anchor empirical estimates must not be substituted for these joint-fit results.

## Main files
- measurements_rebin1.npz and measurements_rebin2.npz: all raw measured multipoles, N/R moments, RR redshift histograms, full and leave-one realizations, joint covariance, and kernel moments.
- measurements.csv: directly measured xi, dxi/dz, d2xi/dz2 for ell=0,2,4.
- kaiser_quadratic_posterior.npz: extended primary posterior and free distance anchors/slopes/curvatures.
- kaiser_linear_posterior.npz: linear-AP comparison.
- linear/quadratic_posterior.npz: deliberately more flexible angular-amplitude comparisons.
- all_model_comparison.csv and model_sensitivity_*.png: model dependence.
- *_summary.json: numerical audits and priors.
- reproduction/: source code. The two-cap catalogue originals remain in the Library archive; this bundle contains analysis products.

Sources: https://data.desi.lbl.gov/doc/releases/dr1/ ; https://data.desi.lbl.gov/doc/releases/dr1/vac/full-shape-bao-clustering/ ; https://arxiv.org/abs/2404.03000 ; https://camb.readthedocs.io/ . DESI data are used under their published license; cite the release and associated analysis papers and use the official acknowledgments in any publication.
'''
    (out/'README_ANALYSIS.md').write_text(text)
    selected=[('xi_ell0.png','Direct monopole; all selected galaxies and random_0'),('dxi_dz_ell0.png','Direct first derivative, 120-region jackknife errors'),('d2xi_dz2_ell0.png','Direct normalized second derivative, 120-region jackknife errors'),('kaiser_quadratic_fit_order0_ell0.png','Joint physical-template fit: ordinary monopole'),('kaiser_quadratic_fit_order1_ell0.png','Joint physical-template fit: first derivative'),('kaiser_quadratic_fit_order2_ell0.png','Joint physical-template fit: second derivative'),('kaiser_quadratic_joint_derivatives.png','Model-conditional joint distance derivatives at z=0.75'),('kaiser_quadratic_flat_geometry.png','Free-anchor internal geometry comparison'),('model_sensitivity_dDM.png','Transverse derivative: nuisance and basis dependence'),('model_sensitivity_dDH.png','Radial derivative: nuisance and basis dependence')]
    page=['<!doctype html><html><head><meta charset="utf-8"><title>Full DESI LRG differential BAO analysis</title><style>body{font:17px system-ui;max-width:1000px;margin:40px auto;padding:20px;line-height:1.55}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid;padding:9px;text-align:left}pre{white-space:pre-wrap}h2{margin-top:45px}</style></head><body><h1>Full DESI DR1 LRG: differential clustering and BAO</h1>']
    page.append(f'<p>{totald:,} galaxies and {totalr:,} random points, NGC+SGC, no thinning. Pre-reconstruction; internal jackknife uncertainties.</p>')
    page.append('<p><strong>The catalogue statistics are directly measured. The BAO distances are conditional on the specified redshift and intrinsic-clustering models; this is not an official DESI likelihood.</strong></p>')
    page.append('<table><tr><th>Quantity at z=0.75</th><th>68% interval: median (+upper, -lower)</th></tr>')
    for k in obs:page.append('<tr><td>'+html.escape(k)+'</td><td>'+interval(primary['distances_and_derivatives'][k])+'</td></tr>')
    page.append('</table>')
    for filename,title in selected:
        raw=base64.b64encode((out/filename).read_bytes()).decode()
        page.append('<h2>'+html.escape(title)+'</h2><img alt="'+html.escape(title)+'" src="data:image/png;base64,'+raw+'">')
    page.append('<h2>Analysis details and caveats</h2><pre>'+html.escape(text)+'</pre></body></html>')
    (out/'OPEN_THIS_REPORT.html').write_text('\n'.join(page))
    print('DELIVERY',json.dumps(delivery),flush=True)

if __name__=='__main__':main()
