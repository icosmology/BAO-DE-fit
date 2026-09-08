#!/usr/bin/env python3
"""Assemble the actual QSO analysis, figures and numerical audit into a PDF.
Does not fabricate unavailable measurements or substitute LRG plots.
"""
import argparse,json,shutil,hashlib,html
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def fmt(q):
    return f"{q['median']:.3f} [{q['p16']:.3f}, {q['p84']:.3f}]"

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',default='qso_measurements');p.add_argument('--fits',default='fits');p.add_argument('--output',default='qso_delivery');a=p.parse_args()
    raw=Path(a.raw);fitroot=Path(a.fits);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plots=out/'figures';plots.mkdir(exist_ok=True)
    summary=json.loads((raw/'measurement_summary.json').read_text())
    report={}
    for model in ['linear','quadratic']:
        path=next(fitroot.rglob(f'{model}_summary.json'));report[model]=json.loads(path.read_text())
        for src in path.parent.glob('*'):
            if src.is_file() and src.suffix in ['.png','.svg','.csv','.json']:
                shutil.copy2(src,plots/src.name)
    for src in raw.glob('*'):
        if src.is_file() and src.suffix in ['.png','.svg','.csv','.json']:shutil.copy2(src,plots/src.name)
    galaxies=sum(v['selected_data_rows'] for v in summary['caps'].values());randoms=sum(v['selected_random_rows'] for v in summary['caps'].values())
    r=np.load(raw/'measurements_rebin1.npz');s=r['s'];R=r['R'][0];H=r['H'][0];mask=(s>=52)&(s<=148)
    # Representative aggregate kernel for visualization only; the likelihood
    # uses separation- and angle-dependent individual kernels.
    rr=R[mask].sum(axis=(0,1));h=H[mask].sum(axis=(0,1));rawm=rr/rr[0];mean=rawm[1];m2=rawm[2]-mean**2
    m3=rawm[3]-3*mean*rawm[2]+2*mean**3;m4=rawm[4]-4*mean*rawm[3]+6*mean**2*rawm[2]-3*mean**4
    nz=len(h);ze=np.linspace(.8,2.1,nz+1);zm=.5*(ze[1:]+ze[:-1]);dz=ze[1]-ze[0]
    z=np.divide(h[:,1],h[:,0],out=zm.copy(),where=h[:,0]>0);K=h[:,0]/rr[0]/dz
    u=z-1.49-mean;q1=u/m2;q2=2*(u*u-m2-(m3/m2)*u)/(m4-m2*m2-m3*m3/m2)
    zfine=np.linspace(.8,2.1,2601);idx=np.minimum(nz-1,((zfine-.8)/dz).astype(int));kf=K[idx]
    A1=kf*np.interp(zfine,z,q1);A2=kf*np.interp(zfine,z,q2)
    W1=-cumulative_trapezoid(A1,zfine,initial=0)
    W2=cumulative_trapezoid(cumulative_trapezoid(A2,zfine,initial=0),zfine,initial=0)
    fig,ax=plt.subplots(figsize=(8.4,5.3));ax.plot(z,K,label='Normalized RR pair-redshift density')
    ax.plot(zfine,W1,label='First-derivative kernel (binned reconstruction)');ax.plot(zfine,W2,label='Second-derivative kernel (binned reconstruction)')
    ax.set(xlabel='Redshift z',ylabel='Kernel density',title='QSO random-pair redshift resolution: representative aggregate')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(plots/'redshift_kernels.png',dpi=180);plt.close(fig)
    for name,q in [('first',q1),('second',q2)]:
        fig,ax=plt.subplots(figsize=(8.4,5.3));ax.plot(z,q);ax.axhline(0,linestyle='--',linewidth=.7)
        ax.set(xlabel='Pair midpoint redshift',ylabel=f'{name}-derivative pair weight',title=f'QSO signed redshift weight: {name} order')
        fig.tight_layout();fig.savefig(plots/f'weight_{name}.png',dpi=180);plt.close(fig)
    np.savetxt(out/'representative_kernel_weights.csv',np.column_stack([z,K,q1,q2]),delimiter=',',header='z,RR_kernel,q1,q2',comments='')
    # Show model dependence at the pivot without asserting independent data.
    for key,label in [('dDM_over_rd_dz',r'$d(D_M/r_d)/dz$'),('dDH_over_rd_dz',r'$d(D_H/r_d)/dz$')]:
        fig,ax=plt.subplots(figsize=(8.4,4.6))
        for j,model in enumerate(['linear','quadratic']):
            q=report[model]['distances_and_derivatives'][key]
            ax.errorbar(q['median'],j,xerr=[[q['median']-q['p16']],[q['p84']-q['median']]],fmt='o',capsize=4)
        ax.set_yticks([0,1],['Linear log-AP; xi and first derivative','Quadratic log-AP; all three channels'])
        ax.set(xlabel=label,title='QSO model/channel sensitivity at z=1.49');fig.tight_layout();fig.savefig(plots/f'model_comparison_{key}.png',dpi=180);plt.close(fig)
    # Official DR1 reports an isotropic QSO distance, not anisotropic slopes.
    fig,ax=plt.subplots(figsize=(8.4,4.6))
    ax.errorbar(26.07,0,xerr=.67,fmt='s',capsize=4,label='Official DR1 post-reconstruction [1]')
    for j,model in enumerate(['linear','quadratic'],1):
        q=report[model]['kinematics']['DV_over_rd'];ax.errorbar(q['median'],j,xerr=[[q['median']-q['p16']],[q['p84']-q['median']]],fmt='o',capsize=4)
    ax.set_yticks([0,1,2],['Published DR1','This analysis: linear','This analysis: quadratic'])
    ax.set(xlabel=r'$D_V(1.49)/r_d$',title='Context comparison, not an independent-data consistency test')
    fig.tight_layout();fig.savefig(plots/'official_DV_context.png',dpi=180);plt.close(fig)
    # Joint raw correlation structure.
    C=r['covariance'];sig=np.sqrt(np.diag(C));cor=C/np.outer(sig,sig)
    fig,ax=plt.subplots(figsize=(7.8,6.6));im=ax.imshow(cor,origin='lower',vmin=-1,vmax=1,interpolation='nearest');fig.colorbar(im,ax=ax,label='Correlation coefficient')
    ax.set(xlabel='Joint data-vector index',ylabel='Joint data-vector index',title='QSO xi, first and second derivatives: full jackknife correlation')
    fig.tight_layout();fig.savefig(plots/'joint_correlation_matrix.png',dpi=180);plt.close(fig)
    font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');bold=font.with_name('DejaVuSans-Bold.ttf')
    if font.exists():
        pdfmetrics.registerFont(TTFont('DejaVu',str(font)));pdfmetrics.registerFont(TTFont('DejaVu-Bold',str(bold)));base='DejaVu';strong='DejaVu-Bold'
    else:base='Helvetica';strong='Helvetica-Bold'
    st=getSampleStyleSheet()
    for key in ['Normal','BodyText','Title','Heading1','Heading2','Heading3']:
        st[key].fontName=base if key in ['Normal','BodyText'] else strong
    st['BodyText'].fontSize=10;st['BodyText'].leading=15;st['BodyText'].spaceAfter=9
    st['Title'].fontSize=25;st['Title'].leading=32
    st['Heading1'].fontSize=17;st['Heading1'].leading=22;st['Heading1'].spaceAfter=13
    st['Heading2'].fontSize=13;st['Heading2'].leading=18
    st.add(ParagraphStyle(name='CaptionSmall',fontName=base,fontSize=9,leading=13,spaceAfter=8))
    st.add(ParagraphStyle(name='Equation',fontName=base,fontSize=11,leading=18,leftIndent=14,spaceBefore=7,spaceAfter=12))
    story=[]
    def P(text,style='BodyText'):return Paragraph(text,st[style])
    def add(text,style='BodyText'):story.append(P(text,style))
    def table(rows,widths):
        tab=Table([[P(str(c),'CaptionSmall') for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EEEEEE')),('LINEBELOW',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#CCCCCC')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(tab);story.append(Spacer(1,10))
    add('DESI DR1 QSO','Title');add('Direct differential clustering<br/>and exploratory BAO inference','Title')
    story.append(Spacer(1,20));add('Full selected NGC + SGC catalogues · complete random_0 · 0.8 ≤ z &lt; 2.1','Heading2')
    add('A single report of measured correlation functions, first and second redshift derivatives, joint distance posteriors, shape-function diagnostics, contours and numerical checks.')
    story.append(Spacer(1,15));add('Status of the results','Heading2')
    add('The clustering points are obtained from actual QSO/random pair counts. Distance and shape-function curves are conditional on a finite log-AP redshift expansion, a Kaiser angular model, nuisance priors and a regularized internal covariance. They are not official DESI likelihood results or model-free point derivatives.')
    add(f'Processed: {galaxies:,} selected QSOs and {randoms:,} selected random points. No object thinning. Pre-reconstruction. Analysis date: 8 September 2026.')
    add('All figures in this PDF were generated by this QSO run. No earlier LRG constraints, rescaled forecasts or published BAO-distance points were used as fitting data.')
    add('Guide: dataset and estimator → measured multipoles → distance likelihood → posterior bands and contours → shape-ratio diagnostics → model and numerical checks.','CaptionSmall')
    story.append(PageBreak())
    add('1. Dataset and directly measured observables','Heading1')
    rows=[['Region','Selected QSOs','Selected randoms','Jackknife regions']]
    for cap,v in summary['caps'].items():rows.append([cap,f"{v['selected_data_rows']:,}",f"{v['selected_random_rows']:,}",v['njack']])
    table(rows,[70,140,155,120])
    add('The inputs are the public DR1 LSS iron v1.5 QSO clustering-ready FITS catalogues and random realization 0 [1]. Standard catalogue correction weights are multiplied by WEIGHT_FKP. The selection is applied identically to data and randoms. NGC and SGC pair counts and normalizations are formed separately before combination.')
    rows=[['Accepted pairs','NGC','SGC']]
    for kind in ['DD','DR','RR']:rows.append([kind,*[f"{summary['caps'][cap]['pair_counts'][kind]:,}" for cap in ['NGC','SGC']]])
    table(rows,[75,205,205])
    add('Pair separations: 40–160 h⁻¹ Mpc; bins: 4 h⁻¹ Mpc and 20 bins in |μ|. Angular pairs below 0.05 degrees are excluded. Each spatial deletion removes every pair touching its region and recomputes catalogue normalizations and RR moments. The cap-wise jackknife covariance contributions are added; no independent-mock covariance correction is asserted.')
    add(f"Raw joint covariance rank: {summary['covariance_rank_rebin1']} for 270 entries. This matrix is not inverted at its full dimension.")
    story.append(PageBreak())
    add('2. Derivative estimators and resolution','Heading1')
    add('In each separation/angle cell, let u be pair midpoint redshift minus its RR-weighted mean and let mₙ = ⟨uⁿ⟩RR. The baseline random-pair density K integrates to one. All numerator pair counts use the same catalogue normalizations; signed moments are never independently normalized.')
    add('q₁ = u / m₂','Equation')
    add('q₂ = 2 [u² − m₂ − (m₃/m₂)u] / [m₄ − m₂² − m₃²/m₂]','Equation')
    add('Measured RR moments through sixth order supply the second-derivative normalization and resolution moments. The q₂ weight has zero constant and linear response, including the skewness correction. Pair-count and analytic polynomial-response tests are included with the reproduction products [2].')
    add('ξ[q₁] = ∫ W₁(z) ∂ξ/∂z dz;   ξ[q₂] = ∫ W₂(z) ∂²ξ/∂z² dz','Equation')
    add('These integration-by-parts identities do not require a Taylor model for ξ(z). Their kernels vary with separation and angle; they should not be interpreted as delta functions at an effective redshift. The plots use s² times the corresponding measured multipole.')
    rows=[['Kernel diagnostic','First derivative','Second derivative']]
    rows.append(['RR-weighted mean redshift',f"{summary['W1_zmean_rebin2']:.5f}",f"{summary['W2_zmean_rebin2']:.5f}"])
    rows.append(['Square root of mean kernel variance',f"{np.sqrt(summary['W1_zvar_rebin2']):.5f}",f"{np.sqrt(summary['W2_zvar_rebin2']):.5f}"])
    table(rows,[245,120,120])
    add('The second-derivative statistic measures curvature of the full clustering signal: geometrical evolution, bias, growth, redshift-space distortions, damping and residual selection effects all contribute. It is not by itself D_H″ or D_M″.')
    story.append(PageBreak())
    add('3. Joint BAO model and uncertainty conventions','Heading1')
    add('The reference oscillatory spectrum is calculated using CAMB and separated from a smooth spectrum. The BAO part is AP-remapped with independent transverse and radial distances, anisotropically damped, transformed to configuration space and averaged through measured RR redshift kernels and separation bins. The smooth baseline has independent broadband freedom. No published QSO BAO distances enter the fit [1,3].')
    add('u = (z − 1.49) / 0.65;   ln α_A(z) = a_A,0 + a_A,1 u + ½ a_A,2 u²','Equation')
    add('The transverse and radial coefficient sets are independent. Both distance normalizations are free. The primary quadratic fit uses ξ, ξ′ and ξ″; the comparison linear fit omits the quadratic AP terms and uses ξ and ξ′. This comparison changes both the model and the channels, so it is not an information-gain measurement.')
    add('The amplitude is positive, A(z), with Kaiser angular form A(z)[1+β(z)μ²]². Log amplitude has quadratic redshift freedom; β is linear and remains positive. Effective damping priors are Σ⊥ = 3 ± 1 and Σ∥ = 8 ± 3 h⁻¹ Mpc. They approximate, rather than separately calibrate, QSO radial smearing. All hard bounds and broadband priors are listed in the machine-readable summaries.')
    add('Fitting uses 8 h⁻¹ Mpc bins over 52–148 h⁻¹ Mpc and the monopole/quadrupole. C_used = 0.8 C_JK + 0.2 diag(C_JK), with best-fit sensitivity checks at diagonal shrinkage 0.1 and 0.4. Figure error bars are raw 1σ jackknife diagonal errors. Shaded distance/shape bands are pointwise posterior quantiles, not simultaneous confidence envelopes.')
    add('Two independently seeded ensembles are run for each redshift model. The report gives autocorrelation and split-chain diagnostics but does not claim mock-calibrated coverage or a discovery significance. Redshift quadrature is checked against the original 0.01-spaced RR histogram.')
    story.append(PageBreak())
    add('4. Distance and differential-distance results','Heading1')
    add('All values below are dimensionless BAO distances or their derivatives per unit redshift, evaluated from the fitted functions at z_p = 1.49. Brackets contain central 68% marginal posterior intervals. The two fits use the same catalogue; they are not independent measurements.')
    rows=[['Quantity','Linear model + ξ, ξ′','Quadratic model + ξ, ξ′, ξ″']]
    lab={'DM_over_rd':'D_M / r_d','DH_over_rd':'D_H / r_d','dDM_over_rd_dz':'d(D_M/r_d)/dz','dDH_over_rd_dz':'d(D_H/r_d)/dz','d2DM_over_rd_dz2':'d²(D_M/r_d)/dz²','d2DH_over_rd_dz2':'d²(D_H/r_d)/dz²'}
    for key,title in lab.items():rows.append([title,fmt(report['linear']['distances_and_derivatives'][key]),fmt(report['quadratic']['distances_and_derivatives'][key])])
    table(rows,[150,167,168])
    add('The second derivatives in the linear-AP fit follow from that restricted model; they are not independent measurements of the second-derivative channel. Neither fit imposes X′ = Y, where X = D_M/r_d and Y = D_H/r_d.')
    rows=[['Derived quantity','Linear','Quadratic']]
    for key in ['DV_over_rd','q_pivot','dDMdz_minus_DH']:rows.append([key,fmt(report['linear']['kinematics'][key]),fmt(report['quadratic']['kinematics'][key])])
    table(rows,[150,167,168])
    add('The published DR1 QSO result is isotropic D_V/r_d = 26.07 ± 0.67 at z_eff = 1.49 [1]. It does not provide official QSO dD_M/dz or dD_H/dz measurements. Our context comparison differs in reconstruction, catalogue version, model and covariance, and shares galaxies with the official measurement; no independent-data discrepancy p-value is calculated.')
    story.append(PageBreak())
    add('5. Shape functions: diagnostic, not an unqualified detection','Heading1')
    add('Use F(a) = a³/[D_H(a)/r_d]² and x = ln a. To avoid extrapolating QSOs to the unobserved present-day boundary, normalize S₀ and S₁ at a_p = 1/(1+1.49), not a = 1 [4].')
    add('S₀,p = (a/a_p)³ − 3[F(a) − F(a_p)] / F_x(a_p)','Equation')
    add('S₁,p = (a_p/a)³ F_x(a) / F_x(a_p);   S₂ = −F_xx(a) / [3F_x(a)]','Equation')
    add('The flat matter+Λ reference has S₀,p = S₁,p = 1 and S₂ = −1. S₀,p and S₁,p have exactly zero spread at the pivot by definition: that pinch is not a precision measurement. The paper’s present-day-normalized S₀ and S₁ cannot be obtained from this QSO-only interval without a continuation assumption or lower-redshift data.')
    add('Each posterior sample is transformed analytically; F_x is not forced to remain positive. Denominator zero crossings can generate poles and heavy tails. The figures show signed-log axes and posterior quantiles together with geometrical-prior quantiles. No arbitrary cut removes singular histories. Means and Gaussian variances are not asserted for these ratio distributions.')
    rows=[['Pole/sensitivity diagnostic','Linear','Quadratic']]
    for key,caption in [('posterior_fraction_Fx_crosses_zero','Posterior fraction with F_x sign crossing'),('prior_fraction_Fx_crosses_zero','Prior fraction with F_x sign crossing')]:rows.append([caption,*[f"{100*report[n]['shape_audit'][key]:.1f}%" for n in ['linear','quadratic']]])
    table(rows,[275,105,105])
    add('A visually structured median of a broad or singular ratio distribution is not evidence for dynamical dark energy. Distances and derivatives remain conditional on the explicit log-AP basis and nuisance model; the shape plots expose, rather than hide, that limitation.')
    story.append(PageBreak())
    add('6. Numerical validation and limitations','Heading1')
    rows=[['Diagnostic','Linear','Quadratic']]
    rows.append(['Estimated effective sample count',*[f"{report[n]['effective_sample_estimate']:.0f}" for n in ['linear','quadratic']]])
    rows.append(['Largest independent-ensemble median shift / posterior σ',*[f"{max(abs(np.array(report[n]['ensemble_median_difference_in_posterior_sigma']))):.3f}" for n in ['linear','quadratic']]])
    rows.append(['Largest redshift-quadrature change / raw data σ',*[f"{report[n]['redshift_quadrature_max_error_in_raw_diagonal_sigma']:.4f}" for n in ['linear','quadratic']]])
    rows.append(['Maximum parameter boundary fraction',*[f"{100*max(report[n]['boundary_fractions'].values()):.2f}%" for n in ['linear','quadratic']]])
    table(rows,[285,100,100])
    add('The exact counter is verified against exhaustive deterministic pair enumeration, including all delete-one contributions, all seven RR moments and disjoint partition completeness. The analytic polynomial-response test checks the unit normalization of ξ″ using the measured RR moments. These are algorithm tests, not cosmological mock validation.')
    add('Remaining limitations: pre-reconstruction; one complete random realization rather than a suite; internal jackknife covariance and shrinkage; finite redshift and angular discretization; Gaussian effective radial damping; nuisance-model dependence; and no injected-cosmology mock coverage tests. Same-galaxy cross-covariances are required before combining this compression with official DESI constraints.')
    add('Robust deliverable: the actual full-sample ξ, ξ′ and normalized ξ″ measurements, their shared covariance and RR response information. Exploratory deliverable: joint BAO and shape-function posterior diagnostics with transparent priors and conditioning.')
    story.append(PageBreak())
    add('7. Sources and reproducibility','Heading1')
    sources=[('[1] DESI DR1 catalogues and BAO analysis','https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/'),('[1a] DESI 2024 III: BAO from galaxies and quasars','https://arxiv.org/abs/2404.03000'),('[2] Optimal redshift weighting and validation','https://arxiv.org/abs/1411.1424'),('[2a] Redshift-weighted BAO after reconstruction','https://arxiv.org/abs/1604.01050'),('[3] CAMB','https://camb.readthedocs.io/'),('[4] Shape-function definitions, Eq. (5)','https://arxiv.org/abs/2504.06118')]
    for title,url in sources:add(f'{title}<br/><link href="{url}">{url}</link>','CaptionSmall')
    add('Reproduction files are in the results archive: adapted pair counter, reducer, QSO likelihood, MCMC driver and this report builder. Input file SHA-256 checksums and selected-row counts are retained in measurement_summary.json; model summaries contain all bounds, priors, numerical diagnostics and covariance sensitivity checks.')
    add('The source branch is chatgpt-qso-dr1-analysis-20260908 in icosmology/BAO-DE-fit. Main was not changed. Figures are generated from this run’s saved arrays and posteriors. The archived input catalogues are public DESI products; any scientific publication must include DESI’s required release citations and acknowledgments.')
    # All scientific figures and contour figures, each on its own spacious page.
    figure_order=[]
    for base_name in ['redshift_kernels','weight_first','weight_second','joint_correlation_matrix']:
        figure_order.append(plots/f'{base_name}.png')
    for group in ['xi','dxi_dz','d2xi_dz2']:
        for ell in [0,2,4]:figure_order.append(plots/f'{group}_ell{ell}.png')
    for model in ['quadratic','linear']:
        figure_order.extend(sorted(plots.glob(f'{model}_fit_order*.png')))
        for key in ['DM','DH','dDM_dz','dDH_dz','d2DM_dz2','d2DH_dz2','joint_anchors','joint_first_derivatives','joint_second_derivatives','flat_geometry','S0','S1','S2','chain_trace']:
            figure_order.append(plots/f'{model}_{key}.png')
    figure_order.extend(sorted(plots.glob('model_comparison_*.png')));figure_order.append(plots/'official_DV_context.png')
    used=[]
    for path in figure_order:
        if not path.exists():raise FileNotFoundError(f'Expected actual figure missing: {path}')
        if path in used:continue
        used.append(path);story.append(PageBreak())
        title=path.stem.replace('_',' ')
        add(f'Figure {len(used)} | {title}','Heading1')
        from PIL import Image as PILImage
        with PILImage.open(path) as im:w,hg=im.size
        width=485;height=width*hg/w
        if height>510:width*=510/height;height=510
        story.append(Image(str(path),width=width,height=height,hAlign='CENTER'))
        if path.stem.startswith(('xi_','dxi_','d2xi_')):caption='Actual full-sample QSO pair-count measurement. Error bars are 1σ diagonal uncertainties from the shared 120-region spatial jackknife covariance. Neighbouring bins and derivative channels are correlated. No signal was simulated.'
        elif '_S' in path.stem:caption='Model-conditional ratio diagnostic. Solid/shaded posterior quantiles are compared with the geometrical prior. Signed-log scaling displays heavy tails. F_x zero crossings are retained. Pivot normalization, not data, fixes S0p and S1p to one at z=1.49.'
        elif 'joint_' in path.stem and 'matrix' not in path.stem:caption='Approximate 68% and 95% marginal highest-density contours from smoothed posterior histograms. They include the stated covariance shrinkage and nuisance priors, not independent-mock uncertainty calibration.'
        elif 'trace' in path.stem:caption='Ensemble-mean chain traces. These supplement, rather than replace, the saved autocorrelation, independent-ensemble and split-chain diagnostics.'
        elif 'weight' in path.stem or 'kernel' in path.stem:caption='Representative RR-weighted aggregate for display. The actual estimator and likelihood retain separation- and angle-dependent redshift responses. Exact normalization uses measured moments, not the plotted binned approximation.'
        else:caption='Exploratory QSO catalogue inference. Redshift basis, nuisance model, effective radial smearing and covariance regularization are specified in the method section. Dashed LCDM curves are references, not independent data.'
        story.append(Spacer(1,14));add(caption,'CaptionSmall')
        add(path.name,'CaptionSmall')
    pdf=out/'DESI_DR1_QSO_Differential_BAO_Report.pdf'
    def decorate(canvas,doc):
        canvas.saveState();canvas.setFont(base,8);canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(48,24,'DESI DR1 QSO | direct catalogue analysis | exploratory inference')
        canvas.drawRightString(A4[0]-48,24,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=45,bottomMargin=45,title='DESI DR1 QSO differential clustering and BAO report',author='Catalogue analysis prepared for the user')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    # Render every page and check the produced PDF programmatically.
    import fitz
    pages=out/'pdf_previews';pages.mkdir(exist_ok=True)
    checks=[]
    with fitz.open(pdf) as pd:
        for i,page in enumerate(pd):
            pix=page.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
            if i<7 or i%5==0:pix.save(pages/f'page_{i+1:03d}.png')
            text=page.get_text();checks.append({'page':i+1,'text_characters':len(text),'rendered_width':pix.width,'rendered_height':pix.height})
            assert len(text)>30,(i,text)
            for word in page.get_text('words'):
                assert word[0]>=-1 and word[1]>=-1 and word[2]<=page.rect.width+1 and word[3]<=page.rect.height+1,('text overflow',i,word)
        npage=len(pd)
    delivery={'source':'Actual DESI DR1 v1.5 QSO NGC+SGC','galaxies':galaxies,'randoms':randoms,'no_thinning':True,'pre_reconstruction':True,'pdf_pages':npage,'scientific_figures':len(used),'pdf_bytes':pdf.stat().st_size,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pdf_render_and_text_checks':'PASS','counts_summary':summary,'model_results':{n:{'distances':report[n]['distances_and_derivatives'],'kinematics':report[n]['kinematics'],'effective_samples':report[n]['effective_sample_estimate'],'shape_audit':report[n]['shape_audit']} for n in report}}
    (out/'delivery_summary.json').write_text(json.dumps(delivery,indent=2));(out/'pdf_quality_checks.json').write_text(json.dumps(checks,indent=2))
    for source in ['prepare_qso.py','run_inference.py','make_report.py']:shutil.copy2(Path('qso_analysis')/source,out/source)
    print('DELIVERY_SUMMARY',json.dumps(delivery),flush=True)

if __name__=='__main__':main()
