# Full DESI DR1 v1.5 LRG differential clustering and BAO analysis

## Actual inputs and scope
Both complete NGC and SGC galaxy catalogues and matching random_0 files were processed. No galaxy or random thinning was performed. The analysis selection is 0.4 <= z < 1.1 with finite coordinates and positive finite WEIGHT*WEIGHT_FKP. Selected objects: 2,138,627 galaxies and 14,346,758 random points. Input checksums and separate-cap counts are in measurement_summary.json.

This is a full-sample **pre-reconstruction** analysis, not an official DESI likelihood and not a mock-validated publication result. All raw measurements are from real catalogue pair counts. No published BAO-distance points, simulated signals or previous illustrative constraints were used as data.

## Raw measurements
The estimator measures xi and normalized first and second redshift derivatives of xi. Exact random-pair moments through order six are retained, including the fourth moment required to normalize the second derivative. The complete random-pair midpoint histogram is retained for the full sample and all deletions. There are 80 NGC and 40 SGC angular jackknife regions; every deletion removes all pairs touching the deleted region and recomputes normalizations. Cap covariances are summed. The 270-component unbinned joint covariance has rank 118 and is not inverted as a 270-dimensional likelihood.

## Joint distance inference
A CAMB-based oscillatory BAO template is Fourier transformed with anisotropic Gaussian damping, AP-remapped, and averaged through the measured RR light-cone kernels and separation bins. The smooth component is kept in fiducial coordinates with broadband freedom. The primary positive-amplitude Kaiser model fits A(z)[1+beta(z)mu^2]^2. Both distance normalizations and their redshift evolution are free. No relation X'=Y is imposed. The alternative flexible angular-amplitude fit deliberately relaxes the Kaiser restriction and exposes nuisance-model dependence.

The pivot is z_p=0.75 and u=(z-z_p)/0.35. Primary model: ln(alpha_perp)=a0+a1*u+a2*u^2/2, with an independent parallel expansion. The comparison model omits the quadratic AP terms. These finite-dimensional choices mean the quoted distance derivatives are **model-conditional point derivatives at the pivot**, not exactly model-independent windowed distance measurements. The raw xi derivatives are finite-window observables.

Primary 68% marginal intervals:

| Quantity | Median (+upper, -lower) |
|---|---|
| DM_over_rd | 17.582 (+0.459, -0.465) |
| DH_over_rd | 20.796 (+1.041, -1.025) |
| dDM_over_rd_dz | 19.217 (+1.723, -1.721) |
| dDH_over_rd_dz | -10.547 (+3.318, -3.413) |
| d2DM_over_rd_dz2 | 13.911 (+19.069, -19.581) |
| d2DH_over_rd_dz2 | -29.923 (+44.738, -49.170) |

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
