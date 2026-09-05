# Completed full DESI DR1 v1.5 LRG analysis

The local ChatGPT execution service failed before executing Python. The analysis was run to completion on clean GitHub Actions runners on an isolated branch; main was unchanged.

## Actual measurements
All 2,138,627 selected LRG galaxies and all 14,346,758 selected random_0 points from NGC and SGC; no thinning. Redshift 0.4 <= z < 1.1. Pre-reconstruction, 0.05 degree angular pair cut. Separation 40--160 Mpc/h in 6 Mpc/h bins and 20 mu bins. 120 spatial delete-one realizations.

xi, first derivative and normalized second derivative are measured directly from pair-redshift moments. RR moments through order six and random pair redshift histograms were computed. These are finite-resolution derivative observables.

## Conditional BAO results
Both AP distance anchors and their evolution were fitted. CAMB-based acoustic template, a light-cone average with the actual s,mu,z random kernels, linear or quadratic log-AP evolution, profiled amplitude/broadband evolution.

The initial fit saturated damping limits, so the final diagnostic fit uses positive endpoint damping in [0,30] Mpc/h and beta endpoints in [0.001,3]. The high-redshift beta still reaches its lower limit in most jackknife fits. Hence the quoted BAO errors are model-conditional diagnostic errors, not a calibrated DESI cosmological likelihood. Curvature/shape-function claims are especially premature.

The fit uses 50-percent diagonal shrinkage of the joint jackknife correlation matrix; parameter uncertainties come from refitting all 120 deletions. Alternative regularization and separation ranges are tabulated.

## Files
plots/: actual measurements with jackknife errors and revised BAO fits.
measured_multipoles.csv: all three derivative orders and ell=0,2,4, with errors.
measured_clustering_all_jackknife_covariance.npz: all leave-one-out multipoles, covariance, exact random moments, and the mean redshift kernels.
robustness_summary.json: final linear/quadratic model fits, cross-covariances, boundary hits and sensitivity checks.
code/: reproduction scripts.

The full original artifact also contains every deletion's complete redshift kernel, omitted from this compact package. Original run 33950255764; revised fit run 33951316368.

No published BAO distance table or simulated clustering was used as the measured data. No post-reconstruction, mock-covariance, fiber-assignment/RIC/AMR validation or official-DESI equivalence is claimed. S0/S1 normalized at z=0 cannot be inferred from this LRG interval without additional information.
