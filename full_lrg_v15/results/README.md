# Full DESI DR1 v1.5 LRG catalogue analysis

All 2,138,627 selected galaxies and all 14,346,758 selected random_0 points, NGC+SGC. No thinning.

## Completed measurements
Ordinary xi, normalized first and second redshift derivatives, all 120 spatial leave-one-out realizations, full cross-covariance, exact random moments through sixth order, and redshift-resolved random kernels.

## BAO inference
Both transverse and radial AP anchors are free. CAMB acoustic template; independent smooth log-AP evolution. Linear and quadratic evolution fits are provided, not silently identified as model-free point derivatives. Errors are from repeated fits to all jackknife deletions.

## Limitations
- Pre-reconstruction only; not the official reconstructed DESI pipeline.
- One complete random realization per sky cap, not all official random realizations.
- Spatial jackknife covariance, not a mock-validated covariance.
- Full correlation matrix regularized by 50% diagonal shrinkage; parameter errors from refitting all jackknife deletions.
- Fiducial CAMB acoustic template and finite AP expansion; damping ratio fixed to 1.6.
- Actual (s,mu,z) RR kernel used; no separately calibrated fiber-assignment/RIC/AMR systematic correction.
- Pivot-normalized shape ratios can be unstable and are diagnostic, not independent measurements.

## Numerical results
```json
{
  "status": "full_catalogue_measurement_and_diagnostic_BAO_fits_completed",
  "input_version": "DESI DR1 v1.5",
  "data_used": 2138627,
  "randoms_used": 14346758,
  "data_fraction": 1.0,
  "random_fraction": 1.0,
  "njack": 120,
  "pivot_redshift": 0.75,
  "rdrag_fid_Mpc": 147.0897795559183,
  "kernel_diagnostics": {
    "pair_z": 0.7625118106676831,
    "first_derivative_z": 0.7461128643691824,
    "second_derivative_z": 0.7464858976144186,
    "pair_sigma_z": 0.1690354997439708,
    "first_derivative_sigma_z": 0.1398379302161262,
    "second_derivative_sigma_z": 0.12122216187311376
  },
  "model_note": "Distance inference is conditional on the explicitly specified smooth AP evolution model; clustering derivative measurements do not require this model.",
  "limitations": [
    "Pre-reconstruction only; not the official reconstructed DESI pipeline.",
    "One complete random realization per sky cap, not all official random realizations.",
    "Spatial jackknife covariance, not a mock-validated covariance.",
    "Full correlation matrix regularized by 50% diagonal shrinkage; parameter errors from refitting all jackknife deletions.",
    "Fiducial CAMB acoustic template and finite AP expansion; damping ratio fixed to 1.6.",
    "Actual (s,mu,z) RR kernel used; no separately calibrated fiber-assignment/RIC/AMR systematic correction.",
    "Pivot-normalized shape ratios can be unstable and are diagnostic, not independent measurements."
  ],
  "linear_AP_evolution": {
    "DM_over_rd": {
      "value": 18.033409597133346,
      "jackknife_sigma": 0.2789071344343018
    },
    "DH_over_rd": {
      "value": 19.99504247377658,
      "jackknife_sigma": 0.6815263796972082
    },
    "dDM_over_rd_dz": {
      "value": 19.971498877163402,
      "jackknife_sigma": 1.1717305398551399
    },
    "dDH_over_rd_dz": {
      "value": -11.029182664039924,
      "jackknife_sigma": 3.243004459403169
    },
    "d2DM_over_rd_dz2": {
      "value": -9.65546777991919,
      "jackknife_sigma": 2.7393262011101522
    },
    "d2DH_over_rd_dz2": {
      "value": 7.0240990853271015,
      "jackknife_sigma": 3.5927498735641077
    },
    "flat_FLRW_Xprime_minus_Y": {
      "value": -0.02354359661317673,
      "jackknife_sigma": 1.3256564317950215
    },
    "jackknife_convergence": 120,
    "boundary_hits": [
      0,
      0,
      0,
      0,
      0,
      117,
      95,
      120
    ],
    "parameter_names": [
      "ln_alpha_perp",
      "ln_alpha_parallel",
      "dln_alpha_perp_dz",
      "dln_alpha_parallel_dz",
      "beta0",
      "dln_beta_dz",
      "Sigma_perp0",
      "dSigma_perp_dz"
    ],
    "all_AP_parameters": [
      -0.02972396821950552,
      0.018128922130222727,
      0.050503178657448376,
      0.05838632637221124,
      0.13192777763946278,
      -2.9999999828719908,
      4.000000000000001,
      -4.999999999999999
    ],
    "geometry_covariance": [
      [
        0.0002390785868505009,
        -0.0003755148425372339,
        -0.0002774931640144736,
        0.0004936378049612452
      ],
      [
        -0.0003755148425372339,
        0.001161896850454601,
        0.0005269874510133343,
        -0.0009342656386849103
      ],
      [
        -0.0002774931640144736,
        0.0005269874510133343,
        0.004547149388338723,
        -0.00549472046415407
      ],
      [
        0.0004936378049612452,
        -0.0009342656386849103,
        -0.00549472046415407,
        0.02484474649277664
      ]
    ]
  },
  "quadratic_AP_evolution": {
    "DM_over_rd": {
      "value": 17.35361255253062,
      "jackknife_sigma": 0.32867802074782176
    },
    "DH_over_rd": {
      "value": 21.13653946640595,
      "jackknife_sigma": 0.7534927943778328
    },
    "dDM_over_rd_dz": {
      "value": 19.214942979314177,
      "jackknife_sigma": 1.1383502617000492
    },
    "dDH_over_rd_dz": {
      "value": -11.727777869674327,
      "jackknife_sigma": 2.163101148542773
    },
    "d2DM_over_rd_dz2": {
      "value": 27.228573597327614,
      "jackknife_sigma": 12.122379645323575
    },
    "d2DH_over_rd_dz2": {
      "value": -50.18081461005013,
      "jackknife_sigma": 31.394891382252286
    },
    "flat_FLRW_Xprime_minus_Y": {
      "value": -1.9215964870917723,
      "jackknife_sigma": 1.4085226695822883
    },
    "jackknife_convergence": 120,
    "boundary_hits": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      118,
      120,
      120
    ],
    "parameter_names": [
      "ln_alpha_perp",
      "ln_alpha_parallel",
      "dln_alpha_perp_dz",
      "dln_alpha_parallel_dz",
      "d2ln_alpha_perp_dz2",
      "d2ln_alpha_parallel_dz2",
      "beta0",
      "dln_beta_dz",
      "Sigma_perp0",
      "dSigma_perp_dz"
    ],
    "all_AP_parameters": [
      -0.06814939343804871,
      0.07364782664896405,
      0.0502899893812941,
      0.055124194008871814,
      2.1049367802801906,
      -2.7290277477580256,
      0.139979448413492,
      -2.9999999694421065,
      4.000000000000001,
      -4.999999999999999
    ],
    "geometry_covariance": [
      [
        0.00035874186554810046,
        -0.00028669439958458524,
        -0.00017247147219441228,
        0.00023095805498414998,
        -0.009261711898685961,
        0.0063917667130313875
      ],
      [
        -0.00028669439958458524,
        0.0012735617144103211,
        0.00015386226351791192,
        -5.5856825383037533e-05,
        0.008982387770485612,
        -0.03973073473698901
      ],
      [
        -0.00017247147219441228,
        0.00015386226351791192,
        0.004246008359750955,
        -0.003110137601392263,
        -0.0012367700910362988,
        0.00133056795246233
      ],
      [
        0.00023095805498414998,
        -5.5856825383037533e-05,
        -0.003110137601392263,
        0.010012017294073654,
        0.005249929888099837,
        -0.018181069315938287
      ],
      [
        -0.009261711898685961,
        0.008982387770485612,
        -0.0012367700910362988,
        0.005249929888099837,
        0.5025106575500942,
        -0.45095589379506734
      ],
      [
        0.0063917667130313875,
        -0.03973073473698901,
        0.00133056795246233,
        -0.018181069315938287,
        -0.45095589379506734,
        1.9651480369109322
      ]
    ]
  },
  "stability_checks": [
    {
      "shrinkage": 0.2,
      "smin": 60,
      "smax": 150,
      "success": true,
      "dDM_over_rd_dz": 19.865783742309503,
      "dDH_over_rd_dz": -11.181782274364188,
      "theta": [
        -0.030973074705301265,
        0.02481017416523534,
        0.04601788425530661,
        0.05447834155704418,
        0.19130432125426397,
        0.49868176214609106,
        4.000000000000001,
        -4.999999999999999
      ],
      "objective": 34.95951969077154
    },
    {
      "shrinkage": 0.8,
      "smin": 60,
      "smax": 150,
      "success": true,
      "dDM_over_rd_dz": 19.987120603553297,
      "dDH_over_rd_dz": -11.142059289764351,
      "theta": [
        -0.02870290229165526,
        0.0166106411083969,
        0.05023833545453912,
        0.051894404598548445,
        0.12526172157324508,
        -2.999999999999985,
        4.052506726418135,
        -4.999999999999972
      ],
      "objective": 17.336057860128165
    },
    {
      "shrinkage": 0.5,
      "smin": 66,
      "smax": 144,
      "success": true,
      "dDM_over_rd_dz": 20.533806301621432,
      "dDH_over_rd_dz": -10.877031204303993,
      "theta": [
        -0.03026670632731979,
        0.018968620179247175,
        0.08230276294357976,
        0.06645237815698272,
        0.1535221790037226,
        -0.41412479291768595,
        5.210436144487463,
        -3.7742752721890778
      ],
      "objective": 14.19460689702991
    }
  ],
  "shape_diagnostics": {
    "present_day_normalization_measured": false,
    "pivot_redshift": 0.75,
    "S1_S0_pivot_denominator_significant_at_2sigma": true,
    "fraction_of_redshift_grid_with_S2_denominator_significant_at_2sigma": 0.5414364640883977,
    "interpretation": "Only formal delta-method, quadratic-log-AP-model conditional diagnostics. A small denominator or model/boundary dependence invalidates a Gaussian shape-constraint interpretation."
  }
}
```