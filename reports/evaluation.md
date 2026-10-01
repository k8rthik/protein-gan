Matrices: 64x64, clip 50 A, split `val` (1043 real windows), 512 generated. NN RMSE and coverage are against the `train` split.

| source | symmetry err (A) | diag err (A) | triangle viol % | bond mean (A) | bond ok % | contact dens | rel contact order | Rg (A) | dist JS | dist W1 (A) | MDS stress-1 | neg eig mass | 3D bond ok % | NN RMSE (A) | self div (A) | coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `dcgan_raw` | 2.01 | 0.46 | 2.54% | 4.24 | 50.2% | 0.030 | 0.285 | 13.8 | 0.007 | 0.39 | 0.122 | 0.195 | 30.1% | 5.28 | 9.69 | 0.86 |
| `dcgan_symmetrized` | 0.00 | 0.00 | 1.93% | 4.25 | 53.4% | 0.023 | 0.258 | 13.7 | 0.006 | 0.51 | 0.122 | 0.195 | 30.1% | 5.04 | 9.43 | 0.86 |
| `real_held_out` | 0.00 | 0.00 | 0.00% | 3.81 | 99.7% | 0.032 | 0.311 | 14.0 | 0.000 | 0.00 | 0.002 | 0.003 | 97.6% | 4.69 | 9.85 | 0.55 |
| `baseline_gaussian` | 10.18 | 17.86 | 12.46% | 17.94 | 1.3% | 0.145 | 0.391 | 14.2 | 0.044 | 1.01 | 0.469 | 0.389 | 2.1% | 11.22 | 12.90 | 0.09 |
| `baseline_shuffled_distances` | 0.00 | 0.00 | 9.57% | 18.12 | 3.1% | 0.121 | 0.390 | 14.0 | 0.000 | 0.03 | 0.475 | 0.410 | 1.6% | 10.39 | 12.51 | 0.17 |
| `baseline_residue_permuted` | 0.00 | 0.00 | 0.00% | 18.11 | 3.1% | 0.121 | 0.391 | 14.0 | 0.000 | 0.03 | 0.004 | 0.005 | 3.1% | 10.28 | 12.51 | 0.37 |
