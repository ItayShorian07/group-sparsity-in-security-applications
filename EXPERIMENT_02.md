# Experiment 02: fully synthetic nonlinearity sweep

Question: as the normal signal becomes less compatible with a linear low-rank
model, does a normal-trained VAE outperform batch RPCA in anomaly detection and
signal recovery? This experiment tests that hypothesis; it does not assume the
answer. Group-sparse estimators are not part of this comparison yet.

## Run

The experiment requires no CIC-IDS2017 files. Run from the repository root:

```bash
.venv/bin/python -m traffic_vae.experiment_02 --config experiment_02.json
```

For a background run on the VM:

```bash
mkdir -p runs
nohup .venv/bin/python -m traffic_vae.experiment_02 --config experiment_02.json \
  > runs/experiment_02.console.log 2>&1 < /dev/null &
tail -f runs/experiment_02.console.log
```

The first experiment's source and outputs are unchanged. Both runs consume CPU;
waiting until experiment 01 finishes avoids resource contention. No GPU is used.
Existing output directories are refused to prevent accidental overwrites.

## Generated data

Arrays have shape `(time, feature)`: 50 feature channels represent 50 synthetic
flows, with consecutive groups of five. These are real-valued deviations, not
literal nonnegative byte/packet counts. Each split has an independent latent
trajectory, with shared generator matrices:

```text
z(t) = [sin(2*pi*t/288 + phase), cos(2*pi*t/288 + phase), load(t)]
load(t) = 0.9*load(t-1) + sqrt(1 - 0.9^2)*epsilon(t)
L_linear = z A
L_nonlinear = sin(2 z B) + 0.3 (z C)^2
L(alpha) = (1 - alpha) L_linear + alpha L_nonlinear
D = L(alpha) + S + N
```

The factor 2 inside sine is fixed, not learned. Each component is centered per
feature and divided by one scalar RMS estimated from training data only. This
matches endpoint energy without changing linear rank (at most three). Intermediate
mixtures need not have equal energy; clean/noise/attack RMS values are reported
so changes in effective signal-to-noise ratio remain visible.

At each seed, A, B, C, trajectories, noise, and attacks are generated once and
reused for every alpha. The VAE initialization seed is also reset for each alpha.
There is no VAE in the data generator and no test-driven choice of alpha.

| Setting | Default | Meaning |
| --- | --- | --- |
| `alphas` | 0, .25, .5, .75, 1 | From entirely linear to entirely nonlinear |
| `seeds` | 17, 42, 2026 | Paired generator and optimization replicates |
| `train_size` | 6000 | Normal-only training time points |
| `validation_size` | 2000 | Normal-only early stopping and threshold calibration |
| `test_size` | 2000 | Held-out signal with injected attacks |
| `noise_std` | .05 | Independent Gaussian noise standard deviation |
| `attack_amplitude` | 2 | Positive additive offset in generator units |
| `attack_duration` | 12 | Consecutive affected time points |
| `attacked_time_fraction` | .1 | Approximate time occupancy after block rounding |
| `group_size` | 5 | Channels affected together in each attack |
| `latent_dim` | 3 | VAE bottleneck matches the known latent dimension |

Attacks occupy randomly selected, nonoverlapping duration-sized time slots. One
random group is attacked in each selected slot. The realized support is saved and
used for evaluation. Most group-time cells remain unmodified. The support is
structured rather than uniformly random, so classical RPCA recovery assumptions
are not guaranteed even at alpha=0.

## Preprocessing and methods

A shared affine scaler is fitted to the noisy normal training observations for each
alpha. No logarithm or clipping is used: additivity is preserved. Models receive
observations, not the hidden clean signal or attack labels. Ground truth is used
only by diagnostics and evaluation.

The VAE reuses experiment 01's model and trainer, with latent dimension three.
It trains on normal observations, restores the best validation reconstruction
checkpoint, and reconstructs test observations at the posterior mean. Its attack
estimate is the residual `D - L_hat`; it includes unrecovered dense noise. This is
a plain VAE baseline, not a group-sparse VAE or an exact probabilistic decomposition.

RPCA solves the penalized noisy decomposition:

```text
min_(L,S) 0.5 ||D - L - S||_F^2 + lambda_* ||L||_* + lambda_1 ||S||_1
lambda_1 = rpca_sparse_penalty = 0.1
lambda_* = lambda_1 * sqrt(max(D.shape))
```

Proximal gradient uses step 1/2, singular-value shrinkage for L, and elementwise
soft thresholding for S. The joint smooth loss has gradient Lipschitz constant 2.
The stopping criterion is the norm of the proximal-gradient mapping divided by
`max(1, ||D||_F)`, with tolerance 1e-5 and up to 1000 iterations. Objective traces,
final ranks, mapping norms, and convergence flags are saved. Reaching the limit is
reported explicitly and does not count as convergence or successful recovery.

This is a penalized stable RPCA baseline, not noiseless equality-constrained PCP.
The nuclear/L1 framework is described in [Candes et al., Robust Principal Component
Analysis?](https://candes.su.domains/publications/downloads/RobustPCA.pdf), and dense
noise motivates [Zhou et al., Stable Principal Component Pursuit](https://arxiv.org/abs/1001.2363).
The fixed penalty magnitude is an initial setting, not a tuned optimum.

**Information access differs:** VAE learns from separate normal data and processes
test samples individually. RPCA decomposes the full validation and test matrices
separately, using context within each batch. It receives no clean test signal but
does adapt to the contaminated test batch. Validation/test sizes are required to
match to keep the RPCA penalty scale and batch size matched. This is a comparison
of these two workflows, not a strictly isolated causal test of representation
nonlinearity. A matched normal-trained linear baseline and penalty sensitivity
study are needed before making broader claims of VAE superiority.

## Evaluation

Both methods use a threshold calibrated on normal validation outputs, with target
FPR 1%. VAE also uses validation for early stopping, so calibration is not independent
of model selection. No test-label hyperparameter tuning is performed.

For each method, anomaly scores are L2 norms of its estimated S at time and group
levels and absolute values at element level. Each level has its own threshold;
group calibration pools normal group-time scores. Scores use the common standardized
space. Recovery errors use the original generator units:

```text
Error_L = ||L_hat - L||_F / ||L||_F
Error_S = ||S_hat - S||_F / ||S||_F
```

Precision, recall, F1, FPR, average precision, trapezoidal PR-AUC, ROC-AUC, prevalence,
and confusion counts are saved at time, group-time, and element levels. These are
cell-level metrics, not attack-event detection rates. Group metrics alone do not
establish a benefit from group sparsity; neither estimator uses that penalty.

The clean test matrix's singular values and ranks capturing 95% and 99% energy are
reported. Alpha=0 is a low-rank sanity case. Numerical energy rank and model
performance are measured, not forced to change monotonically with alpha.

## Outputs

- `config.json`, `provenance.json`: resolved settings, environment, source hashes.
- `seed_*/generator.npz`: latent trajectories, maps, component normalization.
- `seed_*/alpha_*/truth.npz`: alpha, raw L/D/N for every split, test S, affine scaler.
- `seed_*/alpha_*/spectrum.json`: clean-signal singular values and energy ranks.
- `seed_*/alpha_*/vae/`: model, training history, estimates, metrics.
- `seed_*/alpha_*/rpca/`: solver diagnostics, convergence histories, estimates, metrics.
- `metrics.csv`: one row per seed, alpha, and method; updated after each fit.
- `summary.csv`: means and sample standard deviations across seeds.
- `paired_vae_minus_rpca.csv`: per-seed method differences for each alpha.
- `rank_diagnostics.csv`: ranks and component RMS values.
- `completed.json`: completion marker, RPCA convergence status, plot status.

Three seeds characterize only limited variability and are not a confidence interval
for general network traffic. All estimates and truth are local generated artifacts
under `runs/`, excluded from Git.

Optional Matplotlib support creates `comparison.png` and `comparison.pdf` showing
F1, FPR, AP, recovery errors, and energy ranks against alpha. Error bars are sample
standard deviations across paired seeds. To enable plots before running:

```bash
.venv/bin/python -m pip install -e '.[plots]'
```

Without Matplotlib, the full experiment still completes and saves plot-ready CSVs.
The nonlinearity hypothesis can fail: VAE may reconstruct attacks too well, RPCA
may remain competitive, or mixed signals may have altered effective SNR. Such
outcomes must be retained and reported.
