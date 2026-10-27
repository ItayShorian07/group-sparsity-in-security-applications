# Experiment 3: does a group penalty improve a frozen VAE detector?

This extension compares VAE+Lasso with VAE+Group Lasso on both the recorded
CICIDS2017 DDoS flows of Experiment 1 and the synthetic routed network of
Experiment 2. The synthetic arm also includes the unchanged Algorithm-2
OnlineDA baseline from the paper. It does not train another VAE, change either earlier experiment,
or present score denoising as physical attack-amplitude recovery.

## Reproduce

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m traffic_vae.experiment_03 --config experiment-03.json
.venv/bin/python -m traffic_vae.experiment_03_online --source runs/experiment_03_full --output runs/experiment_03_onlineda
.venv/bin/python -m traffic_vae.group_figures --online-source runs/experiment_03_onlineda
```

Both experiment commands refuse to overwrite an existing output directory.
To rerun the VAE comparison, change only `output_dir` in a copied configuration;
for the OnlineDA extension, supply the corresponding `--source` and a new `--output`.
Source data, checkpoints,
and splits must already exist from Experiments 1 and 2. Raw data and arrays
remain local under `runs/`; portable per-seed results, audits, and provenance
are exported to `output/experiment-03/results.json`.

The preliminary endpoint-only run is retained under `runs/experiment_03`.
The final run under `runs/experiment_03_full` covers every original gamma:
`0, 0.25, 0.5, 0.75, 1`. Expanding coverage did not change the penalties,
group definitions, attacked-time budget, or operating-point rule.

## A. Recorded flow data: a score-level experiment

Reuse exactly the three seeds (17, 42, 2026), 114,806 Monday normal validation
scores, and 225,104 Friday test scores from Experiment 1. CSV hashes and source
row IDs are checked against the original run. There is no retraining and no
selection of flows using test labels. CICIDS2017 is recorded laboratory traffic,
not an independently sampled production deployment.

Each score `s_i` is the existing deterministic VAE reconstruction norm.
Define groups by destination IP and a fixed 60-second wall-clock window.
Group membership uses metadata, never attack labels. Windows are processed
after completion; this is an offline, bounded-window procedure, not a causal
per-flow alarm. IP and timestamp are still excluded from VAE input features.

There is no measured link-to-OD routing matrix in these CSVs. Therefore the
real-data arm solves an **identity-design score denoising** problem:

```text
Lasso:       min_b .5 ||s-b||_2^2 + lambda ||b||_1
Group Lasso: min_b .5 ||s-b||_2^2 + lambda sum_g sqrt(|g|) ||b_g||_2
```

Both have exact proximal solutions. For nonnegative scores the Lasso output is
`max(s-lambda, 0)`. Group Lasso scales each block by
`max(1-lambda*sqrt(|g|)/||s_g||_2, 0)`. These are estimated anomaly scores,
not inferred physical attack volumes. This limited real-data test is not
identical to routed recovery in the synthetic arm.

Primary lambda is the median of the normal validation scores, separately for
each frozen VAE. Sensitivity multipliers `0.5, 1, 2` were specified before the
run; multiplier 1 remains primary, irrespective of test performance. No test
label chooses lambda, window size, groups, or detection thresholds.

Flow thresholds are normal-score 99th percentiles. Group scores are block RMS,
with thresholds from normal validation groups. A group is positive if it
contains at least one labeled DDoS flow. All metrics use the dataset labels as
provided. Monday validation was also used for early stopping in Experiment 1:
this extension does **not** retroactively create independent calibration data.
For this identity-design model the output group RMS is exactly the positive
part of the original group RMS minus lambda. Thus group rankings receive no
new information from this particular shrinkage; it is a limited postprocessing
test, not a claim of an end-to-end group-aware representation.

The test has 8,349 groups, 22 containing attacks; all 22 also contain benign
flows. Median size is 2, maximum size is 8,252, and 42.1% are singletons. Thus
an alarm for a group is not a claim that every member flow is attacked. The
sqrt-size weights reduce one obvious size bias but cannot eliminate all effects
of the very unequal group sizes and cross-day distribution shift.

## B. Routed synthetic data: the unchanged Experiment-2 data and matched controls

Reuse all 15 frozen Experiment-2 VAEs (three seeds and five gamma settings),
their exact clean signals, routing matrices, noise, original signed attacks,
and 2,000 normal calibration samples / 4,000 test samples. Reconstructed original
test observations are checked against saved reconstructions. Within the VAE
pair only the decoder changes. Groups are the 15 destination nodes, each containing 14 incoming OD
flows, determined from topology rather than labels.

```text
r_t = observed_link_traffic - frozen_VAE_reconstruction
Lasso:       min_a .5 ||r_t-Ra||_2^2 + .11 ||a||_1
Group Lasso: min_a .5 ||r_t-Ra||_2^2 + .11 sum_g sqrt(14) ||a_g||_2
```

The third method, **OnlineDA**, receives the exact same link observations and
routing matrix, not the VAE residual. It retains the Experiment-2 Algorithm-2
implementation: rank 5, ridge .36, L1 penalty .11, forgetting factor 1, and
post-update nominal output. Its normal-prefix SVD initialization (128 samples)
and normal training are explicit adaptations to the paper's random-basis /
zero-statistics initialization. This is not OnlineDA with a new group penalty.

`experiment_03_online` restores all normal-trained sufficient statistics
(basis, Gram matrices, cross products) saved in Experiment 2. Normal calibration
and each test scenario start from **separate copies of the same trained state**;
updates are causal and continuous within a stream. It replays all 15 original
calibration streams and all 15 original test streams, requiring agreement with
the saved Experiment-2 estimates before accepting the extension. New scenarios
are run on their actual new observations, not inferred from old metrics.
The VAE pair isolates the choice of penalty; the third baseline compares whole
systems. OnlineDA adapts during testing while VAE parameters stay frozen, so
differences do not isolate architecture or sparsity alone. There is no OnlineDA
arm on the recorded flow-score data, which is a different inverse problem.

The three attack scenarios are:

1. `paper_independent`: the exact original Experiment-2 Bernoulli signed attacks.
2. `matched_dispersed`: 300 randomly selected times, each with 14 distinct,
   uniformly sampled OD flows receiving positive unit attacks.
3. `destination_grouped`: the same 300 times and 14 positive unit attacks per
   active time, but all are incoming flows to one uniformly chosen destination.

The latter two cases have exactly the same attack-time support, 4,200 nonzero
flow-time cells (0.5% prevalence), and OD attack energy. The dispersed support
is sampled without replacement conditional on its count; it is not independent
Bernoulli sampling. Routed link energy can differ because of path overlap;
the audit records it explicitly. This is not a claim of a perfectly isolated
grouping effect under equal observed signal-to-noise ratio.

Matched cases are generated once per seed and reused at all gamma values.
Their labels are used only for evaluation. The VAE sees only its original normal
training data; test reconstructions are recomputed for each observation. Each
decoder receives exactly the same residual within a case. The original Lasso
estimates and normal calibration estimates are reused rather than silently
changing the baseline.

Report flow-time, destination-group-time, and time-level detection; average
precision; false positives; and relative attack-amplitude recovery error. Group
scores are L2 norms (all groups have equal size). Thresholds use the relevant
normal calibration scores, targeting 1% FPR. Zero-score ties can make achieved
calibration FPR lower, including zero; achieved rates are saved. F1 across
different attack scenarios is not directly comparable because prevalence and
the number of attacked groups differ. Compare methods within the same case.
The portable export additionally evaluates the fixed absolute flow threshold
0.1 already used in Experiment 2 / the paper as a **secondary** operating point.
It is not optimized on these test labels and does not replace the primary
normal-calibrated results.

## Results

At the primary real-data setting, mean flow F1 is 0.8076 for Lasso and 0.8078
for Group Lasso, while FPR increases from 0.3192 to 0.3312 and AP decreases from
0.6353 to 0.6349. There is no meaningful improvement in this score-level test.

On the exact original synthetic attacks, Group Lasso has worse flow F1 and AP
at every gamma. On destination-grouped attacks it improves flow AP at every
gamma: at gamma=1, AP increases from 0.4543 to 0.8096 and relative OD attack
recovery error decreases from 1.1425 to 0.7311. However, calibrated F1 decreases
from 0.2347 to 0.1565. At the secondary fixed threshold 0.1 it is 0.3768 versus
0.3664. The ranking/recovery gain is not a demonstrated F1 gain.

OnlineDA has higher mean flow F1 and AP than either VAE decoder in all 15
scenario/gamma settings. At gamma=1 on destination-grouped attacks its flow F1
is 0.4949, AP is 0.9998, and relative OD recovery error is 0.1894. Its F1 at the
secondary fixed threshold 0.1 is 0.8268. Thus the grouped VAE's gain over
VAE+Lasso is not superiority to the paper-based online method. This benchmark
still compares a causally adapting tracker with frozen VAE models.

The OnlineDA extension adds 45 seed-level rows (135 synthetic rows in the
combined export), covering 180,000 test and 30,000 distinct normal-calibration
solves. Seventeen test solves and eight normal-calibration solves miss the
strict Lasso convergence flag; maximum normalized KKT residuals are
4.8631e-5 and 4.8811e-5. These cases are retained. All 15 original test streams
and 15 normal-calibration streams reproduce the saved Experiment-2 estimates
within rtol=1e-10, atol=1e-12. Original two-decoder files are not overwritten;
the extension lives under `runs/experiment_03_onlineda` and its hashes,
parameters, replay checks, and solver diagnostics enter the portable export.

All 180,000 grouped test solves and 30,000 distinct grouped calibration solves
meet their KKT tolerance. Twenty of 180,000 test Lasso solves miss the strict
flag; the worst normalized KKT residual is 4.8435e-5. These are disclosed, not
silently omitted. Eighteen unit tests pass, including a primal-dual gap check,
complete OnlineDA state restoration, independent scenario starts, and continuous
chunked updates.

## Numerical checks and limits

The group solver uses proximal gradient acceleration with adaptive restart and
sqrt-size block soft thresholding. It checks the group KKT conditions, normalized
by `max(1, max_g ||R_g.T r||_2)`, rather than using loss stagnation as evidence
of convergence. The tolerance is `1e-5`, with 10,000 iterations maximum. Unit
tests cover exact identity-design solutions, singleton equivalence to Lasso on
a nonorthogonal design, unequal / reordered groups, invalid partitions, and
matched attack budgets.

Numerical convergence is not a guarantee of correct or unique attack recovery.
The routing map has only 52 rows for 210 flow coefficients. A group penalty
does not guarantee within-group sparsity. The fixed `.11` penalties, with the
declared size-weight convention, are not claimed to be independently optimal.
Three model seeds on one capture are not three independently sampled datasets.
There is no end-to-end learned optimization layer or superiority claim over
all possible group-sparse models.

## References

- [CICIDS2017 dataset documentation](https://www.unb.ca/cic/datasets/ids-2017.html).
- [Yuan and Lin: Group Lasso](https://www.columbia.edu/~my2550/papers/glasso.final.pdf).
- [Mardani et al.: Dynamic Anomalography](https://arxiv.org/abs/1208.4043).

The project report contains the concise results and interpretation; the JSON export
preserves every seed, sensitivity setting, threshold, and solver audit.
