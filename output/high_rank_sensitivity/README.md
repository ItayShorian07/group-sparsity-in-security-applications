# High-rank nonlinear stress-test results

This is a completed **exploratory sensitivity test**, not a replacement for
Experiments 1--3 or the project report. The single declared generator uses two latent
coordinates but 16 equal-power nonlinear Fourier features. At gamma=1 all
three seeds need 16 linear components for 95% of clean nominal energy; the
best five normal-training components explain only 32.6--33.4%.

Fifteen new VAEs were trained from scratch with the unchanged architecture and
training rules. The 180 matched seed-level rows cover three attack scenarios,
five gamma settings, three seeds, and four methods. Both VAE decoders use the
same frozen VAE and residual within each case. OnlineDA rank 5 is the original
baseline; rank 16 is a fixed capacity control using the same Algorithm 2,
penalties, observations, and normal-trained start. It is not a new group-aware
OnlineDA algorithm. OnlineDA adapts causally during testing; VAEs remain frozen.

## Primary outcomes

Means across seeds 17, 42, and 2026 at gamma=1. F1 uses independent normal
calibration targeting 1% FPR; AP is threshold-independent. Lower recovery
error is better.

| Attack scenario | Method | Flow F1 | Flow AP | Relative OD recovery error |
| --- | --- | ---: | ---: | ---: |
| Original independent | VAE+Lasso | 0.0960 | 0.2687 | 1.3202 |
| Original independent | VAE+Group Lasso | 0.0562 | 0.1947 | 1.1388 |
| Original independent | OnlineDA rank 5 | 0.3301 | 0.4814 | 1.2018 |
| Original independent | OnlineDA rank 16 | 0.4914 | 0.9395 | 0.3794 |
| Matched dispersed | VAE+Lasso | 0.1507 | 0.1030 | 1.4949 |
| Matched dispersed | VAE+Group Lasso | 0.1339 | 0.1331 | 1.2861 |
| Matched dispersed | OnlineDA rank 5 | 0.2167 | 0.2302 | 1.3254 |
| Matched dispersed | OnlineDA rank 16 | 0.3551 | 0.4086 | 0.8631 |
| Destination grouped | VAE+Lasso | 0.2348 | 0.2862 | 2.0242 |
| Destination grouped | VAE+Group Lasso | 0.1783 | 0.4347 | 1.6044 |
| Destination grouped | OnlineDA rank 5 | 0.4750 | 0.8771 | 1.0570 |
| Destination grouped | OnlineDA rank 16 | 0.7393 | 1.0000 | 0.1400 |

OnlineDA at both ranks has higher **mean calibrated flow F1 and AP** than either
VAE decoder in all 15 scenario/gamma settings. This is not a claim about every
metric: for example, grouped VAE decoding sometimes gives lower amplitude
error than rank-5 OnlineDA despite worse flow F1 and AP. Three seeds do not
establish statistical significance or universal superiority.

## Important secondary operating point

The fixed absolute flow threshold 0.1 is retained from the existing paper
protocol, not chosen on the new test labels. Its mean F1 outcomes at gamma=1 are:

| Attack scenario | VAE+Lasso | VAE+Group Lasso | OnlineDA rank 5 | OnlineDA rank 16 |
| --- | ---: | ---: | ---: | ---: |
| Original independent | 0.1051 | 0.1088 | 0.0991 | 0.8983 |
| Matched dispersed | 0.1601 | 0.1734 | 0.0728 | 0.4604 |
| Destination grouped | 0.2535 | 0.2551 | 0.1212 | 0.9974 |

Thus there **are** VAE F1 gains over rank-5 OnlineDA at this particular fixed
threshold. They must not be hidden, but they do not survive the primary
normal-calibrated operating point or the rank-16 control, and do not indicate
better AP. A common absolute coefficient threshold is not a common FPR.
We therefore do not replace the original experiment with a selected favorable
metric or generator result.

## Checks and limitations

- All 22 unit tests pass. The gamma=0 metrics match the original experiment;
  original routing, noise, attacks, and linear observations match array-for-array.
- Every original VAE checkpoint and truth-array hash matches the pre-existing
  Experiment-3 export. The existing main results are unchanged.
- All 180,000 Group-Lasso test and 30,000 distinct group-calibration solves meet
  their tolerance. Of 180,000 test solves per other method, strict Lasso flags
  fail for 23 VAE+Lasso, 1 OnlineDA rank-5, and 31 rank-16 cases. Ten rank-16
  calibration and three VAE+Lasso calibration solves also miss the strict flag.
  Training flags additionally fail for one rank-5 and 18 rank-16 subproblems.
  No cases are dropped; all
  normalized KKT residuals in these cases are below 5e-5. Detailed training and
  calibration diagnostics are retained in JSON and raw run files.
- The broad-spectrum routing-aware Fourier loading is deliberately artificial,
  not a realistic production-traffic generator. Matching global routed energy
  does not match every marginal distribution or per-link SNR.
- The VAE architecture and 100-epoch budget are not claimed to be optimal for
  this harder manifold. These negative results do not show that all VAEs fail.
- The recorded-data experiment and the real-data arm of Experiment 3 are
  unaffected. Original tracking controls at gamma=0 are not new nonlinear tests.

## Files

- [Complete results and provenance](results.json): all seeds, settings, primary
  and secondary metrics, paired differences, source hashes, and solver audits.
- [Method comparison](method_comparison.png): AP, calibrated flow F1, and group
  F1 across all five gamma values and three attack scenarios; means +/- 1 SD.
- [Normal-signal geometry](normal_geometry.png): effective linear ranks and
  the fixed rank budgets of the two OnlineDA controls.
- [Protocol and reproduction commands](../../HIGH_RANK_SENSITIVITY.md).

Raw arrays and the 15 newly trained model checkpoints remain local under
`runs/high_rank_sensitivity/`. They do not overwrite the original runs.
