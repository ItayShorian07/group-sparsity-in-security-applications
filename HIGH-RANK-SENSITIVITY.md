# Exploratory high-rank nonlinear sensitivity test

## Question and reporting rule

Can a two-latent-dimensional nonlinear normal-traffic generator with a broad
linear spectrum change the comparison between VAE sparse decoding and OnlineDA?
The generator and settings in `high-rank-sensitivity.json` are fixed before
evaluating its new test metrics. This is an exploratory stress test prompted by
the earlier results, not an independently preregistered confirmatory experiment.
No generator search or selection for favorable VAE test scores is performed.
Original Experiments 1--3, their datasets, and the main project-report results are preserved.
A new result is not substituted for the original merely because it is favorable.

## Generator

Retain the exact original network, two Gaussian latent coordinates, noise,
normal splits, and independent signed unit attacks for seeds 17, 42, and 2026.
The linear endpoint is checked array-for-array against the saved Experiment-2
observations, routing matrix, and attacks.

Map each latent coordinate through the standard-normal CDF to a uniform angle:

```text
theta_j = 2*pi*Phi(w_j)
h(w) = sqrt(2) * [sin(theta dot k), cos(theta dot k)] for each declared k
k in {(1,0),(0,1),(1,1),(1,-1),(2,0),(0,2),(2,1),(1,2)}
```

These 16 nonlinear features are orthogonal in population under the unchanged
Gaussian latent distribution. They are deterministic functions of two latent
coordinates, not 16 new independent latent variables or added measurement noise.
The fundamental modes (1,0) and (0,1) retain both angular coordinates.

Create 16 random orthonormal link directions within the column space of the
known routing matrix R. A minimum-norm OD preimage C satisfies R C = Q. The
nonlinear OD signal is `(h(w)-mean_train(h)) C.T`, with a single training-only
scalar matching the original linear endpoint's routed RMS. The namespace 4040
fixes this new topology-conditioned loading without changing old random streams.

This routing-aware construction deliberately produces broad link-level linear
energy. It is an artificial mathematical stress test, not an independently
sampled realistic OD traffic model or evidence about production traffic.
It allows signed nominal deviations, as does the original Gaussian generator.
Fourier order, directions, number of modes, and scaling are not selected using
test labels or comparative method scores.
OD marginal distributions and correlation patterns change as well; matching
global routed RMS does not isolate every statistical property or every
per-link signal-to-noise ratio. A result is specific to this constructed family.

Before training, require normal-training rank95 >= 12 and top-five energy <=
50% at the nonlinear endpoint. Failure stops the run; it does not trigger an
automatic search for another generator. Gamma remains 0, .25, .5, .75, 1,
interpolating between the unchanged linear endpoint and the new nonlinear one.
Intermediate RMS may differ, as in the original interpolation; it is recorded.

## Models and controls

- Train 15 VAEs from scratch on the new normal observations (three seeds by
  five gamma settings). Reuse the original architecture, latent width 2, KL
  weight, optimizer, maximum 100 epochs, patience, and validation-only selection.
  The old VAE checkpoints are not treated as models of this new distribution.
- Apply VAE+Lasso and VAE+Group Lasso to identical frozen-VAE residuals within
  each case, with the original penalties, destination groups, and numerical rules.
- Run the unchanged Algorithm-2 OnlineDA implementation at rank 5, including
  normal training, independent calibration, and causal within-test adaptation.
- Add OnlineDA rank 16 as a fixed capacity control. It changes only the rank,
  not the algorithm, data, penalties, normal splits, or detection thresholds.
  This is not a parameter selected for the best test result.

All methods use the same observations and routing matrix. Each test scenario
and normal-calibration stream starts from a separate normal-trained state.
OnlineDA adapts during testing; VAE weights remain frozen. Equal parameter
counts or independently optimal hyperparameters are not claimed.
In particular, keeping the original architecture and 100-epoch budget does
not establish that a VAE is optimally trained for the new, harder manifold.
The nonlinear endpoint has linear rank 16; intermediate mixtures with the
original two linear modes can span up to 18 dimensions. Thus the rank-16
control is not claimed to be an oracle for every intermediate setting.

## Attack scenarios and evaluation

Repeat the synthetic comparisons of Experiment 3: the original independent
signed attacks, matched dispersed positive attacks, and destination-grouped
positive attacks. The two latter cases retain exactly the original matched
attack times and budgets; noise and topology are also unchanged.

Primary metric: flow F1 with independent normal-calibrated thresholds targeting
1% FPR. Secondary metrics include flow AP, false positives, group/time metrics,
attack-amplitude error, and the pre-existing fixed flow threshold 0.1. Report
all gamma settings, all seeds, and all scenarios, not just favorable endpoints.
Three seeds do not support a strong statistical-significance claim. A mean
improvement in one metric is not automatically an improvement in the others.

Experiment 1 and the real-data arm of Experiment 3 use recorded CICIDS2017,
so this generator does not change or require retraining those models. The
separate Experiment-2 tracking controls were evaluated at gamma=0, whose data
is unchanged; they are not silently relabeled as high-rank nonlinear tests.

## Reproduce

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m traffic_vae.high_rank_sensitivity --config high-rank-sensitivity.json
.venv/bin/python -m traffic_vae.high_rank_results
```

The command refuses to overwrite an existing output directory. To rerun, use
a copied configuration with a new `output_dir`. Raw models, predictions,
normal-training geometry, source hashes, numerical diagnostics, and all result
rows remain under `runs/high_rank_sensitivity/`. There is no test-set early
stopping or test-label tuning.
The second command exports every completed comparison, the unchanged original
linear-control regression check, paired seed differences, solver diagnostics,
and two scientific figures under `output/high-rank-sensitivity/`. It does not
replace `output/experiment-03/results.json`, the main figures, or the project report.

## Completed outcome

The complete run trained all 15 VAEs and evaluated all 180 matched method
rows. OnlineDA at both declared ranks has higher mean normal-calibrated flow
F1 and AP than either VAE decoder in all 15 scenario/gamma settings. There is
an important operating-point exception: at the secondary fixed threshold 0.1,
the VAEs beat rank-5 OnlineDA in some settings, including all three scenarios
at gamma=1. That advantage is absent against rank 16 and does not imply a
better AP or calibrated F1. The full positive and negative outcomes, including
numerical exceptions and limitations, are in the
[result summary](output/high-rank-sensitivity/README.md). Original results and
the main project-report results have not been replaced.
