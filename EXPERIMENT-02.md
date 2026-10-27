# Experiment 02: paper-model linear control and nonlinear sweep

This experiment asks whether a normal-trained VAE gains an advantage over online
Dynamic Anomalography as nominal traffic becomes nonlinear. At `gamma=0`, the
synthetic network follows the model and main settings described in Section VI-A
of [Mardani, Mateos, and Giannakis](https://arxiv.org/abs/1208.4043). Higher gamma
values use our nonlinear extension.

The paper does not provide its sampled node coordinates, random matrices, random
seed, or generated arrays. This code therefore produces new Monte Carlo samples
from the stated model. It matches the published dimensions and distributions;
it does not recreate the authors' exact random realization or reported numbers.

## Run

Validate configuration without generating data or training:

```bash
.venv/bin/python -m traffic_vae.experiment_02 --config experiment-02.json --validate-only
```

Run all gamma values in the background from the repository root:

```bash
mkdir -p runs
nohup .venv/bin/python -m traffic_vae.experiment_02 --config experiment-02.json \
  > runs/experiment_02_paper.console.log 2>&1 < /dev/null &
```

Monitor progress separately:

```bash
tail -f runs/experiment_02_paper.console.log
```

Existing output directories are refused. To run only the linear control in a new
directory:

```bash
.venv/bin/python -m traffic_vae.experiment_02 --config experiment-02.json \
  --gammas 0 --output-dir runs/experiment_02_paper_linear
```

## Paper-model linear endpoint

The synthetic graph has 15 nodes placed uniformly in the unit square. Two nodes
are linked in both directions when their Euclidean distance is below 0.35. A new
connected graph with exactly 52 directed links is sampled for each seed. Every
ordered source-destination pair is an OD flow, giving:

```text
F = N (N - 1) = 15 * 14 = 210 flows
R.shape = (52 links, 210 flows)
```

Minimum-hop paths define the routing matrix `R`. Shortest-path ties use a fixed
sorted-neighbor rule. The graph coordinates, links, OD pairs, routing matrix, and
number of rejected topology draws are saved for audit.

For each time point and split:

```text
W[t, i] ~ Normal(0, 1)
U[f, i] ~ Normal(0, 1/F)
Z_linear = W U.T
A[f, t] in {-1, 0, +1}, with total nonzero probability 0.005
N[l, t] ~ Normal(0, 0.01^2)
D = Z_linear R.T + A R.T + N                  when gamma = 0
```

The nominal rank is two. All 52 link measurements are observed, corresponding to
the paper's `pi=1` case. The static-routing OnlineDA settings use rank bound 5,
forgetting factor 1, L1 penalty 0.11, and ridge penalty 0.36, matching the setting
reported for the paper's static online comparison where applicable.

Notation follows the reference: `alpha` is routing-change probability, `pi` is
link-observation probability, and the OnlineDA forgetting factor is the paper's
`beta`. The VAE KL coefficient is named `vae_kl_weight` to avoid reusing beta.

## Nonlinear extension

The paper's synthetic generator is linear; it does not define our nonlinearity
parameter. We introduce independent maps `B` and `C` and define:

```text
G(W) = sin(2 W B) + 0.3 ((W C)^2 - E[(W C)^2])
Z_nonlinear = c G(W)
Z(gamma) = (1 - gamma) Z_linear + gamma Z_nonlinear
D = Z(gamma) R.T + A R.T + N
```

The training-only scalar `c` matches the routed RMS energy of the two endpoints.
The nonlinear component, attacks, noise, graph, and latent draws are generated
once per seed and reused across gamma values. This pairing isolates gamma within
each seed. Intermediate mixtures can still have different energy, so signal,
noise, attack RMS, and effective ranks are saved.

Following the paper's notation, `alpha` is reserved for the per-step probability
of a routing-link removal/addition event. The main nonlinear sweep sets `alpha=0`
and `pi=1` to isolate representation geometry. Our `gamma` controls only nominal
traffic nonlinearity. The implementation also accepts time-varying `R_t` and
observation masks `Omega_t` for separate tracking stress tests.

## Data splits

| Split | Time points | Purpose |
| --- | ---: | --- |
| Train | 6000 | Normal observations for both methods |
| Validation | 2000 | VAE early stopping |
| Calibration | 2000 | Independent normal detection thresholds |
| Test | 4000 | Held-out observations with sparse signed attacks |

Using separate normal training and validation data adapts the original online
experiment so that the VAE and tracker can begin from a matched normal baseline.
It is extra information relative to the paper's original streaming setup.

## Methods

`VAE_Lasso` trains a VAE on normal 52-dimensional link observations. The VAE has
hidden widths 128 and 64 and a two-dimensional latent bottleneck. After reconstructing
normal link traffic, it maps the link residual back to sparse OD-flow anomalies:

```text
A_hat = argmin_a 0.5 ||(D - L_hat) - R a||_2^2 + 0.11 ||a||_1
```

The VAE is frozen during testing. This method is a VAE followed by a routing-aware
sparse decoder, not a group-sparse VAE.

`OnlineDA` implements the coefficient and subspace updates in Algorithm 2 with
the actual routing matrix, full observations, and static routing. It estimates a
sparse 210-dimensional flow anomaly at each time and updates its link-level basis
causally. The output uses the updated basis, as specified by Algorithm 2.

Initialization is an explicit adaptation: an SVD of the first 128 normal training
observations initializes the basis and sufficient statistics. The remainder of
training is processed once. Calibration and test begin from independent copies of
the trained state, preventing calibration-to-test state leakage. OnlineDA adapts
during each stream; VAE weights remain fixed.

Both methods solve Lasso subproblems with the same numerical implementation and
save normalized KKT residuals, iteration counts, and convergence flags. Fixed
penalties are not independently optimized for every gamma.

The VAE does **not** learn attacks. It is trained only on nominal link traffic and
learns a nonlinear model of normal behavior. At test time, an attack is the part
of the link observation that the nominal reconstruction does not explain; the
known routing matrix and the elementwise Lasso then localize that residual to OD
flows.

## Why group sparsity is not in the main comparison

The reference model penalizes the anomaly matrix with an elementwise L1 norm and
draws its synthetic attack entries independently. It neither defines attack
groups nor uses a Group-Lasso penalty. Consequently, replacing L1 by a grouped
penalty in this experiment would introduce a new assumption unsupported by the
paper and by its generated labels. Group sparsity is therefore reserved for a
separate extension in which groups have an external meaning (for example source,
destination, service, subnet, or a temporal campaign) and attacks are generated
or labeled at that same group level.

## Evaluation

Normal calibration scores set a threshold targeting 1% false-positive rate.
An additional fixed absolute flow-amplitude threshold of 0.1 mirrors the operating
point used in the paper. Absolute values count both positive and negative attacks.

Metrics include precision, recall, F1, false-positive rate, average precision,
and ROC-AUC at two levels:

- `time_*`: a time point is positive if any of its 210 flows is attacked.
- `flow_*`: each flow-time cell is evaluated separately; this is the main anomaly
  localization result.
- `paper_threshold_flow_*`: flow metrics using the fixed 0.1 threshold.

Recovery diagnostics are:

```text
relative_error_L        = ||L_hat - L||_F / ||L||_F
relative_error_S        = ||A_hat - A||_F / ||A||_F
relative_error_routed_S = ||(A_hat - A) R.T||_F / ||A R.T||_F
```

`linear_comparison.csv` contains the aggregate gamma=0 comparison. It tells us how
the methods compare on new data from the paper's linear model, not whether either
method reproduces the exact detection rates printed in the paper.

## Main-sweep results

The completed three-seed sweep does not support the hypothesis that the current
VAE improves OD-flow attack localization merely because nominal traffic becomes
nonlinear. Both methods detect almost every attacked time point, but OnlineDA is
substantially more precise about which of the 210 OD flows was attacked:

| gamma | VAE+Lasso time F1 | OnlineDA time F1 | VAE+Lasso flow F1 | OnlineDA flow F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.9995 | 0.9942 | 0.1261 | 0.5539 |
| 0.25 | 0.9965 | 0.9972 | 0.1314 | 0.4107 |
| 0.50 | 0.9962 | 0.9965 | 0.1106 | 0.4031 |
| 0.75 | 0.9966 | 0.9964 | 0.1147 | 0.4154 |
| 1.00 | 0.9976 | 0.9960 | 0.1391 | 0.4163 |

At `gamma=1`, mean flow average precision is 0.8680 for VAE+Lasso and 0.9295
for OnlineDA; mean flow false-positive rates are 0.0585 and 0.0131,
respectively. The rank needed for 99% of nominal-link energy rises from 2 at the
linear endpoint to a mean of 9.33 at `gamma=1`, so the nonlinear generator does
break the rank-two model. Nevertheless, the configured VAE's denser link
reconstruction residual is amplified by the underdetermined 52-link-to-210-flow
inverse problem. Modeling normal traffic nonlinearly therefore does not by itself
solve sparse localization.

The 10,000-iteration run solved 268,080 Lasso subproblems. Thirty-three (0.0123%)
missed the deliberately strict convergence flag; their worst normalized KKT
residual was `4.89e-5`. The 3,000- and 10,000-iteration runs produced the same
reported aggregate metrics, and both result directories are retained for audit.

## Paper-style tracking stress tests

`experiment-02-tracking.json` follows the four parameter combinations reported
around Figure 4 of the paper at the linear endpoint: routing-change probabilities
`alpha` in `{0.01, 0.1}`, observation probabilities `pi` in `{0.8, 1}`, noise
standard deviations in `{0.01, 1e-5}`, forgetting factor `beta=0.9`, and a
1000-sample test stream. Run it with:

```bash
.venv/bin/python -m traffic_vae.paper_tracking --config experiment-02-tracking.json
```

This remains a matched VAE-versus-OnlineDA adaptation rather than an exact
reproduction of Figure 4: both methods retain the separate normal training,
validation, and calibration splits required by our comparison. The routing and
missing-link processes, however, use the paper's notation and scenario values.

All four scenarios completed across three seeds. Mean flow-localization results
were:

| Scenario `(alpha, pi, sigma)` | VAE+Lasso F1 / AP | OnlineDA F1 / AP |
| --- | ---: | ---: |
| `(0.01, 0.8, 0.01)` | 0.2217 / 0.3793 | 0.2335 / 0.3810 |
| `(0.01, 0.8, 1e-5)` | 0.2213 / 0.3832 | 0.2342 / 0.3819 |
| `(0.10, 0.8, 1e-5)` | 0.2790 / 0.3480 | 0.2748 / 0.3332 |
| `(0.01, 1.0, 1e-5)` | 0.3002 / 0.7560 | 0.3535 / 0.7977 |

Thus the paper-style stress tests do not reveal a stable winner: OnlineDA is
better in three flow-F1 scenarios, while VAE+Lasso is slightly better under fast
routing changes. Both deteriorate sharply relative to static routing. This is
expected for the VAE because it currently models raw link observations and is
not conditioned on `R_t` or the observation mask. Only one Lasso subproblem in
the entire tracking suite missed the strict convergence flag (worst normalized
KKT residual `2.64e-5`).

## Internet2 data availability

The real-data experiment in the paper uses Internet2 traffic matrices and creates
its evaluation target using the paper's own batch estimator. The exact traffic
files, topology snapshot, sampled observation masks, pseudo-ground-truth arrays,
and random seed used for the published figures are not present in this repository
and are not distributed with the article. We therefore do not substitute another
Abilene/Internet2 archive and call it an exact reproduction. Reproducing that
section requires obtaining the authors' exact inputs or an authorized matching
Internet2 archive and documenting any unavoidable discrepancy.

## Outputs

Results are written to `runs/experiment_02_paper/`:

- `config.json`, `provenance.json`: resolved settings, versions, hashes, and deviations.
- `seed_*/generator.npz`: graph, routing matrix, OD pairs, maps, and latent draws.
- `seed_*/gamma_*/truth.npz`: true OD traffic, link traffic, attacks, noise, and observations.
- `*/vae_lasso/`: VAE checkpoint, histories, estimates, metrics, and solver diagnostics.
- `*/onlineda/`: tracker state, estimates, metrics, and solver diagnostics.
- `metrics.csv`: every seed, gamma, and method.
- `summary.csv`: mean and sample standard deviation across seeds.
- `linear_comparison.csv`: gamma=0 rows from the summary.
- `paired_vae_minus_online_da.csv`: paired differences for each gamma.
- `rank_diagnostics.csv`: effective ranks and component energy.
- `completed.json`: written only after all configured runs finish.

The four tracking scenarios write the same per-scenario artifacts plus combined
`metrics.csv`, `summary.csv`, and `completed.json` files under
`runs/experiment_02_tracking/`.

The original 50-channel experiment remains available as
`experiment-02-legacy.json` and writes to `runs/experiment_02_legacy/`. Results
previously saved under `runs/experiment_02/` are unchanged.
