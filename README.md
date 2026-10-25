# Group Sparsity in Security Applications

Project report and implementation for the study of VAE-based anomaly detection
and ordinary versus group-sparse attack localization.

- [Project report (PDF)](output/pdf/group_sparsity_project_report.pdf)
- [Report source (LaTeX)](group_sparsity_project_report.tex)
- [Experiment 2 protocol](EXPERIMENT_02.md)
- [Experiment 3 protocol and results](EXPERIMENT_03.md)

To rebuild the report with a LaTeX installation, run from the repository root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf group_sparsity_project_report.tex
```

The report contains the literature review and the three main experiments.
The exploratory high-rank sensitivity results remain separate and are not
included in the report.

## Experiment 01: normal-only VAE for CIC-IDS2017

Experiment 1 establishes the plain VAE baseline for network-flow anomaly detection. Each row is one observed flow and each retained column is a numeric traffic statistic. This baseline uses real, unchanged flow measurements without a sparse optimization stage. The separate [third experiment](EXPERIMENT_03.md) now compares ordinary and group-sparse decoding on recorded and synthetic data.

## Current project layout

```text
.
├── experiment_01.json       # Experiment settings and data paths
├── README.md
├── pyproject.toml           # Package definition and dependencies
├── requirements-lock.txt    # Verified dependency versions
├── group_sparsity_project_report.tex  # Project report source
├── output/pdf/group_sparsity_project_report.pdf  # Compiled project report
├── data/
│   └── TrafficLabelling/    # Eight extracted CIC-IDS2017 CSV files
├── src/
│   └── traffic_vae/
│       ├── __init__.py
│       ├── data.py          # Loading, splitting, and preprocessing
│       ├── model.py         # VAE, training, and anomaly scoring
│       ├── evaluation.py    # Threshold calibration and metrics
│       └── experiment.py    # Run orchestration and saved outputs
└── .venv/                  # Installed Python environment
```

The CSV files are already extracted under `data/TrafficLabelling/`. No `raw/`
subdirectory is needed. Experiment 01 reads only:

- `Monday-WorkingHours.pcap_ISCX.csv`: normal data for training, validation, and calibration.
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: held-out normal and DDoS evaluation data.

The other six CSV files are available for later experiments and are not loaded by
this configuration. The loader requires Monday to contain only `BENIGN` and the
test file to contain both `BENIGN` and `DDoS`; unexpected labels cause an error.
The data directory is excluded from Git. Dataset source and citation are available
on the [official CIC-IDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html).

## Run with the existing environment

From the repository root, run:

```bash
.venv/bin/traffic-vae --config experiment_01.json
```

The environment is already installed. There is no need to recreate it or reinstall
libraries before each run. This command starts the configured three-seed experiment;
results are written to `runs/experiment_01/`.

Configuration paths are resolved relative to the current working directory. Change
`output_dir` before another run; existing run directories are never overwritten.
A failed run retains its log and partial artifacts and has no completed `summary.json`.

## What is inside .venv?

`.venv` isolates this project's Python libraries from other projects. Its `bin/`
directory contains the Python executable and commands such as `traffic-vae`; its
`lib/` directory contains installed packages such as PyTorch, NumPy, pandas, and
scikit-learn, together with their dependencies. These packages include many modules,
metadata files, and compiled libraries, so many folders are expected.

These are installed dependencies, not additional experiment code. Do not delete
individual folders inside `.venv`: doing so can break imports or training. You can
collapse or hide `.venv` in your editor. It is excluded from Git and can be recreated
as a whole when needed.

## Setup on a new machine only

Python 3.10 or newer is required. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Place the extracted CSV files under `data/TrafficLabelling/`, or edit their paths in
`experiment_01.json`. `requirements-lock.txt` records the dependency versions verified
in the local Python 3.11/macOS ARM environment. To recreate those versions, install
it with `.venv/bin/python -m pip install -r requirements-lock.txt` before installing
this package. Other platforms may require compatible dependency versions.

## Experimental protocol

1. Strip whitespace from column headers and labels. Exclude flow IDs, IP addresses, timestamps, ports, protocol, and unnamed CSV indices. Ports and protocol are categorical; treating their numeric codes as continuous MSE targets is excluded from this baseline.
2. Convert remaining features strictly to numeric. Replace infinities with missing values. Fail on unexpected textual features or incompatible CSV schemas.
3. Remove duplicate normal feature vectors. Divide normal rows in their original CSV order: 75% training and 25% validation. The same validation rows are used for early stopping and threshold calibration; no separate calibration partition is reserved. Rounding remainders are assigned to validation, so all retained normal rows are used. This is a **row-order split, not a verified chronological split**; the supplied files contain timestamps, but this baseline excludes them and does not sort by time.
4. Fit medians, feature retention, signed `log1p`, and standardization using training rows only. Drop all-missing and constant training features. Apply the fitted transformation to other splits without clipping anomalies. Signed `log1p` handles heavy tails and preserves negative values, including dataset sentinel values; interpreting these sentinels is a future preprocessing sensitivity analysis.
5. Exclude test feature vectors that exactly match any normal partition. Preserve remaining test duplicates and report all exclusions. Feature hashing is a practical exact-vector overlap check; it does not prove independence of related flows or sessions. The evaluation population is the retained Friday rows, not every row in the original CSV.
6. Fit a VAE with encoder widths 128 and 64, latent dimension 8, and a mirrored decoder with a linear output. Train on normal examples only. Use Adam, gradient clipping, and early stopping on deterministic validation reconstruction error. Restore the best epoch.
7. Set the threshold to the higher empirical 99th percentile of the normal validation scores, recomputed after restoring the best model. Predict attack only when `score > threshold`. The configured 1% FPR is a calibration target, **not a guarantee of Friday's FPR**.
8. Evaluate the frozen model and threshold on Friday. Repeat using three predetermined initialization seeds with identical data splits. Report each seed and mean/sample standard deviation; seed variation is not a confidence interval over network environments.

The training loss averages over rows and sums within feature/latent dimensions:

```text
loss = mean_i(sum_j((x_ij - decoder(z_i)_j)^2))
       + beta * mean_i(-0.5 * sum_k(1 + logvar_ik - mu_ik^2 - exp(logvar_ik)))
```

`beta = 0.01` is an initial experimental setting, not an optimized claim. Training samples `z` with reparameterization. Evaluation reconstructs from `z = mu` and uses the L2 residual as the score. This deterministic reconstruction score is not an estimated marginal likelihood. Log variance is clamped to [-20, 20] for numerical stability. CPU execution and deterministic PyTorch algorithms are used for a repeatable starting point; bitwise equivalence across library versions and hardware is not promised.

## Outputs

Each run saves:

| Artifact | Purpose |
| --- | --- |
| `config.json`, `provenance.json` | Resolved settings, package versions, input and source SHA-256 hashes |
| `data_audit.json`, `split_manifest.csv` | Cleaning counts, feature list, split sizes, zero-based source row IDs |
| `preprocessor.npz` | Training-only transformation parameters |
| `seed_*/model.pt` | Best model weights, architecture, and threshold |
| `seed_*/training_history.csv` | Training loss, reconstruction, KL, validation reconstruction |
| `seed_*/metrics.json` | Precision, recall, F1, FPR, AP, trapezoidal PR-AUC, ROC-AUC, confusion counts |
| `seed_*/test_predictions.csv` | Row IDs, labels, scores, and decisions |
| `seed_*/calibration_scores.csv`, `seed_*/pr_curve.csv` | Calibration audit and plot-ready precision-recall curve |
| `metrics_by_seed.csv`, `summary.json`, `run.log` | Repeat-level results, aggregate results, readable progress |

Threshold calibration reuses the model-selection validation set and is therefore not independent of model selection. This reuse is recorded in `data_audit.json` and the run log. `calibration_scores.csv` contains validation source row IDs and scores; the split manifest lists each row only once. Friday remains reserved for final evaluation.

Average precision (AP) and trapezoidal PR-AUC are saved under separate names because their numerical definitions differ. Attack prevalence is included to contextualize precision and AP. No test-label tuning or test-set early stopping is performed.

## Scope and interpretation

This baseline measures flow-level detection on a specific cross-day dataset. Monday-to-Friday distribution shift can cause false positives unrelated to attack structure. Strong performance alone would not establish a contribution from group sparsity or superiority to RPCA. Real data does not provide ground-truth additive `L` and `S`, so decomposition recovery errors cannot be reported.

Experiment 3 implements a controlled comparison between a frozen VAE followed by
Lasso or Group Lasso. It does not claim to solve a joint nonlinear VAE/sparse
objective exactly or train an end-to-end differentiable optimization layer.
The synthetic arm additionally benchmarks both against the original OnlineDA
implementation on the same observations.

The configured CSV files are present locally. Full CSVs are loaded into RAM, so memory grows with dataset size.

Dataset citation: I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” ICISSP, 2018.

## Experiment 02: paper-model synthetic data

The independent [second experiment](EXPERIMENT_02.md) uses the 15-node,
52-directed-link, 210-OD-flow synthetic model described in Section VI-A of
*Dynamic Anomalography* at gamma=0. Our nonlinear extension supplies the higher
gamma values. Following the paper, alpha is reserved for routing-change probability.
A normal-trained VAE with routing-aware sparse decoding is compared
with Algorithm-2-based online anomalography on the same link observations.
The VAE learns only nominal traffic; attacks are inferred from its residual by
the same elementwise L1 routing decoder used for the comparison. The paper does
not define attack groups, so Group Lasso is deliberately left out of this main
paper-faithful experiment.

The code creates new samples from the paper's stated random model; the paper does
not provide the exact sampled graph, matrices, arrays, or random seed. No dataset
download is required. Results are written to `runs/experiment_02_paper/`.

```bash
.venv/bin/python -m traffic_vae.experiment_02 --config experiment_02.json
```

The routing-change, missing-link, and noise scenarios reported around Figure 4
are configured separately:

```bash
.venv/bin/python -m traffic_vae.paper_tracking --config experiment_02_tracking.json
```

See `EXPERIMENT_02.md` for the protocol, configuration, comparison limits, and outputs.
The previous 50-channel generator remains available in `experiment_02_legacy.json`.

## Experiment 03: ordinary versus grouped sparse decoding

The [third experiment](EXPERIMENT_03.md) reuses the frozen VAE models and exact
data from both earlier experiments. On CICIDS2017, destination/time groups are
used for identity-design anomaly-score denoising; no routing matrix is invented
for these flow CSVs. On the synthetic network, destination groups are used in
a routing-aware Group-Lasso decoder, compared with the original Lasso decoder
over all five gamma settings and three seeds. Matched dispersed and grouped
attack controls are included alongside the original independent attacks.
OnlineDA is a third synthetic baseline, with unchanged Experiment-2 parameters
and an independent copy of its normal-trained state for each test stream.

```bash
.venv/bin/python -m traffic_vae.experiment_03 --config experiment_03.json
.venv/bin/python -m traffic_vae.experiment_03_online
.venv/bin/python -m traffic_vae.group_figures --online-source runs/experiment_03_onlineda
```

Raw runs remain local under `runs/experiment_03_full/`. The portable
`output/experiment_03/results.json` retains all seed-level results and audits.
The additional OnlineDA arrays, replay checks, and diagnostics remain under
`runs/experiment_03_onlineda/`; the original VAE results are not overwritten.
The project report includes the primary results, two new figures, numerical checks,
and the limitations of the real-data and synthetic arms.

## Exploratory high-rank nonlinear sensitivity

The separate [high-rank sensitivity protocol](HIGH_RANK_SENSITIVITY.md) tests a
two-latent-dimensional, 16-feature Fourier manifold with a broad linear spectrum.
It retrains the synthetic VAEs and repeats routed Lasso / Group-Lasso / OnlineDA
comparisons, with a fixed OnlineDA rank-16 capacity control. This is an artificial
stress test, not a replacement selected for favorable VAE test performance.
The original experiments, main figures, and project-report results are preserved.
Completed outcomes, both favorable and unfavorable operating points, and
separate figures are in the [result summary](output/high_rank_sensitivity/README.md).

```bash
.venv/bin/python -m traffic_vae.high_rank_sensitivity
.venv/bin/python -m traffic_vae.high_rank_results
```
