# Experiment 01: normal-only VAE for CIC-IDS2017

This experiment establishes the plain VAE baseline for network-flow anomaly detection. Each row is one observed flow and each retained column is a numeric traffic statistic. The experiment uses real, unchanged flow measurements. It does not yet implement RPCA, group sparsity, or synthetic attacks.

## Current project layout

```text
.
├── experiment_01.json       # Experiment settings and data paths
├── README.md
├── pyproject.toml           # Package definition and dependencies
├── requirements-lock.txt    # Verified dependency versions
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

The next controlled comparison can introduce RPCA, group-RPCA, and a carefully specified group-sparse VAE. The alternating shrinkage sketch in the background notes is not assumed to solve the full nonlinear VAE objective exactly. Those methods need a separate derivation and matched evaluation protocol.

The configured CSV files are present locally. Full CSVs are loaded into RAM, so memory grows with dataset size.

Dataset citation: I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” ICISSP, 2018.

## Experiment 02: fully synthetic data

The independent [second experiment](EXPERIMENT_02.md) generates `D = L(alpha) + S + N`
and compares a normal-trained VAE with batch RPCA and an Algorithm-2-based online
anomalography tracker across five nonlinearity levels.
It requires no dataset downloads and writes to `runs/experiment_02/`.

```bash
.venv/bin/python -m traffic_vae.experiment_02 --config experiment_02.json
```

See `EXPERIMENT_02.md` for the protocol, configuration, comparison limits, and outputs.
