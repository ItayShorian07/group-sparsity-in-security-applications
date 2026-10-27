# Experiment 3 results

`results.json` contains the actual completed run, not illustrative values:

- Recorded CICIDS2017 flows from Experiment 1: 225,104 test flows, 114,806 normal
  validation scores, destination/60-second groups, three frozen VAE seeds.
- The exact Experiment-2 routed synthetic data: all five gamma settings and
  three frozen VAE seeds, plus matched dispersed and destination-grouped attacks.
- A third synthetic baseline: the unchanged Algorithm-2 OnlineDA implementation,
  with the same inputs and separate normal-trained state copies per stream.
- Every seed-level metric, fixed sensitivity setting, calibration operating
  point, source hash, and grouped/Lasso solver-quality summary.
- A secondary evaluation at the pre-existing paper flow threshold 0.1, without
  choosing a threshold from these test labels.

The recorded-data arm is **identity-design score denoising**, not physical
OD-flow recovery. The synthetic arm is routing-aware sparse attack recovery.
Neither trains a new VAE or differentiates through an optimization layer.

## Main findings

| Setting | VAE+Lasso | VAE+Group Lasso | OnlineDA |
| --- | ---: | ---: | ---: |
| Recorded data: flow F1 | 0.8076 | 0.8078 | Not applicable |
| Recorded data: benign-flow FPR | 0.3192 | 0.3312 | Not applicable |
| Original synthetic attacks, gamma=1: flow AP | 0.8680 | 0.7181 | 0.9295 |
| Grouped synthetic attacks, gamma=1: flow AP | 0.4543 | 0.8096 | 0.9998 |
| Grouped synthetic attacks, gamma=1: relative OD recovery error | 1.1425 | 0.7311 | 0.1894 |
| Grouped synthetic attacks, gamma=1: calibrated flow F1 | 0.2347 | 0.1565 | 0.4949 |
| Grouped synthetic attacks, gamma=1: fixed-0.1 flow F1 | 0.3768 | 0.3664 | 0.8268 |

All rows are means over seeds 17, 42, and 2026. These seeds do not represent
independently collected traffic datasets. Better ranking/recovery on structured
attacks is not a demonstrated improvement in thresholded F1.
OnlineDA has higher mean flow F1 and AP than either VAE method in every
scenario/gamma setting here; it adapts causally during testing, unlike the frozen VAE.

All 180,000 grouped test solves and 30,000 distinct grouped calibration solves
meet the declared KKT tolerance. Twenty test Lasso solves miss the strict flag;
the worst normalized KKT residual is 4.8435e-5. No cases were dropped.
OnlineDA adds 180,000 test and 30,000 normal-calibration solves: 17 and 8 miss
the strict flag, with worst residual below 4.89e-5. The export includes these
audits and all 15 original-test / 15 normal-calibration replay checks.

The complete method, assumptions, limitations, and reproduction commands are in
[EXPERIMENT-03.md](../../EXPERIMENT-03.md). Large raw arrays remain in the local
`runs/experiment_03_full/` directory and are not committed by default.
The separate `runs/experiment_03_onlineda/` directory retains the third method's
arrays and histories; the old two-decoder run is unchanged. The combined JSON
contains 135 synthetic seed-level records and the unchanged recorded-data results.

## Project report editing scope

Experiment 3 receives its own methods, real-data and synthetic results tables,
two figures, and limitations. To make room, Experiment 1 retains its main
metric table and score/PR figure; the duplicate metric/confusion plot and the
additional shift/threshold figures are omitted from the main PDF, with their
key numerical findings retained in prose. Experiment 2 retains both main sweep
figures and the tracking table, but omits the tracking plot that repeats it.
Original figure files and raw results remain available. The literature-review section
and the earlier requested convergence corrections were preserved.
