"""Create the paper-protocol Experiment 02 figures from completed CSV outputs."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"VAE_Lasso": "#D95F02", "OnlineDA": "#1B6CA8"}
LABELS = {"VAE_Lasso": "VAE + Lasso", "OnlineDA": "OnlineDA"}


def _save(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _metric_lines(axis, summary, metric, title, ylabel=None):
    for method in ("VAE_Lasso", "OnlineDA"):
        rows = summary[summary["method"] == method].sort_values("gamma")
        axis.errorbar(rows["gamma"], rows[f"{metric}_mean"],
                      yerr=rows[f"{metric}_std"], marker="o", capsize=3,
                      linewidth=2, label=LABELS[method], color=COLORS[method])
    axis.set_title(title)
    axis.set_xlabel(r"Nonlinearity $\gamma$")
    axis.set_ylabel(ylabel or title)
    axis.set_xticks(sorted(summary["gamma"].unique()))
    axis.grid(alpha=.25)


def main_sweep(summary, ranks, output):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    _metric_lines(axes[0, 0], summary, "time_f1", "Attack-time F1", "F1")
    _metric_lines(axes[0, 1], summary, "flow_f1", "Flow localization F1", "F1")
    _metric_lines(axes[1, 0], summary, "flow_average_precision",
                  "Flow localization average precision", "Average precision")
    _metric_lines(axes[1, 1], summary, "flow_false_positive_rate",
                  "Flow false-positive rate", "False-positive rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.035),
                  ncol=2, frameon=False)
    _save(figure, output / "experiment_02_paper_detection.png")

    rank_summary = ranks.groupby("gamma")[["rank_95", "rank_99"]].agg(["mean", "std"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for rank, color in (("rank_95", "#66A61E"), ("rank_99", "#7570B3")):
        axes[0].errorbar(rank_summary.index, rank_summary[(rank, "mean")],
                         yerr=rank_summary[(rank, "std")], marker="o", capsize=3,
                         linewidth=2, label=rank.replace("_", " ").upper(), color=color)
    axes[0].set(title="Effective linear rank of nominal link traffic",
                xlabel=r"Nonlinearity $\gamma$", ylabel="Rank")
    axes[0].set_xticks(rank_summary.index)
    axes[0].grid(alpha=.25)
    axes[0].legend(frameon=False)
    for method in ("VAE_Lasso", "OnlineDA"):
        rows = summary[summary["method"] == method].sort_values("gamma")
        axes[1].errorbar(rows["gamma"], rows["relative_error_S_mean"],
                         yerr=rows["relative_error_S_std"], marker="o", capsize=3,
                         linewidth=2, label=LABELS[method], color=COLORS[method])
    axes[1].set(title="OD anomaly recovery error", xlabel=r"Nonlinearity $\gamma$",
                ylabel=r"$\|\hat A-A\|_F/\|A\|_F$")
    axes[1].set_xticks(sorted(summary["gamma"].unique()))
    axes[1].grid(alpha=.25)
    axes[1].legend(frameon=False)
    _save(figure, output / "experiment_02_paper_rank_recovery.png")


def tracking(summary, output):
    configured_order = ["slow_noisy_missing", "slow_low_noise_missing",
                        "fast_low_noise_missing", "slow_low_noise_full"]
    scenarios = [item for item in configured_order if item in set(summary["scenario"])]
    display = {
        "slow_noisy_missing": r"$\alpha=.01,\ \pi=.8,\ \sigma=.01$",
        "slow_low_noise_missing": r"$\alpha=.01,\ \pi=.8,\ \sigma=10^{-5}$",
        "fast_low_noise_missing": r"$\alpha=.1,\ \pi=.8,\ \sigma=10^{-5}$",
        "slow_low_noise_full": r"$\alpha=.01,\ \pi=1,\ \sigma=10^{-5}$",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(scenarios))
    width = .36
    for method_index, method in enumerate(("VAE_Lasso", "OnlineDA")):
        rows = summary[summary["method"] == method].set_index("scenario").loc[scenarios]
        offset = (method_index - .5) * width
        axes[0].bar(x + offset, rows["flow_f1_mean"], width,
                    yerr=rows["flow_f1_std"], capsize=3,
                    color=COLORS[method], label=LABELS[method])
        axes[1].bar(x + offset, rows["flow_average_precision_mean"], width,
                    yerr=rows["flow_average_precision_std"], capsize=3,
                    color=COLORS[method], label=LABELS[method])
    for axis, title, ylabel in zip(axes, ("Flow F1", "Flow average precision"),
                                   ("F1", "Average precision")):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [display.get(item, item) for item in scenarios], rotation=18,
                        ha="right")
        axis.grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.04),
                  ncol=2, frameon=False)
    _save(figure, output / "experiment_02_paper_tracking.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-dir", type=Path, default=Path("runs/experiment_02_paper"))
    parser.add_argument("--tracking-dir", type=Path,
                        default=Path("runs/experiment_02_tracking"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("figures"))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    main_sweep(pd.read_csv(args.main_dir / "summary.csv"),
               pd.read_csv(args.main_dir / "rank_diagnostics.csv"), args.output_dir)
    tracking(pd.read_csv(args.tracking_dir / "summary.csv"), args.output_dir)


if __name__ == "__main__":
    main()
