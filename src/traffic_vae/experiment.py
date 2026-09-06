"""Run experiment 01 and persist everything needed to audit and repeat it."""
import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

from .data import prepare, sha256
from .evaluation import calibrate, evaluate
from .model import VAE, score, train

DEFAULTS = {
    "seeds": [17, 42, 2026], "hidden_dims": [128, 64], "latent_dim": 8,
    "epochs": 100, "batch_size": 512, "learning_rate": 0.001,
    "beta": 0.01, "patience": 10, "target_fpr": 0.01,
    "train_fraction": 0.7, "validation_fraction": 0.15, "threads": 4,
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict:
    supplied = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(supplied) - set(DEFAULTS) - {"normal_csv", "test_csv", "output_dir"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    config = DEFAULTS | supplied
    for key in ("normal_csv", "test_csv", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"A nonempty {key} path is required.")
    for key in ("latent_dim", "epochs", "batch_size", "patience", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("seeds", "hidden_dims"):
        if not isinstance(config[key], list) or not config[key]:
            raise ValueError(f"{key} must be a nonempty list.")
        minimum = 0 if key == "seeds" else 1
        if any(type(x) is not int or x < minimum for x in config[key]):
            raise ValueError(f"Invalid {key} values.")
    if len(set(config["seeds"])) != len(config["seeds"]) or max(config["seeds"]) >= 2**32:
        raise ValueError("Seeds must be unique integers below 2**32.")
    for key in ("learning_rate", "beta", "target_fpr", "train_fraction", "validation_fraction"):
        if not isinstance(config[key], (int, float)) or not np.isfinite(config[key]):
            raise ValueError(f"{key} must be a finite number.")
    if config["learning_rate"] <= 0 or config["beta"] < 0:
        raise ValueError("Require learning_rate > 0 and beta >= 0.")
    if not 0 < config["target_fpr"] < 1:
        raise ValueError("Require 0 < target_fpr < 1.")
    if not (0 < config["train_fraction"] < 1 and 0 < config["validation_fraction"] < 1
            and config["train_fraction"] + config["validation_fraction"] < 1):
        raise ValueError("Split fractions must be positive and sum to less than one.")
    for key in ("normal_csv", "test_csv", "output_dir"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    for key in ("normal_csv", "test_csv"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"Missing {key}: {config[key]}. See README.md for data setup.")
    if config["normal_csv"] == config["test_csv"]:
        raise ValueError("Normal and test files must differ.")
    return config


def run(config: dict) -> Path:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"traffic_vae.{output}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(output / "run.log")]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        write_json(output / "config.json", config)
        logger.info("Experiment 01 | normal-only VAE | CPU | %d seeds", len(config["seeds"]))
        logger.info("Loading normal and DDoS evaluation CSV files")
        normal_path, test_path = Path(config["normal_csv"]), Path(config["test_csv"])
        arrays, labels, processor, manifest, audit = prepare(
            normal_path, test_path, config["train_fraction"], config["validation_fraction"])
        write_json(output / "data_audit.json", audit)
        processor.save(output / "preprocessor.npz")
        manifest.to_csv(output / "split_manifest.csv", index=False)
        source_root = Path(__file__).parent
        write_json(output / "provenance.json", {
            "python": sys.version, "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("numpy", "pandas", "torch", "scikit-learn")},
            "input_sha256": {str(path): sha256(path) for path in (normal_path, test_path)},
            "source_sha256": {p.name: sha256(p) for p in sorted(source_root.glob("*.py"))},
            "device": "cpu", "deterministic_algorithms": True,
        })
        logger.info("Split sizes: %s | retained features: %d", audit["split_sizes"], len(processor.columns))
        logger.info("Removed %d normal duplicates and %d overlapping test rows",
                    audit["normal_duplicate_rows_removed"], audit["test_overlap_rows_removed"])
        if len(arrays["calibration"]) * config["target_fpr"] < 10:
            logger.warning("Calibration tail contains fewer than 10 expected examples; threshold may be unstable")
        torch.set_num_threads(config["threads"])
        torch.use_deterministic_algorithms(True)
        all_metrics = []
        for seed in config["seeds"]:
            started = time.perf_counter()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            directory = output / f"seed_{seed}"
            directory.mkdir()
            logger.info("Seed %d | training started", seed)
            model = VAE(len(processor.columns), config["hidden_dims"], config["latent_dim"])
            history = train(model, arrays["train"], arrays["validation"], config | {"seed": seed}, logger)
            pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
            calibration_scores = score(model, arrays["calibration"], config["batch_size"])
            threshold = calibrate(calibration_scores, config["target_fpr"])
            test_scores = score(model, arrays["test"], config["batch_size"])
            metrics = evaluate(labels, test_scores, threshold)
            metrics.update(seed=seed, epochs_completed=len(history),
                           best_epoch=min(history, key=lambda row: row["validation_squared_l2"])["epoch"],
                           calibration_fpr=float(np.mean(calibration_scores > threshold)),
                           elapsed_seconds=time.perf_counter() - started)
            write_json(directory / "metrics.json", metrics)
            pd.DataFrame({"score": calibration_scores}).to_csv(directory / "calibration_scores.csv", index=False)
            pd.DataFrame({"source_row": manifest.loc[manifest.split == "test", "source_row"].to_numpy(),
                          "label": labels, "score": test_scores,
                          "prediction": (test_scores > threshold).astype(int)}).to_csv(
                              directory / "test_predictions.csv", index=False)
            precision, recall, _ = precision_recall_curve(labels, test_scores)
            pd.DataFrame({"precision": precision, "recall": recall}).to_csv(directory / "pr_curve.csv", index=False)
            torch.save({"state_dict": model.state_dict(), "input_dim": len(processor.columns),
                        "hidden_dims": config["hidden_dims"], "latent_dim": config["latent_dim"],
                        "threshold": threshold}, directory / "model.pt")
            all_metrics.append(metrics)
            logger.info("Seed %d | precision %.4f | recall %.4f | F1 %.4f | AP %.4f | FPR %.4f",
                        seed, metrics["precision"], metrics["recall"], metrics["f1"],
                        metrics["average_precision"], metrics["false_positive_rate"])
        table = pd.DataFrame(all_metrics)
        table.to_csv(output / "metrics_by_seed.csv", index=False)
        summary = {}
        for key in ("precision", "recall", "f1", "false_positive_rate", "average_precision", "pr_auc_trapezoidal", "roc_auc"):
            summary[key] = {"mean": float(table[key].mean()),
                            "std_across_seeds": float(table[key].std(ddof=1)) if len(table) > 1 else None}
        write_json(output / "summary.json", summary)
        logger.info("Completed | F1 %.4f | AP %.4f | artifacts: %s",
                    summary["f1"]["mean"], summary["average_precision"]["mean"], output)
        return output
    except Exception:
        logger.exception("Experiment failed; partial artifacts are retained for diagnosis")
        raise
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiment_01.json"))
    args = parser.parse_args()
    try:
        run(load_config(args.config))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
