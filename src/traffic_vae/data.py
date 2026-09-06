"""Strict CSV ingestion and training-only feature transformations."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

# Identifiers and categorical ports are excluded from the numeric baseline.
EXCLUDED = {
    "flow id", "source ip", "destination ip", "src ip", "dst ip",
    "source port", "destination port", "src port", "dst port", "protocol",
    "timestamp", "label",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_flows(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate column names after whitespace normalization: {path}")
    if "Label" not in frame:
        raise ValueError(f"Required Label column is missing: {path}")
    if frame["Label"].isna().any():
        raise ValueError(f"Missing labels in {path}")
    labels = frame.pop("Label").astype(str).str.strip().str.upper()
    columns = [c for c in frame if c.lower() not in EXCLUDED
               and not c.lower().startswith("unnamed:")]
    if not columns:
        raise ValueError("No numeric feature candidates remain.")
    numeric = frame[columns].apply(pd.to_numeric, errors="raise")
    # Normalize dtypes so equal values hash identically across separate CSV files.
    numeric = numeric.astype(np.float64).replace([np.inf, -np.inf], np.nan)
    return numeric, labels


@dataclass
class Preprocessor:
    """Median imputation, signed log1p, and scaling fitted only on training rows."""
    columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Preprocessor":
        usable = frame.loc[:, frame.notna().any(axis=0)]
        median = usable.median().to_numpy(dtype=np.float64)
        values = usable.fillna(dict(zip(usable.columns, median))).to_numpy(dtype=np.float64)
        values = np.sign(values) * np.log1p(np.abs(values))
        scale = values.std(axis=0)
        keep = scale > 1e-12
        if not keep.any():
            raise ValueError("Training data has no nonconstant, finite features.")
        return cls(usable.columns[keep].tolist(), median[keep],
                   values.mean(axis=0)[keep], scale[keep])

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.columns].to_numpy(dtype=np.float64, copy=True)
        values = np.where(np.isfinite(values), values, self.median)
        values = np.sign(values) * np.log1p(np.abs(values))
        result = ((values - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite values after preprocessing.")
        return result

    def save(self, path: Path) -> None:
        np.savez(path, columns=np.asarray(self.columns), median=self.median,
                 mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        with np.load(path, allow_pickle=False) as values:
            return cls(values["columns"].tolist(), values["median"],
                       values["mean"], values["scale"])


def prepare(normal_path: Path, test_path: Path, train_fraction: float,
            validation_fraction: float):
    normal, normal_labels = read_flows(normal_path)
    test, test_labels = read_flows(test_path)
    if set(normal_labels) != {"BENIGN"}:
        raise ValueError("The normal CSV must contain BENIGN rows only.")
    if set(test_labels) != {"BENIGN", "DDOS"}:
        raise ValueError("The test CSV must contain exactly BENIGN and DDoS labels.")
    if set(normal.columns) != set(test.columns):
        raise ValueError("Normal and test CSV feature schemas differ.")
    test = test[normal.columns]
    original_normal = len(normal)
    # Keep identical feature vectors in only one normal split.
    normal = normal.drop_duplicates()
    n_train = int(len(normal) * train_fraction)
    n_validation = int(len(normal) * validation_fraction)
    if min(n_train, n_validation, len(normal) - n_train - n_validation) < 2:
        raise ValueError("Each normal split must contain at least two unique rows.")
    train = normal.iloc[:n_train]
    validation = normal.iloc[n_train:n_train + n_validation]
    calibration = normal.iloc[n_train + n_validation:]
    # Hash features without labels or row indices to exclude exact cross-day overlap.
    normal_hash = pd.util.hash_pandas_object(normal, index=False)
    test_hash = pd.util.hash_pandas_object(test, index=False)
    overlap = test_hash.isin(normal_hash)
    original_test = len(test)
    test, test_labels = test.loc[~overlap], test_labels.loc[~overlap]
    if set(test_labels) != {"BENIGN", "DDOS"}:
        raise ValueError("Both test classes must remain after overlap removal.")
    processor = Preprocessor.fit(train)
    frames = {"train": train, "validation": validation, "calibration": calibration, "test": test}
    arrays = {name: processor.transform(frame) for name, frame in frames.items()}
    manifest = pd.concat([
        pd.DataFrame({"split": name, "source_row": frame.index,
                      "source": "test" if name == "test" else "normal"})
        for name, frame in frames.items()
    ], ignore_index=True)
    audit = {
        "original_normal_rows": original_normal,
        "normal_duplicate_rows_removed": original_normal - len(normal),
        "original_test_rows": original_test,
        "test_overlap_rows_removed": int(overlap.sum()),
        "split_sizes": {key: len(value) for key, value in arrays.items()},
        "test_label_counts": test_labels.value_counts().to_dict(),
        "retained_features": processor.columns,
        "dropped_features": [c for c in normal if c not in processor.columns],
        "missing_values_before_imputation": {k: int(v.isna().sum().sum()) for k, v in frames.items()},
        "split_policy": "Normal CSV row order, deduplicated before splitting; not verified chronological.",
    }
    return arrays, (test_labels == "DDOS").to_numpy(dtype=np.int64), processor, manifest, audit
