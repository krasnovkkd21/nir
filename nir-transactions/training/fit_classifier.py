"""Refit the published 188-feature XGBoost with a frozen transaction encoder."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
BATCH_ROWS = 65_536
MAX_FPR = 0.0041052216702632885
STRICT_FPR = 0.001
MAX_ROUNDS = 400
PATIENCE = 40
BASE_PARAMS = {"max_depth": 6, "min_child_weight": 5, "reg_lambda": 1.0,
               "learning_rate": 0.05, "max_bin": 512, "scale_pos_weight": 1.0,
               "colsample_bytree": 0.8, "subsample": 1.0, "seed": 42,
               "tree_method": "hist", "device": "cpu", "objective": "binary:logistic",
               "eval_metric": "aucpr"}
SPLIT_NAMESPACE = "nir-ru-pooled-clients-v1/stop-calibration/client/"


def negative_only_threshold(negative_scores, max_fpr):
    """Choose a >= cut using only negatives; tied scores can only reduce the budget spent."""
    scores = np.asarray(negative_scores, dtype=np.float64)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or not 0 <= max_fpr < 1):
        raise ValueError("Finite negative scores and a false-positive budget in [0, 1) are required.")
    allowed = int(np.floor(float(max_fpr) * len(scores)))
    boundary = float(np.partition(scores, len(scores) - allowed - 1)[len(scores) - allowed - 1])
    threshold = float(np.nextafter(boundary, np.inf))
    if not np.isfinite(threshold):
        raise ValueError("No finite strict threshold is available.")
    spent = int(np.count_nonzero(scores >= threshold))
    if spent > allowed:
        raise RuntimeError("Negative-only threshold exceeded its false-positive budget.")
    return threshold, {"negative_rows": len(scores), "max_fpr": float(max_fpr),
                       "allowed_false_positive": allowed, "false_positive": spent,
                       "false_positive_rate": spent / len(scores), "boundary": boundary,
                       "threshold": threshold, "threshold_rule": "nextafter_negative_boundary_up",
                       "positive_scores_used": False}


def first_episode_rows(episodes, times, labels):
    fraud = np.flatnonzero(labels == 1)
    if np.any(episodes[fraud] <= 0) or np.any(episodes[labels == 0] != 0):
        raise ValueError("Source episode metadata disagree with the binary labels.")
    ordered = fraud[np.lexsort((fraud, times[fraud], episodes[fraud]))]
    first = np.zeros(len(labels), dtype=bool)
    if len(ordered):
        first[ordered[np.r_[True, episodes[ordered[1:]] != episodes[ordered[:-1]]]]] = True
    return first


class PooledFeatures:
    """Read only requested rows from several disk-backed matrices, without a full copy."""

    def __init__(self, matrices, columns):
        self.matrices = matrices
        self.columns = columns
        self.offsets = np.cumsum([0, *[len(matrix) for matrix in matrices]], dtype=np.int64)

    def rows(self, rows):
        rows = np.asarray(rows)
        if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
                or (len(rows) and (rows.min() < 0 or rows.max() >= self.offsets[-1]))):
            raise ValueError("Invalid pooled row indices.")
        output = np.empty((len(rows), len(self.columns)), dtype=np.float32)
        for index, matrix in enumerate(self.matrices):
            mask = (rows >= self.offsets[index]) & (rows < self.offsets[index + 1])
            if mask.any():
                output[mask] = matrix[rows[mask] - self.offsets[index]]
        if np.isinf(output).any():
            raise ValueError("Infinite pooled feature value.")
        return output


class TrainingBatches(xgb.DataIter):
    def __init__(self, features, rows, labels):
        super().__init__(release_data=True)
        self.features, self.indices, self.labels = features, rows, labels
        self.offset = 0

    def reset(self):
        self.offset = 0

    def next(self, input_data):
        if self.offset >= len(self.indices):
            return False
        rows = self.indices[self.offset:self.offset + BATCH_ROWS]
        self.offset += len(rows)
        input_data(data=self.features.rows(rows), label=self.labels[rows],
                   feature_names=self.features.columns)
        return True


def quality_tuple(labels, first, scores):
    threshold, _ = negative_only_threshold(scores[labels == 0], MAX_FPR)
    alerts = scores >= threshold
    return (int(np.count_nonzero((labels == 1) & alerts)),
            int(np.count_nonzero(first & alerts)),
            float(average_precision_score(labels, scores)))


def fit_candidate(features, train_rows, stop_rows, labels, first, params):
    options = {"nthread": params["nthread"], "max_bin": params["max_bin"]}
    training = xgb.QuantileDMatrix(TrainingBatches(features, train_rows, labels), **options)
    stopping = xgb.QuantileDMatrix(TrainingBatches(features, stop_rows, labels), ref=training, **options)
    stop_labels, stop_first = labels[stop_rows], first[stop_rows]

    def metric(scores, matrix):
        found, early, ap = quality_tuple(stop_labels, stop_first, scores)
        # Integer part encodes lexicographic (fraud found, first found); AP breaks ties only.
        return "detection_quality", found * (int(stop_first.sum()) + 1) + early + 0.5 * ap

    history = {}
    model = xgb.train(params, training, num_boost_round=MAX_ROUNDS, evals=[(stopping, "stop")],
                      custom_metric=metric, evals_result=history, verbose_eval=25,
                      callbacks=[xgb.callback.EarlyStopping(
                          rounds=PATIENCE, data_name="stop", metric_name="detection_quality",
                          maximize=True, save_best=True)])
    if model.num_boosted_rounds() != model.best_iteration + 1:
        raise RuntimeError("The checkpoint must retain exactly the best stopping iteration.")
    del training, stopping
    gc.collect()
    return model, history["stop"]


def predict_model(model, features, rows):
    if model.feature_names != features.columns:
        raise ValueError("Training model feature order changed.")
    scores = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), BATCH_ROWS):
        batch = rows[start:start + BATCH_ROWS]
        scores[start:start + len(batch)] = model.inplace_predict(features.rows(batch))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite candidate scores.")
    return scores


def calibrated_thresholds(negative_scores):
    low, low_audit = negative_only_threshold(negative_scores, MAX_FPR)
    high, high_audit = negative_only_threshold(negative_scores, STRICT_FPR)
    if low > high:
        raise RuntimeError("Calibrated policy thresholds are reversed.")
    return {"confirmation_threshold": low, "manual_review_threshold": high}, {
        "confirmation_or_review": low_audit, "strict_manual_review": high_audit}


def load_period(feature_dir, data_dir, period, expected_columns):
    schema_file = feature_dir / "columns.json"
    if schema_file.is_file():
        columns = json.loads(schema_file.read_text(encoding="utf-8"))
    else:
        columns = json.loads((feature_dir / "config.json").read_text(encoding="utf-8"))["columns"]
    if columns != expected_columns:
        raise ValueError(f"Unexpected schema in {feature_dir}")

    matrix = np.load(feature_dir / "X.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(feature_dir / "ids.npy", mmap_mode="r", allow_pickle=False)
    events = pq.read_table(data_dir / "transactions.parquet", columns=["transaction_id", "client_id", "timestamp"])
    truth = pq.read_table(data_dir / "labels.parquet", columns=["transaction_id", "period", "is_fraud", "episode_id"])
    if not events["transaction_id"].equals(truth["transaction_id"]):
        raise ValueError(f"Misaligned transactions and labels in {data_dir}")
    rows = np.flatnonzero(truth["period"].to_numpy() == period)
    transaction_ids = events["transaction_id"].to_numpy()
    if (matrix.shape != (len(rows), len(columns)) or matrix.dtype != np.float32
            or not len(rows) or not np.array_equal(ids, transaction_ids[rows])
            or np.any(transaction_ids[1:] <= transaction_ids[:-1])):
        raise ValueError(f"Expected every transaction of period {period} in source order at {feature_dir}")

    labels = truth["is_fraud"].to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be binary")
    times = pc.cast(events["timestamp"], "int64").to_numpy()
    first = first_episode_rows(truth["episode_id"].to_numpy(), times, labels)
    return matrix, labels[rows], first[rows], events["client_id"].to_numpy()[rows]


def development_split(clients):
    unique_clients = np.unique(clients)
    if len(unique_clients) < 2:
        raise ValueError("At least two development clients are needed")

    def split_key(client):
        raw = (SPLIT_NAMESPACE + str(int(client))).encode()
        return hashlib.sha256(raw).hexdigest(), int(client)

    ranked = sorted(unique_clients, key=split_key)
    stop = np.isin(clients, ranked[:len(ranked) // 2])
    return np.flatnonzero(stop), np.flatnonzero(~stop)


def run(args):
    policy = json.loads((ROOT / "models" / "policy.json").read_text(encoding="utf-8"))
    columns = policy["input_feature_columns"]
    if args.threads < 1:
        raise ValueError("Thread count must be positive")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    train = [load_period(features, source, 2, columns) for features, source in args.train]
    dev = load_period(*args.development, 3, columns)
    cohorts = [*train, dev]
    seen_clients = set()
    for _, _, _, clients in cohorts:
        this_cohort = set(np.unique(clients))
        if not seen_clients.isdisjoint(this_cohort):
            raise ValueError("Training and development cohorts must have distinct clients")
        seen_clients.update(this_cohort)

    features = PooledFeatures([cohort[0] for cohort in cohorts], columns)
    labels = np.concatenate([c[1] for c in cohorts])
    first = np.concatenate([c[2] for c in cohorts])
    training_count = sum(len(c[1]) for c in train)
    training = np.arange(training_count)
    stop, calibration = development_split(dev[3])
    stop += training_count
    calibration += training_count
    if any(np.unique(labels[r]).size != 2 for r in (training, stop)):
        raise ValueError("Training and stopping subsets must both contain fraud and legitimate transactions")
    negatives = calibration[labels[calibration] == 0]
    if not len(negatives):
        raise ValueError("Calibration needs at least one legitimate operation")
    model, _ = fit_candidate(features, training, stop, labels, first, {**BASE_PARAMS, "nthread": args.threads})
    cuts, _ = calibrated_thresholds(predict_model(model, features, negatives))

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir.parent) as tmp:
        bundle = Path(tmp) / "bundle"
        bundle.mkdir()
        model_file = bundle / policy["model_file"]
        model.save_model(model_file)
        digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
        result = {**policy, **cuts, "name": "refitted_full188", "model_sha256": digest}
        (bundle / "policy.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        bundle.rename(args.output_dir)
    print(f"Selected {model.num_boosted_rounds()} trees on {len(stop)} stopping operations; "
          f"thresholds calibrated on {len(negatives)} legitimate operations. Saved in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs=2, action="append", metavar=("FEATURES", "DATA"), required=True, type=Path)
    parser.add_argument("--development", nargs=2, metavar=("FEATURES", "DATA"), required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
