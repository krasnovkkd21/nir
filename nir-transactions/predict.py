"""Score prepared transaction features with the published fraud detector."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb


MODEL_DIR = Path(__file__).resolve().parent / "models"
BATCH_ROWS = 32_768


class RiskModel:
    """Load the frozen feature schema, classifier and two decision thresholds."""

    def __init__(self, threads=4, model_dir=MODEL_DIR):
        if threads < 1:
            raise ValueError("threads must be positive")
        model_dir = Path(model_dir)
        self.policy_path = model_dir / "policy.json"
        policy = json.loads(self.policy_path.read_text(encoding="utf-8"))
        weights = self.weights_path = model_dir / policy["model_file"]
        if hashlib.sha256(weights.read_bytes()).hexdigest() != policy["model_sha256"]:
            raise ValueError("Classifier checksum differs from policy.json")
        self.model = xgb.Booster()
        self.model.load_model(weights)
        self.model.set_param({"nthread": threads, "device": "cpu"})
        self.columns = policy["input_feature_columns"]
        if self.model.feature_names != self.columns:
            raise ValueError("Classifier feature names differ from policy.json")
        self.lower = policy["confirmation_threshold"]
        self.upper = policy["manual_review_threshold"]
        self.actions = policy["actions"]

    def predict(self, features):
        """Score a DataFrame by column name and preserve its row order."""
        if not features.columns.is_unique:
            raise ValueError("Input column names must be unique")
        missing = [name for name in self.columns if name not in features.columns]
        if missing:
            raise ValueError(f"Missing prepared feature columns: {', '.join(missing)}")
        ordered = features[self.columns]
        matrix = ordered.to_numpy(dtype=np.float32, na_value=np.nan)
        if np.isinf(matrix).any():
            raise ValueError("Feature values must be finite or missing")
        if len(matrix):
            scores = self.model.inplace_predict(np.ascontiguousarray(matrix)).astype(np.float64)
        else:
            scores = np.empty(0, dtype=np.float64)
        # The calibrated thresholds use float64, including at the decision boundary.
        bands = np.searchsorted([self.lower, self.upper], scores, side="right")
        result = pd.DataFrame({"risk_score": scores, "action": np.asarray(self.actions)[bands]},
                              index=features.index)
        if "transaction_id" in features:
            result.insert(0, "transaction_id", features["transaction_id"])
        return result


def predict(features, model_dir=MODEL_DIR):
    """Score one DataFrame. Reuse RiskModel for repeated batches."""
    return RiskModel(model_dir=model_dir).predict(features)


@contextmanager
def prepared_batches(source, model):
    if source.is_dir():
        names = json.loads((source / "columns.json").read_text(encoding="utf-8"))
        if isinstance(names, dict):
            names = names["columns"]
        if not isinstance(names, list) or len(names) != len(model.columns) or set(names) != set(model.columns):
            raise ValueError("columns.json differs from the published feature schema")
        matrix = np.load(source / "X.npy", mmap_mode="r", allow_pickle=False)
        ids = np.load(source / "ids.npy", mmap_mode="r", allow_pickle=False)
        if (matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != len(names)
                or ids.shape != (len(matrix),) or not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("Expected a float32 feature matrix and one integer transaction ID per row")

        def frames():
            for start in range(0, len(matrix), BATCH_ROWS):
                batch = slice(start, start + BATCH_ROWS)
                frame = pd.DataFrame(matrix[batch], columns=names)
                frame.insert(0, "transaction_id", ids[batch])
                yield frame

        yield pa.field("transaction_id", pa.from_numpy_dtype(ids.dtype)), frames()
        return

    with pq.ParquetFile(source) as reader:
        schema = reader.schema_arrow
        if len(set(schema.names)) != len(schema.names):
            raise ValueError("Input column names must be unique")
        missing = [name for name in model.columns if name not in schema.names]
        if missing:
            raise ValueError(f"Missing prepared feature columns: {', '.join(missing)}")
        columns = list(model.columns)
        id_field = None
        if "transaction_id" in schema.names:
            columns.append("transaction_id")
            id_field = schema.field("transaction_id")
        batches = reader.iter_batches(batch_size=BATCH_ROWS, columns=columns)
        yield id_field, (batch.to_pandas(types_mapper=pd.ArrowDtype) for batch in batches)


def score_file(source, destination, threads=4, model_dir=MODEL_DIR):
    """Score prepared Parquet or a directory of arrays without loading the full dataset."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must differ")
    if source.is_dir() and destination.resolve() in {
            (source / name).resolve() for name in ("X.npy", "ids.npy", "columns.json")}:
        raise ValueError("The output path would overwrite an input file")
    model = RiskModel(threads, model_dir)
    if destination.resolve() in (model.policy_path.resolve(), model.weights_path.resolve()):
        raise ValueError("The output path would overwrite the classifier or policy")
    with prepared_batches(source, model) as (id_field, batches):
        fields = [pa.field("risk_score", pa.float64()), pa.field("action", pa.string())]
        if id_field is not None:
            fields.insert(0, id_field)
        out_schema = pa.schema(fields)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as temporary:
            staged = Path(temporary) / "decisions.parquet"
            with pq.ParquetWriter(staged, out_schema) as writer:
                for features in batches:
                    result = model.predict(features)
                    writer.write_table(pa.Table.from_pandas(result, schema=out_schema,
                                                            preserve_index=False))
            staged.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Feature Parquet or directory with X.npy, ids.npy and columns.json")
    parser.add_argument("output", type=Path, help="Output Parquet file with risk scores and actions")
    parser.add_argument("--threads", type=int, default=4, help="CPU prediction threads")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="Directory containing policy.json and trained classifier weights")
    args = parser.parse_args()
    score_file(args.input, args.output, args.threads, args.model_dir)
