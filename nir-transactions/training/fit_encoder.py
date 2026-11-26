"""Refit the 54-input Transformer on chronological P1 data (run as a module)."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
import torch
from torch import nn

from pipeline.ru_sequence_training import FittedSequenceEncoder, TransactionTransformer, row_indices
from pipeline.run_ru_fraud_experiment import index_history
from training.fit_classifier import first_episode_rows

ROOT = Path(__file__).resolve().parents[1]


def fit_normalization(tokens, train_rows):
    columns = tokens.shape[1]
    count = np.zeros(columns, dtype=np.int64)
    sums = np.zeros(columns, dtype=np.float64)
    squares = np.zeros(columns, dtype=np.float64)
    for start in range(0, len(train_rows), 65_536):
        values = tokens[train_rows[start:start + 65_536]].astype(np.float64)
        observed = np.isfinite(values)
        values[~observed] = 0
        count += observed.sum(axis=0)
        sums += values.sum(axis=0)
        squares += np.einsum("ij,ij->j", values, values)
    divisor = count.clip(min=1)
    mean = sums / divisor
    variance = np.maximum(0, squares / divisor - mean ** 2)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1
    return mean.astype(np.float32), scale.astype(np.float32), np.flatnonzero(count != len(train_rows))


def training_labels(labels, rows):
    truth = labels[rows]
    if not len(rows) or not np.isin(truth, (0, 1)).all() or np.unique(truth).size != 2:
        raise ValueError("В каждом периоде Transformer нужны оба класса и двоичные метки.")
    return truth


def fit_encoder(tokens, timestamps_ms, history_start, history_end, labels, first_fraud, train_rows, valid_rows,
                output_dir, *, seed, threads, max_epochs, patience, negative_ratio, batch_size, sequence_length):
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or 0 in tokens.shape or tokens.dtype != np.float32:
        raise ValueError("Вход Transformer должен быть непустой двумерной матрицей float32.")
    timestamps_ms, history_start, history_end, labels, first_fraud = map(
        np.asarray, (timestamps_ms, history_start, history_end, labels, first_fraud),
    )
    count = len(tokens)
    if any(array.shape != (count,) for array in (timestamps_ms, history_start, history_end, labels, first_fraud)):
        raise ValueError("Метаданные должны содержать по одному значению на операцию.")
    if any(not np.issubdtype(array.dtype, np.signedinteger) for array in (timestamps_ms, history_start, history_end)):
        raise ValueError("Время и позиции истории должны быть целыми числами со знаком.")
    if (min(threads, max_epochs, patience, batch_size, sequence_length, negative_ratio) <= 0
            or not np.isfinite(negative_ratio)):
        raise ValueError("Настройки Transformer должны быть положительными.")
    train_rows = row_indices(train_rows, count)
    valid_rows = row_indices(valid_rows, count)
    train_truth = training_labels(labels, train_rows)
    valid_truth = training_labels(labels, valid_rows)
    if timestamps_ms[train_rows].max() >= timestamps_ms[valid_rows].min():
        raise ValueError("Период проверки Transformer должен следовать за обучением.")
    if np.any(first_fraud[train_rows].astype(bool) & (train_truth != 1)):
        raise ValueError("Первые операции эпизодов должны иметь метку мошенничества.")

    positive = train_rows[train_truth == 1]
    negative = train_rows[train_truth == 0]
    negative_count = min(len(negative), max(1, int(negative_ratio * len(positive))))
    mean, scale, missing_columns = fit_normalization(tokens, train_rows)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = TransactionTransformer(tokens.shape[1] + len(missing_columns) + 1, sequence_length)
    fitted = FittedSequenceEncoder(model, tokens, timestamps_ms, history_start, history_end, mean, scale,
                                   missing_columns, batch_size)
    learning_rate, weight_decay = 3e-4, 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    objective = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = []
    best_ap, best_epoch, best_parameters = -1.0, 0, None
    for epoch in range(1, max_epochs + 1):
        sampled = negative if negative_count == len(negative) else rng.choice(negative, size=negative_count, replace=False)
        order = rng.permutation(np.concatenate((positive, sampled)))
        model.train()
        accumulated_loss = 0.0
        for offset in range(0, len(order), batch_size):
            rows = order[offset:offset + batch_size]
            targets = torch.from_numpy(labels[rows].astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(*fitted.batch(rows))
            loss = objective(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            accumulated_loss += float(loss.detach()) * len(rows)

        validation_ap = float(average_precision_score(valid_truth, fitted.predict_logits(valid_rows)))
        train_loss = accumulated_loss / len(order)
        epochs.append({"epoch": epoch, "train_loss": train_loss, "validation_average_precision": validation_ap})
        print(f"Transformer, эпоха {epoch}: loss={train_loss:.5f}, validation AP={validation_ap:.5f}.", flush=True)
        if validation_ap > best_ap:
            best_epoch, best_ap = epoch, validation_ap
            best_parameters = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_parameters)
    model.eval().requires_grad_(False)
    fitted.metrics = {
        "selected_epoch": best_epoch,
        "validation_average_precision": best_ap,
        "train_rows": len(train_rows),
        "train_fraud": len(positive),
        "train_first_fraud": int(np.count_nonzero(first_fraud[positive])),
        "train_negatives_per_epoch": negative_count,
        "validation_rows": len(valid_rows),
        "validation_fraud": int(valid_truth.sum()),
        "epochs": epochs,
    }
    checkpoint = {
        "format_version": 1,
        "model_config": model.config,
        "model_state": best_parameters,
        "feature_columns": list(fitted.columns),
        "normalization": {
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
            "missing_columns": torch.from_numpy(missing_columns),
        },
        "training_config": {"seed": seed, "batch_size": batch_size, "negative_ratio": negative_ratio,
                            "max_epochs": max_epochs, "patience": patience, "learning_rate": learning_rate,
                            "weight_decay": weight_decay},
        "metrics": fitted.metrics,
    }
    with TemporaryDirectory(dir=output_dir) as temporary:
        staged = Path(temporary) / "sequence_encoder.pt"
        torch.save(checkpoint, staged)
        staged.replace(output_dir / staged.name)
    return fitted


def run(args):
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    config = json.loads((ROOT / "models/encoder/config.json").read_text(encoding="utf-8"))
    columns = json.loads((args.features / "columns.json").read_text(encoding="utf-8"))
    inputs = np.load(args.features / "X.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(args.features / "ids.npy", mmap_mode="r", allow_pickle=False)
    events = pq.read_table(args.data / "transactions.parquet", columns=["transaction_id", "client_id", "timestamp"])
    truth = pq.read_table(args.data / "labels.parquet", columns=["transaction_id", "is_fraud", "episode_id", "period"])
    if (not events["transaction_id"].equals(truth["transaction_id"])
            or not np.array_equal(events["transaction_id"].to_numpy(), ids)
            or inputs.shape != (len(ids), len(columns)) or inputs.dtype != np.float32
            or np.any(ids[1:] <= ids[:-1]) or len(columns) != len(set(columns))
            or events["timestamp"].type != pa.timestamp("ms", "UTC")):
        raise ValueError("Expected complete aligned features and source transactions, including warmup")
    period = truth["period"].to_numpy()
    source_rows = np.flatnonzero(np.isin(period, (0, 1)))
    token_names = config["token_columns"]
    tokens = np.empty((len(source_rows), len(token_names)), dtype=np.float32)
    token_columns = np.array([columns.index(name) for name in token_names])
    for start in range(0, len(source_rows), 65536):
        batch = slice(start, start + 65536)
        tokens[batch] = inputs[np.ix_(source_rows[batch], token_columns)]

    clients = events["client_id"].to_numpy()[source_rows]
    if np.any(clients[1:] < clients[:-1]):
        raise ValueError("Client histories must be contiguous and ordered")
    timestamps = pc.cast(events["timestamp"], pa.int64()).to_numpy()[source_rows]
    period = period[source_rows]
    labels = truth["is_fraud"].to_numpy()[source_rows]
    first = first_episode_rows(truth["episode_id"].to_numpy()[source_rows], timestamps, labels)
    starts, ends = np.empty(len(labels), dtype=np.int64), np.empty(len(labels), dtype=np.int64)
    index_history(clients, timestamps, starts, ends, 0)

    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    p1 = next(entry for entry in manifest["periods"] if entry["period"] == 1)
    begin, end = [int(datetime.fromisoformat(p1[key]).timestamp() * 1000) for key in ("start", "end_exclusive")]
    if not np.all((timestamps[period == 1] >= begin) & (timestamps[period == 1] < end)):
        raise ValueError("P1 time range and period labels disagree")
    cutoff = begin + (end - begin) * 2 // 3
    train = np.flatnonzero((period == 1) & (timestamps < cutoff))
    valid = np.flatnonzero((period == 1) & (timestamps >= cutoff))
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / "encoder"
        result = fit_encoder(tokens, timestamps, starts, ends, labels, first, train, valid, stage,
                             seed=42, threads=args.threads, max_epochs=args.max_epochs, patience=6,
                             negative_ratio=64, batch_size=256, sequence_length=16)
        summary = {"token_columns": token_names, "embedding_columns": list(result.columns),
                   "history_policy": "strictly_before_current_timestamp", "metrics": result.metrics}
        (stage / "config.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        stage.rename(args.output_dir)
    print(f"Selected epoch {result.metrics['selected_epoch']}; saved in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=40)
    run(parser.parse_args())
