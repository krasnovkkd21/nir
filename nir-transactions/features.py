"""Generate a synthetic cohort or transform its observed history into frozen model inputs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RECIPIENT_INPUT = [
    "transaction_id", "client_id", "timestamp", "operation_type", "direction",
    "is_self_transfer", "counterparty_id", "recipient_bank_id", "amount_kopecks",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def generate_cohort(output_dir, clients, seed):
    from pipeline.generate_ru_episode_dataset import parameters_from_file
    from pipeline.generate_stress_data_ru_v3 import Config, generate
    from pipeline.replay_ru_mobile_observations import MobileEpisodeTelemetry

    parameters = parameters_from_file(ROOT / "models/preprocessing/episode_parameters.json")
    return generate(output_dir, Config(clients=clients, seed=seed), telemetry=MobileEpisodeTelemetry(parameters))


def complete_client_batches(transactions):
    carry = None
    for record_batch in transactions.iter_batches(batch_size=150_000):
        table = pa.Table.from_batches([record_batch])
        if carry is not None:
            table = pa.concat_tables([carry, table])
        clients = table["client_id"].to_numpy()
        if np.any(clients[1:] < clients[:-1]):
            raise ValueError("Operations must be sorted by client, timestamp and transaction ID.")
        stop = int(np.searchsorted(clients, clients[-1]))
        if stop:
            yield table.slice(0, stop)
        carry = table.slice(stop)
    if carry is not None:
        yield carry


def matrix(table, ids, columns):
    if table.column_names != ["transaction_id", *columns] or not np.array_equal(table["transaction_id"].to_numpy(), ids):
        raise ValueError("Feature names or transaction IDs differ from the trained schema.")
    data = np.column_stack([table[name].to_numpy() for name in columns])
    if data.dtype != np.float32 or np.isinf(data).any():
        raise ValueError("Expected float32 inputs without infinite values.")
    return data


def journal_slice(journal, ids):
    table, journal_ids = journal
    start = int(np.searchsorted(journal_ids, ids[0]))
    stop = int(np.searchsorted(journal_ids, ids[-1], side="right"))
    return table.slice(start, stop - start)


def load_journal(path, operation_ids):
    table = pq.read_table(path)
    ids = table["transaction_id"].to_numpy()
    positions = np.searchsorted(operation_ids, ids)
    if (np.any(ids[1:] <= ids[:-1]) or np.any(positions >= len(operation_ids))
            or not np.array_equal(operation_ids[positions], ids)):
        raise ValueError(f"Duplicate, unsorted or unknown transaction ID in {path.name}.")
    return table, ids


def features_for_batch(operations, context, recipients, journals, rows, columns, encoder_config, checkpoint):
    from pipeline import ru_mobile_session_features as mobile
    from pipeline import ru_physical_relationship_features as physical
    from pipeline import ru_predecision_session as session
    from pipeline import ru_online_amount_risk as online
    from pipeline import ru_recipient_geo_risk as geo
    from pipeline.ru_volume_batch_features import append_features, tabular_features
    from pipeline.ru_profile_v2 import build_profile
    from pipeline.ru_sequence_training import load_encoder
    from pipeline.run_ru_fraud_experiment import index_history

    ids = operations["transaction_id"].to_numpy()
    times = pc.cast(operations["timestamp"], pa.int64()).to_numpy()
    base = append_features(tabular_features(operations), build_profile(operations))
    groups = [
        (base, columns[:104]),
        (session.build_features(operations, journal_slice(journals[0], ids)), session.FEATURE_COLUMNS),
        (recipients, geo.RECIPIENT_COLUMNS),
        (geo.build_geo_features(operations, context), geo.GEO_COLUMNS),
        (online.build_online_features(operations, context), online.ONLINE_COLUMNS),
        (physical.build_features(operations, context), physical.FEATURE_COLUMNS),
        (mobile.build_features(operations, journal_slice(journals[1], ids)), mobile.FEATURE_COLUMNS),
    ]
    arrays = [matrix(table, ids, names) for table, names in groups]
    inputs = {name: array[:, i] for (_, names), array in zip(groups, arrays) for i, name in enumerate(names)}
    tokens = np.column_stack([inputs[name] for name in encoder_config["token_columns"]])
    starts = np.empty(len(ids), dtype=np.int64)
    ends = np.empty_like(starts)
    index_history(operations["client_id"].to_numpy(), times, starts, ends, 0)
    encoder = load_encoder(checkpoint, tokens, times, starts, ends, batch_size=32)
    encoded = encoder.encode(rows)
    names = [*groups[0][1], *("fresh_" + name for name in encoder.columns),
             *(name for _, group_names in groups[1:] for name in group_names)]
    if names != columns:
        raise ValueError("The assembled feature order differs from the classifier policy.")
    return np.column_stack([arrays[0][rows], encoded, *(array[rows] for array in arrays[1:])])


def selected_rows(data_dir, timestamps, period):
    if period is None:
        return np.ones(len(timestamps), dtype=bool)
    schedule = read_json(data_dir / "manifest.json")["periods"]
    entry = next(item for item in schedule if item["period"] == period)
    start, end = [int(datetime.fromisoformat(entry[name]).timestamp() * 1_000)
                  for name in ("start", "end_exclusive")]
    return (start <= timestamps) & (timestamps < end)


def build_features(data_dir, output_dir, period=None, encoder_dir=ROOT / "models/encoder"):
    import torch
    from pipeline.ru_recipient_geo_risk import build_recipient_features

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output already exists: {output_dir}")
    torch.set_num_threads(1)
    columns = read_json(ROOT / "models/policy.json")["input_feature_columns"]
    checkpoint = encoder_dir / "sequence_encoder.pt"
    encoder_config = read_json(encoder_dir / "config.json")
    transactions = pq.ParquetFile(data_dir / "transactions.parquet")
    stream = transactions.read(columns=RECIPIENT_INPUT)
    ids = stream["transaction_id"].to_numpy()
    if not len(ids) or np.any(ids[1:] <= ids[:-1]) or stream["timestamp"].type != pa.timestamp("ms", "UTC"):
        raise ValueError("Expected sorted unique transaction IDs and UTC timestamps in milliseconds.")
    recipients = build_recipient_features(stream)
    times = pc.cast(stream["timestamp"], pa.int64()).to_numpy()
    mask = selected_rows(data_dir, times, period)
    expected = int(mask.sum())
    if not expected:
        raise ValueError("The requested period contains no transactions.")
    context = pq.read_table(data_dir / "risk_context.parquet")
    if not context["transaction_id"].equals(stream["transaction_id"]):
        raise ValueError("Risk context and transactions are not aligned.")
    journals = [load_journal(data_dir / name, ids) for name in ("session_telemetry.parquet", "mobile_telemetry.parquet")]
    del stream
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent, prefix=".prepare-features-") as temporary:
        stage = Path(temporary) / "features"
        stage.mkdir()
        values = np.lib.format.open_memmap(stage / "X.npy", mode="w+", dtype=np.float32, shape=(expected, len(columns)))
        offset = written = 0
        for operations in complete_client_batches(transactions):
            count = len(operations)
            rows = np.flatnonzero(mask[offset:offset+count])
            if len(rows):
                values[written:written+len(rows)] = features_for_batch(
                    operations, context.slice(offset, count), recipients.slice(offset, count), journals,
                    rows, columns, encoder_config, checkpoint,
                )
                written += len(rows)
            offset += count
        if written != expected or offset != len(ids):
            raise RuntimeError("Extraction did not cover the selected operations.")
        values.flush()
        del values
        np.save(stage / "ids.npy", ids[mask], allow_pickle=False)
        (stage / "columns.json").write_text(json.dumps(columns, indent=2) + "\n", encoding="utf-8")
        stage.rename(output_dir)
    print(f"{written:,} transactions, {len(columns)} features: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    generate = subcommands.add_parser("generate", help="Generate labeled scenarios and observable telemetry.")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--clients", type=int, default=30)
    generate.add_argument("--seed", type=int, default=42)
    build = subcommands.add_parser("build", help="Convert observations to the 188 frozen model inputs.")
    build.add_argument("--data-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--encoder-dir", type=Path, default=ROOT / "models/encoder",
                       help="Directory with sequence_encoder.pt and token config.json.")
    build.add_argument("--period", type=int, choices=range(5), help="Emit only this period, retaining its earlier history.")
    args = parser.parse_args()
    if args.command == "generate":
        generate_cohort(args.output_dir, args.clients, args.seed)
    else:
        build_features(args.data_dir, args.output_dir, args.period, args.encoder_dir)


if __name__ == "__main__":
    main()
