"""Observable native-mobile session history available before a debit transfer."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from .ru_predecision_session import TELEMETRY_SCHEMA, _failed_auth

ID = "transaction_id"

DAY_MS = 86_400_000

TIMESTAMP = pa.timestamp("ms", "UTC")

INPUT_TYPES = {
    ID: pa.int64(), "client_id": pa.int64(), "timestamp": TIMESTAMP,
    "operation_type": pa.string(), "channel": pa.string(), "direction": pa.string(),
}

FEATURE_COLUMNS = (
    "mobile_telemetry_observed", "mobile_device_seen_before", "mobile_device_age_days",
    "mobile_device_prior_count", "mobile_device_prior_share", "mobile_device_minutes_since_seen",
    "mobile_device_is_most_recent", "mobile_device_distinct_seen", "mobile_session_elapsed_seconds",
    "mobile_failed_auth_count", "mobile_seconds_since_last_failed_auth",
    "mobile_device_changed_within_session",
)


def eligible_rows(operations):
    result = pc.and_(pc.and_(pc.equal(operations["operation_type"], "transfer"),
                            pc.equal(operations["channel"], "mobile_app")),
                     pc.equal(operations["direction"], "debit"))
    return pc.fill_null(result, False).to_numpy(zero_copy_only=False)


def _keys(clients, times, devices=None):
    names = [("client", np.int64)]
    if devices is not None:
        names.append(("device", np.int64))
    names.append(("time", np.int64))
    keys = np.empty(len(times), dtype=names)
    keys["client"], keys["time"] = clients, times
    if devices is not None:
        keys["device"] = devices
    return keys


def _device_history(clients, times, devices, ids):
    """Seven client-scoped device columns; ties are excluded together."""
    n = len(times)
    result = np.full((7, n), np.nan, dtype=np.float32)
    if not n:
        return result
    # Tuple keys avoid multiplying raw client IDs (which can approach int64 limits).
    order = np.lexsort((ids, times, devices, clients))
    c, d, t = clients[order], devices[order], times[order]
    keys = _keys(c, t, d)
    beginning = np.r_[True, (c[1:] != c[:-1]) | (d[1:] != d[:-1])]
    starts = np.maximum.accumulate(np.where(beginning, np.arange(n), 0))
    end = np.searchsorted(keys, keys, side="left")
    count = end - starts
    known = count > 0
    result[0, order] = known
    result[1, order[known]] = (t[known] - t[starts[known]]) / DAY_MS
    result[2, order] = count
    result[4, order[known]] = (t[known] - t[end[known] - 1]) / 60_000

    introduced = np.zeros(n, dtype=np.int64)
    introduced[order[beginning]] = 1
    chronological = np.lexsort((ids, times, clients))
    c, t, d = clients[chronological], times[chronological], devices[chronological]
    keys = _keys(c, t)
    beginning = np.r_[True, c[1:] != c[:-1]]
    starts = np.maximum.accumulate(np.where(beginning, np.arange(n), 0))
    end = np.searchsorted(keys, keys, side="left")
    prior = end - starts
    counts = result[2, chronological]
    share = np.full(n, np.nan, dtype=np.float32)
    np.divide(counts, prior, out=share, where=prior > 0)
    result[3, chronological] = share
    prefix = np.r_[0, np.cumsum(introduced[chronological])]
    result[6, chronological] = prefix[end] - prefix[starts]

    bucket_starts = np.flatnonzero(beginning | np.r_[True, t[1:] != t[:-1]])
    bucket_number = np.repeat(np.arange(len(bucket_starts)), np.diff(np.r_[bucket_starts, n]))
    minimum = np.minimum.reduceat(d, bucket_starts)
    maximum = np.maximum.reduceat(d, bucket_starts)
    rows = np.flatnonzero(prior > 0)
    previous_bucket = bucket_number[end[rows] - 1]
    usable = minimum[previous_bucket] == maximum[previous_bucket]
    rows, previous_bucket = rows[usable], previous_bucket[usable]
    result[5, chronological[rows]] = d[rows] == minimum[previous_bucket]
    return result


def build_features(operations: pa.Table, telemetry: pa.Table) -> pa.Table:
    """Return twelve float32 columns, without labels, scenarios or private actor state."""
    for name, expected in INPUT_TYPES.items():
        column = operations[name]
        if column.type != expected or column.null_count:
            raise ValueError(f"Incorrect operation type or missing value in {name}.")
    if telemetry.schema != TELEMETRY_SCHEMA or any(c.null_count for c in telemetry.columns):
        raise ValueError("Mobile observations have an incorrect schema or null values.")
    ids = operations[ID].to_numpy()
    if np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Operation IDs must be unique and increasing.")
    observed_ids = telemetry[ID].to_numpy()
    rows = np.searchsorted(ids, observed_ids)
    if (np.any(observed_ids[1:] <= observed_ids[:-1]) or np.any(rows >= len(ids))
            or np.any(ids[rows] != observed_ids)):
        raise ValueError("Each mobile observation must refer to a unique matching operation ID.")
    eligible = eligible_rows(operations)
    if not eligible[rows].all():
        raise ValueError("Mobile observations apply only to outgoing transfers in the mobile app.")
    times = pc.cast(operations["timestamp"], pa.int64()).to_numpy()[rows]
    clients = operations["client_id"].to_numpy()[rows]
    observed = pc.cast(telemetry["observed_at"], pa.int64()).to_numpy()
    started = pc.cast(telemetry["session_started_at"], pa.int64()).to_numpy()
    devices = telemetry["device_ref"].to_numpy()
    initial = telemetry["initial_device_ref"].to_numpy()
    if (np.any(started >= observed) or np.any(observed >= times)
            or np.any(devices <= 0) or np.any(initial <= 0)):
        raise ValueError("Mobile start and observation must precede the decision; devices must be positive.")
    counts, with_attempts, last = _failed_auth(telemetry["failed_auth_times"].combine_chunks(), started, observed)
    values = np.full((len(FEATURE_COLUMNS), len(ids)), np.nan, dtype=np.float32)
    values[0, eligible] = 0
    values[0, rows] = 1
    values[1:8, rows] = _device_history(clients, times, devices, observed_ids)
    values[8, rows] = (times - started) / 1_000
    values[9, rows] = counts
    values[10, rows[with_attempts]] = (times[with_attempts] - last) / 1_000
    values[11, rows] = devices != initial
    if np.isinf(values).any():
        raise ValueError("Mobile features contain an infinite value.")
    return pa.table({ID: operations[ID], **dict(zip(FEATURE_COLUMNS, values, strict=True))})
