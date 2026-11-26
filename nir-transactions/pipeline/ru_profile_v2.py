from __future__ import annotations
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

MINUTE_MS = 60_000

HOUR_MS = 60 * MINUTE_MS

HISTORY_MS = 30 * 24 * HOUR_MS

FEATURE_COLUMNS = (
    "profile_amount_context_count_30d",
    "profile_amount_context_z_30d",
    "profile_recipient_count_30d",
    "profile_recipient_minutes_since_previous",
    "profile_new_recipient_share_30d",
    "profile_city_channel_count_30d",
    "profile_city_channel_minutes_since_previous",
    "profile_channel_hour_share_30d",
)


def _codes(column):
    encoded = pc.dictionary_encode(column.combine_chunks())
    return pc.fill_null(encoded.indices, -1).to_numpy()


def _matches(column, value):
    return pc.fill_null(pc.equal(column, value), False).to_numpy(zero_copy_only=False)


def _history(clients, times, eligible, *keys):
    rows = np.flatnonzero(eligible)
    if not len(rows):
        return rows, rows, rows, rows

    sort_keys = (times[rows], *(key[rows] for key in reversed(keys)), clients[rows])
    rows = rows[np.lexsort(sort_keys)]
    new_group = np.r_[True, clients[rows[1:]] != clients[rows[:-1]]]
    for key in keys:
        new_group[1:] |= key[rows[1:]] != key[rows[:-1]]

    positions = np.arange(len(rows))
    group_start = np.maximum.accumulate(np.where(new_group, positions, 0))
    group = np.cumsum(new_group) - 1
    ordered_times = times[rows]
    minimum_time = int(ordered_times.min())
    stride = int(ordered_times.max()) - minimum_time + HISTORY_MS + 1
    if stride * (int(group[-1]) + 1) <= np.iinfo(np.int64).max:
        lookup = group * stride + (ordered_times - minimum_time)
        stop = np.searchsorted(lookup, lookup, side="left")
        start = np.searchsorted(lookup, lookup - HISTORY_MS, side="left")
        return rows, start, stop, group_start

    lookup = np.empty(len(rows), dtype=[("group", np.int64), ("time", np.int64)])
    lookup["group"] = group
    lookup["time"] = ordered_times
    stop = np.searchsorted(lookup, lookup, side="left")
    queries = lookup.copy()
    queries["time"] = np.maximum(ordered_times, np.iinfo(np.int64).min + HISTORY_MS) - HISTORY_MS
    start = np.searchsorted(lookup, queries, side="left")
    return rows, start, stop, group_start


def _sums(values, start, stop):
    prefix = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    return prefix[stop] - prefix[start]


def _minutes_since_previous(output, times, rows, stop, group_start):
    has_history = stop > group_start
    current = rows[has_history]
    previous = rows[stop[has_history] - 1]
    output[current] = (times[current] - times[previous]) / MINUTE_MS


def build_profile(operations: pa.Table) -> pa.Table:
    ids = operations["transaction_id"]
    if operations["timestamp"].type != pa.timestamp("ms", tz="UTC"):
        raise ValueError("Профилю нужны отметки времени в UTC с миллисекундной точностью.")

    clients = operations["client_id"].to_numpy()
    times = pc.cast(operations["timestamp"], pa.int64()).to_numpy()
    channel = _codes(operations["channel"])
    direction = _codes(operations["direction"])
    category = _codes(operations["mcc"])
    currency = _codes(operations["currency"])
    recipient = _codes(operations["counterparty_id"])
    city = _codes(operations["merchant_city"])

    debit = _matches(operations["direction"], "debit")
    purchase = debit & _matches(operations["operation_type"], "purchase")
    external_transfer = (
        debit
        & _matches(operations["operation_type"], "transfer")
        & _matches(operations["is_self_transfer"], False)
    )

    features = {name: np.full(len(operations), np.nan, dtype=np.float32) for name in FEATURE_COLUMNS}

    amount = operations["amount_kopecks"].to_numpy()
    known_amount = (amount > 0) & (currency >= 0) & (channel >= 0)
    comparable = known_amount & ((purchase & (category >= 0)) | external_transfer)
    amount_group = np.where(purchase, category, -1)
    rows, start, stop, _ = _history(clients, times, comparable, channel, currency, amount_group)
    counts = stop - start
    features["profile_amount_context_count_30d"][rows] = counts
    enough = counts >= 5
    log_amount = np.log(amount[rows].astype(np.float64))
    averages = _sums(log_amount, start[enough], stop[enough]) / counts[enough]
    variance = _sums(log_amount**2, start[enough], stop[enough]) / counts[enough] - averages**2
    scale = np.sqrt(np.maximum(variance, 0.25**2))
    features["profile_amount_context_z_30d"][rows[enough]] = (log_amount[enough] - averages) / scale

    known_recipient = external_transfer & (recipient >= 0)
    rows, start, stop, group_start = _history(clients, times, known_recipient, recipient)
    features["profile_recipient_count_30d"][rows] = stop - start
    _minutes_since_previous(features["profile_recipient_minutes_since_previous"], times, rows, stop, group_start)

    new_recipient = np.zeros(len(operations), dtype=bool)
    new_recipient[rows] = stop == group_start
    rows, start, stop, _ = _history(clients, times, known_recipient)
    past_transfers = stop - start
    has_history = past_transfers > 0
    features["profile_new_recipient_share_30d"][rows[has_history]] = (
        _sums(new_recipient[rows], start[has_history], stop[has_history]) / past_transfers[has_history]
    )

    physical = _matches(operations["channel"], "pos") | _matches(operations["channel"], "qr")
    rows, start, stop, group_start = _history(clients, times, purchase & physical & (city >= 0), channel, city)
    features["profile_city_channel_count_30d"][rows] = stop - start
    _minutes_since_previous(features["profile_city_channel_minutes_since_previous"], times, rows, stop, group_start)

    known_channel = (channel >= 0) & (direction >= 0)
    rows, start, stop, _ = _history(clients, times, known_channel, channel, direction)
    channel_counts = np.zeros(len(operations), dtype=np.int64)
    channel_counts[rows] = stop - start
    moscow_hours = (times // HOUR_MS + 3) % 24
    rows, start, stop, _ = _history(clients, times, known_channel, channel, direction, moscow_hours // 4)
    has_history = channel_counts[rows] > 0
    features["profile_channel_hour_share_30d"][rows[has_history]] = (
        (stop[has_history] - start[has_history]) / channel_counts[rows[has_history]]
    )

    return pa.table({"transaction_id": ids, **features})
