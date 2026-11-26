"""Подсчитать недавние операции клиента для условного эксперимента RU v3."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ID = "transaction_id"

SHORT_WINDOWS = (5 * 60_000, 15 * 60_000)

PAYEE_WINDOWS = (*SHORT_WINDOWS, 30 * 86_400_000)

FEATURE_COLUMNS = (
    "recent_operation_count_5m",
    "recent_operation_count_15m",
    "recent_same_channel_direction_count_5m",
    "recent_same_channel_direction_count_15m",
    "recent_same_payee_count_5m",
    "recent_same_payee_count_15m",
    "recent_same_payee_count_30d",
)


def window_counts(times, windows):
    """Границы окна: [timestamp - window, timestamp)."""
    end = np.searchsorted(times, times, side="left")
    return [end - np.searchsorted(times, times - window, side="left") for window in windows]


def grouped_counts(times, codes, windows):
    result = np.full((len(windows), len(times)), np.nan, dtype=np.float32)
    known = np.flatnonzero(codes >= 0)
    if len(known):
        rows = known[np.argsort(codes[known], kind="stable")]
        ordered_codes = codes[rows]
        edges = np.r_[0, 1 + np.flatnonzero(ordered_codes[1:] != ordered_codes[:-1]), len(rows)]
        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            selected = rows[start:stop]
            result[:, selected] = window_counts(times[selected], windows)
    return result


def encode(values):
    encoded = pc.dictionary_encode(values.combine_chunks())
    return pc.fill_null(encoded.indices, -1).to_numpy(), len(encoded.dictionary)


def payee_codes(operations):
    merchant, merchants = encode(operations["merchant_id"])
    counterparty, _ = encode(operations["counterparty_id"])
    purchase = pc.equal(operations["operation_type"], "purchase").to_numpy(zero_copy_only=False)
    debit_transfer = pc.and_(
        pc.equal(operations["operation_type"], "transfer"), pc.equal(operations["direction"], "debit")
    ).to_numpy(zero_copy_only=False)
    # Одинаковый числовой код у продавца и получателя перевода не означает совпадение.
    return np.where(purchase, merchant, np.where(debit_transfer & (counterparty >= 0),
                                                 merchants + counterparty, -1))


def build_features(operations):
    ids = operations[ID].to_numpy()
    clients = operations["client_id"].to_numpy()
    times = pc.cast(operations["timestamp"], pa.int64()).to_numpy()
    channel, _ = encode(operations["channel"])
    direction, directions = encode(operations["direction"])
    channel_direction = channel * directions + direction
    payees = payee_codes(operations)
    order = np.lexsort((ids, times, clients))
    sorted_clients = clients[order]
    edges = np.r_[0, 1 + np.flatnonzero(sorted_clients[1:] != sorted_clients[:-1]), len(order)]
    features = np.empty((len(FEATURE_COLUMNS), len(ids)), dtype=np.float32)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        rows = order[start:stop]
        client_times = times[rows]
        features[:2, rows] = window_counts(client_times, SHORT_WINDOWS)
        features[2:4, rows] = grouped_counts(client_times, channel_direction[rows], SHORT_WINDOWS)
        features[4:, rows] = grouped_counts(client_times, payees[rows], PAYEE_WINDOWS)
    return pa.table({ID: ids, **dict(zip(FEATURE_COLUMNS, features, strict=True))})
