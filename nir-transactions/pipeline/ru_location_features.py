"""Рассчитать признаки смены города по истории физических покупок клиента."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from .ru_recent_features import ID, encode, grouped_counts, window_counts

FEATURE_COLUMNS = (
    "physical_minutes_since_previous",
    "physical_city_changed",
    "physical_different_city_count_24h",
)

MINUTE_MS = 60_000

DAY_MS = 1_440 * MINUTE_MS


def client_location_history(times, cities):
    prior = np.searchsorted(times, times, side="left")
    minutes = np.full(len(times), np.nan, dtype=np.float32)
    changed = np.full(len(times), np.nan, dtype=np.float32)
    rows = np.flatnonzero(prior > 0)
    previous = prior[rows] - 1
    minutes[rows] = (times[rows] - times[previous]) / MINUTE_MS

    # У последней отметки времени могут быть покупки в нескольких городах.
    starts = np.r_[True, times[1:] != times[:-1]]
    groups = np.cumsum(starts) - 1
    edges = np.flatnonzero(starts)
    unambiguous = np.minimum.reduceat(cities, edges) == np.maximum.reduceat(cities, edges)
    valid = unambiguous[groups[previous]]
    changed[rows[valid]] = cities[rows[valid]] != cities[previous[valid]]

    all_counts, = window_counts(times, (DAY_MS,))
    same_city, = grouped_counts(times, cities, (DAY_MS,))
    return minutes, changed, all_counts - same_city


def build_features(operations):
    ids = operations[ID].to_numpy()
    clients = operations["client_id"].to_numpy()
    times = pc.cast(operations["timestamp"], pa.int64()).to_numpy()
    city_codes, _ = encode(operations["merchant_city"])
    eligible = pc.and_(
        pc.and_(pc.equal(operations["operation_type"], "purchase"),
                pc.is_in(operations["channel"], value_set=pa.array(["pos", "qr"]))),
        pc.fill_null(pc.not_equal(operations["merchant_city"], ""), False),
    ).to_numpy(zero_copy_only=False)
    selected = np.flatnonzero(eligible)
    features = np.full((len(FEATURE_COLUMNS), len(ids)), np.nan, dtype=np.float32)
    if len(selected):
        order = selected[np.lexsort((ids[selected], times[selected], clients[selected]))]
        sorted_clients = clients[order]
        edges = np.r_[0, 1 + np.flatnonzero(sorted_clients[1:] != sorted_clients[:-1]), len(order)]
        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            rows = order[start:stop]
            features[:, rows] = client_location_history(times[rows], city_codes[rows])
    return pa.table({ID: ids, **dict(zip(FEATURE_COLUMNS, features, strict=True))})
