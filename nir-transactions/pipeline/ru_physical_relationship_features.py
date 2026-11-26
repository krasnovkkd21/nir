"""Causal terminal and city/zone relationships for POS and QR debit purchases.

Only events with timestamps strictly before the decision are available. Simultaneous
events never update each other's history. Identifiers are grouping keys, not features.
"""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from .ru_recent_features import encode, grouped_counts

ID = "transaction_id"

DAY_MS = 86_400_000

FEATURE_COLUMNS = (
    "physical_terminal_age_days",
    "physical_terminal_count_1d",
    "physical_terminal_count_7d",
    "physical_terminal_count_30d",
    "physical_terminal_share_7d_in_30d",
    "physical_zone_age_days",
    "physical_zone_count_1d",
    "physical_zone_count_7d",
    "physical_zone_share_7d_in_30d",
    "physical_new_terminal_count_7d",
    "physical_new_terminal_share_7d",
    "physical_zone_changed",
)


def _boolean(values):
    return pc.fill_null(values, False).to_numpy(zero_copy_only=False)


def _known_codes(values):
    """Dictionary encode nullable keys; an empty text key also means unknown."""
    codes, count = encode(values)
    codes = codes.astype(np.int64, copy=False)
    if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
        codes[_boolean(pc.equal(values, ""))] = -1
    return codes, count


def _ratio(numerator, denominator):
    result = np.full(len(numerator), np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _first_observations(times, codes):
    """Return first relationship age and one introduction row per known relationship."""
    ages = np.full(len(times), np.nan, dtype=np.float32)
    rows = np.flatnonzero(codes >= 0)
    if not len(rows):
        return ages, rows
    rows = rows[np.argsort(codes[rows], kind="stable")]
    ordered = codes[rows]
    starts = np.r_[0, 1 + np.flatnonzero(ordered[1:] != ordered[:-1])]
    ends = np.r_[starts[1:], len(rows)]
    first_rows = rows[starts]
    first_times = np.repeat(times[first_rows], ends - starts)
    # A relationship first seen at t remains unseen for ALL its rows at t.
    earlier = first_times < times[rows]
    ages[rows[earlier]] = (times[rows[earlier]] - first_times[earlier]) / DAY_MS
    return ages, first_rows


def build_features(operations, risk_context):
    """Return transaction IDs and twelve float32 columns in original input order.

    Input ordering is arbitrary. History consists only of eligible physical purchases,
    including warmup rows supplied by the caller. No label/scenario fields are accessed.
    """
    ids = operations[ID].to_numpy()
    if not np.array_equal(ids, risk_context[ID].to_numpy()):
        raise ValueError("Контекст риска и операции имеют разные идентификаторы или порядок.")
    for name in (ID, "client_id", "timestamp"):
        if operations[name].null_count:
            raise ValueError(f"Отсутствует обязательное поле {name}.")
    if not pa.types.is_timestamp(operations["timestamp"].type):
        raise ValueError("Поле timestamp должно иметь тип timestamp.")
    times = pc.cast(pc.cast(operations["timestamp"], pa.timestamp("ms", "UTC")), pa.int64()).to_numpy()
    eligible = _boolean(pc.and_(
        pc.and_(pc.equal(operations["operation_type"], "purchase"),
                pc.equal(operations["direction"], "debit")),
        pc.is_in(operations["channel"], value_set=pa.array(["pos", "qr"])),
    ))
    output = np.full((len(FEATURE_COLUMNS), len(ids)), np.nan, dtype=np.float32)
    selected = np.flatnonzero(eligible)
    if len(selected):
        clients_raw = operations["client_id"].to_numpy()[selected]
        order = np.lexsort((ids[selected], times[selected], clients_raw))
        selected = selected[order]
        times = times[selected]
        # Compress before multiplying: generated client IDs can exceed 3e15.
        _, clients = np.unique(clients_raw[order], return_inverse=True)
        clients = clients.astype(np.int64, copy=False)
        take = pa.array(selected)
        terminal, terminal_count = _known_codes(operations["terminal_id"].take(take))
        city, city_count = _known_codes(operations["merchant_city"].take(take))
        zone_values = risk_context["terminal_zone"].take(take)
        if not pa.types.is_integer(zone_values.type):
            raise ValueError("terminal_zone должен быть целым номером района или null.")
        zone, zone_count = _known_codes(zone_values)
        zone[_boolean(pc.less(zone_values, 0))] = -1
        terminals = np.where(terminal >= 0, clients * terminal_count + terminal, -1)
        known_zone = (city >= 0) & (zone >= 0)
        zones = np.where(known_zone, (clients * city_count + city) * zone_count + zone, -1)

        values = np.full((len(FEATURE_COLUMNS), len(selected)), np.nan, dtype=np.float32)
        values[0], introductions = _first_observations(times, terminals)
        values[1:4] = grouped_counts(times, terminals, (DAY_MS, 7 * DAY_MS, 30 * DAY_MS))
        values[4] = _ratio(values[2], values[3])
        values[5], _ = _first_observations(times, zones)
        zone_counts = grouped_counts(times, zones, (DAY_MS, 7 * DAY_MS, 30 * DAY_MS))
        values[6:8] = zone_counts[:2]
        values[8] = _ratio(zone_counts[1], zone_counts[2])

        # Introductions are distinct terminal relationships, even if several purchases
        # at the first timestamp share a terminal. Query windows still exclude ties.
        introduced = np.zeros(len(selected), dtype=np.int64)
        introduced[introductions] = 1
        edges = np.r_[0, 1 + np.flatnonzero(clients[1:] != clients[:-1]), len(clients)]
        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            ct = times[start:stop]
            known_terminal = terminals[start:stop] >= 0
            lower = np.searchsorted(ct, ct - 7 * DAY_MS, side="left")
            upper = np.searchsorted(ct, ct, side="left")
            prefix_new = np.r_[0, np.cumsum(introduced[start:stop])]
            prefix_known = np.r_[0, np.cumsum(known_terminal)]
            new_counts = (prefix_new[upper] - prefix_new[lower]).astype(np.float32)
            known_counts = prefix_known[upper] - prefix_known[lower]
            values[9, start:stop] = np.where(known_terminal, new_counts, np.nan)
            values[10, start:stop] = np.where(known_terminal, _ratio(new_counts, known_counts), np.nan)

            # A single previous timestamp can contain multiple simultaneous zones.
            # Its zone is usable only when every physical row agrees and is known.
            bucket_starts = np.r_[0, 1 + np.flatnonzero(ct[1:] != ct[:-1])]
            z = zones[start:stop]
            minimum = np.minimum.reduceat(z, bucket_starts)
            maximum = np.maximum.reduceat(z, bucket_starts)
            previous_bucket = np.searchsorted(ct[bucket_starts], ct, side="left") - 1
            previous = np.maximum(previous_bucket, 0)
            usable = ((z >= 0) & (previous_bucket >= 0) & (minimum[previous] >= 0)
                      & (minimum[previous] == maximum[previous]))
            change = np.full(len(ct), np.nan, dtype=np.float32)
            change[usable] = z[usable] != minimum[previous[usable]]
            values[11, start:stop] = change
        output[:, selected] = values
    return pa.table({ID: ids, **dict(zip(FEATURE_COLUMNS, output, strict=True))})
