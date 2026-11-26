"""Признаки истории получателя по предшествующим наблюдаемым переводам."""
import numpy as np

MINUTE_MS = 60_000

WINDOWS = {
    "recipient_other_senders_1h": 60,
    "recipient_other_senders_24h": 24 * 60,
    "recipient_other_senders_7d": 7 * 24 * 60,
}

AGE = "recipient_age_minutes"

FEATURE_COLUMNS = (*WINDOWS, AGE)


def recipient_history(times, target_ids, sender_ids, eligible):
    """Считает других отправителей в интервалах [t - окно, t).

    Возраст учитывает первое строго предшествующее наблюдение получателя
    у любого отправителя. При его отсутствии возраст неизвестен.
    """
    times = np.asarray(times, dtype=np.int64)
    target_ids = np.asarray(target_ids)
    sender_ids = np.asarray(sender_ids)
    eligible = np.asarray(eligible, dtype=bool)
    if times.ndim != 1 or any(
        values.shape != times.shape for values in (target_ids, sender_ids, eligible)
    ):
        raise ValueError("Время, получатели, отправители и маска должны иметь одинаковую длину.")

    features = {name: np.full(len(times), np.nan, dtype=np.float32) for name in FEATURE_COLUMNS}
    rows = np.flatnonzero(eligible)
    if not len(rows):
        return features
    rows = rows[np.lexsort((times[rows], target_ids[rows]))]
    time = times[rows]
    target = target_ids[rows]
    sender = sender_ids[rows]
    size = len(rows)

    new_target = np.r_[True, target[1:] != target[:-1]]
    new_time = new_target | np.r_[True, time[1:] != time[:-1]]
    target_start = np.maximum.accumulate(np.where(new_target, np.arange(size), 0))
    observed = time > time[target_start]
    features[AGE][rows[observed]] = (
        time[observed].astype(np.float64) - time[target_start[observed]].astype(np.float64)
    ) / MINUTE_MS
    time_group_edges = np.r_[np.flatnonzero(new_time), size]
    after_time = np.repeat(time_group_edges[1:], np.diff(time_group_edges))

    # Для каждой пары отправитель–получатель считаем только последнюю операцию до запроса.
    pair_order = np.lexsort((sender, target))
    pair_target, pair_sender, pair_time = target[pair_order], sender[pair_order], time[pair_order]
    same_pair = (pair_target[1:] == pair_target[:-1]) & (pair_sender[1:] == pair_sender[:-1])
    next_operation_start = np.full(size, size, dtype=np.intp)
    next_operation_start[pair_order[:-1][same_pair]] = after_time[pair_order[1:][same_pair]]

    pair_pos = np.arange(size)
    pair_start = np.maximum.accumulate(np.where(np.r_[True, ~same_pair], pair_pos, 0))
    new_pair_time = np.r_[True, (~same_pair) | (pair_time[1:] != pair_time[:-1])]
    pair_time_start = np.maximum.accumulate(np.where(new_pair_time, pair_pos, 0))
    has_prior_sender = pair_time_start > pair_start
    prior_sender_time = pair_time[pair_time_start[has_prior_sender] - 1]
    sender_rows = pair_order[has_prior_sender]

    keys = np.empty(size, dtype=[("target", target.dtype), ("time", time.dtype)])
    keys["target"], keys["time"] = target, time
    expiry_keys = keys.copy()
    arrivals = np.bincount(after_time, minlength=size + 1)
    max_time = np.iinfo(np.int64).max
    for name, minutes in WINDOWS.items():
        window = minutes * MINUTE_MS
        expiry_keys["time"] = np.minimum(time, max_time - window) + window
        expiry = np.searchsorted(keys, expiry_keys, side="right")
        departure = np.minimum(expiry, next_operation_start)
        # Вклад операции прекращается после окна или следующего перевода от того же отправителя.
        counts = np.cumsum(arrivals - np.bincount(departure, minlength=size + 1))[:-1]
        own_recent = time[sender_rows] <= np.minimum(prior_sender_time, max_time - window) + window
        counts[sender_rows[own_recent]] -= 1
        features[name][rows] = counts
    return features
