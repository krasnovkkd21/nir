"""Признаки российских синтетических транзакций на момент операции."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .build_behavior_features import DAY_NS, HOUR_NS, amount_stats, divide
from .ru_transaction_catalog import CATEGORIES, CITIES

MINUTE_NS = HOUR_NS // 60

MOSCOW_OFFSET_NS = 3 * HOUR_NS

CURRENT_CATEGORIES = {
    "operation_type": (("purchase", "purchase"), ("transfer", "transfer"), ("refund", "refund")),
    "rail": (("card", "card"), ("sbp", "sbp")),
    "channel": tuple((value, value) for value in ("pos", "ecommerce", "qr", "mobile_app", "refund")),
    "direction": (("debit", "debit"), ("credit", "credit")),
    "entry_mode": (("contactless", "contactless"), ("chip", "chip")),
    "mcc": tuple((mcc, mcc) for mcc, *_ in CATEGORIES),
    "merchant_city": tuple((city, str(number)) for number, (city, _) in enumerate(CITIES, 1)),
}

MATCHED_FIELDS = {
    "merchant": "merchant_id",
    "terminal": "terminal_id",
    "mcc": "mcc",
    "city": "merchant_city",
    "counterparty": "counterparty_id",
    "other_bank": "other_bank",
    "hour": "hour",
}

PROFILE_COLUMNS = (
    "history_count", "history_age_days", "minutes_since_previous",
    "count_1h", "count_24h", "count_7d", "amount_sum_24h",
    "purchase_count_24h", "transfer_count_24h", "transfer_count_7d", "transfer_amount_sum_24h",
    "channel_amount_count_30d", "channel_amount_mean_30d", "channel_amount_std_30d",
    "channel_amount_ratio_30d", "minutes_since_previous_same_channel",
    *(f"{name}_{feature}" for name in MATCHED_FIELDS for feature in ("seen_before", "prior_share")),
)

REQUIRED_COLUMNS = (
    "transaction_id", "client_id", "timestamp", "amount_kopecks", *CURRENT_CATEGORIES,
    "merchant_id", "terminal_id", "sender_bank_id", "recipient_bank_id", "counterparty_id",
    "is_self_transfer",
)


def _prefix(values):
    return np.r_[0, np.cumsum(values)]


def _elapsed_minutes(times, prior):
    valid = prior > 0
    result = np.full(len(times), np.nan, dtype=np.float32)
    result[valid] = (times[valid] - times[prior[valid] - 1]) / MINUTE_NS
    return result


def _same_channel_amount(amounts, times, channel_codes):
    results = np.full((len(times), 5), np.nan, dtype=np.float32)
    for code in np.unique(channel_codes):
        rows = np.flatnonzero(channel_codes == code)
        selected_times = times[rows]
        selected_amounts = amounts[rows]
        end = np.searchsorted(selected_times, selected_times, side="left")
        start = np.searchsorted(selected_times, selected_times - 30 * DAY_NS, side="left")
        mean, std = amount_stats(_prefix(selected_amounts), _prefix(selected_amounts ** 2), start, end)
        results[rows, 0] = end - start
        results[rows, 1] = mean
        results[rows, 2] = std
        results[rows, 3] = divide(selected_amounts, mean)
        results[rows, 4] = _elapsed_minutes(selected_times, end)
    return results


def _matched_history(codes, times, eligible, end):
    """Найти предыдущие совпадения внутри клиента, исключив всю текущую минуту при равенстве времени."""
    seen = np.full(len(times), np.nan, dtype=np.float32)
    shares = np.full(len(times), np.nan, dtype=np.float32)
    known = np.flatnonzero(codes >= 0)
    if not len(known):
        return seen, shares

    # Стабильная сортировка по значению сохраняет хронологический порядок внутри группы.
    order = known[np.argsort(codes[known], kind="stable")]
    positions = np.arange(len(order))
    sorted_codes = codes[order]
    sorted_times = times[order]
    new_value = np.r_[True, sorted_codes[1:] != sorted_codes[:-1]]
    new_time = new_value | np.r_[True, sorted_times[1:] != sorted_times[:-1]]
    value_start = np.maximum.accumulate(np.where(new_value, positions, 0))
    time_start = np.maximum.accumulate(np.where(new_time, positions, 0))
    matched_prefix = _prefix(eligible[order])
    counts = matched_prefix[time_start] - matched_prefix[value_start]
    prior_known = _prefix(eligible & (codes >= 0))[end[order]]
    seen[order] = counts > 0
    shares[order] = divide(counts, prior_known)
    return seen, shares


def _client_profile(times, amounts, channels, purchase, transfer, matches):
    end = np.searchsorted(times, times, side="left")
    past_hour = np.searchsorted(times, times - HOUR_NS, side="left")
    past_day = np.searchsorted(times, times - DAY_NS, side="left")
    past_week = np.searchsorted(times, times - 7 * DAY_NS, side="left")
    amount_prefix = _prefix(amounts)
    purchase_prefix = _prefix(purchase)
    transfer_prefix = _prefix(transfer)
    transfer_amount_prefix = _prefix(np.where(transfer, amounts, 0.0))
    channel_amount = _same_channel_amount(amounts, times, channels)
    all_ops = np.ones(len(times), dtype=bool)
    matched = []
    for name, codes in matches.items():
        eligible = transfer if name in ("counterparty", "other_bank") else all_ops if name == "hour" else purchase
        matched.extend(_matched_history(codes, times, eligible, end))

    return np.column_stack((
        end, (times - times[0]) / DAY_NS, _elapsed_minutes(times, end),
        end - past_hour, end - past_day, end - past_week,
        amount_prefix[end] - amount_prefix[past_day],
        purchase_prefix[end] - purchase_prefix[past_day],
        transfer_prefix[end] - transfer_prefix[past_day],
        transfer_prefix[end] - transfer_prefix[past_week],
        transfer_amount_prefix[end] - transfer_amount_prefix[past_day],
        channel_amount, *matched,
    ))


def _encode_categories(operations, features):
    for field, vocabulary in CURRENT_CATEGORIES.items():
        values = operations[field]
        for value, suffix in vocabulary:
            features[f"{field}_{suffix}"] = values.eq(value).to_numpy(dtype=np.uint8, na_value=0)
        features[f"{field}_missing"] = values.isna().to_numpy(dtype=np.uint8)
        features[f"{field}_other"] = (values.notna() & ~values.isin([value for value, _ in vocabulary])).to_numpy(dtype=np.uint8)


def build_features(operations: pd.DataFrame) -> pd.DataFrame:
    """Вернуть числовые признаки в исходном порядке, включая int64-ключ transaction_id."""
    missing = set(REQUIRED_COLUMNS).difference(operations.columns)
    if missing:
        raise ValueError(f"Отсутствуют поля транзакций: {', '.join(sorted(missing))}")
    for field in ("transaction_id", "client_id", "amount_kopecks"):
        if not pd.api.types.is_integer_dtype(operations[field]) or operations[field].isna().any():
            raise ValueError(f"Поле {field} должно содержать целые числа без пропусков.")

    transaction_ids = operations["transaction_id"].to_numpy(dtype=np.int64)
    clients = operations["client_id"].to_numpy(dtype=np.int64)
    timestamps = pd.to_datetime(operations["timestamp"], utc=True)
    if timestamps.isna().any():
        raise ValueError("Время операции обязательно.")
    times = timestamps.to_numpy(dtype="datetime64[ns]").view(np.int64)
    amounts = operations["amount_kopecks"].to_numpy(dtype=np.float64) / 100.0
    if (amounts <= 0).any():
        raise ValueError("Сумма операции должна быть положительной.")
    local_time = times + MOSCOW_OFFSET_NS
    hours = local_time // HOUR_NS % 24
    weekdays = (local_time // DAY_NS + 3) % 7

    sender = operations["sender_bank_id"]
    recipient = operations["recipient_bank_id"]
    transfer = operations["operation_type"].eq("transfer").to_numpy(dtype=bool, na_value=False)
    purchase = operations["operation_type"].eq("purchase").to_numpy(dtype=bool, na_value=False)
    debit = operations["direction"].eq("debit").to_numpy(dtype=bool, na_value=False)
    same_bank = sender.eq(recipient).where(transfer & sender.notna() & recipient.notna())
    other_bank = recipient.where(debit, sender).where(transfer)
    self_transfer = pd.array(operations["is_self_transfer"], dtype="boolean")
    features = {
        "transaction_id": transaction_ids,
        "amount_rub": amounts.astype(np.float32),
        "log_amount_rub": np.log1p(amounts).astype(np.float32),
        "hour_msk_sin": np.sin(hours * 2 * np.pi / 24).astype(np.float32),
        "hour_msk_cos": np.cos(hours * 2 * np.pi / 24).astype(np.float32),
        "weekday_msk": weekdays.astype(np.uint8),
        "is_weekend_msk": (weekdays >= 5).astype(np.uint8),
        "same_bank_transfer": same_bank.to_numpy(dtype=np.float32, na_value=np.nan),
        "is_self_transfer": self_transfer.to_numpy(dtype=np.float32, na_value=np.nan),
    }
    _encode_categories(operations, features)

    match_values = {"other_bank": other_bank, "hour": hours}
    for field in MATCHED_FIELDS.values():
        if field not in match_values:
            match_values[field] = operations[field]
    order = np.lexsort((transaction_ids, times, clients))
    profile = np.empty((len(order), len(PROFILE_COLUMNS)), dtype=np.float32)
    if len(order):
        sorted_clients = clients[order]
        limits = np.r_[0, np.flatnonzero(sorted_clients[1:] != sorted_clients[:-1]) + 1, len(order)]
        # Код категории нужен только для сравнения значений внутри истории.
        channel_codes = pd.factorize(pd.MultiIndex.from_frame(operations[["channel", "direction"]]))[0]
        match_codes = {name: pd.factorize(match_values[field])[0] for name, field in MATCHED_FIELDS.items()}
        for left, right in zip(limits[:-1], limits[1:]):
            rows = order[left:right]
            profile[rows] = _client_profile(
                times[rows], amounts[rows], channel_codes[rows], purchase[rows], transfer[rows],
                {name: codes[rows] for name, codes in match_codes.items()},
            )
    features.update(zip(PROFILE_COLUMNS, profile.T, strict=True))
    return pd.DataFrame(features, index=operations.index)
