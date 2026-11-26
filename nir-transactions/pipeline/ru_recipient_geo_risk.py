"""Признаки получателей и географии из истории, доступной до операции."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from .ru_recent_features import encode
from .ru_recipient_features import FEATURE_COLUMNS as RECIPIENT_COLUMNS, recipient_history

ID = "transaction_id"

DAY_MS = 86_400_000

MONTH_MS = 30 * DAY_MS

GEO_COLUMNS = (
    "geo_history_purchase_count_30d",
    "geo_city_purchase_count_30d",
    "geo_city_purchase_share_30d",
    "geo_zone_purchase_count_30d",
    "geo_zone_share_within_city_30d",
    "geo_days_since_zone",
)


def _boolean(values):
    return pc.fill_null(values, False).to_numpy()


def _timestamps(operations):
    values = operations["timestamp"]
    if values.type != pa.timestamp("ms", "UTC") or values.null_count:
        raise ValueError("Для истории требуется время каждой операции в UTC с миллисекундной точностью.")
    return pc.cast(values, pa.int64()).to_numpy()


def build_recipient_features(operations: pa.Table) -> pa.Table:
    """Подсчитать историю получателей на потоке переводов всех известных клиентов."""
    counterparty, parties = encode(operations["counterparty_id"])
    bank, _ = encode(operations["recipient_bank_id"])
    eligible = (
        _boolean(pc.equal(operations["operation_type"], "transfer"))
        & _boolean(pc.equal(operations["direction"], "debit"))
        & _boolean(pc.equal(operations["is_self_transfer"], False))
        & _boolean(pc.not_equal(operations["counterparty_id"], ""))
        & _boolean(pc.not_equal(operations["recipient_bank_id"], ""))
    )
    recipient = bank.astype(np.int64) * parties + counterparty
    features = recipient_history(
        _timestamps(operations), recipient, operations["client_id"].to_numpy(), eligible,
    )
    return pa.table({ID: operations[ID], **features})


def _last_month(times, groups, *, include_age=False):
    """Количество событий группы в [t - 30 дней, t) и возраст последнего события."""
    order = np.lexsort((times, groups))
    keys = np.empty(len(times), dtype=[("group", np.int64), ("time", np.int64)])
    keys["group"], keys["time"] = groups[order], times[order]
    cutoff = keys.copy()
    cutoff["time"] -= MONTH_MS
    end = np.searchsorted(keys, keys, side="left")
    start = np.searchsorted(keys, cutoff, side="left")
    counts = np.empty(len(times), dtype=np.float32)
    counts[order] = end - start
    if not include_age:
        return counts

    ages = np.full(len(times), np.nan, dtype=np.float32)
    possible = np.flatnonzero(end > 0)
    previous = end[possible] - 1
    same_group = keys["group"][possible] == keys["group"][previous]
    rows, previous = possible[same_group], previous[same_group]
    ages[order[rows]] = (keys["time"][rows] - keys["time"][previous]) / DAY_MS
    return counts, ages


def _divide(numerator, denominator):
    ratio = np.full(len(numerator), np.nan, dtype=np.float32)
    np.divide(numerator, denominator, where=denominator > 0, out=ratio)
    return ratio


def build_geo_features(operations: pa.Table, risk_context: pa.Table) -> pa.Table:
    """Сопоставить город и зону очной покупки с предшествующими покупками клиента."""
    ids = operations[ID]
    if not ids.equals(risk_context[ID]):
        raise ValueError("Наблюдения географии должны соответствовать операциям в исходном порядке.")
    zone_values = risk_context["terminal_zone"]
    if zone_values.type != pa.int16() or _boolean(pc.less(zone_values, 0)).any():
        raise ValueError("Коды зон должны быть неотрицательными int16 или пропуском.")
    eligible = (
        _boolean(pc.equal(operations["operation_type"], "purchase"))
        & _boolean(pc.equal(operations["direction"], "debit"))
        & _boolean(pc.is_in(operations["channel"], value_set=pa.array(["pos", "qr"])))
        & _boolean(pc.not_equal(operations["merchant_city"], ""))
    )
    zones = pc.fill_null(zone_values, -1).to_numpy()
    if np.any((zones >= 0) & ~eligible):
        raise ValueError("Зона терминала доступна только для очной покупки с известным городом.")

    values = np.full((len(GEO_COLUMNS), len(ids)), np.nan, dtype=np.float32)
    selected = np.flatnonzero(eligible)
    if len(selected):
        times = _timestamps(operations)[selected]
        _, clients = np.unique(operations["client_id"].to_numpy()[selected], return_inverse=True)
        cities, city_count = encode(operations["merchant_city"].take(pa.array(selected)))
        client_city = clients * city_count + cities
        count = _last_month(times, clients)
        city_count = _last_month(times, client_city)
        values[0, selected] = count
        values[1, selected] = city_count
        values[2, selected] = _divide(city_count, count)

        scoped_zones = zones[selected]
        known = scoped_zones >= 0
        zone_count, age = _last_month(
            times[known], client_city[known] * (np.iinfo(np.int16).max + 1) + scoped_zones[known],
            include_age=True,
        )
        geo_city_count = _last_month(times[known], client_city[known])
        values[3, selected[known]] = zone_count
        values[4, selected[known]] = _divide(zone_count, geo_city_count)
        values[5, selected[known]] = age

    return pa.table({ID: ids, **dict(zip(GEO_COLUMNS, values, strict=True))})
