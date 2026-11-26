"""Получить признаки интернет-сессии из событий до решения по покупке."""
from __future__ import annotations
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ID = "transaction_id"

TIMESTAMP = pa.timestamp("ms", tz="UTC")

DAY_MS = 86_400_000

TELEMETRY_SCHEMA = pa.schema([
    pa.field(ID, pa.int64(), nullable=False),
    pa.field("observed_at", TIMESTAMP, nullable=False),
    pa.field("device_ref", pa.int64(), nullable=False),
    pa.field("session_started_at", TIMESTAMP, nullable=False),
    pa.field("initial_device_ref", pa.int64(), nullable=False),
    pa.field("failed_auth_times", pa.list_(TIMESTAMP), nullable=False),
])

FEATURE_COLUMNS = (
    "session_telemetry_observed",
    "session_device_seen_before",
    "session_device_age_days",
    "session_device_changed",
    "session_elapsed_seconds",
    "session_failed_auth_count",
    "session_seconds_since_last_failed_auth",
)

OPERATION_TYPES = {
    ID: pa.int64(), "client_id": pa.int64(), "timestamp": TIMESTAMP,
    "operation_type": pa.string(), "channel": pa.string(), "direction": pa.string(),
}


def _ordered_rows(operations, telemetry):
    for name, expected in OPERATION_TYPES.items():
        field = operations[name]
        if field.type != expected or field.null_count:
            raise ValueError(f"Некорректный тип или пропуск в поле операции {name}.")
    if telemetry.schema != TELEMETRY_SCHEMA or any(column.null_count for column in telemetry.columns):
        raise ValueError("Схема событий сессии не соответствует формату наблюдений.")

    ids = operations[ID].to_numpy()
    clients = operations["client_id"].to_numpy()
    times = pc.cast(operations["timestamp"], pa.int64()).to_numpy()
    if (np.any(ids[1:] <= ids[:-1]) or np.any(clients[1:] < clients[:-1])
            or np.any((clients[1:] == clients[:-1]) & (times[1:] < times[:-1]))):
        raise ValueError("Покупки должны идти последовательно в полных историях клиентов.")

    observed_ids = telemetry[ID].to_numpy()
    rows = np.searchsorted(ids, observed_ids)
    if (np.any(observed_ids[1:] <= observed_ids[:-1]) or np.any(rows >= len(ids))
            or np.any(ids[rows] != observed_ids)):
        raise ValueError("Каждое наблюдение должно соответствовать одной покупке из той же порции.")

    eligible = (pc.equal(operations["operation_type"], "purchase").to_numpy()
                & pc.equal(operations["channel"], "ecommerce").to_numpy()
                & pc.equal(operations["direction"], "debit").to_numpy())
    if not np.all(eligible[rows]):
        raise ValueError("Наблюдения требуются только для исходящих интернет-покупок.")
    return ids, clients[rows], times[rows], rows, eligible


def _failed_auth(attempts, started, observed):
    offsets = attempts.offsets.to_numpy()
    counts = np.diff(offsets)
    roots = np.repeat(np.arange(len(counts)), counts)
    values = attempts.values.slice(int(offsets[0]), int(offsets[-1] - offsets[0]))
    if values.null_count:
        raise ValueError("Время неудачных подтверждений должно быть заполнено.")
    times = pc.cast(values, pa.int64()).to_numpy()
    if (np.any((times <= started[roots]) | (times > observed[roots]))
            or np.any((roots[1:] == roots[:-1]) & (times[1:] <= times[:-1]))):
        raise ValueError("Время попыток должно возрастать после начала сессии и до её наблюдения.")
    known = counts > 0
    last = times[offsets[1:][known] - offsets[0] - 1]
    return counts, known, last


def _device_age_days(clients, times, devices):
    if not len(times):
        return np.empty(0, dtype=np.float32)
    order = np.lexsort((times, devices, clients))
    ordered_clients = clients[order]
    ordered_devices = devices[order]
    ordered_times = times[order]
    new = np.r_[True, (ordered_clients[1:] != ordered_clients[:-1])
                | (ordered_devices[1:] != ordered_devices[:-1])]
    first = np.maximum.accumulate(np.where(new, np.arange(len(order)), 0))
    elapsed = ordered_times - ordered_times[first]
    values = np.full(len(times), np.nan, dtype=np.float32)
    seen = elapsed > 0
    values[order[seen]] = elapsed[seen] / DAY_MS
    return values


def build_features(operations: pa.Table, telemetry: pa.Table) -> pa.Table:
    """Рассчитать признаки на полных историях клиентов без доступа к меткам.

    Прежнее устройство учитывается по предыдущим завершённым покупкам
    клиента со строго меньшим временем. Пропуск телеметрии оставляет
    признаки сессии пустыми, а признак её наличия равным нулю.
    """
    ids, clients, times, rows, eligible = _ordered_rows(operations, telemetry)
    observed = pc.cast(telemetry["observed_at"], pa.int64()).to_numpy()
    started = pc.cast(telemetry["session_started_at"], pa.int64()).to_numpy()
    devices = telemetry["device_ref"].to_numpy()
    initial = telemetry["initial_device_ref"].to_numpy()
    if (np.any(started > observed) or np.any(observed >= times)
            or np.any(devices <= 0) or np.any(initial <= 0)):
        raise ValueError("Сессия и её наблюдение должны предшествовать покупке, устройство должно быть указано.")

    counts, with_attempts, last = _failed_auth(telemetry["failed_auth_times"].combine_chunks(), started, observed)
    age = _device_age_days(clients, times, devices)
    values = np.full((len(FEATURE_COLUMNS), len(ids)), np.nan, dtype=np.float32)
    values[0, eligible] = 0
    values[0, rows] = 1
    values[1, rows] = np.isfinite(age)
    values[2, rows] = age
    values[3, rows] = devices != initial
    values[4, rows] = (times - started) / 1_000
    values[5, rows] = counts
    values[6, rows[with_attempts]] = (times[with_attempts] - last) / 1_000
    return pa.table({ID: ids, **dict(zip(FEATURE_COLUMNS, values, strict=True))})
