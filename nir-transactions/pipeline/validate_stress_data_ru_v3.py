"""Проверка структуры и согласованности синтетических операций РФ v3."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TX_SCHEMA = pa.schema([
    pa.field("transaction_id", pa.int64(), nullable=False),
    pa.field("client_id", pa.int64(), nullable=False),
    pa.field("timestamp", pa.timestamp("ms", tz="UTC"), nullable=False),
    pa.field("amount_kopecks", pa.int64(), nullable=False),
    *[pa.field(name, pa.string(), nullable=False) for name in (
        "currency", "operation_type", "rail", "channel", "direction", "account_id",
    )],
    *[pa.field(name, pa.string()) for name in (
        "card_id", "merchant_id", "terminal_id", "merchant_city", "merchant_region", "mcc", "entry_mode",
        "sender_bank_id", "recipient_bank_id", "counterparty_id",
    )],
    pa.field("is_self_transfer", pa.bool_()),
    pa.field("original_transaction_id", pa.int64()),
])

LABEL_SCHEMA = pa.schema([
    pa.field("transaction_id", pa.int64(), nullable=False),
    pa.field("is_fraud", pa.int8(), nullable=False),
    pa.field("scenario", pa.string(), nullable=False),
    pa.field("episode_id", pa.int64(), nullable=False),
    pa.field("period", pa.int8(), nullable=False),
])

CHANNELS = ("pos", "ecommerce", "qr", "mobile_app", "refund")

REFUND_MATCH_FIELDS = ("rail", "account_id", "card_id", "merchant_id", "merchant_city", "merchant_region", "mcc")

DAY_MS = 86_400_000

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_files(data_dir, manifest, verify_hashes):
    files = manifest.get("files", {})
    _require(isinstance(files, dict) and {"transactions.parquet", "labels.parquet", "client_profiles.json"} <= files.keys(),
             "В манифесте отсутствуют файлы транзакций, меток или профилей.")
    for name, checksum in files.items():
        path = (data_dir / name).resolve()
        _require(path.is_relative_to(data_dir) and path.is_file() and isinstance(checksum, str) and len(checksum) == 64,
                 f"Некорректная запись о файле: {name}.")
        if verify_hashes:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            _require(actual == checksum, f"Контрольная сумма файла изменилась: {name}.")


def _period_edges(manifest):
    periods = manifest.get("periods")
    _require(isinstance(periods, list) and len(periods) == 5, "Ожидалась начальная история и четыре периода.")
    edges = []
    for number, part in enumerate(periods):
        _require(isinstance(part, dict) and part.get("period") == number, "Нарушен порядок периодов.")
        role = part.get("role")
        _require(role == "warmup" if number == 0 else role in ("evaluation", "test"), "Некорректная роль периода.")
        try:
            first = datetime.fromisoformat(part["start"])
            last = datetime.fromisoformat(part["end_exclusive"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Некорректная дата начала или окончания периода.") from exc
        _require(first.utcoffset() == last.utcoffset() == timedelta(0) and first < last,
                 "Границы периода должны возрастать и быть заданы в UTC.")
        _require((last - first) % timedelta(days=1) == timedelta(0), "Период должен содержать целое число суток.")
        left = (first - EPOCH) // timedelta(milliseconds=1)
        right = (last - EPOCH) // timedelta(milliseconds=1)
        _require(not edges or left == edges[-1], "Периоды должны следовать подряд без разрывов.")
        if not edges:
            edges.append(left)
        edges.append(right)
    return np.asarray(edges, dtype=np.int64)


def _profiles(data_dir, total_days):
    profiles = _json(data_dir / "client_profiles.json")
    _require(isinstance(profiles, list) and profiles, "Ожидался список профилей клиентов.")
    result = {}
    for profile in profiles:
        _require(isinstance(profile, dict), "Профиль клиента должен быть объектом.")
        client = profile.get("client_id")
        home = (profile.get("home_city"), profile.get("home_region"))
        attacker = (profile.get("attacker_city"), profile.get("attacker_region"))
        banks = (profile.get("primary_bank_id"), profile.get("secondary_bank_id"))
        _require(type(client) is int and client >= 0 and client not in result, "Некорректный ID в профиле клиента.")
        _require(all(isinstance(field, str) and field for field in (*home, *attacker, *banks)) and banks[0] != banks[1],
                 f"У клиента {client} некорректные места или банки.")
        trip_days = {}
        previous_return = 0
        trips = profile.get("trips")
        _require(isinstance(trips, list), f"У клиента {client} отсутствует календарь поездок.")
        for trip in trips:
            _require(isinstance(trip, dict), f"У клиента {client} некорректное описание поездки.")
            depart, arrive, leave, returned = (trip.get(name) for name in (
                "departure_day", "arrival_day", "leave_day", "return_day",
            ))
            destination = (trip.get("city"), trip.get("region"))
            _require(all(type(day) is int for day in (depart, arrive, leave, returned))
                     and previous_return <= depart < arrive < leave < returned <= total_days,
                     f"У клиента {client} некорректный календарь поездок.")
            _require(destination != home and all(isinstance(field, str) and field for field in destination),
                     f"У клиента {client} некорректное место поездки.")
            trip_days.update({day: None for day in range(depart, arrive)})
            trip_days.update({day: destination for day in range(arrive, leave)})
            trip_days.update({day: None for day in range(leave, returned)})
            previous_return = returned
        result[client] = (home, attacker, banks, trip_days)
    return result


def _aligned_batches(transactions, labels, batch_size):
    """Сопоставить потоки даже при разных границах групп строк Parquet."""
    raw_iter = transactions.iter_batches(batch_size=batch_size, use_threads=False)
    label_iter = labels.iter_batches(batch_size=batch_size, use_threads=False)
    raw, truth = next(raw_iter, None), next(label_iter, None)
    while raw is not None and truth is not None:
        count = min(raw.num_rows, truth.num_rows)
        yield raw.slice(0, count), truth.slice(0, count)
        raw = next(raw_iter, None) if count == raw.num_rows else raw.slice(count)
        truth = next(label_iter, None) if count == truth.num_rows else truth.slice(count)
    _require(raw is None and truth is None, "Число транзакций и меток не совпадает.")


def _arrays(batch):
    for index, field in enumerate(batch.schema):
        _require(field.nullable or batch.column(index).null_count == 0, f"Пропуск обязательного поля {field.name}.")
    return {field.name: batch.column(index).to_numpy(zero_copy_only=False)
            for index, field in enumerate(batch.schema) if field.name != "original_transaction_id"}


def _validate_channels(raw, links):
    channel = raw["channel"]
    pos, online, qr, transfer, refund = [channel == value for value in CHANNELS]
    purchase = pos | online | qr
    card = raw["rail"] == "card"
    debit = raw["direction"] == "debit"
    _require(np.isin(channel, CHANNELS).all(), "Обнаружен неизвестный канал операции.")
    _require(np.all(raw["operation_type"] == np.select([purchase, transfer], ["purchase", "transfer"], default="refund")),
             "Тип операции не соответствует её каналу.")
    _require(np.isin(raw["rail"], ("card", "sbp")).all() and np.all(card[pos | online]) and not card[qr | transfer].any(),
             "Платёжная система не соответствует каналу операции.")
    _require(np.isin(raw["direction"], ("debit", "credit")).all() and debit[purchase].all() and not debit[refund].any(),
             "Некорректное направление списания или зачисления.")

    for name, expected in (
        ("card_id", card), ("merchant_id", purchase | refund), ("mcc", purchase | refund),
        ("terminal_id", pos), ("entry_mode", pos), ("sender_bank_id", transfer),
        ("recipient_bank_id", transfer), ("counterparty_id", transfer), ("is_self_transfer", transfer),
    ):
        _require(np.array_equal(np.not_equal(raw[name], None), expected),
                 f"Поле {name} заполнено для неподходящего типа операции.")
    city_present = np.not_equal(raw["merchant_city"], None)
    _require(np.array_equal(city_present, np.not_equal(raw["merchant_region"], None)) and city_present[pos | qr].all()
             and not city_present[online | transfer].any(), "Некорректное заполнение географии торговой точки.")
    _require(np.isin(raw["entry_mode"][pos], ("chip", "contactless")).all(), "Неизвестный режим предъявления карты.")
    _require(np.all(raw["sender_bank_id"][transfer] != raw["recipient_bank_id"][transfer]),
             "Межбанковский перевод связывает один и тот же банк.")
    _require(np.array_equal(np.asarray([item is not None for item in links]), refund),
             "Ссылка на исходную операцию должна быть только у возврата.")


def _validate_client(raw, truth, links, start, stop, profile, first_ms, purchases, returned, terminals):
    home, attacker, banks, trip_days = profile
    primary, secondary = banks
    channel = raw["channel"][start:stop]
    transfer_positions = start + np.flatnonzero(channel == "mobile_app")
    if len(transfer_positions):
        debit = raw["direction"][transfer_positions] == "debit"
        sender = raw["sender_bank_id"][transfer_positions]
        recipient = raw["recipient_bank_id"][transfer_positions]
        same_owner = np.asarray(raw["is_self_transfer"][transfer_positions], dtype=bool)
        _require(np.all(np.where(debit, sender, recipient) == primary)
                 and np.all(np.where(debit, recipient, sender)[same_owner] == secondary),
                 "Банки перевода не согласованы с профилем клиента.")

    for index in start + np.flatnonzero((channel == "pos") | (channel == "qr")):
        day = int((raw["timestamp"][index] - first_ms) // DAY_MS)
        planned = trip_days.get(day, home)
        observed = (raw["merchant_city"][index], raw["merchant_region"][index])
        allowed = (home, planned, attacker) if truth["is_fraud"][index] else (planned,)
        _require(observed in allowed, "География физической покупки не согласована с профилем и поездками.")
        terminal = raw["terminal_id"][index]
        if terminal is not None:
            identity = (raw["merchant_id"][index], raw["mcc"][index], *observed)
            _require(terminals.setdefault(terminal, identity) == identity,
                     "Один терминал связан с несколькими торговыми точками или MCC.")

    for index in range(start, stop):
        operation = raw["operation_type"][index]
        if operation == "transfer" or truth["is_fraud"][index]:
            _require(operation != "refund", "Возврат размечен как мошенничество.")
            continue
        details = tuple(raw[name][index] for name in REFUND_MATCH_FIELDS)
        timestamp = raw["timestamp"][index]
        amount = raw["amount_kopecks"][index]
        if operation == "purchase":
            purchases[int(raw["transaction_id"][index])] = (timestamp, amount, details)
            continue
        original = links[index]
        previous = purchases.get(original)
        _require(previous is not None and original not in returned,
                 "Возврат ссылается на отсутствующую, мошенническую или уже возвращённую покупку.")
        previous_time, previous_amount, previous_details = previous
        _require(timestamp > previous_time and amount <= previous_amount and details == previous_details,
                 "Возврат не согласован с исходной покупкой по времени, сумме или реквизитам.")
        returned.add(original)


def validate(data_dir, *, manifest=None, verify_hashes=True, batch_size=100_000):
    """Проверить файлы без изменений и вернуть сводку для манифеста."""
    data_dir = Path(data_dir).expanduser().resolve()
    if manifest is None:
        manifest = _json(data_dir / "manifest.json")
    _require(manifest.get("generator_version") == 3, "Ожидался российский набор версии 3.")
    _require(type(batch_size) is int and batch_size > 0, "batch-size должен быть положительным целым числом.")
    _verify_files(data_dir, manifest, verify_hashes)
    edges = _period_edges(manifest)
    profiles = _profiles(data_dir, int((edges[-1] - edges[0]) // DAY_MS))
    scenarios = manifest.get("scenario_definitions")
    _require(isinstance(scenarios, dict) and scenarios, "Отсутствует описание сценариев.")
    scenario_labels = {name: details.get("label") for name, details in scenarios.items()}
    _require(all(type(label) is int and label in (0, 1) for label in scenario_labels.values()),
             "Метки сценариев должны быть равны 0 или 1.")
    transactions = pq.ParquetFile(data_dir / "transactions.parquet")
    labels = pq.ParquetFile(data_dir / "labels.parquet")
    _require(transactions.schema_arrow.equals(TX_SCHEMA) and labels.schema_arrow.equals(LABEL_SCHEMA),
             "Схема Parquet не соответствует формату v3.")
    _require(transactions.metadata.num_rows == labels.metadata.num_rows > 0,
             "Набор пуст или число строк транзакций и меток отличается.")

    rows = fraud_rows = refunds = 0
    by_period = {str(period): {"rows": 0, "fraud_rows": 0} for period in range(5)}
    by_channel = {name: {"rows": 0, "fraud_rows": 0} for name in CHANNELS}
    by_scenario = Counter()
    observed_clients = set()
    episodes = {}
    terminals = {}
    previous_id = previous_client = previous_time = None
    purchases, returned = {}, set()

    for raw_batch, label_batch in _aligned_batches(transactions, labels, batch_size):
        raw, truth = _arrays(raw_batch), _arrays(label_batch)
        links = raw_batch.column(raw_batch.schema.get_field_index("original_transaction_id")).to_pylist()
        ids, clients, fraud = raw["transaction_id"], raw["client_id"], truth["is_fraud"]
        timestamp = raw["timestamp"].astype("datetime64[ms]").astype(np.int64)
        raw["timestamp"] = timestamp
        _require(np.array_equal(ids, truth["transaction_id"]), "ID транзакций и меток не совпадают.")
        _require(ids[0] >= 0 and np.all(np.diff(ids) == 1) and (previous_id is None or ids[0] == previous_id + 1),
                 "ID транзакций должны возрастать последовательно без повторов.")
        _require(np.all(clients[1:] >= clients[:-1])
                 and not np.any((clients[1:] == clients[:-1]) & (timestamp[1:] < timestamp[:-1]))
                 and (previous_client is None or (clients[0], timestamp[0]) >= (previous_client, previous_time)),
                 "Операции должны быть отсортированы по клиенту, времени и идентификатору.")
        previous_id, previous_client, previous_time = int(ids[-1]), int(clients[-1]), int(timestamp[-1])
        _require(np.all(raw["amount_kopecks"] > 0) and np.all(raw["currency"] == "RUB"),
                 "Суммы должны быть положительными целыми в копейках, валюта — RUB.")
        period = np.searchsorted(edges, timestamp, side="right") - 1
        _require(np.all((period >= 0) & (period < 5)) and np.array_equal(period, truth["period"]),
                 "Дата операции не соответствует границам периода.")
        episode = truth["episode_id"]
        _require(np.isin(fraud, (0, 1)).all() and np.all(episode[fraud == 0] == 0)
                 and np.all(episode[fraud == 1] > 0) and not fraud[period == 0].any(),
                 "Неверная метка, ID эпизода или мошенничество в начальной истории.")
        names, counts = np.unique(truth["scenario"], return_counts=True)
        for name, count in zip(names, counts):
            _require(name in scenario_labels and np.all(fraud[truth["scenario"] == name] == scenario_labels[name]),
                     f"Метка не соответствует сценарию {name}.")
            by_scenario[name] += int(count)
        for index in np.flatnonzero(fraud):
            key = int(episode[index])
            client = int(clients[index])
            _require(episodes.setdefault(key, client) == client, "Один эпизод связан с несколькими клиентами.")
        _validate_channels(raw, links)

        starts = np.r_[0, np.flatnonzero(clients[1:] != clients[:-1]) + 1, len(clients)]
        for start, stop in zip(starts[:-1], starts[1:]):
            client = int(clients[start])
            _require(client in profiles, "У транзакции отсутствует профиль клиента.")
            if client not in observed_clients:
                purchases, returned = {}, set()
                observed_clients.add(client)
            _validate_client(raw, truth, links, start, stop, profiles[client], edges[0], purchases, returned, terminals)

        for number in range(5):
            selected = period == number
            by_period[str(number)]["rows"] += int(selected.sum())
            by_period[str(number)]["fraud_rows"] += int(fraud[selected].sum())
        for name in CHANNELS:
            selected = raw["channel"] == name
            by_channel[name]["rows"] += int(selected.sum())
            by_channel[name]["fraud_rows"] += int(fraud[selected].sum())
        rows += len(ids)
        fraud_rows += int(fraud.sum())
        refunds += int((raw["operation_type"] == "refund").sum())

    _require(rows == transactions.metadata.num_rows, "Число проверенных операций не совпадает с Parquet.")
    if "rows" in manifest:
        _require(manifest["rows"] == rows, "Число строк в манифесте отличается от Parquet.")
    if "clients" in manifest:
        _require(manifest["clients"] == len(profiles), "Число профилей не совпадает с манифестом.")
    return {
        "rows": rows,
        "fraud_rows": fraud_rows,
        "clients_with_operations": len(observed_clients),
        "fraud_episodes": len(episodes),
        "refunds": refunds,
        "terminals": len(terminals),
        "counts_by_period": by_period,
        "counts_by_channel": by_channel,
        "counts_by_scenario": dict(sorted(by_scenario.items())),
        "checks": ["schemas", "ids", "chronology", "periods", "fraud_labels", "channel_fields",
                   "bank_pairs", "physical_locations", "terminals", "refund_links"],
    }
