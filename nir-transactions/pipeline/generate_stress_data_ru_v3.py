"""Создать размеченные синтетические операции в рублях для экспериментов НИР."""
from __future__ import annotations
from bisect import bisect_right
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .ru_transaction_catalog import BANK_IDS, BANK_WEIGHTS, CITIES, CITY_UTC_OFFSETS, CITY_WEIGHTS, build_catalog
from .validate_stress_data_ru_v3 import LABEL_SCHEMA, TX_SCHEMA, validate

DAY_MS = 86_400_000

MOSCOW = timezone(timedelta(hours=3))

PHYSICAL_CHANNELS = {"pos", "qr"}

PURCHASE_CHANNELS = ("pos", "ecommerce", "qr")

HOUR_WEIGHTS = (1, 1, 1, 1, 1, 1, 3, 6, 8, 8, 9, 10, 12, 11, 10, 9, 10, 13, 16, 17, 13, 8, 5, 2)

SCENARIOS = {
    "routine_purchase": {"label": 0, "description": "Обычная покупка клиента."},
    "travel_purchase": {"label": 0, "description": "Покупка клиента во время поездки."},
    "large_purchase": {"label": 0, "description": "Крупная законная покупка."},
    "routine_transfer": {"label": 0, "description": "Перевод знакомому получателю или от знакомого отправителя."},
    "new_recipient_transfer": {"label": 0, "description": "Законный исходящий перевод новому получателю."},
    "own_transfer": {"label": 0, "description": "Перевод между счетами клиента в разных банках."},
    "legitimate_refund": {"label": 0, "description": "Полный или частичный возврат по законной покупке."},
    "unauthorized_purchase": {"label": 1, "description": "Покупка, инициированная виртуальным злоумышленником."},
    "unauthorized_transfer": {"label": 1, "description": "Исходящий перевод, инициированный виртуальным злоумышленником."},
}


@dataclass(frozen=True)
class Config:
    clients: int = 2000
    seed: int = 20261112
    start_date: str = "2025-01-01"
    warmup_days: int = 31
    period_days: int = 30
    max_rows: int = 10_000_000
    daily_attack_probability: float = 0.0015
    travel_probability: float = 0.14
    refund_probability: float = 0.02


def validated_start(config: Config) -> datetime:
    for name, low, high in (("clients", 1, 100_000), ("seed", 0, 1_000_000_000),
                            ("warmup_days", 1, 366), ("period_days", 1, 366),
                            ("max_rows", 1, 50_000_000)):
        value = getattr(config, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name}: требуется целое число от {low} до {high}.")
    for name in ("daily_attack_probability", "travel_probability", "refund_probability"):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name}: требуется значение от 0 до 1.")
    start = date.fromisoformat(config.start_date)
    if not 2020 <= start.year <= 2090:
        raise ValueError("Для сценарного календаря выбери дату с 2020 по 2090 год.")
    return datetime.combine(start, time(), MOSCOW).astimezone(timezone.utc)


class WeightedChoice:
    def __init__(self, items, weights):
        self.items = tuple(items)
        self.cumulative = np.cumsum(weights).tolist()

    def choose(self, rng):
        return self.items[bisect_right(self.cumulative, rng.random() * self.cumulative[-1])]


def build_trip_plan(home, config, rng, city_choice):
    total_days = config.warmup_days + 4 * config.period_days
    route = [home] * total_days
    trips = []
    if config.period_days < 4:
        return route, trips
    for period in range(4):
        if rng.random() >= config.travel_probability:
            continue
        stay = int(rng.integers(2, min(5, config.period_days - 2) + 1))
        first = config.warmup_days + period * config.period_days
        departure = first + int(rng.integers(config.period_days - stay - 1))
        arrival, leave, returned = departure + 1, departure + 1 + stay, departure + 2 + stay
        destination = city_choice.choose(rng)
        while destination == home:
            destination = city_choice.choose(rng)
        route[departure:returned] = [None] + [destination] * stay + [None]
        trips.append({"departure_day": departure, "arrival_day": arrival, "leave_day": leave,
                      "return_day": returned, "city": destination[0], "region": destination[1]})
    return route, trips


class ClientSimulation:
    def __init__(self, client_id, config, rng, city_choice, bank_choice, merchants, online_merchants):
        self.client_id = client_id
        self.config = config
        self.rng = rng
        self.bank_choice = bank_choice
        self.merchants = merchants
        self.online_merchants = online_merchants
        self.home = city_choice.choose(rng)
        self.attacker = self.home if rng.random() < 0.65 else city_choice.choose(rng)
        self.route, self.trips = build_trip_plan(self.home, config, rng, city_choice)
        self.primary = bank_choice.choose(rng)
        self.secondary = self.other_bank()
        self.account = f"RU-A-{client_id}-01"
        self.other_account = f"RU-A-{client_id}-02"
        self.cards = (f"RU-C-{client_id}-01", f"RU-C-{client_id}-02")
        self.contacts = []
        self.next_contact = 0
        for _ in range(int(rng.integers(2, 7))):
            self.contacts.append(self.new_contact())
        self.contact_weights = 1 / np.arange(1, len(self.contacts) + 1) ** 0.8
        self.amount_factor = float(np.clip(rng.lognormal(0, 0.48), 0.25, 4))
        self.purchase_rate = float(np.clip(rng.lognormal(0.68, 0.62), 0.18, 11))
        self.transfer_rate = float(np.clip(rng.lognormal(-1.14, 0.82), 0.02, 3.2))
        self.channel_choice = WeightedChoice(PURCHASE_CHANNELS, rng.dirichlet([12, 4, 3]))
        self.night_shift = int(rng.choice((0, 1, 2, 4), p=(0.63, 0.22, 0.12, 0.03)))
        self.favourites = {city: tuple(merchants[city].choose(rng) for _ in range(7))
                           for city in sorted({self.home, *(city for city in self.route if city)})}
        self.online_favourites = tuple(online_merchants[self.home].choose(rng) for _ in range(5))
        self.events = []

    def other_bank(self):
        bank = self.bank_choice.choose(self.rng)
        while bank == self.primary:
            bank = self.bank_choice.choose(self.rng)
        return bank

    def new_contact(self):
        self.next_contact += 1
        return f"RU-P-{self.client_id}-{self.next_contact}", self.other_bank()

    def event_time(self, day, city):
        # The period/day axis is Moscow time. Rotate the activity hours by
        # local offset so operations in the east do not use Moscow evenings.
        offset = CITY_UTC_OFFSETS[city] - 3
        hour = (HOURS.choose(self.rng) + self.night_shift - offset) % 24
        return day * DAY_MS + hour * 3_600_000 + int(self.rng.integers(3_600_000))

    def add(self, timestamp_ms, amount, scenario, episode=0, **fields):
        event = {"timestamp": timestamp_ms, "amount_kopecks": amount, "client_id": self.client_id,
                 "currency": "RUB", "account_id": self.account, **fields}
        self.events.append((event, scenario, episode))
        return event

    def purchase_merchant(self, channel, city, episode):
        if channel == "ecommerce":
            pool = self.online_merchants[self.home]
            favourite = self.online_favourites
        else:
            pool = self.merchants[city]
            favourite = self.favourites.get(city)
        if favourite and self.rng.random() < 0.84:
            return favourite[int(self.rng.integers(len(favourite)))]
        return pool.choose(self.rng)

    def purchase_amount(self, merchant, channel, scenario, episode):
        scale = 5 if scenario == "large_purchase" else 1
        return self.amount(merchant.median_rub, 0.72, scale)

    def purchase(self, day, timestamp, channel, city, scenario, episode):
        if channel in PHYSICAL_CHANNELS and city is None:
            channel = "ecommerce"
        city = city or self.home
        merchant = self.purchase_merchant(channel, city, episode)
        physical = channel in PHYSICAL_CHANNELS
        pos = channel == "pos"
        card = channel != "qr"
        amount = self.purchase_amount(merchant, channel, scenario, episode)
        purchase = self.add(timestamp, amount, scenario, episode, operation_type="purchase",
                            rail="card" if card else "sbp", channel=channel, direction="debit",
                            card_id=self.cards[int(self.rng.random() < 0.14)] if card else None,
                            merchant_id=merchant.merchant_id, mcc=merchant.mcc,
                            merchant_city=city[0] if physical else None,
                            merchant_region=city[1] if physical else None,
                            terminal_id=merchant.terminal_id if pos else None,
                            entry_mode=("contactless" if self.rng.random() < 0.87 else "chip") if pos else None)
        if episode == 0 and self.rng.random() < self.config.refund_probability:
            refund_day = day + int(self.rng.integers(1, 11))
            if refund_day < len(self.route):
                returned = amount if self.rng.random() < 0.8 else max(1, int(amount * self.rng.uniform(0.15, 0.90)))
                original = {name: purchase[name] for name in
                            ("rail", "card_id", "merchant_id", "merchant_city", "merchant_region", "mcc")}
                self.add(self.event_time(refund_day, self.home), returned, "legitimate_refund",
                         operation_type="refund", channel="refund", direction="credit", _original=purchase, **original)

    def amount(self, median, sigma, scale=1):
        rubles = median * self.amount_factor * scale * self.rng.lognormal(0, sigma)
        return int(round(min(max(rubles, 1), 300_000) * 100))

    def transfer(self, timestamp, episode):
        choice = self.rng.random()
        if not episode and choice < 0.24:
            scenario, contact, other, own = "own_transfer", self.other_account, self.secondary, True
            direction, median = ("credit" if self.rng.random() < 0.33 else "debit"), 6100
        else:
            own = False
            new = self.rng.random() < (0.57 if episode else 0.12)
            if new:
                contact, other = self.new_contact()
                if not episode and len(self.contacts) < 24 and self.rng.random() < 0.50:
                    self.contacts.append((contact, other))
                    self.contact_weights = 1 / np.arange(1, len(self.contacts) + 1) ** 0.8
            else:
                probabilities = self.contact_weights / self.contact_weights.sum()
                contact, other = self.contacts[int(self.rng.choice(len(self.contacts), p=probabilities))]
            scenario = ("unauthorized_transfer" if episode else
                        "new_recipient_transfer" if new else "routine_transfer")
            direction = "credit" if not (episode or new) and self.rng.random() < 0.23 else "debit"
            median = 2050 if episode else 1900
        sender, recipient = (self.primary, other) if direction == "debit" else (other, self.primary)
        self.add(timestamp, self.amount(median, 0.95), scenario, episode, operation_type="transfer",
                 rail="sbp", channel="mobile_app", direction=direction, sender_bank_id=sender,
                 recipient_bank_id=recipient, counterparty_id=contact, is_self_transfer=own)

    def simulate(self, start_day, episode_count):
        rng = self.rng
        for day, city in enumerate(self.route):
            weekend = (start_day + day) % 7 >= 5
            schedule = []
            for _ in range(rng.poisson(self.purchase_rate * (1.11 if weekend else 0.96))):
                channel = self.channel_choice.choose(rng)
                if rng.random() < 0.012:
                    scenario = "large_purchase"
                elif channel in PHYSICAL_CHANNELS and city is not None and city != self.home:
                    scenario = "travel_purchase"
                else:
                    scenario = "routine_purchase"
                schedule.append((self.event_time(day, city or self.home), channel, city, scenario, 0))
            for _ in range(rng.poisson(self.transfer_rate)):
                schedule.append((self.event_time(day, city or self.home), "mobile_app", city, None, 0))
            if day >= self.config.warmup_days and rng.random() < self.config.daily_attack_probability:
                episode_count += 1
                channel = ATTACK_CHANNELS.choose(rng)
                target_city = (city or self.attacker) if rng.random() < 0.55 else self.attacker
                delay = int(rng.integers(60_000, 260_000))
                size = min(5, 1 + rng.poisson(1.4))
                start = min(self.event_time(day, target_city), (day + 1) * DAY_MS - 1 - (size - 1) * delay)
                for index in range(size):
                    schedule.append((start + index * delay, channel, target_city, "unauthorized_purchase", episode_count))
            # A contact introduced today can only recur at a later event.
            for when, channel, location, scenario, episode in sorted(schedule, key=lambda event: event[0]):
                if channel == "mobile_app":
                    self.transfer(when, episode)
                else:
                    self.purchase(day, when, channel, location, scenario, episode)
        self.events.sort(key=lambda item: item[0]["timestamp"])
        return episode_count

    def profile(self):
        return {"client_id": self.client_id, "home_city": self.home[0], "home_region": self.home[1],
                "attacker_city": self.attacker[0], "attacker_region": self.attacker[1],
                "primary_bank_id": self.primary, "secondary_bank_id": self.secondary, "trips": self.trips}

HOURS = WeightedChoice(range(24), HOUR_WEIGHTS)

ATTACK_CHANNELS = WeightedChoice((*PURCHASE_CHANNELS, "mobile_app"), (0.20, 0.25, 0.10, 0.45))


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def period_schedule(config, start):
    boundaries = [0, *(config.warmup_days + i * config.period_days for i in range(5))]
    return [{"period": i, "role": "warmup" if i == 0 else "evaluation",
             "start": (start + timedelta(days=left)).isoformat(),
             "end_exclusive": (start + timedelta(days=right)).isoformat()}
            for i, (left, right) in enumerate(zip(boundaries, boundaries[1:]))]


def generate(output_dir: Path, config: Config, *, telemetry=None) -> dict:
    start = validated_start(config)
    start_ms = int(start.timestamp()) * 1000
    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Папка уже существует: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    city_choice = WeightedChoice(CITIES, CITY_WEIGHTS)
    bank_choice = WeightedChoice(BANK_IDS, BANK_WEIGHTS)
    catalog = build_catalog()
    merchants = {city: WeightedChoice(entries, [merchant.weight for merchant in entries])
                 for city, entries in catalog.items()}
    online_catalog = {city: [merchant for merchant in entries if merchant.mcc not in {"4111", "5541"}]
                      for city, entries in catalog.items()}
    online_merchants = {city: WeightedChoice(entries, [merchant.weight for merchant in entries])
                        for city, entries in online_catalog.items()}
    client_base = 3_000_000_000_000_000 + config.seed * 100_000
    tx_base = 3_000_000_000_000_000_000 + config.seed * 50_000_000
    profiles, tx_rows, label_rows = [], [], []
    rows = episodes = 0
    client_type = ClientSimulation if telemetry is None else telemetry.client_type
    with tempfile.TemporaryDirectory(prefix=".stress-ru-v3-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        with (pq.ParquetWriter(stage / "transactions.parquet", TX_SCHEMA, compression="zstd") as transactions,
              pq.ParquetWriter(stage / "labels.parquet", LABEL_SCHEMA, compression="zstd") as labels,
              telemetry.open(stage) if telemetry is not None else nullcontext() as journal):
            for index in range(config.clients):
                client = client_type(client_base + index, config, rng, city_choice, bank_choice,
                                     merchants, online_merchants)
                episodes = client.simulate(start.astimezone(MOSCOW).weekday(), episodes)
                profiles.append(client.profile())
                if rows + len(client.events) > config.max_rows:
                    raise ValueError("Превышен max-rows. Уменьши число клиентов или длительность.")
                for event, scenario, episode in client.events:
                    event["transaction_id"] = tx_base + rows
                    original = event.pop("_original", None)
                    if original is not None:
                        event["original_transaction_id"] = original["transaction_id"]
                    day = event["timestamp"] // DAY_MS
                    event["timestamp"] += start_ms
                    period = 0 if day < config.warmup_days else 1 + (day - config.warmup_days) // config.period_days
                    if journal is not None:
                        journal.record(event, start_ms)
                    tx_rows.append(event)
                    label_rows.append({"transaction_id": event["transaction_id"], "period": period,
                                       "is_fraud": SCENARIOS[scenario]["label"], "scenario": scenario,
                                       "episode_id": episode})
                    rows += 1
                if len(tx_rows) >= 25_000 or index + 1 == config.clients:
                    if tx_rows:
                        transactions.write_table(pa.Table.from_pylist(tx_rows, schema=TX_SCHEMA))
                        labels.write_table(pa.Table.from_pylist(label_rows, schema=LABEL_SCHEMA))
                        if journal is not None:
                            journal.flush()
                        tx_rows.clear()
                        label_rows.clear()
                if (index + 1) % 250 == 0 or index + 1 == config.clients:
                    print(f"Клиентов: {index + 1:,}/{config.clients:,}. Операций: {rows:,}.", flush=True)
        if not rows:
            raise ValueError("Не сгенерировано ни одной операции. Увеличь число клиентов или длительность.")
        write_json(stage / "client_profiles.json", profiles)
        code = [Path(__file__), Path(__file__).with_name("ru_transaction_catalog.py"),
                Path(__file__).with_name("validate_stress_data_ru_v3.py")]
        if telemetry is not None:
            code.extend(telemetry.source_files)
        for source in code:
            shutil.copyfile(source, stage / source.name)
        manifest = {
            "format_version": 1, "generator_version": 3, "dataset_kind": "synthetic_ru_retail_payments",
            "synthetic": True, "empirically_calibrated": False, "currency": "RUB", "amount_unit": "kopeck",
            "config": asdict(config), "periods": period_schedule(config, start),
            "rows": rows, "clients": config.clients, "attack_episodes": episodes,
            "row_order": "client_id,timestamp,transaction_id", "timestamp_timezone": "UTC",
            "period_timezone": "Europe/Moscow", "scenario_definitions": SCENARIOS,
            "ground_truth_rule": "Label 1 for transactions initiated by the simulated attacker, 0 for legitimate transactions.",
            "model_used_for_generation": False, "labels_are_features": False,
            "limitations": ["Scenario assumptions, not fitted to private Russian bank data.",
                            "One primary account perspective per client; external counterparties are not duplicated in the ledger.",
                            "Completed events only; no balance, authorisation response or customer confirmation model."],
            "versions": {"numpy": np.__version__, "pyarrow": pa.__version__},
            "files": {path.name: file_hash(path) for path in sorted(stage.iterdir())},
        }
        if journal is not None:
            manifest["session_telemetry"] = journal.metadata()
        print("Проверяю типы операций, связи возвратов, периоды и метки…", flush=True)
        audit = validate(stage, manifest=manifest)
        write_json(stage / "quality_checks.json", audit)
        manifest["files"]["quality_checks.json"] = file_hash(stage / "quality_checks.json")
        write_json(stage / "manifest.json", manifest)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Папка уже существует: {output_dir}")
        stage.rename(output_dir)
    print(f"Российский синтетический набор сохранён: {output_dir}", flush=True)
    print(f"Операций: {rows:,}. Мошеннических: {audit['fraud_rows']:,}.", flush=True)
    return manifest
