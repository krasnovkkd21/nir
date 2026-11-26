"""Сгенерировать операции РФ с короткими и многодневными эпизодами мошенничества."""
from collections import Counter
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .generate_ru_channel_risk_dataset import ChannelRiskClient, ChannelRiskJournal, ChannelRiskTelemetry, RISK_CONTEXT_SCHEMA, RISK_FILE, WEBSITE_CATALOG_FILE, WEBSITE_REGISTRY
from .generate_ru_session_dataset import TELEMETRY_FILE
from .generate_stress_data_ru_v3 import DAY_MS, HOURS, PHYSICAL_CHANNELS, write_json
from .ru_predecision_session import TELEMETRY_SCHEMA
from .ru_transaction_catalog import CITY_UTC_OFFSETS

EPISODE_FILE = "episodes.parquet"

PARAMETER_FILE = "episode_parameters.json"

EPISODE_SCHEMA = pa.schema([
    pa.field("episode_id", pa.int64(), nullable=False),
    pa.field("client_id", pa.int64(), nullable=False),
    pa.field("kind", pa.string(), nullable=False),
    pa.field("planned_operations", pa.int32(), nullable=False),
    pa.field("observed_operations", pa.int32(), nullable=False),
    pa.field("started_at", pa.timestamp("ms", "UTC"), nullable=False),
    pa.field("planned_ended_at", pa.timestamp("ms", "UTC"), nullable=False),
    pa.field("observed_ended_at", pa.timestamp("ms", "UTC"), nullable=False),
    pa.field("right_censored", pa.bool_(), nullable=False),
    pa.field("channels", pa.list_(pa.string()), nullable=False),
])

DEFAULT_PARAMETERS = {
    "rng_stream": 913_127,
    "patterns": {
        "single": {"probability": 0.30, "operation_count": [1, 1]},
        "burst": {"probability": 0.35, "operation_count": [2, 8],
                  "gap_seconds": [5, 480]},
        "distributed": {"probability": 0.25, "operation_count": [3, 15],
                        "duration_hours": [6, 72], "minimum_gap_seconds": 30},
        "extended": {"probability": 0.10, "operation_count": [8, 40],
                     "duration_hours": [72, 240], "minimum_gap_seconds": 600},
    },
    "channel_modes": {
        "pos": 0.20, "ecommerce": 0.175, "qr": 0.10,
        "mobile_app": 0.375, "account_takeover": 0.15,
    },
    "takeover_channels": ["ecommerce", "mobile_app"],
}


def parameters_from_file(path):
    parameters = json.loads(json.dumps(DEFAULT_PARAMETERS))
    if path is not None:
        supplied = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict) or set(supplied) != set(parameters):
            raise ValueError("JSON параметров должен содержать rng_stream, patterns, channel_modes и takeover_channels.")
        parameters = supplied
    if type(parameters["rng_stream"]) is not int or not 0 <= parameters["rng_stream"] <= 1_000_000_000:
        raise ValueError("rng_stream должен быть неотрицательным целым числом.")
    patterns = parameters["patterns"]
    if not isinstance(patterns, dict) or set(patterns) != set(DEFAULT_PARAMETERS["patterns"]):
        raise ValueError("Требуются четыре сценария: single, burst, distributed, extended.")
    probabilities = []
    for name, pattern in patterns.items():
        if not isinstance(pattern, dict) or set(pattern) != set(DEFAULT_PARAMETERS["patterns"][name]):
            raise ValueError(f"{name}: неизвестные или пропущенные параметры сценария.")
        probability = pattern["probability"]
        if not isinstance(probability, (int, float)) or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"{name}: некорректная вероятность.")
        probabilities.append(probability)
        limits = pattern["operation_count"]
        if (not isinstance(limits, list) or len(limits) != 2
                or any(type(value) is not int for value in limits) or not 1 <= limits[0] <= limits[1] <= 100):
            raise ValueError(f"{name}: число операций должно находиться в пределах 1–100.")
        if name == "single" and limits != [1, 1]:
            raise ValueError("Сценарий single должен содержать одну операцию.")
        if name != "single" and limits[0] < 2:
            raise ValueError(f"{name}: необходимо как минимум две операции.")
        if name == "burst":
            low, high = pattern["gap_seconds"]
            if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in (low, high)) or not 0 < low <= high:
                raise ValueError("Интервалы burst должны быть положительными.")
        if name in ("distributed", "extended"):
            low, high = pattern["duration_hours"]
            minimum = pattern["minimum_gap_seconds"]
            if (not all(isinstance(value, (int, float)) and np.isfinite(value) for value in (low, high, minimum))
                    or not 0 < low <= high <= 24 * 30 or minimum <= 0
                    or low * 3600 < (limits[1] - 1) * minimum):
                raise ValueError(f"{name}: длительность не вмещает минимальные интервалы между операциями.")
    modes = parameters["channel_modes"]
    if not isinstance(modes, dict) or set(modes) != set(DEFAULT_PARAMETERS["channel_modes"]):
        raise ValueError("Некорректный набор режимов каналов.")
    if (not all(isinstance(value, (int, float)) and np.isfinite(value) and 0 <= value <= 1 for value in modes.values())
            or not np.isclose(sum(modes.values()), 1) or not np.isclose(sum(probabilities), 1)):
        raise ValueError("Вероятности сценариев и режимов каналов должны суммироваться в единицу.")
    if parameters["takeover_channels"] != ["ecommerce", "mobile_app"]:
        raise ValueError("Сценарий захвата аккаунта использует интернет-покупки и переводы в приложении.")
    return parameters


class EpisodeClient(ChannelRiskClient):
    def __init__(self, *args, episode_parameters, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_parameters = episode_parameters
        self.episode_rng = np.random.default_rng([
            episode_parameters["rng_stream"], self.config.seed, self.client_id,
        ])
        self.episode_devices = {}
        self.episode_recipients = {}
        self.episode_plans = {}

    def choose_checkout_device(self, episode):
        if episode and episode in self.episode_devices:
            self.active_attack = self.episode_devices[episode]
        result = super().choose_checkout_device(episode)
        if episode:
            self.episode_devices[episode] = self.active_attack
        return result

    def transfer(self, timestamp, episode):
        if episode and episode in self.episode_recipients:
            self.attack_recipient = (episode, self.episode_recipients[episode])
        super().transfer(timestamp, episode)
        if episode:
            event = self.events[-1][0]
            self.episode_recipients[episode] = (event["recipient_bank_id"], event["counterparty_id"])

    def episode_offsets(self, kind, count):
        rng = self.episode_rng
        pattern = self.episode_parameters["patterns"][kind]
        if kind == "single":
            return np.zeros(1, dtype=np.int64)
        if kind == "burst":
            low, high = pattern["gap_seconds"]
            gaps = np.exp(rng.uniform(np.log(low), np.log(high), count - 1))
            return np.r_[0, np.cumsum(np.maximum(1, np.rint(gaps * 1000).astype(np.int64)))]
        low, high = pattern["duration_hours"]
        duration = int(round(rng.uniform(low, high) * 3_600_000))
        minimum = int(round(pattern["minimum_gap_seconds"] * 1000))
        remainder = duration - (count - 1) * minimum
        weights = rng.dirichlet(np.ones(count - 1))
        extra = np.floor(weights * remainder).astype(np.int64)
        extra[-1] += remainder - int(extra.sum())
        return np.r_[0, np.cumsum(extra + minimum)]

    def attack_plan(self, day, episode):
        rng = self.episode_rng
        patterns = self.episode_parameters["patterns"]
        names = tuple(patterns)
        kind = str(rng.choice(names, p=[patterns[name]["probability"] for name in names]))
        low, high = patterns[kind]["operation_count"]
        count = int(rng.integers(low, high + 1))
        modes = self.episode_parameters["channel_modes"]
        mode = str(rng.choice(tuple(modes), p=tuple(modes.values())))
        channels = [mode] * count
        if mode == "account_takeover":
            candidates = self.episode_parameters["takeover_channels"]
            channels = rng.choice(candidates, size=count).tolist()
            if count > 1:
                # A multi-operation takeover may use both permitted capabilities.
                channels[:2] = rng.permutation(candidates).tolist()
        offset = CITY_UTC_OFFSETS[self.attacker] - 3
        hour = (HOURS.choose(rng) + self.night_shift - offset) % 24
        start = day * DAY_MS + hour * 3_600_000 + int(rng.integers(3_600_000))
        times = start + self.episode_offsets(kind, count)
        horizon = len(self.route) * DAY_MS
        observed = times < horizon
        if not observed[0] or np.any(np.diff(times) <= 0):
            raise ValueError("План эпизода должен начинаться в интервале наблюдения и возрастать по времени.")
        observed_channels = [channel for channel, keep in zip(channels, observed, strict=True) if keep]
        self.episode_plans[episode] = {
            "episode_id": episode, "client_id": self.client_id, "kind": kind,
            "planned_operations": count, "observed_operations": int(observed.sum()),
            "started_at": int(times[0]), "planned_ended_at": int(times[-1]),
            "observed_ended_at": int(times[observed][-1]), "right_censored": bool(not observed.all()),
            "channels": sorted(set(observed_channels)),
        }
        return [(int(when), channel, self.attacker, "unauthorized_purchase", episode)
                for when, channel, keep in zip(times, channels, observed, strict=True) if keep]

    def simulate(self, start_day, episode_count):
        pending = {}
        for day, city in enumerate(self.route):
            weekend = (start_day + day) % 7 >= 5
            schedule = pending.pop(day, [])
            for _ in range(self.rng.poisson(self.purchase_rate * (1.11 if weekend else 0.96))):
                channel = self.channel_choice.choose(self.rng)
                if self.rng.random() < 0.012:
                    scenario = "large_purchase"
                elif channel in PHYSICAL_CHANNELS and city is not None and city != self.home:
                    scenario = "travel_purchase"
                else:
                    scenario = "routine_purchase"
                schedule.append((self.event_time(day, city or self.home), channel, city, scenario, 0))
            for _ in range(self.rng.poisson(self.transfer_rate)):
                schedule.append((self.event_time(day, city or self.home), "mobile_app", city, None, 0))
            if day >= self.config.warmup_days and self.episode_rng.random() < self.config.daily_attack_probability:
                episode_count += 1
                for event in self.attack_plan(day, episode_count):
                    event_day = event[0] // DAY_MS
                    if event_day == day:
                        schedule.append(event)
                    else:
                        pending.setdefault(event_day, []).append(event)
            for when, channel, location, scenario, episode in sorted(schedule, key=lambda event: event[0]):
                if channel == "mobile_app":
                    self.transfer(when, episode)
                else:
                    self.purchase(day, when, channel, location, scenario, episode)
        if pending:
            raise RuntimeError("После окончания симуляции остались наблюдаемые события.")
        self.events.sort(key=lambda item: item[0]["timestamp"])
        first = set()
        for event, _, episode in self.events:
            if episode and episode not in first:
                event["_episode_record"] = self.episode_plans[episode]
                first.add(episode)
        if first != set(self.episode_plans):
            raise RuntimeError("Не все запланированные эпизоды содержат наблюдаемую операцию.")
        return episode_count


class EpisodeJournal(ChannelRiskJournal):
    def __init__(self, session_writer, context_writer, episode_writer, parameters):
        super().__init__(session_writer, context_writer)
        self.episode_writer = episode_writer
        self.parameters = parameters
        self.pending_episodes = []
        self.kind_counts = Counter()
        self.censored = 0
        self.mixed = 0
        self.multi_day = 0

    def record(self, event, start_ms):
        record = event.pop("_episode_record", None)
        if record is not None:
            values = dict(record)
            for name in ("started_at", "planned_ended_at", "observed_ended_at"):
                values[name] += start_ms
            self.pending_episodes.append(values)
            self.kind_counts[values["kind"]] += 1
            self.censored += int(values["right_censored"])
            self.mixed += int(len(values["channels"]) > 1)
            self.multi_day += int(record["started_at"] // DAY_MS != record["observed_ended_at"] // DAY_MS)
        super().record(event, start_ms)

    def flush(self):
        super().flush()
        if self.pending_episodes:
            table = pa.Table.from_pylist(self.pending_episodes, schema=EPISODE_SCHEMA)
            self.episode_writer.write_table(table)
            self.pending_episodes.clear()

    def metadata(self):
        return {**super().metadata(), "episode_generation": {
            "file": EPISODE_FILE, "rows": int(sum(self.kind_counts.values())),
            "kind_counts": dict(self.kind_counts), "right_censored": self.censored,
            "mixed_channel_episodes": self.mixed, "multi_day_episodes": self.multi_day,
            "parameters": self.parameters, "parameter_file": PARAMETER_FILE,
            "ground_truth_only": True, "empirically_calibrated": False,
            "first_operation_rule": "First observed attacker transaction of a globally identified episode.",
            "censoring_rule": "Planned operations at or after the dataset end are unobserved and recorded as right-censored.",
            "chronology_rule": "Legitimate and attacker operations are processed together in timestamp order per client.",
            "state_rule": "Attacker device and recipient are retained by episode across interleaving and day boundaries.",
        }}


class EpisodeTelemetry:
    source_files = (*ChannelRiskTelemetry.source_files, Path(__file__))

    def __init__(self, parameters):
        self.parameters = parameters
        self.client_type = partial(EpisodeClient, episode_parameters=parameters)

    @contextmanager
    def open(self, stage):
        write_json(stage / WEBSITE_CATALOG_FILE, WEBSITE_REGISTRY)
        write_json(stage / PARAMETER_FILE, self.parameters)
        with (pq.ParquetWriter(stage / TELEMETRY_FILE, TELEMETRY_SCHEMA, compression="zstd") as sessions,
              pq.ParquetWriter(stage / RISK_FILE, RISK_CONTEXT_SCHEMA, compression="zstd") as context,
              pq.ParquetWriter(stage / EPISODE_FILE, EPISODE_SCHEMA, compression="zstd") as episodes):
            journal = EpisodeJournal(sessions, context, episodes, self.parameters)
            yield journal
            journal.flush()
