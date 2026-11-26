"""Сгенерировать транзакции RU v3 и отдельный журнал событий интернет-оплаты."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .generate_stress_data_ru_v3 import ClientSimulation, PHYSICAL_CHANNELS
from .ru_predecision_session import TELEMETRY_SCHEMA

TELEMETRY_FILE = "session_telemetry.parquet"

SETTINGS = {
    "rng_stream": 743_301,
    "missing_probability": 0.08,
    "owner_device_replacement_probability": 0.008,
    "owner_spare_device_probability": 0.10,
    "owner_guest_device_probability": 0.015,
    "attacker_recognized_device_probability": 0.47,
    "familiar_device_handoff_probability": 0.012,
    "unfamiliar_device_handoff_probability": 0.065,
    "owner_prepared_probability": 0.91,
    "attacker_prepared_on_recognized_device_probability": 0.78,
    "attacker_prepared_on_unrecognized_device_probability": 0.48,
    "prepared_retry_probabilities": [0.963, 0.031, 0.005, 0.001],
    "unprepared_retry_probabilities": [0.59, 0.29, 0.10, 0.02],
    "elapsed_base_milliseconds": [12_000, 260_000],
    "elapsed_milliseconds_per_retry": 11_000,
    "observation_lag_milliseconds": [100, 2_000],
}


class SessionClient(ClientSimulation):
    def __init__(self, client_id, config, rng, city_choice, bank_choice, merchants, online_merchants):
        super().__init__(client_id, config, rng, city_choice, bank_choice, merchants, online_merchants)
        # The transactional simulation keeps its original random sequence.
        self.session_rng = np.random.default_rng([SETTINGS["rng_stream"], config.seed, client_id])
        self.primary_device = 1
        self.spare_device = 2
        self.next_device = 3
        self.active_attack = None

    def allocate_device(self):
        device = self.next_device
        self.next_device += 1
        return device

    def choose_checkout_device(self, episode):
        rng = self.session_rng
        if episode:
            # An attack can reuse an already recognized device or arrive on
            # a separate device that persists across purchases in the episode.
            if self.active_attack is None or self.active_attack[0] != episode:
                recognized = rng.random() < SETTINGS["attacker_recognized_device_probability"]
                device = self.primary_device if recognized else self.allocate_device()
                self.active_attack = (episode, device, recognized)
            _, device, recognized = self.active_attack
            preparedness_key = ("attacker_prepared_on_recognized_device_probability" if recognized
                                else "attacker_prepared_on_unrecognized_device_probability")
            return device, recognized, SETTINGS[preparedness_key]

        if rng.random() < SETTINGS["owner_device_replacement_probability"]:
            self.primary_device = self.allocate_device()
            return self.primary_device, False, SETTINGS["owner_prepared_probability"]
        channel_draw = rng.random()
        spare = SETTINGS["owner_spare_device_probability"]
        if channel_draw < spare:
            return self.spare_device, True, SETTINGS["owner_prepared_probability"]
        if channel_draw < spare + SETTINGS["owner_guest_device_probability"]:
            return self.allocate_device(), False, SETTINGS["owner_prepared_probability"]
        return self.primary_device, True, SETTINGS["owner_prepared_probability"]

    def checkout(self, timestamp, episode):
        rng = self.session_rng
        device, recognized, preparedness = self.choose_checkout_device(episode)
        handoff_key = "familiar_device_handoff_probability" if recognized else "unfamiliar_device_handoff_probability"
        initial_device = device
        if rng.random() < SETTINGS[handoff_key]:
            initial_device = self.spare_device if device == self.primary_device else self.primary_device

        prepared = rng.random() < preparedness
        retry_probabilities = SETTINGS["prepared_retry_probabilities" if prepared else "unprepared_retry_probabilities"]
        retry_count = int(rng.choice(len(retry_probabilities), p=retry_probabilities))
        observed = timestamp - int(rng.integers(*SETTINGS["observation_lag_milliseconds"]))
        duration = int(rng.integers(*SETTINGS["elapsed_base_milliseconds"]))
        duration += retry_count * SETTINGS["elapsed_milliseconds_per_retry"]
        started = observed - duration
        retries = [started + step * duration // (retry_count + 1) for step in range(1, retry_count + 1)]
        if rng.random() < SETTINGS["missing_probability"]:
            return None
        return {
            "observed_at": observed, "device_ref": device,
            "session_started_at": started, "initial_device_ref": initial_device,
            "failed_auth_times": retries,
        }

    def purchase(self, day, timestamp, channel, city, scenario, episode):
        online = channel == "ecommerce" or (channel in PHYSICAL_CHANNELS and city is None)
        if not online:
            return super().purchase(day, timestamp, channel, city, scenario, episode)
        # Create the checkout observation from the simulated actor and device
        # state before the purchase and its ground-truth record are written.
        observation = self.checkout(timestamp, episode)
        position = len(self.events)
        super().purchase(day, timestamp, channel, city, scenario, episode)
        self.events[position][0]["_session_observation"] = observation


class SessionJournal:
    def __init__(self, writer):
        self.writer = writer
        self.pending = []
        self.eligible_rows = 0
        self.rows = 0
        self.previous_id = 0

    def record(self, event, start_ms):
        if "_session_observation" not in event:
            return
        self.eligible_rows += 1
        observation = event.pop("_session_observation")
        if observation is None:
            return
        if (event["channel"], event["operation_type"], event["direction"]) != ("ecommerce", "purchase", "debit"):
            raise ValueError("Событие журнала должно относиться к исходящей интернет-покупке.")
        transaction_id = event["transaction_id"]
        started = observation["session_started_at"] + start_ms
        observed = observation["observed_at"] + start_ms
        attempts = [time + start_ms for time in observation["failed_auth_times"]]
        sequence = [started, *attempts, observed, event["timestamp"]]
        if transaction_id <= self.previous_id or any(left >= right for left, right in zip(sequence, sequence[1:])):
            raise ValueError("События должны предшествовать покупке и идти в исходном порядке.")
        self.previous_id = transaction_id
        self.pending.append({
            **observation, "transaction_id": transaction_id, "session_started_at": started,
            "observed_at": observed, "failed_auth_times": attempts,
        })
        self.rows += 1

    def flush(self):
        if self.pending:
            self.writer.write_table(pa.Table.from_pylist(self.pending, schema=TELEMETRY_SCHEMA))
            self.pending.clear()

    def metadata(self):
        return {
            "file": TELEMETRY_FILE, "rows": self.rows, "eligible_rows": self.eligible_rows,
            "synthetic": True, "empirically_calibrated": False,
            "event_rule": "Events are generated before transaction serialization from simulated actor, device and checkout state.",
            "decision_time": "transaction timestamp",
            "device_id_scope": "client",
            "missing_rows": "No checkout observation captured.",
            "parameters": SETTINGS,
        }


class SessionTelemetry:
    client_type = SessionClient
    source_files = (Path(__file__), Path(__file__).with_name("ru_predecision_session.py"))

    @contextmanager
    def open(self, stage):
        with pq.ParquetWriter(stage / TELEMETRY_FILE, TELEMETRY_SCHEMA, compression="zstd") as writer:
            journal = SessionJournal(writer)
            yield journal
            journal.flush()
