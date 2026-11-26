"""Replay the same synthetic ledger with an independent pre-decision mobile journal.

The sidecar is published only after the five existing transaction/ground-truth/
context files have been reproduced byte for byte. No label file is read to create
mobile devices, authentication attempts, or sessions.
"""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .generate_ru_channel_risk_dataset import RISK_CONTEXT_SCHEMA, RISK_FILE, WEBSITE_CATALOG_FILE, WEBSITE_REGISTRY
from .generate_ru_episode_dataset import EPISODE_FILE, EPISODE_SCHEMA, PARAMETER_FILE, EpisodeClient, EpisodeJournal, EpisodeTelemetry
from .generate_ru_session_dataset import SETTINGS as CHECKOUT_SETTINGS, TELEMETRY_FILE
from .generate_stress_data_ru_v3 import write_json
from .ru_predecision_session import TELEMETRY_SCHEMA

MOBILE_FILE = "mobile_telemetry.parquet"

RNG_STREAM = 954_419

MOBILE_SETTINGS = {**CHECKOUT_SETTINGS, "rng_stream": RNG_STREAM}


class MobileEpisodeClient(EpisodeClient):
    """Native-app state is independent of transaction and browser RNG/state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mobile_rng = np.random.default_rng([RNG_STREAM, self.config.seed, self.client_id])
        self.mobile_primary_device = 1
        self.mobile_spare_device = 2
        self.mobile_next_device = 3
        self.mobile_attack_devices = {}

    def allocate_mobile_device(self):
        value = self.mobile_next_device
        self.mobile_next_device += 1
        return value

    def choose_mobile_device(self, episode):
        rng, settings = self.mobile_rng, MOBILE_SETTINGS
        if episode:
            if episode not in self.mobile_attack_devices:
                recognized = rng.random() < settings["attacker_recognized_device_probability"]
                device = self.mobile_primary_device if recognized else self.allocate_mobile_device()
                self.mobile_attack_devices[episode] = (device, recognized)
            device, recognized = self.mobile_attack_devices[episode]
            key = ("attacker_prepared_on_recognized_device_probability" if recognized else
                   "attacker_prepared_on_unrecognized_device_probability")
            return device, recognized, settings[key]
        if rng.random() < settings["owner_device_replacement_probability"]:
            self.mobile_primary_device = self.allocate_mobile_device()
            return self.mobile_primary_device, False, settings["owner_prepared_probability"]
        draw = rng.random()
        spare = settings["owner_spare_device_probability"]
        if draw < spare:
            return self.mobile_spare_device, True, settings["owner_prepared_probability"]
        if draw < spare + settings["owner_guest_device_probability"]:
            return self.allocate_mobile_device(), False, settings["owner_prepared_probability"]
        return self.mobile_primary_device, True, settings["owner_prepared_probability"]

    def mobile_checkout(self, timestamp, episode):
        rng, settings = self.mobile_rng, MOBILE_SETTINGS
        device, recognized, preparedness = self.choose_mobile_device(episode)
        initial = device
        key = "familiar_device_handoff_probability" if recognized else "unfamiliar_device_handoff_probability"
        if rng.random() < settings[key]:
            initial = (self.mobile_spare_device if device == self.mobile_primary_device else
                       self.mobile_primary_device)
        prepared = rng.random() < preparedness
        retry_key = "prepared_retry_probabilities" if prepared else "unprepared_retry_probabilities"
        retries = int(rng.choice(len(settings[retry_key]), p=settings[retry_key]))
        observed = timestamp - int(rng.integers(*settings["observation_lag_milliseconds"]))
        duration = int(rng.integers(*settings["elapsed_base_milliseconds"]))
        duration += retries * settings["elapsed_milliseconds_per_retry"]
        started = observed - duration
        attempts = [started + step * duration // (retries + 1) for step in range(1, retries + 1)]
        if rng.random() < settings["missing_probability"]:
            return None
        return {"observed_at": observed, "device_ref": device,
                "session_started_at": started, "initial_device_ref": initial,
                "failed_auth_times": attempts}

    def transfer(self, timestamp, episode):
        # The base draft decides whether this is an outgoing app action. Incoming
        # transfers must not create an app session or mutate mobile device state.
        position = len(self.events)
        super().transfer(timestamp, episode)
        if len(self.events) != position + 1:
            raise RuntimeError("A transfer must append exactly one transaction.")
        event = self.events[position][0]
        if (event["channel"], event["operation_type"], event["direction"]) == (
                "mobile_app", "transfer", "debit"):
            # Actor state was known at action time; no scenario or label is read.
            # All event timestamps precede the decision, and the journal and
            # serialized ground-truth row are still not written.
            event["_mobile_session_observation"] = self.mobile_checkout(timestamp, episode)


class MobileEpisodeJournal(EpisodeJournal):
    def __init__(self, sessions, context, episodes, mobile, parameters):
        super().__init__(sessions, context, episodes, parameters)
        self.mobile_writer = mobile
        self.mobile_pending = []
        self.mobile_eligible_rows = 0
        self.mobile_rows = 0
        self.mobile_previous_id = 0

    def record(self, event, start_ms):
        if "_mobile_session_observation" in event:
            observation = event.pop("_mobile_session_observation")
            if (event["channel"], event["operation_type"], event["direction"]) != (
                    "mobile_app", "transfer", "debit"):
                raise ValueError("Mobile observations require an outgoing native-app transfer.")
            self.mobile_eligible_rows += 1
            if observation is not None:
                started = observation["session_started_at"] + start_ms
                observed = observation["observed_at"] + start_ms
                attempts = [value + start_ms for value in observation["failed_auth_times"]]
                sequence = [started, *attempts, observed, event["timestamp"]]
                transaction_id = event["transaction_id"]
                if (transaction_id <= self.mobile_previous_id
                        or any(left >= right for left, right in zip(sequence, sequence[1:]))):
                    raise ValueError("Mobile events must precede the transfer and have ordered IDs.")
                self.mobile_previous_id = transaction_id
                self.mobile_pending.append({**observation, "transaction_id": transaction_id,
                                            "session_started_at": started, "observed_at": observed,
                                            "failed_auth_times": attempts})
                self.mobile_rows += 1
        super().record(event, start_ms)

    def flush(self):
        super().flush()
        if self.mobile_pending:
            self.mobile_writer.write_table(pa.Table.from_pylist(self.mobile_pending, schema=TELEMETRY_SCHEMA))
            self.mobile_pending.clear()

    def metadata(self):
        return {**super().metadata(), "mobile_session_observations": {
            "file": MOBILE_FILE, "rows": self.mobile_rows, "eligible_rows": self.mobile_eligible_rows,
            "synthetic": True, "empirically_calibrated": False,
            "eligible": "mobile_app transfer debit", "decision_time": "transaction timestamp",
            "device_id_scope": "client, native-mobile namespace; not a browser fingerprint",
            "event_rule": "Independent actor and device state sampled for outgoing app actions, with timestamps before the decision and before journal/ground-truth serialization; no labels or scenarios are read.",
            "state_rule": "Native-app device persists by episode across overlaps; legitimate replacement/spare/guest devices are allowed.",
            "missing_rows": "No mobile observation captured.", "parameters": MOBILE_SETTINGS,
            "parameter_origin": "Reused existing checkout simulation probabilities; not fitted to real banks or model scores.",
        }}


class MobileEpisodeTelemetry:
    source_files = (*EpisodeTelemetry.source_files, Path(__file__))

    def __init__(self, parameters):
        self.parameters = parameters
        self.client_type = partial(MobileEpisodeClient, episode_parameters=parameters)

    @contextmanager
    def open(self, stage):
        write_json(stage / WEBSITE_CATALOG_FILE, WEBSITE_REGISTRY)
        write_json(stage / PARAMETER_FILE, self.parameters)
        with (pq.ParquetWriter(stage / TELEMETRY_FILE, TELEMETRY_SCHEMA, compression="zstd") as sessions,
              pq.ParquetWriter(stage / RISK_FILE, RISK_CONTEXT_SCHEMA, compression="zstd") as context,
              pq.ParquetWriter(stage / EPISODE_FILE, EPISODE_SCHEMA, compression="zstd") as episodes,
              pq.ParquetWriter(stage / MOBILE_FILE, TELEMETRY_SCHEMA, compression="zstd") as mobile):
            journal = MobileEpisodeJournal(sessions, context, episodes, mobile, self.parameters)
            yield journal
            journal.flush()
