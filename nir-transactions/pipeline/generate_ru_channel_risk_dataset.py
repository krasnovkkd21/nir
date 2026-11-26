"""Сгенерировать RU v3 с наблюдениями о получателях, сайтах и районах оплаты."""
from contextlib import contextmanager
from pathlib import Path
from zlib import crc32
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .generate_ru_session_dataset import SessionClient, SessionJournal, SessionTelemetry, TELEMETRY_FILE
from .generate_stress_data_ru_v3 import PHYSICAL_CHANNELS, WeightedChoice, write_json
from .ru_predecision_session import TELEMETRY_SCHEMA
from .ru_transaction_catalog import BANK_IDS, build_catalog

RISK_FILE = "risk_context.parquet"

WEBSITE_CATALOG_FILE = "website_catalog.json"

RISK_CONTEXT_SCHEMA = pa.schema([
    pa.field("transaction_id", pa.int64(), nullable=False),
    pa.field("terminal_zone", pa.int16()),
    pa.field("website_ref", pa.string()),
    pa.field("website_tier", pa.int8()),
])

SETTINGS = {
    "rng_stream": 813_124,
    "zones_per_city": 6,
    "owner_favourite_outlets_in_one_zone": 5,
    "attacker_other_zone_probability": 0.55,
    "website_tiers": ["verified", "unverified", "elevated"],
    "owner_website_tier_probabilities": [0.94, 0.052, 0.008],
    "attacker_website_tier_probabilities": [0.61, 0.24, 0.15],
    "owner_extra_large_web_purchase_probability": 0.018,
    "attacker_large_web_purchase_probability": 0.54,
    "owner_extra_large_web_purchase_amount_multiplier_range": [1.5, 4.5],
    "attacker_large_web_purchase_amount_multiplier_range": [1.6, 6.0],
    "general_shared_recipients_per_bank": 48,
    "concentrated_shared_recipients_per_bank": 3,
    "recipient_routes": ["private", "general_shared", "concentrated_shared"],
    "owner_recipient_route_probabilities": [0.955, 0.035, 0.010],
    "attacker_recipient_route_probabilities": [0.32, 0.06, 0.62],
}


def terminal_zone(terminal_id):
    return 1 + crc32(terminal_id.encode("ascii")) % SETTINGS["zones_per_city"]


def build_market():
    zones = {}
    all_merchants = set()
    for city, outlets in build_catalog().items():
        by_zone = {}
        for merchant in outlets:
            by_zone.setdefault(terminal_zone(merchant.terminal_id), []).append(merchant)
            if merchant.mcc not in {"4111", "5541"}:
                all_merchants.add(merchant.merchant_id)
        zones[city] = {
            zone: WeightedChoice(pool, [merchant.weight for merchant in pool])
            for zone, pool in by_zone.items()
        }
    registry = {}
    by_merchant = {}
    for index, merchant_id in enumerate(sorted(all_merchants), start=1):
        sites = []
        for tier, name in enumerate(SETTINGS["website_tiers"]):
            site = f"shop-{index:03d}-{chr(97 + tier)}.example.invalid"
            registry[site] = {"merchant_id": merchant_id, "tier": tier, "verification": name}
            sites.append(site)
        by_merchant[merchant_id] = tuple(sites)
    return zones, registry, by_merchant


def build_recipient_pools():
    general = SETTINGS["general_shared_recipients_per_bank"]
    concentrated = SETTINGS["concentrated_shared_recipients_per_bank"]
    rng = np.random.default_rng(SETTINGS["rng_stream"])
    pools = {}
    for index, bank in enumerate(BANK_IDS, start=1):
        accounts = tuple(f"RU-NET-{index:02d}-{code:03d}"
                         for code in rng.permutation(general + concentrated))
        pools[bank] = (accounts[:general], accounts[general:])
    return pools

MARKET_ZONES, WEBSITE_REGISTRY, WEBSITES_BY_MERCHANT = build_market()

RECIPIENT_POOLS = build_recipient_pools()


class ChannelRiskClient(SessionClient):
    def __init__(self, client_id, config, rng, city_choice, bank_choice, merchants, online_merchants):
        super().__init__(client_id, config, rng, city_choice, bank_choice, merchants, online_merchants)
        self.risk_rng = np.random.default_rng([SETTINGS["rng_stream"], config.seed, client_id])
        self.typical_zones = {city: terminal_zone(favourites[0].terminal_id)
                              for city, favourites in self.favourites.items()}
        count = SETTINGS["owner_favourite_outlets_in_one_zone"]
        for city, zone in self.typical_zones.items():
            local = MARKET_ZONES[city][zone]
            self.favourites[city] = tuple(local.choose(self.risk_rng) for _ in range(count)) + self.favourites[city][count:]
        self.selected_terminal = None
        self.attack_recipient = (None, None)

    def purchase_merchant(self, channel, city, episode):
        merchant = super().purchase_merchant(channel, city, episode)
        typical = self.typical_zones.get(city)
        if (episode and channel in PHYSICAL_CHANNELS and typical is not None
                and self.risk_rng.random() < SETTINGS["attacker_other_zone_probability"]):
            zones = MARKET_ZONES[city]
            outside = [zone for zone in zones if zone != typical]
            zone = outside[int(self.risk_rng.integers(len(outside)))]
            merchant = zones[zone].choose(self.risk_rng)
        self.selected_terminal = merchant.terminal_id
        return merchant

    def purchase_amount(self, merchant, channel, scenario, episode):
        if channel == "ecommerce" and scenario != "large_purchase":
            actor = "attacker" if episode else "owner"
            event = "large_web_purchase" if episode else "extra_large_web_purchase"
            key = f"{actor}_{event}"
            if self.risk_rng.random() < SETTINGS[f"{key}_probability"]:
                scale = self.risk_rng.uniform(*SETTINGS[f"{key}_amount_multiplier_range"])
                return self.amount(merchant.median_rub, 0.72, scale)
        return super().purchase_amount(merchant, channel, scenario, episode)

    def purchase(self, day, timestamp, channel, city, scenario, episode):
        position = len(self.events)
        super().purchase(day, timestamp, channel, city, scenario, episode)
        transaction = self.events[position][0]
        if transaction["channel"] in PHYSICAL_CHANNELS:
            transaction["_terminal_zone"] = terminal_zone(self.selected_terminal)
        else:
            actor = "attacker" if episode else "owner"
            probabilities = SETTINGS[f"{actor}_website_tier_probabilities"]
            tier = int(self.risk_rng.choice(len(probabilities), p=probabilities))
            transaction["_website_ref"] = WEBSITES_BY_MERCHANT[transaction["merchant_id"]][tier]

    def recipient(self, bank, probabilities):
        route = int(self.risk_rng.choice(len(probabilities), p=probabilities))
        if route == 0:
            return None
        accounts = RECIPIENT_POOLS[bank][route - 1]
        return bank, accounts[int(self.risk_rng.integers(len(accounts)))]

    def transfer(self, timestamp, episode):
        super().transfer(timestamp, episode)
        transaction = self.events[-1][0]
        if transaction["is_self_transfer"] or transaction["direction"] != "debit":
            return
        bank = transaction["recipient_bank_id"]
        if episode:
            active_episode, recipient = self.attack_recipient
            if active_episode != episode:
                recipient = self.recipient(bank, SETTINGS["attacker_recipient_route_probabilities"])
                self.attack_recipient = (episode, recipient)
        else:
            recipient = self.recipient(bank, SETTINGS["owner_recipient_route_probabilities"])
        if recipient is not None:
            transaction["recipient_bank_id"], transaction["counterparty_id"] = recipient


class ChannelRiskJournal(SessionJournal):
    def __init__(self, session_writer, context_writer):
        super().__init__(session_writer)
        self.context_writer = context_writer
        self.pending_context = []
        self.context_rows = 0
        self.zone_rows = 0
        self.website_rows = [0] * len(SETTINGS["website_tiers"])
        self.shared_recipient_rows = 0
        self.shared_recipients = set()

    def record(self, event, start_ms):
        super().record(event, start_ms)
        zone = event.pop("_terminal_zone", None)
        site = event.pop("_website_ref", None)
        channel = event["channel"]
        if (zone is not None) != (channel in PHYSICAL_CHANNELS) or (site is not None) != (channel == "ecommerce"):
            raise ValueError("География и сайт должны соответствовать каналу операции.")
        if zone is not None and not 1 <= zone <= SETTINGS["zones_per_city"]:
            raise ValueError("Неизвестный район торговой точки.")
        tier = None
        if site is not None:
            registered = WEBSITE_REGISTRY[site]
            if registered["merchant_id"] != event["merchant_id"]:
                raise ValueError("Сайт не соответствует торговой точке.")
            tier = registered["tier"]
            self.website_rows[tier] += 1
        if channel == "mobile_app" and event["counterparty_id"].startswith("RU-NET-"):
            self.shared_recipient_rows += 1
            self.shared_recipients.add(event["counterparty_id"])
        self.context_rows += 1
        self.zone_rows += int(zone is not None)
        self.pending_context.append({
            "transaction_id": event["transaction_id"], "terminal_zone": zone,
            "website_ref": site, "website_tier": tier,
        })

    def flush(self):
        super().flush()
        if self.pending_context:
            self.context_writer.write_table(pa.Table.from_pylist(self.pending_context, schema=RISK_CONTEXT_SCHEMA))
            self.pending_context.clear()

    def metadata(self):
        return {**super().metadata(), "risk_context": {
            "file": RISK_FILE, "rows": self.context_rows, "physical_zone_rows": self.zone_rows,
            "website_catalog": WEBSITE_CATALOG_FILE,
            "website_rows_by_verification": dict(zip(SETTINGS["website_tiers"], self.website_rows, strict=True)),
            "shared_recipient_rows": self.shared_recipient_rows,
            "shared_recipients_observed": len(self.shared_recipients),
            "terminal_zone_scope": "fictional outlet zone within merchant city",
            "website_catalog_rule": "fictional fixed registry, independent of transaction labels",
            "recipient_history_rule": "cross-client history observable through prior synthetic transfers",
            "parameters": SETTINGS,
        }}


class ChannelRiskTelemetry:
    client_type = ChannelRiskClient
    source_files = (*SessionTelemetry.source_files, Path(__file__))

    @contextmanager
    def open(self, stage):
        write_json(stage / WEBSITE_CATALOG_FILE, WEBSITE_REGISTRY)
        with (pq.ParquetWriter(stage / TELEMETRY_FILE, TELEMETRY_SCHEMA, compression="zstd") as sessions,
              pq.ParquetWriter(stage / RISK_FILE, RISK_CONTEXT_SCHEMA, compression="zstd") as context):
            journal = ChannelRiskJournal(sessions, context)
            yield journal
            journal.flush()
