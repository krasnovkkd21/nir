"""Профиль сумм интернет-покупок и оценка условного правила блокировки."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from .ru_ecommerce_features import category_groups
from .ru_recent_features import ID, encode

RULE_SETTINGS = {
    "history_days": 30,
    "minimum_history_purchases": {"website": 3, "category": 4, "all_online": 5},
    "website_tiers": {
        "verified": {"code": 0, "min_rubles": 15_000, "min_amount_multiple": 4.5},
        "unverified": {"code": 1, "min_rubles": 10_000, "min_amount_multiple": 4.0},
        "elevated": {"code": 2, "min_rubles": 6_000, "min_amount_multiple": 2.8},
        "missing": {"code": None, "min_rubles": 18_000, "min_amount_multiple": 5.0},
    },
}

HISTORY_MS = RULE_SETTINGS["history_days"] * 86_400_000

MAX_MEDIAN_CELLS = 250_000

SITE_COUNT = "web_site_purchase_count_30d"

SITE_RATIO = "web_site_amount_to_median_30d"

CATEGORY_COUNT = "web_category_purchase_count_30d"

CATEGORY_RATIO = "web_category_amount_to_median_30d"

CLIENT_COUNT = "web_client_purchase_count_30d"

CLIENT_RATIO = "web_client_amount_to_median_30d"

TIER = "web_registry_tier"

SITE_SEEN = "web_site_seen_before"

PROFILE_RATIO = "web_amount_to_profile_30d"

PROFILE_LEVEL = "web_amount_profile_level"

ONLINE_COLUMNS = (
    TIER, SITE_SEEN, SITE_COUNT, SITE_RATIO, CATEGORY_COUNT,
    CATEGORY_RATIO, CLIENT_COUNT, CLIENT_RATIO, PROFILE_RATIO, PROFILE_LEVEL,
)


def rolling_profile(times, amounts, minimum):
    end = np.searchsorted(times, times, side="left")
    start = np.searchsorted(times, times - HISTORY_MS, side="left")
    counts = end - start
    selected = np.flatnonzero(counts >= minimum)
    ratios = np.full(len(times), np.nan, dtype=np.float32)
    if len(selected):
        width = int(counts[selected].max())
        offsets = np.arange(width)
        chunk_size = max(1, MAX_MEDIAN_CELLS // width)
        for offset in range(0, len(selected), chunk_size):
            batch = selected[offset:offset + chunk_size]
            positions = start[batch, None] + offsets
            history = amounts[np.minimum(positions, len(amounts) - 1)].astype(np.float64)
            history[positions >= end[batch, None]] = np.nan
            ratios[batch] = amounts[batch] / np.nanmedian(history, axis=1)
    return counts, ratios, end > 0


def build_online_features(operations: pa.Table, risk_context: pa.Table) -> pa.Table:
    """Рассчитать признаки по покупкам с предшествующими временными метками."""
    if not risk_context[ID].equals(operations[ID]):
        raise ValueError("Интернет-контекст должен соответствовать строкам операций в том же порядке.")
    if risk_context["website_ref"].type != pa.string() or risk_context["website_tier"].type != pa.int8():
        raise ValueError("Ожидаются website_ref типа string и website_tier типа int8.")

    ids = operations[ID].to_numpy()
    result = {name: np.full(len(ids), np.nan, dtype=np.float32) for name in ONLINE_COLUMNS}
    eligible = pc.and_(
        pc.and_(pc.equal(operations["operation_type"], "purchase"), pc.equal(operations["channel"], "ecommerce")),
        pc.equal(operations["direction"], "debit"),
    )
    selected = np.flatnonzero(pc.fill_null(eligible, False).to_numpy())
    if len(selected):
        indices = pa.array(selected)
        online = operations.take(indices)
        context = risk_context.take(indices)
        clients = online["client_id"].to_numpy()
        times = pc.cast(online["timestamp"], pa.int64()).to_numpy()
        amounts = online["amount_kopecks"].to_numpy()
        if not np.isfinite(amounts).all() or np.any(amounts <= 0):
            raise ValueError("В интернет-покупке нужна положительная сумма.")
        websites, _ = encode(context["website_ref"])
        categories, _ = encode(online["mcc"])
        tiers = context["website_tier"].to_numpy(zero_copy_only=False)
        if np.any(np.isfinite(tiers) & ~np.isin(tiers, (0, 1, 2))):
            raise ValueError("В каталоге встретилась неизвестная категория сайта.")
        result[TIER][selected] = np.where(websites >= 0, tiers, np.nan)
        order = np.lexsort((ids[selected], times, clients))
        edges = np.r_[0, np.flatnonzero(clients[order[1:]] != clients[order[:-1]]) + 1, len(order)]
        minimum = RULE_SETTINGS["minimum_history_purchases"]

        for start, end in zip(edges[:-1], edges[1:], strict=True):
            rows = order[start:end]
            counts, ratios, _ = rolling_profile(times[rows], amounts[rows], minimum["all_online"])
            result[CLIENT_COUNT][selected[rows]] = counts
            result[CLIENT_RATIO][selected[rows]] = ratios
            for local in category_groups(websites[rows]):
                group = rows[local]
                counts, ratios, seen = rolling_profile(times[group], amounts[group], minimum["website"])
                result[SITE_COUNT][selected[group]] = counts
                result[SITE_RATIO][selected[group]] = ratios
                result[SITE_SEEN][selected[group]] = seen
            for local in category_groups(categories[rows]):
                group = rows[local]
                counts, ratios, _ = rolling_profile(times[group], amounts[group], minimum["category"])
                result[CATEGORY_COUNT][selected[group]] = counts
                result[CATEGORY_RATIO][selected[group]] = ratios

    ratios = (result[SITE_RATIO], result[CATEGORY_RATIO], result[CLIENT_RATIO])
    have_profile = tuple(np.isfinite(ratio) for ratio in ratios)
    result[PROFILE_RATIO] = np.select(have_profile, ratios, default=np.float32(np.nan))
    result[PROFILE_LEVEL] = np.select(
        have_profile, (np.float32(2), np.float32(1), np.float32(0)), default=np.float32(np.nan),
    )
    return pa.table({ID: ids, **result})
