"""Stable fictional geography, banks and merchant outlets for the Russian simulator.

The city, bank and category weights and median amounts are scenario assumptions.
They are neither estimates of market shares nor statistics from real payments.
"""
from dataclasses import dataclass

CITIES = (
    ("Москва", "Москва"),
    ("Санкт-Петербург", "Санкт-Петербург"),
    ("Казань", "Республика Татарстан"),
    ("Екатеринбург", "Свердловская область"),
    ("Нижний Новгород", "Нижегородская область"),
    ("Новосибирск", "Новосибирская область"),
    ("Ростов-на-Дону", "Ростовская область"),
    ("Краснодар", "Краснодарский край"),
    ("Владивосток", "Приморский край"),
    ("Уфа", "Республика Башкортостан"),
)

CITY_WEIGHTS = (0.21, 0.14, 0.09, 0.09, 0.08, 0.09, 0.08, 0.08, 0.06, 0.08)

CITY_UTC_OFFSETS = dict(zip(CITIES, (3, 3, 3, 5, 3, 7, 3, 3, 10, 5), strict=True))

BANK_IDS = (
    "RU_BANK_01", "RU_BANK_02", "RU_BANK_03", "RU_BANK_04",
    "RU_BANK_05", "RU_BANK_06", "RU_BANK_07",
)

BANK_WEIGHTS = (0.28, 0.21, 0.16, 0.12, 0.09, 0.08, 0.06)

CATEGORIES = (
    ("5411", "Продуктовые магазины", 850.0, 0.28),
    ("5812", "Кафе и рестораны", 1100.0, 0.16),
    ("5912", "Аптеки", 650.0, 0.10),
    ("5541", "Автозаправочные станции", 2100.0, 0.08),
    ("4111", "Городской и пригородный транспорт", 90.0, 0.10),
    ("5200", "Товары для дома", 1450.0, 0.08),
    ("5651", "Одежда", 2500.0, 0.07),
    ("5732", "Электроника", 4700.0, 0.04),
    ("5999", "Прочие специализированные магазины", 950.0, 0.07),
    ("7011", "Гостиницы", 5200.0, 0.02),
)

_CITY_AMOUNT_FACTORS = {"Москва": 1.20, "Санкт-Петербург": 1.12, "Владивосток": 1.08}

_OUTLETS = ((0.45, 1.00, 2), (0.33, 0.86, 1), (0.22, 1.18, 1))


@dataclass(frozen=True)
class Merchant:
    merchant_id: str
    terminal_id: str
    mcc: str
    median_rub: float
    weight: float


def build_catalog() -> dict[tuple[str, str], list[Merchant]]:
    """Return outlet records by city, with one record per fictional terminal."""
    catalog = {}
    for city_number, city in enumerate(CITIES, start=1):
        amount_factor = _CITY_AMOUNT_FACTORS.get(city[0], 1.0)
        merchants = []
        for mcc, _, median_rub, category_weight in CATEGORIES:
            for outlet_number, (share, amount_multiplier, terminal_count) in enumerate(_OUTLETS):
                merchant_id = (
                    f"RU_M_{mcc}_CHAIN" if outlet_number == 0
                    else f"RU_M_{mcc}_{city_number:02d}_{outlet_number}"
                )
                for terminal_number in range(1, terminal_count + 1):
                    merchants.append(Merchant(
                        merchant_id=merchant_id,
                        terminal_id=f"RU_T_{city_number:02d}_{mcc}_{outlet_number}_{terminal_number}",
                        mcc=mcc,
                        median_rub=median_rub * amount_factor * amount_multiplier,
                        weight=category_weight * share / terminal_count,
                    ))
        catalog[city] = merchants
    return catalog
