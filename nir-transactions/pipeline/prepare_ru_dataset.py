"""Разбить российские транзакции по времени и рассчитать поведенческие признаки."""
from __future__ import annotations
import pyarrow as pa

PROHIBITED_FEATURES = {"client_id", "card_id", "account_id", "original_transaction_id", "is_fraud",
                       "scenario", "episode_id", "period", "timestamp"}


def behavioral_table(history: pa.Table) -> pa.Table:
    from .ru_behavior_features import build_features

    # Сохраняем точные целые ID при преобразовании столбцов с пропусками.
    frame = history.to_pandas(integer_object_nulls=True)
    computed = build_features(frame)
    if computed.columns.duplicated().any() or PROHIBITED_FEATURES.intersection(computed.columns):
        raise ValueError("В вычисленные признаки попали запрещённые или повторяющиеся поля.")
    feature_table = pa.Table.from_pandas(computed, preserve_index=False).replace_schema_metadata(None)
    if feature_table["transaction_id"].type != pa.int64() or not feature_table["transaction_id"].equals(history["transaction_id"]):
        raise ValueError("Вычисление признаков изменило порядок или тип ID операций.")
    for field in feature_table.schema:
        if not (pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_boolean(field.type)):
            raise ValueError(f"Модельный признак должен быть числовым: {field.name}.")
    return feature_table
