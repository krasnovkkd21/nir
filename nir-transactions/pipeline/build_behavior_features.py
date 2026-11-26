"""Статистический профиль по строго предшествующей истории клиента.

Запуск: python build_behavior_features.py
Ни метки мошенничества, ни будущие операции не участвуют в расчёте признаков.
История берётся целиком, даже если операция не выбрана для обучения модели.
"""
import numpy as np

HOUR_NS = 3_600_000_000_000

DAY_NS = 24 * HOUR_NS


def divide(numerator, denominator, valid=None):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    mask = denominator > 0
    if valid is not None:
        mask &= valid
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=mask)
    return result


def amount_stats(prefix, squares, start, end):
    count = end - start
    mean = divide(prefix[end] - prefix[start], count)
    second = divide(squares[end] - squares[start], count)
    variance = np.maximum(second - mean * mean, 0.0)
    std = np.sqrt(variance)
    std[count < 2] = np.nan
    return mean, std
