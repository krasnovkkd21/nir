import pyarrow as pa
from . import ru_location_features
from . import ru_recent_features
from .prepare_ru_dataset import behavioral_table
from .ru_behavior_features import REQUIRED_COLUMNS

ID = "transaction_id"


def _base_features(operations):
    if operations["timestamp"].type != pa.timestamp("ms", tz="UTC"):
        raise ValueError("Ожидались метки времени в UTC с миллисекундной точностью.")
    return behavioral_table(operations.select(REQUIRED_COLUMNS))


def _causal_context(operations):
    return ru_recent_features.build_features(operations), ru_location_features.build_features(operations)


def tabular_features(operations):
    return append_features(_base_features(operations), *_causal_context(operations))


def append_features(features, *groups):
    for group in groups:
        if not group[ID].equals(features[ID]):
            raise ValueError("Порядок операций в группах признаков различается.")
        for name in group.column_names:
            if name != ID:
                features = features.append_column(name, group[name])
    return features
