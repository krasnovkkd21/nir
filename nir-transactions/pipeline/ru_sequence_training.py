import numpy as np
import torch
from torch import nn

HIDDEN_DIM = 32

ENCODER_COLUMNS = tuple(f"seq_emb_{i:03d}" for i in range(HIDDEN_DIM)) + ("seq_logit",)


class TransactionTransformer(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_dim=HIDDEN_DIM, heads=4, dropout=0.1):
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "sequence_length": sequence_length,
            "hidden_dim": hidden_dim,
            "heads": heads,
            "dropout": dropout,
        }
        self.project = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Embedding(sequence_length, hidden_dim)
        self.register_buffer("position_ids", torch.arange(sequence_length), persistent=False)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, padding):
        positions = self.project(inputs) + self.position(self.position_ids)
        values = self.attention_norm(positions)
        attended, _ = self.attention(values[:, -1:], values, values, key_padding_mask=padding, need_weights=False)
        current = positions[:, -1] + self.dropout(attended[:, 0])
        vector = self.output_norm(current + self.dropout(self.feedforward(current)))
        return vector, self.classifier(vector).squeeze(-1)


class FittedSequenceEncoder:
    def __init__(self, model, tokens, timestamps, history_start, history_end, mean, scale, missing_columns, batch_size):
        self.model = model
        self.tokens = tokens
        self.timestamps = timestamps
        self.history_start = history_start
        self.history_end = history_end
        self.mean = mean
        self.scale = scale
        self.missing_columns = missing_columns
        self.batch_size = batch_size
        self.offsets = np.arange(model.config["sequence_length"] - 1, -1, -1, dtype=np.int64)
        self.columns = ENCODER_COLUMNS

    def batch(self, rows):
        indices = self.history_end[rows, None] - self.offsets
        padding = indices < self.history_start[rows, None]
        indices[:, -1] = rows
        padding[:, -1] = False
        indices[padding] = 0
        elapsed = self.timestamps[rows, None] - self.timestamps[indices]
        elapsed[padding] = 0
        values = self.tokens[indices]
        available = np.isfinite(values)
        values -= self.mean
        values /= self.scale
        values[~available] = 0
        np.clip(values, -8.0, 8.0, out=values)
        elapsed_feature = np.log1p(elapsed.astype(np.float32) / 60_000.0) / 10.0
        inputs = np.concatenate((values, ~available[:, :, self.missing_columns], elapsed_feature[:, :, None]), axis=-1)
        inputs[padding] = 0
        return torch.from_numpy(inputs), torch.from_numpy(padding)

    @torch.inference_mode()
    def predict_logits(self, rows):
        rows = row_indices(rows, len(self.tokens))
        scores = np.empty(len(rows), dtype=np.float32)
        self.model.eval()
        for offset in range(0, len(rows), self.batch_size):
            chunk = rows[offset:offset + self.batch_size]
            _, logits = self.model(*self.batch(chunk))
            scores[offset:offset + len(chunk)] = logits.numpy()
        return scores

    @torch.inference_mode()
    def encode(self, rows):
        rows = row_indices(rows, len(self.tokens))
        vectors = np.empty((len(rows), len(self.columns)), dtype=np.float32)
        self.model.eval()
        for offset in range(0, len(rows), self.batch_size):
            chunk = rows[offset:offset + self.batch_size]
            vector, logits = self.model(*self.batch(chunk))
            vectors[offset:offset + len(chunk), :-1] = vector.numpy()
            vectors[offset:offset + len(chunk), -1] = logits.numpy()
        if not np.isfinite(vectors).all():
            raise FloatingPointError("Transformer вернул неконечные признаки.")
        return vectors


def row_indices(rows, count):
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Нужен одномерный массив позиций операций.")
    if len(rows) and (rows.min() < 0 or rows.max() >= count):
        raise ValueError("Позиция операции вне диапазона.")
    return rows


def load_encoder(checkpoint_path, tokens, timestamps_ms, history_start, history_end, *, batch_size):
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    config = checkpoint["model_config"]
    if (checkpoint["format_version"] != 1 or tuple(checkpoint["feature_columns"]) != ENCODER_COLUMNS
            or config["hidden_dim"] != HIDDEN_DIM):
        raise ValueError("Несовместимая схема сохранённого Transformer.")
    mean, scale, missing = (checkpoint["normalization"][key].numpy() for key in ("mean", "scale", "missing_columns"))
    tokens = np.asarray(tokens)
    timestamps_ms, history_start, history_end = map(np.asarray, (timestamps_ms, history_start, history_end))
    if (tokens.ndim != 2 or tokens.dtype != np.float32 or tokens.shape[1] != len(mean) or batch_size <= 0
            or any(array.shape != (len(tokens),) for array in (timestamps_ms, history_start, history_end))):
        raise ValueError("Входные операции не соответствуют сохранённой схеме Transformer.")
    model = TransactionTransformer(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    fitted = FittedSequenceEncoder(model, tokens, timestamps_ms, history_start, history_end, mean, scale, missing, batch_size)
    fitted.metrics = checkpoint["metrics"]
    return fitted
