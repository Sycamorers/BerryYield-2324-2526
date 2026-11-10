"""Small, fixed-budget canopy sequence models for the weekly benchmark.

No target-season labels, validation scores or test features are used to choose
architecture, training duration or normalization parameters. Inputs are three
canopy observations in chronological order; the coordinator records observation
dates and any left padding. Returned forecasts are in the original yield units
and deliberately remain unclipped so the coordinator can apply one common rule.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn


MODEL_NAMES = ("RNN", "LSTM", "Transformer")
SEQUENCE_LENGTH = 3
HIDDEN_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.001
GRADIENT_CLIP = 1.0
DROPOUT = 0.1


def get_config() -> dict[str, Any]:
    """Return the exact fixed settings used by :func:`predict_neural`."""
    return {
        "model_names": list(MODEL_NAMES),
        "input": "Three canopy-size observations, chronological oldest to latest",
        "sequence_length": SEQUENCE_LENGTH,
        "input_features_per_step": 1,
        "hidden_size_or_d_model": HIDDEN_SIZE,
        "recurrent_layers": 1,
        "rnn_nonlinearity": "tanh",
        "recurrent_readout": "Final recurrent output; dropout 0.1; linear 16-to-1",
        "transformer_encoder_layers": 1,
        "transformer_attention_heads": 2,
        "transformer_feedforward_width": 32,
        "transformer_activation": "gelu",
        "transformer_normalization": "Pre-layernorm and final LayerNorm",
        "transformer_positional_encoding": "Fixed sinusoidal encoding",
        "transformer_readout": "Final position; linear 16-to-1",
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "batching": "Full training batch",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "Mean squared error on standardized yield",
        "gradient_clip_norm": GRADIENT_CLIP,
        "input_normalization": "Single mean and population SD of training X only",
        "target_normalization": "Mean and population SD of training y only",
        "near_constant_scale_rule": "Replace SD <= 1e-8 by 1.0",
        "model_selection": "None; fixed architecture and epochs",
        "device": "cpu",
        "torch_threads": 1,
        "dtype": "float32",
        "prediction_clipping": "None in this module",
        "torch_version": str(torch.__version__),
    }


class RecurrentRegressor(nn.Module):
    def __init__(self, kind: str, input_size: int = 1) -> None:
        super().__init__()
        if kind == "RNN":
            self.recurrent = nn.RNN(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=1,
                nonlinearity="tanh",
                batch_first=True,
            )
        elif kind == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=1,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown recurrent kind: {kind}")
        self.readout = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(HIDDEN_SIZE, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        return self.readout(output[:, -1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, input_size: int = 1) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, HIDDEN_SIZE)
        position = torch.arange(SEQUENCE_LENGTH, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, HIDDEN_SIZE, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / HIDDEN_SIZE)
        )
        encoding = torch.zeros(SEQUENCE_LENGTH, HIDDEN_SIZE)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=2,
            dim_feedforward=32,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            norm=nn.LayerNorm(HIDDEN_SIZE),
            enable_nested_tensor=False,
        )
        self.readout = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection(x) + self.position_encoding[:, : x.shape[1]]
        # Every input belongs to the observed history at the forecast origin;
        # bidirectional attention within that available prefix is permissible.
        output = self.encoder(embedded)
        return self.readout(output[:, -1]).squeeze(-1)


def _checked_arrays(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(X_train, dtype=np.float64)
    target = np.asarray(y_train, dtype=np.float64)
    test = np.asarray(X_test, dtype=np.float64)
    if train.ndim not in (2, 3) or train.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"X_train must have shape (N, {SEQUENCE_LENGTH}[, features])")
    if test.ndim != train.ndim or test.shape[1:] != train.shape[1:]:
        raise ValueError("X_test must have the same sequence and feature dimensions as X_train")
    if target.ndim != 1 or len(target) != len(train) or len(train) == 0:
        raise ValueError("y_train must be a nonempty vector matching X_train")
    if not all(np.isfinite(array).all() for array in (train, target, test)):
        raise ValueError("Neural model arrays must contain only finite values")
    return train, target, test


def predict_neural(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int = 42,
    bundle_path=None,
) -> np.ndarray:
    """Fit one fixed-seed model and forecast yields, without test-set selection.

    Normalization uses the training arrays exclusively. The test arrays undergo
    only the stored affine transformation and a single evaluation-mode forward
    pass. Each call fits a fresh model, suitable for strict chronological refits.
    """
    canonical = {name.lower(): name for name in MODEL_NAMES}.get(str(model_name).lower())
    if canonical is None:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}, got {model_name!r}")
    train, target, test = _checked_arrays(X_train, y_train, X_test)
    if len(test) == 0:
        return np.empty(0, dtype=np.float64)

    # Preserve scalar normalization for the original single-channel benchmark.
    # Multichannel inputs require separate scales for pixels and roundness.
    axes = (0, 1) if train.ndim == 3 else None
    x_mean, x_scale = train.mean(axis=axes), train.std(axis=axes, ddof=0)
    y_mean, y_scale = float(target.mean()), float(target.std(ddof=0))
    x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)
    y_scale = y_scale if y_scale > 1e-8 else 1.0
    normalized_train = (train - x_mean) / x_scale
    normalized_test = (test - x_mean) / x_scale
    if train.ndim == 2:
        normalized_train = normalized_train[..., None]
        normalized_test = normalized_test[..., None]
    train_tensor = torch.tensor(normalized_train, dtype=torch.float32)
    target_tensor = torch.tensor((target - y_mean) / y_scale, dtype=torch.float32)
    test_tensor = torch.tensor(normalized_test, dtype=torch.float32)

    torch.set_num_threads(1)
    # Preserve the coordinator's torch RNG state while making each fit repeatable.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model: nn.Module = (
            TransformerRegressor(train_tensor.shape[-1]) if canonical == "Transformer"
            else RecurrentRegressor(canonical, train_tensor.shape[-1])
        )
        model = model.cpu()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.MSELoss()
        model.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_tensor), target_tensor)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {canonical} training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(test_tensor).cpu().numpy().astype(np.float64)
        if bundle_path is not None:
            torch.save(dict(model_name=canonical, input_size=train_tensor.shape[-1],
                state_dict=model.state_dict(), x_mean=np.asarray(x_mean).tolist(),
                x_scale=np.asarray(x_scale).tolist(), y_mean=y_mean, y_scale=y_scale,
                seed=int(seed), config=get_config() | {
                    "input_features_per_step": train_tensor.shape[-1],
                    "input_normalization": "Per-channel training mean and SD"}), bundle_path)
    result = prediction * y_scale + y_mean
    if not np.isfinite(result).all():
        raise FloatingPointError(f"Non-finite {canonical} predictions")
    return result


if __name__ == "__main__":
    # A small diagnostic, not a performance estimate or a tuning procedure.
    rng = np.random.default_rng(9)
    x = rng.uniform(100_000, 2_000_000, size=(32, SEQUENCE_LENGTH))
    y = 5 + x @ np.array([0.00002, -0.00001, 0.00004])
    test_x = x[:4].copy()
    swapped = test_x[:, [1, 0, 2]]  # Same latest canopy, different observed order.
    for name in MODEL_NAMES:
        combined = predict_neural(name, x, y, np.concatenate([test_x, swapped]), seed=42)
        order_effect = float(np.mean(np.abs(combined[:4] - combined[4:])))
        assert np.isfinite(combined).all() and combined.shape == (8,)
        assert order_effect > 1e-6, f"{name} failed order-sensitivity diagnostic"
        repeated = predict_neural(name, x, y, np.concatenate([test_x, swapped]), seed=42)
        assert np.array_equal(combined, repeated), f"{name} is not repeatable"
        print(f"{name}: finite, deterministic; mean order effect={order_effect:.6f} g")
