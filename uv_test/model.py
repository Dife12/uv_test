from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class ExportedModel:
    window_size: int
    feature_names: list[str]
    means: list[float]
    scales: list[float]
    coefficients: list[float]
    intercept: float

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExportedModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


class LogisticRegressionModel:
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iterations: int = 4000,
        l2_strength: float = 1e-2,
        tolerance: float = 1e-7,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.l2_strength = l2_strength
        self.tolerance = tolerance
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticRegressionModel":
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float)
        if x.ndim != 2:
            raise ValueError("features must be a 2D array")
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("labels must be a 1D array aligned with features")

        self.means_ = x.mean(axis=0)
        scales = x.std(axis=0)
        scales[scales == 0.0] = 1.0
        self.scales_ = scales
        x_scaled = (x - self.means_) / self.scales_

        weights = np.zeros(x_scaled.shape[1], dtype=float)
        bias = 0.0
        previous_loss = float("inf")

        for _ in range(self.max_iterations):
            logits = x_scaled @ weights + bias
            probs = _sigmoid(logits)
            error = probs - y

            weight_grad = (x_scaled.T @ error) / len(x_scaled) + self.l2_strength * weights
            bias_grad = float(np.mean(error))

            weights -= self.learning_rate * weight_grad
            bias -= self.learning_rate * bias_grad

            loss = self._loss(x_scaled, y, weights, bias)
            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss

        self.coefficients_ = weights
        self.intercept_ = bias
        return self

    def _loss(self, x_scaled: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float) -> float:
        logits = x_scaled @ weights + bias
        probs = np.clip(_sigmoid(logits), 1e-9, 1.0 - 1e-9)
        cross_entropy = -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
        penalty = 0.5 * self.l2_strength * float(np.sum(weights ** 2))
        return float(cross_entropy + penalty)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self._check_is_fit()
        x = np.asarray(features, dtype=float)
        x_scaled = (x - self.means_) / self.scales_
        logits = x_scaled @ self.coefficients_ + self.intercept_
        positive = _sigmoid(logits)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)

    def export(self, feature_names: list[str], window_size: int) -> ExportedModel:
        self._check_is_fit()
        return ExportedModel(
            window_size=window_size,
            feature_names=feature_names,
            means=self.means_.tolist(),
            scales=self.scales_.tolist(),
            coefficients=self.coefficients_.tolist(),
            intercept=float(self.intercept_),
        )

    def _check_is_fit(self) -> None:
        if (
            self.means_ is None
            or self.scales_ is None
            or self.coefficients_ is None
            or self.intercept_ is None
        ):
            raise RuntimeError("Model has not been fit yet")

    @classmethod
    def from_export(cls, exported: ExportedModel) -> "LogisticRegressionModel":
        model = cls()
        model.means_ = np.array(exported.means, dtype=float)
        model.scales_ = np.array(exported.scales, dtype=float)
        model.coefficients_ = np.array(exported.coefficients, dtype=float)
        model.intercept_ = float(exported.intercept)
        return model
