from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import WINDOW_FEATURE_NAMES, WindowedDataset, build_windowed_dataset
from .model import ExportedModel, LogisticRegressionModel


@dataclass(frozen=True)
class FoldMetrics:
    participant_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class EvaluationResult:
    window_size: int
    weighted_accuracy: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1_score: float
    folds: list[FoldMetrics]
    exported_model: ExportedModel


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = len(y_true)

    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return accuracy, precision, recall, f1_score


def evaluate_windowed_dataset(dataset: WindowedDataset, window_size: int) -> EvaluationResult:
    participants = np.unique(dataset.participants)
    folds: list[FoldMetrics] = []

    for participant_id in participants:
        test_mask = dataset.participants == participant_id
        train_mask = ~test_mask
        model = LogisticRegressionModel().fit(dataset.features[train_mask], dataset.labels[train_mask])
        predictions = model.predict(dataset.features[test_mask])
        accuracy, precision, recall, f1_score = _binary_metrics(dataset.labels[test_mask], predictions)
        folds.append(
            FoldMetrics(
                participant_id=str(participant_id),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                support=int(np.sum(test_mask)),
            )
        )

    supports = np.array([fold.support for fold in folds], dtype=float)
    if np.sum(supports) == 0:
        raise ValueError("No fold support found during evaluation")

    def weighted(metric_name: str) -> float:
        values = np.array([getattr(fold, metric_name) for fold in folds], dtype=float)
        return float(np.sum(values * supports) / np.sum(supports))

    final_model = LogisticRegressionModel().fit(dataset.features, dataset.labels)
    return EvaluationResult(
        window_size=window_size,
        weighted_accuracy=weighted("accuracy"),
        weighted_precision=weighted("precision"),
        weighted_recall=weighted("recall"),
        weighted_f1_score=weighted("f1_score"),
        folds=folds,
        exported_model=final_model.export(list(WINDOW_FEATURE_NAMES), window_size),
    )


def select_best_window(samples: list, window_sizes: list[int]) -> EvaluationResult:
    if not window_sizes:
        raise ValueError("window_sizes must not be empty")

    best_result: EvaluationResult | None = None
    for window_size in window_sizes:
        dataset = build_windowed_dataset(samples, window_size=window_size)
        result = evaluate_windowed_dataset(dataset, window_size=window_size)
        if best_result is None or result.weighted_f1_score > best_result.weighted_f1_score:
            best_result = result
    assert best_result is not None
    return best_result
