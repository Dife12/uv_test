from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .data import Sample

BASE_COLUMNS = ("uva", "uvb", "pitch", "roll")
STAT_NAMES = ("min", "max", "mean", "std")
WINDOW_FEATURE_NAMES = tuple(f"{column}_{stat}" for column in BASE_COLUMNS for stat in STAT_NAMES)


@dataclass(frozen=True)
class WindowedDataset:
    features: np.ndarray
    labels: np.ndarray
    participants: np.ndarray


def _window_to_features(window: list[Sample]) -> np.ndarray:
    matrix = np.array(
        [[sample.uva, sample.uvb, sample.pitch, sample.roll] for sample in window],
        dtype=float,
    )
    stats = []
    for column_index in range(matrix.shape[1]):
        values = matrix[:, column_index]
        stats.extend(
            [
                float(np.min(values)),
                float(np.max(values)),
                float(np.mean(values)),
                float(np.std(values)),
            ]
        )
    return np.array(stats, dtype=float)


def build_windowed_dataset(samples: list[Sample], window_size: int) -> WindowedDataset:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    by_participant: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_participant[sample.participant_id].append(sample)

    features = []
    labels = []
    participants = []
    for participant_id, participant_samples in by_participant.items():
        for start in range(0, len(participant_samples) - window_size + 1, window_size):
            window = participant_samples[start : start + window_size]
            features.append(_window_to_features(window))
            labels.append(window[-1].label)
            participants.append(participant_id)

    if not features:
        raise ValueError("No windows were generated; check window_size and input data length")

    return WindowedDataset(
        features=np.vstack(features),
        labels=np.array(labels, dtype=int),
        participants=np.array(participants, dtype=object),
    )
