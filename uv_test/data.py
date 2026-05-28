from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    participant_id: str
    timestamp: str
    uva: float
    uvb: float
    pitch: float
    roll: float
    label: int


def normalize_label(value: str) -> int:
    lowered = value.strip().lower()
    if lowered in {"outdoor", "1", "+1", "true"}:
        return 1
    if lowered in {"indoor", "-1", "0", "false"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def load_samples(csv_path: str | Path) -> list[Sample]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        required = {"participant_id", "timestamp", "uva", "uvb", "pitch", "roll", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")

        samples: list[Sample] = []
        for row in reader:
            samples.append(
                Sample(
                    participant_id=row["participant_id"],
                    timestamp=row["timestamp"],
                    uva=float(row["uva"]),
                    uvb=float(row["uvb"]),
                    pitch=float(row["pitch"]),
                    roll=float(row["roll"]),
                    label=normalize_label(row["label"]),
                )
            )
    return samples
