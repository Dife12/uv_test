from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import load_samples
from .evaluate import select_best_window
from .features import build_windowed_dataset
from .firmware_export import write_firmware_header
from .model import ExportedModel, LogisticRegressionModel


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="uv_test command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the uv_test pipeline")
    train_parser.add_argument("--csv", required=True, help="Input CSV file")
    train_parser.add_argument(
        "--window-sizes",
        nargs="+",
        required=True,
        type=int,
        help="Candidate non-overlapping window sizes in samples",
    )
    train_parser.add_argument("--export", help="Optional path to write the exported model JSON")
    train_parser.add_argument(
        "--firmware-header",
        help="Optional path to write a generated C++ header for on-device inference",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference from an exported model")
    predict_parser.add_argument("--csv", required=True, help="Input CSV file")
    predict_parser.add_argument("--model", required=True, help="Exported model JSON")

    export_parser = subparsers.add_parser(
        "export-firmware",
        help="Convert an exported model JSON into a C++ header for firmware inference",
    )
    export_parser.add_argument("--model", required=True, help="Exported model JSON")
    export_parser.add_argument("--output", required=True, help="Output header path")

    return parser


def _train(
    csv_path: str,
    window_sizes: list[int],
    export_path: str | None,
    firmware_header_path: str | None,
) -> int:
    samples = load_samples(csv_path)
    result = select_best_window(samples, window_sizes)
    payload = {
        "best_window_size": result.window_size,
        "weighted_accuracy": result.weighted_accuracy,
        "weighted_precision": result.weighted_precision,
        "weighted_recall": result.weighted_recall,
        "weighted_f1_score": result.weighted_f1_score,
        "folds": [
            {
                "participant_id": fold.participant_id,
                "accuracy": fold.accuracy,
                "precision": fold.precision,
                "recall": fold.recall,
                "f1_score": fold.f1_score,
                "support": fold.support,
            }
            for fold in result.folds
        ],
    }
    print(json.dumps(payload, indent=2))

    if export_path:
        result.exported_model.save(export_path)
        print(f"Exported model to {Path(export_path)}")
    if firmware_header_path:
        write_firmware_header(result.exported_model, firmware_header_path)
        print(f"Generated firmware header at {Path(firmware_header_path)}")
    return 0


def _predict(csv_path: str, model_path: str) -> int:
    exported = ExportedModel.load(model_path)
    model = LogisticRegressionModel.from_export(exported)
    samples = load_samples(csv_path)
    dataset = build_windowed_dataset(samples, exported.window_size)
    probabilities = model.predict_proba(dataset.features)[:, 1]
    predictions = model.predict(dataset.features)
    rows = [
        {
            "participant_id": participant_id,
            "label": int(label),
            "prediction": int(prediction),
            "outdoor_probability": float(probability),
        }
        for participant_id, label, prediction, probability in zip(
            dataset.participants,
            dataset.labels,
            predictions,
            probabilities,
            strict=True,
        )
    ]
    print(json.dumps(rows, indent=2))
    return 0


def _export_firmware(model_path: str, output_path: str) -> int:
    exported = ExportedModel.load(model_path)
    write_firmware_header(exported, output_path)
    print(f"Generated firmware header at {Path(output_path)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "train":
        return _train(args.csv, args.window_sizes, args.export, args.firmware_header)
    if args.command == "predict":
        return _predict(args.csv, args.model)
    if args.command == "export-firmware":
        return _export_firmware(args.model, args.output)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
