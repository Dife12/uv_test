from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from uv_test.cli import main as cli_main
from uv_test.data import load_samples
from uv_test.evaluate import select_best_window
from uv_test.features import WINDOW_FEATURE_NAMES, build_windowed_dataset
from uv_test.model import ExportedModel, LogisticRegressionModel


def _write_fixture_csv(path: Path) -> None:
    rows = ["participant_id,timestamp,uva,uvb,pitch,roll,label"]
    for participant_index, participant_id in enumerate(("P1", "P2", "P3"), start=1):
        for step in range(8):
            rows.append(
                f"{participant_id},t{participant_index}-{step},{0.10 + 0.01 * step:.3f},{0.05 + 0.01 * step:.3f},-10,2,indoor"
            )
        for step in range(8):
            rows.append(
                f"{participant_id},t{participant_index}-{step + 8},{2.00 + 0.02 * step:.3f},{1.00 + 0.02 * step:.3f},15,12,outdoor"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class UvTestPipelineTests(unittest.TestCase):
    def test_windowed_features_match_paper_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "samples.csv"
            _write_fixture_csv(csv_path)
            samples = load_samples(csv_path)
            dataset = build_windowed_dataset(samples, window_size=4)
            self.assertEqual(dataset.features.shape[1], 16)
            self.assertEqual(len(WINDOW_FEATURE_NAMES), 16)

    def test_lopo_selection_and_export_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "samples.csv"
            model_path = Path(tmp_dir) / "model.json"
            _write_fixture_csv(csv_path)

            samples = load_samples(csv_path)
            result = select_best_window(samples, [2, 4])
            self.assertGreaterEqual(result.weighted_f1_score, 0.95)

            result.exported_model.save(model_path)
            restored = ExportedModel.load(model_path)
            self.assertEqual(restored.window_size, result.window_size)

            dataset = build_windowed_dataset(samples, result.window_size)
            model = LogisticRegressionModel.from_export(restored)
            predictions = model.predict(dataset.features)
            self.assertEqual(len(predictions), len(dataset.labels))
            self.assertTrue(set(predictions.tolist()).issubset({0, 1}))

            payload = json.loads(model_path.read_text(encoding="utf-8"))
            self.assertIn("coefficients", payload)

    def test_cli_train_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "samples.csv"
            model_path = Path(tmp_dir) / "model.json"
            _write_fixture_csv(csv_path)

            train_code = cli_main(
                [
                    "train",
                    "--csv",
                    str(csv_path),
                    "--window-sizes",
                    "2",
                    "4",
                    "--export",
                    str(model_path),
                ]
            )
            self.assertEqual(train_code, 0)
            self.assertTrue(model_path.exists())

            predict_code = cli_main(
                [
                    "predict",
                    "--csv",
                    str(csv_path),
                    "--model",
                    str(model_path),
                ]
            )
            self.assertEqual(predict_code, 0)

    def test_cli_can_generate_firmware_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "samples.csv"
            model_path = Path(tmp_dir) / "model.json"
            header_path = Path(tmp_dir) / "TrainedModel.h"
            _write_fixture_csv(csv_path)

            train_code = cli_main(
                [
                    "train",
                    "--csv",
                    str(csv_path),
                    "--window-sizes",
                    "2",
                    "--export",
                    str(model_path),
                    "--firmware-header",
                    str(header_path),
                ]
            )
            self.assertEqual(train_code, 0)
            self.assertTrue(header_path.exists())
            header_text = header_path.read_text(encoding="utf-8")
            self.assertIn("kTrainedFeatureCount = 16", header_text)
            self.assertIn("kModelAvailable = true", header_text)

            export_code = cli_main(
                [
                    "export-firmware",
                    "--model",
                    str(model_path),
                    "--output",
                    str(header_path),
                ]
            )
            self.assertEqual(export_code, 0)


if __name__ == "__main__":
    unittest.main()
