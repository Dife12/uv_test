from __future__ import annotations

from pathlib import Path

from .model import ExportedModel


def _format_float_list(values: list[float]) -> str:
    return ", ".join(f"{value:.9g}f" for value in values)


def render_firmware_header(exported: ExportedModel) -> str:
    feature_count = len(exported.feature_names)
    if not (
        feature_count == len(exported.means) == len(exported.scales) == len(exported.coefficients)
    ):
        raise ValueError("Exported model arrays must all have the same length")

    means = _format_float_list(exported.means)
    scales = _format_float_list(exported.scales)
    coefficients = _format_float_list(exported.coefficients)
    intercept = f"{exported.intercept:.9g}f"
    return f"""#pragma once

#include <Arduino.h>

namespace uv_test {{

constexpr bool kModelAvailable = true;
constexpr size_t kTrainedFeatureCount = {feature_count};
constexpr size_t kTrainedWindowSize = {exported.window_size};

constexpr const char* kTrainedFeatureNames[kTrainedFeatureCount] = {{
    {", ".join(f'"{name}"' for name in exported.feature_names)}
}};

constexpr float kFeatureMeans[kTrainedFeatureCount] = {{{means}}};
constexpr float kFeatureScales[kTrainedFeatureCount] = {{{scales}}};
constexpr float kFeatureCoefficients[kTrainedFeatureCount] = {{{coefficients}}};
constexpr float kFeatureIntercept = {intercept};

}}  // namespace uv_test
"""


def write_firmware_header(exported: ExportedModel, path: str | Path) -> None:
    Path(path).write_text(render_firmware_header(exported), encoding="utf-8")
