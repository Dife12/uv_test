#pragma once

#include <Arduino.h>

namespace uv_test {

constexpr bool kModelAvailable = false;
constexpr size_t kTrainedFeatureCount = 16;
constexpr size_t kTrainedWindowSize = 32;

constexpr const char* kTrainedFeatureNames[kTrainedFeatureCount] = {
    "uva_min",   "uva_max",   "uva_mean",   "uva_std",
    "uvb_min",   "uvb_max",   "uvb_mean",   "uvb_std",
    "pitch_min", "pitch_max", "pitch_mean", "pitch_std",
    "roll_min",  "roll_max",  "roll_mean",  "roll_std",
};

constexpr float kFeatureMeans[kTrainedFeatureCount] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};
constexpr float kFeatureScales[kTrainedFeatureCount] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
};
constexpr float kFeatureCoefficients[kTrainedFeatureCount] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};
constexpr float kFeatureIntercept = 0.0f;

}  // namespace uv_test
