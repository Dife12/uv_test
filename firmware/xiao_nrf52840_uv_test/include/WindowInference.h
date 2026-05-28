#pragma once

#include <Arduino.h>

#include "AppConfig.h"
#include "SampleTypes.h"
#include "TrainedModel.h"

namespace uv_test {

class WindowInference {
 public:
  void update(SampleFrame& sample) {
    push(sample);

    if (!kModelAvailable) {
      return;
    }

    float outdoor_probability = NAN;
    int32_t prediction = -1;
    if (!infer(outdoor_probability, prediction)) {
      return;
    }

    sample.outdoor_probability = outdoor_probability;
    sample.prediction = prediction;
    sample.status |= kStatusInferenceReady;
  }

 private:
  struct ChannelStats {
    float minimum = NAN;
    float maximum = NAN;
    float mean = NAN;
    float standard_deviation = NAN;
  };

  void push(const SampleFrame& sample) {
    if (count_ < kTrainedWindowSize) {
      frames_[(head_ + count_) % kTrainedWindowSize] = sample;
      ++count_;
      return;
    }

    frames_[head_] = sample;
    head_ = (head_ + 1) % kTrainedWindowSize;
  }

  bool infer(float& outdoor_probability, int32_t& prediction) const {
    if (count_ < kTrainedWindowSize || !window_is_valid()) {
      return false;
    }

    float features[kTrainedFeatureCount];
    compute_features(features);

    float logit = kFeatureIntercept;
    for (size_t index = 0; index < kTrainedFeatureCount; ++index) {
      const float standardized = (features[index] - kFeatureMeans[index]) / kFeatureScales[index];
      logit += standardized * kFeatureCoefficients[index];
    }

    outdoor_probability = sigmoid(logit);
    prediction = outdoor_probability >= 0.5f ? 1 : 0;
    return true;
  }

  bool window_is_valid() const {
    for (size_t index = 0; index < count_; ++index) {
      const SampleFrame& sample = frames_[(head_ + index) % kTrainedWindowSize];
      if ((sample.status & kStatusUvOk) == 0 || (sample.status & kStatusImuOk) == 0) {
        return false;
      }
      if (isnan(sample.uva_uw_cm2) || isnan(sample.uvb_uw_cm2) || isnan(sample.pitch_deg) ||
          isnan(sample.roll_deg)) {
        return false;
      }
    }
    return true;
  }

  void compute_features(float* features) const {
    const ChannelStats uva = compute_channel_stats(&SampleFrame::uva_uw_cm2);
    const ChannelStats uvb = compute_channel_stats(&SampleFrame::uvb_uw_cm2);
    const ChannelStats pitch = compute_channel_stats(&SampleFrame::pitch_deg);
    const ChannelStats roll = compute_channel_stats(&SampleFrame::roll_deg);

    const ChannelStats stats[] = {uva, uvb, pitch, roll};
    size_t offset = 0;
    for (const ChannelStats& channel : stats) {
      features[offset++] = channel.minimum;
      features[offset++] = channel.maximum;
      features[offset++] = channel.mean;
      features[offset++] = channel.standard_deviation;
    }
  }

  ChannelStats compute_channel_stats(float SampleFrame::*member) const {
    ChannelStats stats;
    if (count_ == 0) {
      return stats;
    }

    float sum = 0.0f;
    stats.minimum = INFINITY;
    stats.maximum = -INFINITY;
    for (size_t index = 0; index < count_; ++index) {
      const float value = frames_[(head_ + index) % kTrainedWindowSize].*member;
      stats.minimum = min(stats.minimum, value);
      stats.maximum = max(stats.maximum, value);
      sum += value;
    }

    stats.mean = sum / static_cast<float>(count_);

    float squared_error_sum = 0.0f;
    for (size_t index = 0; index < count_; ++index) {
      const float value = frames_[(head_ + index) % kTrainedWindowSize].*member;
      const float centered = value - stats.mean;
      squared_error_sum += centered * centered;
    }
    stats.standard_deviation = sqrtf(squared_error_sum / static_cast<float>(count_));
    return stats;
  }

  float sigmoid(float value) const {
    const float clipped = constrain(value, -50.0f, 50.0f);
    return 1.0f / (1.0f + expf(-clipped));
  }

  SampleFrame frames_[kTrainedWindowSize];
  size_t head_ = 0;
  size_t count_ = 0;
};

}  // namespace uv_test
