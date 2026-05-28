#pragma once

#include <Arduino.h>

namespace uv_test {

constexpr uint32_t kSampleRateHz = 4;
constexpr uint32_t kSamplePeriodMs = 1000 / kSampleRateHz;
constexpr uint32_t kStartupDelayMs = 1500;
constexpr size_t kLogBufferCapacity = 128;
constexpr uint8_t kImuI2cAddress = 0x6A;

constexpr uint32_t kStatusUvOk = 1u << 0;
constexpr uint32_t kStatusImuOk = 1u << 1;
constexpr uint32_t kStatusBufferOverflow = 1u << 2;
constexpr uint32_t kStatusInferenceReady = 1u << 3;

}  // namespace uv_test
