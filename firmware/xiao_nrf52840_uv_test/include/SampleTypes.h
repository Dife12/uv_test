#pragma once

#include <Arduino.h>

namespace uv_test {

struct SampleFrame {
  uint32_t sequence = 0;
  uint32_t timestamp_ms = 0;
  float uva_uw_cm2 = NAN;
  float uvb_uw_cm2 = NAN;
  float uvc_uw_cm2 = NAN;
  float accel_x_g = NAN;
  float accel_y_g = NAN;
  float accel_z_g = NAN;
  float gyro_x_dps = NAN;
  float gyro_y_dps = NAN;
  float gyro_z_dps = NAN;
  float pitch_deg = NAN;
  float roll_deg = NAN;
  float temperature_c = NAN;
  float outdoor_probability = NAN;
  int32_t prediction = -1;
  uint32_t status = 0;
};

}  // namespace uv_test
