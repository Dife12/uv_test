#pragma once

#include <Arduino.h>
#include <math.h>

namespace uv_test {

struct TiltEstimate {
  float pitch_deg = NAN;
  float roll_deg = NAN;
};

inline TiltEstimate estimate_tilt_from_accel(float ax_g, float ay_g, float az_g) {
  constexpr float kRadToDeg = 57.2957795f;

  TiltEstimate tilt;
  tilt.pitch_deg = atan2f(-ax_g, sqrtf(ay_g * ay_g + az_g * az_g)) * kRadToDeg;
  tilt.roll_deg = atan2f(ay_g, az_g) * kRadToDeg;
  return tilt;
}

}  // namespace uv_test
