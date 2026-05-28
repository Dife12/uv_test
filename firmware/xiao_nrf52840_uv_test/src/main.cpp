#include <Arduino.h>
#include <LSM6DS3.h>
#include <SparkFun_AS7331.h>
#include <Wire.h>

#include "AppConfig.h"
#include "CsvLogger.h"
#include "ImuMath.h"
#include "RingBuffer.h"
#include "SampleTypes.h"
#include "WindowInference.h"

namespace {

using uv_test::CsvLogger;
using uv_test::RingBuffer;
using uv_test::SampleFrame;

LSM6DS3 g_imu(I2C_MODE, uv_test::kImuI2cAddress);
SfeAS7331ArdI2C g_uv_sensor;
CsvLogger g_logger;
RingBuffer<SampleFrame, uv_test::kLogBufferCapacity> g_pending_samples;
uv_test::WindowInference g_inference;

bool g_imu_ready = false;
bool g_uv_ready = false;
bool g_buffer_overflow = false;
uint32_t g_sequence = 0;
uint32_t g_next_sample_ms = 0;

bool initialize_imu() {
  return g_imu.begin() == 0;
}

bool initialize_uv_sensor() {
  if (!g_uv_sensor.begin()) {
    return false;
  }

  if (!g_uv_sensor.prepareMeasurement(MEAS_MODE_CMD)) {
    return false;
  }

  return true;
}

bool read_uv(float& uva, float& uvb, float& uvc) {
  if (ksfTkErrOk != g_uv_sensor.setStartState(true)) {
    return false;
  }

  delay(2 + g_uv_sensor.getConversionTimeMillis());
  if (ksfTkErrOk != g_uv_sensor.readAllUV()) {
    return false;
  }

  uva = g_uv_sensor.getUVA();
  uvb = g_uv_sensor.getUVB();
  uvc = g_uv_sensor.getUVC();
  return true;
}

bool read_imu(SampleFrame& sample) {
  sample.accel_x_g = g_imu.readFloatAccelX();
  sample.accel_y_g = g_imu.readFloatAccelY();
  sample.accel_z_g = g_imu.readFloatAccelZ();
  sample.gyro_x_dps = g_imu.readFloatGyroX();
  sample.gyro_y_dps = g_imu.readFloatGyroY();
  sample.gyro_z_dps = g_imu.readFloatGyroZ();
  sample.temperature_c = g_imu.readTempC();

  if (isnan(sample.accel_x_g) || isnan(sample.accel_y_g) || isnan(sample.accel_z_g)) {
    return false;
  }

  const uv_test::TiltEstimate tilt = uv_test::estimate_tilt_from_accel(
      sample.accel_x_g, sample.accel_y_g, sample.accel_z_g);
  sample.pitch_deg = tilt.pitch_deg;
  sample.roll_deg = tilt.roll_deg;
  return true;
}

SampleFrame acquire_sample() {
  SampleFrame sample;
  sample.sequence = g_sequence++;
  sample.timestamp_ms = millis();

  if (g_uv_ready) {
    if (read_uv(sample.uva_uw_cm2, sample.uvb_uw_cm2, sample.uvc_uw_cm2)) {
      sample.status |= uv_test::kStatusUvOk;
    }
  }

  if (g_imu_ready && read_imu(sample)) {
    sample.status |= uv_test::kStatusImuOk;
  }

  if (g_buffer_overflow) {
    sample.status |= uv_test::kStatusBufferOverflow;
    g_buffer_overflow = false;
  }

  g_inference.update(sample);
  return sample;
}

void enqueue_sample(const SampleFrame& sample) {
  if (!g_pending_samples.push(sample)) {
    g_buffer_overflow = true;
    SampleFrame discarded;
    g_pending_samples.pop(discarded);
    g_pending_samples.push(sample);
  }
}

void flush_logs() {
  while (!g_pending_samples.empty()) {
    SampleFrame sample;
    if (!g_pending_samples.pop(sample)) {
      break;
    }
    g_logger.writeSample(sample);
  }
}

void print_startup_banner() {
  Serial.println("uv_test_xiao_nrf52840");
  Serial.println("mode=sample_and_log");
  Serial.println("sample_rate_hz=4");
}

}  // namespace

void setup() {
  Serial.begin(SERIAL_LOG_BAUD);
  delay(uv_test::kStartupDelayMs);

  Wire.begin();
  g_logger.begin(Serial);
  print_startup_banner();

  g_imu_ready = initialize_imu();
  g_uv_ready = initialize_uv_sensor();

  Serial.print("imu_ready=");
  Serial.println(g_imu_ready ? "true" : "false");
  Serial.print("uv_ready=");
  Serial.println(g_uv_ready ? "true" : "false");

  g_next_sample_ms = millis();
}

void loop() {
  const uint32_t now = millis();
  if (static_cast<int32_t>(now - g_next_sample_ms) >= 0) {
    g_next_sample_ms += uv_test::kSamplePeriodMs;
    enqueue_sample(acquire_sample());
  }

  flush_logs();
}
