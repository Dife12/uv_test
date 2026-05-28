#pragma once

#include <Arduino.h>

#include "SampleTypes.h"

namespace uv_test {

class CsvLogger {
 public:
  void begin(Stream& stream) {
    stream_ = &stream;
  }

  void writeHeaderIfNeeded() {
    if (stream_ == nullptr || header_written_) {
      return;
    }

    stream_->println(
        "sequence,timestamp_ms,uva_uw_cm2,uvb_uw_cm2,uvc_uw_cm2,"
        "accel_x_g,accel_y_g,accel_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps,"
        "pitch_deg,roll_deg,temperature_c,outdoor_probability,prediction,status");
    header_written_ = true;
  }

  void writeSample(const SampleFrame& sample) {
    if (stream_ == nullptr) {
      return;
    }

    writeHeaderIfNeeded();
    stream_->print(sample.sequence);
    stream_->print(',');
    stream_->print(sample.timestamp_ms);
    stream_->print(',');
    printFloat(sample.uva_uw_cm2);
    stream_->print(',');
    printFloat(sample.uvb_uw_cm2);
    stream_->print(',');
    printFloat(sample.uvc_uw_cm2);
    stream_->print(',');
    printFloat(sample.accel_x_g);
    stream_->print(',');
    printFloat(sample.accel_y_g);
    stream_->print(',');
    printFloat(sample.accel_z_g);
    stream_->print(',');
    printFloat(sample.gyro_x_dps);
    stream_->print(',');
    printFloat(sample.gyro_y_dps);
    stream_->print(',');
    printFloat(sample.gyro_z_dps);
    stream_->print(',');
    printFloat(sample.pitch_deg);
    stream_->print(',');
    printFloat(sample.roll_deg);
    stream_->print(',');
    printFloat(sample.temperature_c);
    stream_->print(',');
    printFloat(sample.outdoor_probability);
    stream_->print(',');
    stream_->print(sample.prediction);
    stream_->print(',');
    stream_->println(sample.status);
  }

 private:
  void printFloat(float value) {
    if (isnan(value)) {
      stream_->print("nan");
    } else {
      stream_->print(value, 6);
    }
  }

  Stream* stream_ = nullptr;
  bool header_written_ = false;
};

}  // namespace uv_test
