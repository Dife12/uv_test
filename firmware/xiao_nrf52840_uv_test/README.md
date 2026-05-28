# XIAO nRF52840 uv_test firmware scaffold

This is a `PlatformIO + Arduino` firmware scaffold for a uv_test device built from:

- Seeed Studio XIAO nRF52840 Sense
- AS7331 UV sensor on I2C
- onboard LSM6DS3 IMU

## What it does

- Samples at `4 Hz`
- Reads `UVA`, `UVB`, `UVC` from the AS7331
- Reads accelerometer, gyroscope, and temperature from the LSM6DS3
- Estimates `pitch` and `roll` from the accelerometer
- Emits one CSV row per sample over USB serial
- Buffers samples in RAM so short host-side serial stalls do not immediately drop data
- Runs windowed logistic-regression inference on-device when `include/TrainedModel.h` has been generated from a trained model

## CSV output

```text
sequence,timestamp_ms,uva_uw_cm2,uvb_uw_cm2,uvc_uw_cm2,accel_x_g,accel_y_g,accel_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps,pitch_deg,roll_deg,temperature_c,outdoor_probability,prediction,status
0,1532,0.412000,0.161000,0.002000,-0.034000,0.014000,1.004000,0.122000,-0.035000,0.011000,1.939332,0.798054,29.187500,nan,-1,3
```

`status` is a bitmask:

- `1`: UV sample succeeded
- `2`: IMU sample succeeded
- `4`: a previous sample was dropped because the RAM queue overflowed
- `8`: a full inference window was available and the prediction fields are valid

## Important assumptions

- This scaffold uses an accelerometer-only tilt estimate for `pitch` and `roll`.
- This scaffold currently focuses on reliable wired CSV logging. The queue and logger layers are separated so BLE and flash sinks can be added next.

## Build and upload

1. Train a model in the Python project and generate `include/TrainedModel.h`, or keep the checked-in placeholder header if you only want raw logging.
2. Install PlatformIO.
3. Open this folder as a PlatformIO project.
4. Connect the XIAO nRF52840 Sense.
5. Build and upload:

```bash
pio run -t upload
pio device monitor -b 115200
```

If the board does not enter upload mode normally, double-tap reset on the XIAO to enter the bootloader, then upload again.

## Isolated PlatformIO core directory

If your global `~/.platformio` has permission issues or you want to keep this project isolated from other embedded projects, use a project-local PlatformIO core directory:

```bash
mkdir -p .platformio-core
env PLATFORMIO_CORE_DIR=$(pwd)/.platformio-core ~/.platformio/penv/bin/pio run
env PLATFORMIO_CORE_DIR=$(pwd)/.platformio-core ~/.platformio/penv/bin/pio run -t upload
env PLATFORMIO_CORE_DIR=$(pwd)/.platformio-core ~/.platformio/penv/bin/pio device monitor -b 115200
```

This keeps downloaded platforms, libraries, and lock files inside this project instead of reusing the global PlatformIO home.

## Generating the device model header

From the repository root:

```bash
python3 -m uv_test.cli train \
  --csv data/uv_test_samples.csv \
  --window-sizes 10 20 32 40 \
  --export model.json \
  --firmware-header firmware/xiao_nrf52840_uv_test/include/TrainedModel.h
```

Or reuse an existing exported model:

```bash
python3 -m uv_test.cli export-firmware \
  --model model.json \
  --output firmware/xiao_nrf52840_uv_test/include/TrainedModel.h
```

## Wiring notes

- The XIAO nRF52840 Sense already exposes the onboard LSM6DS3 internally.
- Connect the AS7331 to the same I2C bus:
  - `3V3 -> 3V3`
  - `GND -> GND`
  - `SDA -> SDA`
  - `SCL -> SCL`

## Next useful additions

- BLE transport for wireless phone-side logging
- Internal flash or external QSPI buffering for disconnected logging
- Timestamp synchronization with a host phone
- A small host-side tool to convert the richer firmware CSV back into the software pipeline input schema
