# uv_test

This repository contains a software pipeline for wrist-based indoor/outdoor sensing with UV and orientation features.

## What is reproduced

- Input channels: `UVA`, `UVB`, `pitch`, `roll`
- Non-overlapping sliding windows
- Window statistics: `min`, `max`, `mean`, `std`
- `16` engineered features per window
- Leave-one-participant-out cross-validation
- Lightweight logistic regression suitable for on-device export

## Current limitations

- No real participant dataset is included in this repository
- Pitch/roll preprocessing is implemented as a practical approximation
- Device-side coefficients depend on your own collected training data
- Transition-point evaluation is not yet implemented as a separate benchmark

## Expected CSV format

One row per sensor sample:

```csv
participant_id,timestamp,uva,uvb,pitch,roll,label
P1,2026-05-01T10:00:00,0.42,0.16,-18.3,5.7,indoor
P1,2026-05-01T10:00:00.250,0.41,0.15,-18.1,5.8,indoor
```

- `participant_id`: identifier used for leave-one-participant-out splitting
- `timestamp`: any string; only kept for traceability
- `label`: `indoor`/`outdoor`, `-1`/`1`, or `0`/`1`

## Quick start

Run the included tests:

```bash
python3 -m unittest discover -s tests -v
```

Train and evaluate across candidate window sizes:

```bash
python3 -m uv_test.cli train \
  --csv data/uv_test_samples.csv \
  --window-sizes 10 20 32 40 \
  --export model.json \
  --firmware-header firmware/xiao_nrf52840_uv_test/include/TrainedModel.h
```

Score a new recording with an exported model:

```bash
python3 -m uv_test.cli predict \
  --csv data/new_samples.csv \
  --model model.json
```

Convert an already-exported model JSON into a firmware header later:

```bash
python3 -m uv_test.cli export-firmware \
  --model model.json \
  --output firmware/xiao_nrf52840_uv_test/include/TrainedModel.h
```

## Notes

The default firmware sampling rate is `4 Hz`. A window size such as `32` samples corresponds to about `8` seconds of context. Window sizing remains configurable so you can tune the system against your own collected data.

## Firmware scaffold

A first-pass XIAO nRF52840 Sense firmware scaffold now lives in [firmware/xiao_nrf52840_uv_test](/Users/tekihi/Documents/uv_test/firmware/xiao_nrf52840_uv_test/README.md:1). It targets `PlatformIO + Arduino`, samples the onboard `LSM6DS3` plus an external `AS7331`, emits CSV logs at `4 Hz`, and can now run the exported logistic-regression model on-device once `include/TrainedModel.h` is generated from a trained model.
