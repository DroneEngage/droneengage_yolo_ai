# AGENTS.md — droneengage_yolo_ai

DroneEngage YOLO AI detector (`de_ai`, binary `de_ai_tracker.so`).
Runs YOLOv8 inference for object detection on a Raspberry Pi 5 with a
Hailo8L TPU, or via a UDP-AI source (e.g. IMX500). Acts as a
recognition/detection provider in the video pipeline — reads tracked
video from upstream (e.g. `de_tracker`) and writes detections to the
bus. See parent `../AGENTS.md` for the **virtual video device pipeline**
contract and `de_common` vendoring.

## Build

    ./build.sh                 # DEBUG
    ./build_release.sh         # RELEASE
    ./build_ddebug.sh          # DEBUG + DDEBUG=ON
    ./build_imx500.sh          # IMX500 / UDP-AI source variant

Out-of-source in `build/`. Binary: `bin/de_ai_tracker.so`.

### CMake options

- `DDEBUG` — detailed debug.
- `HAILO_AI_DETECTION` (default ON) — use Hailo8L TPU board.
- `UDP_AI_DETECTION` (default OFF) — use IMX500 or any UDP-AI source
  instead of Hailo.
- `TEST_MODE_NO_HAILO_LINK` (default OFF) — fake source; do not link
  the Hailo library. Use this to build on non-Pi / non-Hailo hosts.
  Note: if neither `TEST_MODE_NO_HAILO_LINK` nor `UDP_AI_DETECTION` is
  set, CMake forces `TEST_MODE_NO_HAILO_LINK` ON automatically.
- Auto-increment build number from `.version` on RELEASE
  (MAJOR.MINOR.BUGFIX.BUILD). Bump via `MAJOR_VERSION`/`MINOR_VERSION`/
  `BUGFIX_VERSION` in `CMakeLists.txt`.

### Dependencies

OpenCV 4.5+, Threads, HailoRT (`hailort`, `hailofw` —
`sudo apt install hailofw hailort`) when `HAILO_AI_DETECTION=ON` and
`TEST_MODE_NO_HAILO_LINK=OFF`. Model: `yolov8s.hef` (Hailo HEF model,
repo root). Raspberry Pi OS recommended over Ubuntu for RPi5 firmware
support of the Hailo8L TPU.

## Config

- `de_ai_tracker.config.module.json` — module config (WebClient UI).
- `de_ai_tracker.config.local` — instance identity.
- `de_yolo_ai.config.local` — alternate local config.

## Source layout & docs

`src/` — `main.cpp`, `de_common/` (vendored), `advancedxx/`. `wiki/` —
docs. `yolov8_cam_ai2r2_streamer_Working2.cpp` — reference streamer.
`TODOHEFNY.txt` — open work items. `readme.md` — Hailo8L setup walkthrough.
