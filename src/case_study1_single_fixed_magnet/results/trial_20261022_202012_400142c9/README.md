# CS1 — one fixed magnet, manually controlled free magnet

cs1 Orientation aware

Trial: trial_20261022_202012_400142c9

Mode: CAN hardware. Input: keyboard.

Start: 2026-10-22T12:20:12.801512+00:00. Duration: 121.237 s. Stop: Driver action: zero_currents.

Control stopped 119.221 s after start (Driver action: zero_currents); recording continued after the stop.

Pose at arming [x_m, y_m, heading_rad]: [-0.034251061795490916, -0.01321516266193258, 1.4090435297555846].

Initial pose and calibration: see session.json. Position is in EM meters; headings in radians; currents in amperes.

Files:
- ui.avi: complete main UI, including camera, TFW, controls and current table (MJPEG).
- video_timestamps.csv: captured/duplicated frame timing; video_complete.json: frame/drop/error counts.
- telemetry.csv: controller-rate pose, input, commanded/measured coil currents and freshness.
- events.jsonl: received pose, continuous heading, tracking status, commands, CAN feedback and image timestamps.
- camera/: original received JPEG previews.
- session.json: configuration, calibration, TFW provenance and initial-pose snapshot.

Counts: 153911 data events; 565 raw camera frames; 607 UI video frames.

Timing uses host UTC and a common host monotonic clock. Camera/pose/video are not hardware synchronized. Missing current feedback is blank/null, not measured zero.
