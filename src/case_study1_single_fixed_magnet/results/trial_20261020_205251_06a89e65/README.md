# CS1 — one fixed magnet, manually controlled free magnet

First try

Trial: trial_20261020_205251_06a89e65

Mode: CAN hardware. Input: keyboard.

Start: 2026-10-20T12:52:51.628709+00:00. Duration: 135.013 s. Stop: Driver action: zero_currents.

Pose at arming [x_m, y_m, heading_rad]: [-0.03324006828050013, -0.04184390733288269, 1.719959470435072].

Initial pose and calibration: see session.json. Position is in EM meters; headings in radians; currents in amperes.

Files:
- ui.avi: complete main UI, including camera, TFW, controls and current table (MJPEG).
- video_timestamps.csv: captured/duplicated frame timing; video_complete.json: frame/drop/error counts.
- telemetry.csv: controller-rate pose, input, commanded/measured coil currents and freshness.
- events.jsonl: received pose, continuous heading, tracking status, commands, CAN feedback and image timestamps.
- camera/: original received JPEG previews.
- session.json: configuration, calibration, TFW provenance and initial-pose snapshot.

Counts: 171760 data events; 629 raw camera frames; 676 UI video frames.

Timing uses host UTC and a common host monotonic clock. Camera/pose/video are not hardware synchronized. Missing current feedback is blank/null, not measured zero.
