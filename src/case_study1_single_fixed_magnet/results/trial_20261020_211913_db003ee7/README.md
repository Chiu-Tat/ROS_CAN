# CS1 — one fixed magnet, manually controlled free magnet

Third

Trial: trial_20261020_211913_db003ee7

Mode: CAN hardware. Input: keyboard.

Start: 2026-10-20T13:19:13.869331+00:00. Duration: 251.726 s. Stop: Sideways drift exceeded limit.

Control stopped 236.722 s after start (Sideways drift exceeded limit); recording continued after the stop.

Pose at arming [x_m, y_m, heading_rad]: [-0.03039786911517115, -0.03845898876972802, 1.2133654930322268].

Initial pose and calibration: see session.json. Position is in EM meters; headings in radians; currents in amperes.

Files:
- ui.avi: complete main UI, including camera, TFW, controls and current table (MJPEG).
- video_timestamps.csv: captured/duplicated frame timing; video_complete.json: frame/drop/error counts.
- telemetry.csv: controller-rate pose, input, commanded/measured coil currents and freshness.
- events.jsonl: received pose, continuous heading, tracking status, commands, CAN feedback and image timestamps.
- camera/: original received JPEG previews.
- session.json: configuration, calibration, TFW provenance and initial-pose snapshot.

Counts: 317673 data events; 1174 raw camera frames; 1259 UI video frames.

Timing uses host UTC and a common host monotonic clock. Camera/pose/video are not hardware synchronized. Missing current feedback is blank/null, not measured zero.
