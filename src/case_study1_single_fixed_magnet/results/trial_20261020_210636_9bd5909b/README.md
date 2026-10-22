# CS1 — one fixed magnet, manually controlled free magnet

Second

Trial: trial_20261020_210636_9bd5909b

Mode: CAN hardware. Input: keyboard.

Start: 2026-10-20T13:06:36.709305+00:00. Duration: 81.712 s. Stop: measured pose outside guarded model-in TFW.

Pose at arming [x_m, y_m, heading_rad]: [-0.031482403280898436, -0.031636814536487895, 1.6634076902378976].

Initial pose and calibration: see session.json. Position is in EM meters; headings in radians; currents in amperes.

Files:
- ui.avi: complete main UI, including camera, TFW, controls and current table (MJPEG).
- video_timestamps.csv: captured/duplicated frame timing; video_complete.json: frame/drop/error counts.
- telemetry.csv: controller-rate pose, input, commanded/measured coil currents and freshness.
- events.jsonl: received pose, continuous heading, tracking status, commands, CAN feedback and image timestamps.
- camera/: original received JPEG previews.
- session.json: configuration, calibration, TFW provenance and initial-pose snapshot.

Counts: 103955 data events; 381 raw camera frames; 409 UI video frames.

Timing uses host UTC and a common host monotonic clock. Camera/pose/video are not hardware synchronized. Missing current feedback is blank/null, not measured zero.
