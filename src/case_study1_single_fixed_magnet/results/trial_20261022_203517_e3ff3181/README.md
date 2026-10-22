# CS1 — one fixed magnet, manually controlled free magnet

cs1 entering outside TFW

Trial: trial_20261022_203517_e3ff3181

Mode: CAN hardware. Input: keyboard.

Start: 2026-10-22T12:35:17.431840+00:00. Duration: 41.796 s. Stop: tracking lost; initialize heading again.

Control stopped 32.140 s after start (tracking lost; initialize heading again); recording continued after the stop.

Pose at arming [x_m, y_m, heading_rad]: [-0.02530260164876824, -0.029154457033274864, 2.1513913294924873].

Initial pose and calibration: see session.json. Position is in EM meters; headings in radians; currents in amperes.

Files:
- ui.avi: complete main UI, including camera, TFW, controls and current table (MJPEG).
- video_timestamps.csv: captured/duplicated frame timing; video_complete.json: frame/drop/error counts.
- telemetry.csv: controller-rate pose, input, commanded/measured coil currents and freshness.
- events.jsonl: received pose, continuous heading, tracking status, commands, CAN feedback and image timestamps.
- camera/: original received JPEG previews.
- session.json: configuration, calibration, TFW provenance and initial-pose snapshot.

Counts: 52701 data events; 195 raw camera frames; 209 UI video frames.

Timing uses host UTC and a common host monotonic clock. Camera/pose/video are not hardware synchronized. Missing current feedback is blank/null, not measured zero.
