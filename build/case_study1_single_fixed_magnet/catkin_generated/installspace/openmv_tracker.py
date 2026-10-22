#!/usr/bin/env python3
"""Calibrated line tracking with manually initialized directed heading."""
import csv
from datetime import datetime
import json
from pathlib import Path
import math
import threading
import time

import numpy as np
import rospy
import serial
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, Float64, String
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger, TriggerResponse

from cs1_experiment.core import PACKAGE, calibrated_line, fixed_separation, load_config, unwrap_axis, wrap
from cs1_experiment.tracking import DETECTOR, green_candidate, positive_y_heading
from cs1_experiment.camera_stream import JsonFrameBuffer, decode_preview


class LineTracker:
    def __init__(self):
        self.config = load_config(rospy.get_param('~config', str(PACKAGE/'config/experiment.json')))
        self.calibration_path = rospy.get_param('~calibration', str(PACKAGE/'config/camera_calibration.json'))
        with open(self.calibration_path) as stream:
            data = json.load(stream)
        self.affine = data['affine']
        self.image_size = data.get('image_size_px')
        if data.get('frame_id') != 'emns_world':
            raise ValueError('Calibration must map to emns_world in meters')
        calibrated_line({'x':0,'y':0,'axis_rad':0}, self.affine)
        self.results_dir = Path(rospy.get_param('~results_dir', str(PACKAGE/'results')))
        self.legacy_tracking_log = rospy.get_param('~legacy_tracking_log', False)
        self.lock = threading.RLock()
        self.raw = self.previous_xy = self.heading = self.acquisition_xy = None
        self.raw_time = self.last_valid_time = 0.0
        self.stable_frames = 0
        self.acquisition_axis = None
        self.last_sequence = None
        self.last_size = None
        self.session_log = self.session_writer = None
        self.initial_heading = None
        self.ser = None
        self.pub = rospy.Publisher('/cs1/pose', PoseStamped, queue_size=1)
        self.valid = rospy.Publisher('/cs1/tracking_valid', Bool, queue_size=1, latch=True)
        self.fps = rospy.Publisher('/cs1/fps', Float32, queue_size=1)
        self.heading_pub = rospy.Publisher('/cs1/heading_unwrapped', Float64, queue_size=1)
        self.initial_pose_pub = rospy.Publisher('/cs1/initial_pose', PoseStamped, queue_size=1, latch=True)
        self.initial_metadata_pub = rospy.Publisher('/cs1/initial_pose_metadata', String, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher('/cs1/tracking_status', String, queue_size=1, latch=True)
        self.image_pub = rospy.Publisher('/cs1/camera/image/compressed', CompressedImage, queue_size=1)
        self.service = rospy.Service('/cs1/initialize_heading', Trigger, self.initialize)
        self.field_heading, self.field_time = None, 0.0
        rospy.Subscriber('/cs1/field_heading', Float64, self.on_field_heading, queue_size=1)
        rospy.on_shutdown(self.close)
        self.invalidate('Place the green magnet with N in the +y half-plane, then initialize heading')

    FIELD_FRESH_S = 0.3

    def on_field_heading(self, msg):
        if math.isfinite(msg.data):
            with self.lock:
                self.field_heading, self.field_time = float(msg.data), time.monotonic()

    def fresh_field_heading(self, now):
        field = getattr(self, 'field_heading', None)
        if field is None or now-getattr(self, 'field_time', 0.0) > self.FIELD_FRESH_S:
            return None
        return field

    def status(self, reason):
        self.status_pub.publish(String(data=json.dumps({
            'reason': reason, 'valid': self.heading is not None and self.raw is not None,
            'stable_frames': self.stable_frames, 'size': self.last_size,
            'heading_unwrapped_rad': self.heading,
            'candidate_pose': None if self.raw is None else [float(self.raw[0][0]),
                float(self.raw[0][1]), float(self.raw[1]) % math.pi],
            # Why the last initialized track was dropped; stays visible after
            # the next frame overwrites 'reason'.
            'last_loss': getattr(self, 'last_loss', None)})))

    def invalidate(self, reason='Tracking lost; place N in the +y half-plane and initialize again'):
        if getattr(self, 'heading', None) is not None:
            self.last_loss = {'reason': reason, 'wall_time_s': time.time(),
                              'heading_deg': math.degrees(self.heading),
                              'xy_mm': None if self.previous_xy is None else
                                       [float(v)*1000 for v in self.previous_xy]}
            rospy.logwarn('Tracking reset: %s (last heading %.1f deg)', reason, math.degrees(self.heading))
        if self.session_log is not None:
            self.session_writer.writerow([time.time(), self.last_sequence, 0, '', '', '', '', reason])
            self.session_log.close()
            self.session_log = self.session_writer = None
        self.heading = self.previous_xy = self.raw = self.acquisition_xy = None
        self.raw_time = 0
        self.stable_frames = 0
        self.acquisition_axis = None
        self.last_size = None
        self.valid.publish(Bool(data=False))
        self.status(reason)

    def missing_detection(self, reason):
        # Keep the last polarity branch only across a bounded, short segmentation
        # dropout. Publish no pose during that gap, but do not publish
        # tracking_valid=False: the controller disarms on any False. Once the
        # grace period ends, invalidate() publishes False. dropout_grace_s must
        # stay below the controller's pose_timeout_s.
        if self.heading is None or time.monotonic()-self.last_valid_time > self.config['tracking']['dropout_grace_s']:
            self.invalidate(reason+'; known-polarity initialization required')
            return
        self.last_rejection = reason
        self.raw = self.acquisition_xy = self.acquisition_axis = None
        self.raw_time = 0
        self.stable_frames = 0
        if self.session_writer is not None:
            self.session_writer.writerow([time.time(), self.last_sequence, 0, '', '', '', '', 'temporary segmentation dropout'])
            self.session_log.flush()
        self.status('Temporary segmentation dropout: no valid pose; holding last polarity branch briefly')

    def make_pose(self, xy, heading):
        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = rospy.Time.now(), 'emns_world'
        pose.pose.position.x, pose.pose.position.y = map(float, xy)
        pose.pose.position.z = self.config['model']['z_m']
        pose.pose.orientation.z = math.sin(heading/2)
        pose.pose.orientation.w = math.cos(heading/2)
        return pose

    def log_pose(self, xy, heading, reason='tracking'):
        if self.session_writer is not None:
            self.session_writer.writerow([time.time(), self.last_sequence, 1, float(xy[0]),
                                          float(xy[1]), heading,
                                          (heading-self.initial_heading)/(2*math.pi), reason])
            self.session_log.flush()

    def initialize(self, _request):
        with self.lock:
            if self.raw is None or time.monotonic()-self.raw_time > self.config['control']['pose_timeout_s']:
                return TriggerResponse(False, 'Need one unambiguous fresh green cylinder detection')
            if self.stable_frames < self.config['tracking']['initialization_frames']:
                return TriggerResponse(False, 'Wait for three consecutive stable green detections')
            try:
                heading = positive_y_heading(self.raw[1],
                    self.config['tracking']['initialization_boundary_margin_rad'])
            except ValueError as exc:
                return TriggerResponse(False, str(exc))
            xy = self.raw[0].copy()
            pose = self.make_pose(xy, heading)
            # Save each initialization as a new immutable reference, including recalibrations.
            stamp = datetime.now().astimezone()
            token = stamp.strftime('%Y%m%d_%H%M%S_%f')
            metadata = {'created_at': stamp.isoformat(), 'frame_id': 'emns_world',
                        'layout':self.config['model'].get('layout','single'),
                        'position_m': [float(xy[0]), float(xy[1]), self.config['model']['z_m']],
                        'heading_unwrapped_rad': heading,
                        'initialization_rule': 'north_pole_in_positive_y_half_plane',
                        'initial_heading_measured_deg': math.degrees(heading),
                        'observed_axis_rad': self.raw[1], 'camera_sequence': self.last_sequence,
                        'calibration_file': str(self.calibration_path), 'affine': self.affine,
                        'detector': DETECTOR, 'measured_size': self.last_size,
                        'nominal_length_m': .011, 'nominal_diameter_m': .004,
                        'trajectory_file': 'tracking_%s.csv' % token}
            new_log = writer = None
            legacy = getattr(self,'legacy_tracking_log',True)
            if not legacy:metadata['trajectory_file']=None
            try:
                self.results_dir.mkdir(parents=True, exist_ok=True)
                if legacy:
                    new_log = open(str(self.results_dir/metadata['trajectory_file']), 'x', newline='')
                    writer = csv.writer(new_log)
                    writer.writerow(['wall_time_s','camera_sequence','valid','x_m','y_m',
                                     'heading_unwrapped_rad','turns_from_initial','reason'])
                reference_dir=self.results_dir/'initial_poses'
                reference_dir.mkdir(exist_ok=True)
                with open(str(reference_dir/('initial_pose_%s.json' % token)), 'x') as stream:
                    json.dump(metadata, stream, indent=2, allow_nan=False)
                    stream.write('\n')
            except (OSError, ValueError) as exc:
                if new_log is not None:
                    new_log.close()
                return TriggerResponse(False, 'Cannot save initial pose: '+str(exc))
            if self.session_log is not None:
                self.session_log.close()
            self.session_log, self.session_writer = new_log, writer
            self.initial_heading = self.heading = heading
            self.previous_xy = xy
            self.last_valid_time = time.monotonic()
            self.initial_pose_pub.publish(pose)
            if hasattr(self,'initial_metadata_pub'):
                self.initial_metadata_pub.publish(String(data=json.dumps(metadata)))
            self.log_pose(xy, heading, 'initial_pose')
            self.status('Initial pose saved; verify N direction before arming')
            return TriggerResponse(True, 'Initial pose saved; continuous heading starts at %.1f degrees' % math.degrees(heading))

    def process(self, frame):
        now = time.monotonic()
        if not isinstance(frame, dict):
            raise ValueError('Expected camera JSON object')
        sequence = frame.get('sequence')
        if not isinstance(sequence, int) or sequence == self.last_sequence:
            raise ValueError('Invalid/repeated camera sequence')
        if self.last_sequence is not None and sequence < self.last_sequence:
            self.invalidate('Camera restarted; initialize again')
        self.last_sequence = sequence
        fps = float(frame.get('fps', 0))
        if math.isfinite(fps):
            self.fps.publish(Float32(data=fps))
        self.publish_preview(frame)
        if frame.get('detector') != DETECTOR:
            self.invalidate('Upload current GREEN cylinder camera script')
            return
        if self.image_size and frame.get('image_size_px') != self.image_size:
            self.invalidate('Camera dimensions differ from calibration')
            return
        if frame.get('detection_overflow'):
            self.invalidate('Too many green objects; restrict ROI or remove clutter')
            return
        if self.heading is not None and now-self.last_valid_time > min(
                self.config['control']['pose_timeout_s'], self.config['tracking']['dropout_grace_s']):
            self.invalidate('No accepted detection for %.0f ms (last rejection: %s); initialize again'
                            % (1000*(now-self.last_valid_time), getattr(self, 'last_rejection', None) or 'no camera frame'))
        candidates, rejection = [], 'No green magnet of the expected size'
        for blob in frame.get('lines', []):
            try:
                candidate = green_candidate(blob, self.affine, self.config['tracking'])
            except (ValueError, KeyError, TypeError) as exc:
                rejection = str(exc)
                continue
            xy, axis, size = candidate
            distance = fixed_separation(xy,self.config['model'])
            if np.max(np.abs(xy)) > 0.05:
                rejection = 'Green magnet outside the +/-50 mm workspace'
            elif distance < self.config['control']['min_separation_m']:
                rejection = 'Green magnet closer than %.0f mm to the fixed magnet' % (
                    1000*self.config['control']['min_separation_m'])
            else:
                candidates.append(candidate)
        if self.previous_xy is not None:
            limit = min(self.config['control']['max_pose_jump_m'], self.config['tracking']['association_max_distance_m'])
            near = [c for c in candidates if np.linalg.norm(c[0]-self.previous_xy) <= limit]
            if candidates and not near:
                rejection = 'Magnet moved %.1f mm in one frame (limit %.1f mm)' % (
                    1000*min(np.linalg.norm(c[0]-self.previous_xy) for c in candidates), 1000*limit)
            candidates = near
        if len(candidates) > 1:
            self.invalidate('Multiple green magnets; identity ambiguous')
            return
        if not candidates:
            self.missing_detection(rejection)
            return
        xy, axis, self.last_size = candidates[0]
        if (self.acquisition_xy is None
                or np.linalg.norm(xy-self.acquisition_xy) > self.config['tracking']['initialization_max_shift_m']
                or abs(unwrap_axis(axis, self.acquisition_axis)-self.acquisition_axis)
                    > self.config['tracking']['initialization_max_axis_change_rad']):
            self.stable_frames = 1
        else:
            self.stable_frames += 1
        self.acquisition_xy, self.acquisition_axis = xy.copy(), axis
        self.raw, self.raw_time = (xy, axis), now
        field = self.fresh_field_heading(now)
        if field is not None:
            # While the controller applies its holding field, that field
            # defines the magnetic polarity (as in Case 6): N aligns with it,
            # so take the undirected axis branch nearest the field direction.
            # No frame-to-frame polarity test is needed, so an alignment flip
            # at startup does not reset tracking.
            heading = unwrap_axis(axis, field)
            self.heading, self.previous_xy, self.last_valid_time = heading, xy, now
            self.last_rejection = None
            self.valid.publish(Bool(data=True))
            self.pub.publish(self.make_pose(xy, heading))
            self.heading_pub.publish(Float64(data=heading))
            self.status('Tracking green magnet; polarity from the applied field')
            return
        if self.heading is None:
            self.valid.publish(Bool(data=False))
            self.status('Green magnet found; place N in the +y half-plane and Initialize heading')
            return
        heading = unwrap_axis(axis, self.heading)
        allowed = min(self.config['control']['max_heading_jump_rad'],
                      self.config['tracking']['max_heading_rate_rad_s']*(now-self.last_valid_time)
                      # The difference contains error from BOTH observations.
                      + 2*self.config['tracking']['heading_noise_allowance_rad'])
        if abs(heading-self.heading) > allowed:
            # Treat a single implausible angle as a bad detection, not a reset.
            # The frame is NOT accepted, so the polarity branch is unchanged;
            # the allowance grows with time since the last accepted frame (capped
            # well below 90 deg). Persisting beyond dropout_grace_s still resets.
            self.missing_detection('Angular jump too fast to preserve polarity (%.0f deg in %.0f ms, limit %.0f deg)'
                                   % (math.degrees(abs(heading-self.heading)), 1000*(now-self.last_valid_time), math.degrees(allowed)))
            return
        self.heading, self.previous_xy, self.last_valid_time = heading, xy, now
        self.last_rejection = None
        self.valid.publish(Bool(data=True))
        self.pub.publish(self.make_pose(xy, heading))
        self.heading_pub.publish(Float64(data=heading))
        self.log_pose(xy, heading)
        self.status('Tracking green magnet with continuous N-pole heading')

    def publish_preview(self, frame):
        if frame.get('preview_error'):
            rospy.logwarn_throttle(5,'OpenMV preview: %s',frame['preview_error'])
        try:
            jpeg = decode_preview(frame)
        except ValueError as exc:
            rospy.logwarn_throttle(5,'Preview rejected (pose processing continues): %s',exc)
            return
        if jpeg is not None:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()  # Host arrival time, not hardware-synchronized capture time.
            msg.header.frame_id = 'openmv_camera'
            msg.format,msg.data = 'jpeg',jpeg
            self.image_pub.publish(msg)

    def close(self):
        if self.session_log is not None:
            self.session_log.close()
            self.session_log = self.session_writer = None
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def run(self):
        camera = self.config['camera']
        packets = JsonFrameBuffer(skip_invalid_lines=True)
        last_complete = time.monotonic()
        while not rospy.is_shutdown():
            try:
                if self.ser is None:
                    self.ser = serial.Serial(rospy.get_param('~port', camera['port']),
                                             camera['baudrate'], timeout=0.1, exclusive=True)
                    self.ser.reset_input_buffer()
                    packets.clear()
                    last_complete = time.monotonic()
                    with self.lock:
                        self.last_sequence = None
                        self.invalidate()
                # Read chunks, not one byte at a time: JPEG lines are much larger
                # than the old pose-only packets. Preserve incomplete packets.
                chunk = self.ser.read(max(1,min(self.ser.in_waiting,65536)))
                frames = packets.feed(chunk)
                for _ in range(3):
                    if not self.ser.in_waiting:
                        break
                    frames.extend(packets.feed(self.ser.read(min(self.ser.in_waiting,65536))))
                with self.lock:
                    # A streaming camera can produce more bytes while we drain.
                    # Keep the bounded parser's partial packet for the next read;
                    # pending USB bytes do not imply stale or invalid tracking.
                    if not frames:
                        if time.monotonic()-last_complete > self.config['control']['pose_timeout_s']:
                            self.invalidate()
                        continue
                    last_complete = time.monotonic()
                    # Keep newest pose; retain the newest available video frame
                    # even if a subsequent pose-only packet arrived in this batch.
                    for older in reversed(frames[:-1]):
                        if 'jpeg_b64' in older and 'jpeg_b64' not in frames[-1]:
                            self.publish_preview(older)
                            break
                    self.process(frames[-1])
            except (OSError, serial.SerialException) as exc:
                with self.lock:
                    self.invalidate()
                self.close()
                rospy.logwarn_throttle(2, 'Camera reconnect: %s', exc)
                rospy.sleep(1)
            except (ValueError, KeyError, TypeError) as exc:
                with self.lock:
                    self.invalidate()
                packets.clear()
                rospy.logwarn_throttle(2, 'Invalid camera frame: %s', exc)


if __name__ == '__main__':
    rospy.init_node('cs1_openmv_tracker')
    LineTracker().run()
