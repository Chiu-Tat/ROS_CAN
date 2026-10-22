"""Metric identity checks and polarity continuity over complete revolutions."""
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import rospy
from tracking_fixture import blob, frame, tracker
from cs1_experiment.tracking import green_candidate


class GreenTrackingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.tracker = tracker(self.folder.name)
        self.clock = 100.0
        self.sequence = 0
        self.patches = [mock.patch('openmv_tracker.time.monotonic', side_effect=lambda:self.clock),
                        mock.patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(1)),
                        mock.patch.object(rospy, 'get_param', return_value=90)]
        for patch in self.patches: patch.start()

    def tearDown(self):
        self.tracker.close()
        for patch in self.patches: patch.stop()
        self.folder.cleanup()

    def observe(self, degrees=90, lines=None):
        self.clock += .05
        self.sequence += 1
        self.tracker.process(frame(self.sequence, degrees, lines))

    def initialize(self):
        for _ in range(3): self.observe()
        self.assertTrue(self.tracker.initialize(None).success)

    def test_size_and_green_identity_at_every_heading(self):
        for degrees in range(0,360,10):
            xy, angle, size = green_candidate(blob(degrees), self.tracker.affine,
                                               self.tracker.config['tracking'])
            self.assertAlmostEqual(size['length_m'], .011)
            self.assertAlmostEqual(size['diameter_m'], .004)
        for invalid in [blob(kind='dark'), blob(length_px=30), blob(width_px=12),
                        blob(length_px=5), blob(width_px=.5)]:
            with self.assertRaises(ValueError):
                green_candidate(invalid, self.tracker.affine, self.tracker.config['tracking'])
        invalid = blob(); invalid['corners_px'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            green_candidate(invalid, self.tracker.affine, self.tracker.config['tracking'])

    def test_initial_pose_requires_stable_observations_and_is_saved(self):
        self.observe()
        self.assertFalse(self.tracker.initialize(None).success)
        self.observe(); self.observe()
        self.assertTrue(self.tracker.initialize(None).success)
        files = list(Path(self.folder.name).glob('initial_poses/initial_pose_*.json'))
        self.assertEqual(len(files),1)
        initial = json.loads(files[0].read_text())
        self.assertAlmostEqual(initial['heading_unwrapped_rad'], math.pi/2)
        self.assertEqual(initial['initial_heading_measured_deg'], 90)
        self.assertEqual(initial['initialization_rule'], 'north_pole_in_positive_y_half_plane')
        np.testing.assert_allclose(initial['position_m'], [-.03,-.03,.082])
        self.assertTrue((Path(self.folder.name)/initial['trajectory_file']).is_file())
        self.tracker.initial_pose_pub.publish.assert_called_once()

    def test_initial_heading_is_measured_anywhere_in_positive_y_half_plane(self):
        for angle in [15, 35, 90, 130, 165, -30, 210]:
            self.tracker.invalidate()
            for _ in range(3): self.observe(angle)
            self.assertTrue(self.tracker.initialize(None).success)
            self.assertAlmostEqual(math.degrees(self.tracker.heading), angle % 180)
        for angle in [0, 1, 179, 180]:
            self.tracker.invalidate()
            for _ in range(3): self.observe(angle)
            self.assertFalse(self.tracker.initialize(None).success)
            self.assertIsNone(self.tracker.heading)

    def test_complete_revolutions_both_directions_without_pi_flip(self):
        self.initialize()
        for degrees in range(100,811,10):
            self.observe(degrees)
            self.assertAlmostEqual(math.degrees(self.tracker.heading), degrees)
        for degrees in range(800,-271,-10):
            self.observe(degrees)
            self.assertAlmostEqual(math.degrees(self.tracker.heading), degrees)
        self.assertAlmostEqual(self.tracker.heading_pub.publish.call_args[0][0].data, -1.5*math.pi)
        trajectory = next(Path(self.folder.name).glob('tracking_*.csv'))
        with trajectory.open() as stream: rows = list(csv.DictReader(stream))
        self.assertAlmostEqual(float(rows[-1]['turns_from_initial']), -1)
        self.assertAlmostEqual(float(rows[-1]['heading_unwrapped_rad']), -1.5*math.pi)

    def test_loss_or_ambiguous_identity_requires_new_initialization(self):
        self.initialize(); self.observe(100)
        self.clock += self.tracker.config['tracking']['dropout_grace_s']+.03
        self.observe(lines=[])
        self.assertIsNone(self.tracker.heading)
        for _ in range(3): self.observe(90)
        self.assertIsNone(self.tracker.heading)
        self.assertTrue(self.tracker.initialize(None).success)
        self.observe(lines=[blob(90),blob(90,center=(24,20))])
        self.assertIsNone(self.tracker.heading)
        self.assertFalse(self.tracker.valid.publish.call_args[0][0].data)

    def test_short_segmentation_dropout_preserves_branch_without_publishing_pose(self):
        self.initialize(); self.observe(100)
        published = self.tracker.pub.publish.call_count
        valid_published = self.tracker.valid.publish.call_count
        self.observe(lines=[])
        self.assertAlmostEqual(math.degrees(self.tracker.heading), 100)
        self.assertIsNone(self.tracker.raw)
        # A short dropout publishes neither a pose nor tracking_valid=False,
        # so the controller is not disarmed by one missed frame.
        self.assertEqual(self.tracker.valid.publish.call_count, valid_published)
        self.assertEqual(self.tracker.pub.publish.call_count, published)
        self.observe(110)
        self.assertAlmostEqual(math.degrees(self.tracker.heading), 110)
        self.assertTrue(self.tracker.valid.publish.call_args[0][0].data)
        self.clock += self.tracker.config['tracking']['dropout_grace_s']+.03
        self.observe(120)
        self.assertIsNone(self.tracker.heading)

    def test_fast_rotation_gap_reset_and_wrong_protocol_invalidate(self):
        self.initialize()
        # A 60 deg jump is rejected without accepting the frame or publishing a
        # pose; only a jump that persists past the dropout grace resets.
        published = self.tracker.pub.publish.call_count
        self.observe(150)
        self.assertAlmostEqual(math.degrees(self.tracker.heading), 90)
        self.assertEqual(self.tracker.pub.publish.call_count, published)
        for _ in range(6): self.observe(150)
        self.assertIsNone(self.tracker.heading)
        self.assertIn('Angular jump', self.tracker.last_loss['reason'])
        for _ in range(3): self.observe()
        self.assertTrue(self.tracker.initialize(None).success)
        self.clock += .5; self.observe()
        self.assertIsNone(self.tracker.heading)
        for _ in range(3): self.observe()
        self.assertTrue(self.tracker.initialize(None).success)
        self.tracker.process(frame(0))
        self.assertIsNone(self.tracker.heading)
        self.tracker.process({'sequence':1, 'lines':[blob()]})
        self.assertIsNone(self.tracker.raw)

    def test_stationary_angle_noise_between_two_frames_does_not_lose_polarity(self):
        self.initialize()
        for angle in [84,96,84,96,84,96]:
            self.clock += .01
            self.sequence += 1
            self.tracker.process(frame(self.sequence,angle))
            self.assertIsNotNone(self.tracker.heading)
            self.assertTrue(self.tracker.valid.publish.call_args[0][0].data)
        # The hard jump limit and branch protection remain active: a single
        # outlier frame is skipped (heading unchanged), then tracking resumes.
        heading = self.tracker.heading
        self.clock += .01
        self.sequence += 1
        self.tracker.process(frame(self.sequence,150))
        self.assertEqual(self.tracker.heading, heading)
        self.clock += .01
        self.sequence += 1
        self.tracker.process(frame(self.sequence,90))
        self.assertAlmostEqual(math.degrees(self.tracker.heading), 90)
        self.assertTrue(self.tracker.valid.publish.call_args[0][0].data)
        # A jump that persists beyond the grace period still resets.
        for _ in range(8):
            self.clock += .05
            self.sequence += 1
            self.tracker.process(frame(self.sequence,150))
        self.assertIsNone(self.tracker.heading)

    def test_dimension_mismatch_and_overflow_are_rejected(self):
        for key, value in [('image_size_px',[640,480]),('detection_overflow',True)]:
            data = frame(self.sequence+1)
            self.sequence += 1
            data[key] = value
            self.tracker.process(data)
            self.assertIsNone(self.tracker.raw)


if __name__ == '__main__':
    unittest.main()
