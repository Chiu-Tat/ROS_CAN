"""Field-led (Case 6 style) controller: polarity from the field, push along axis."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from cs1_experiment.core import load_config, wrap
from cs1_experiment.field_steer import FieldSteerController

START = np.array([-.0321, -.0413, math.radians(89.6)])


class FieldSteerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.core = FieldSteerController(self.config)
        self.core.begin_hold(START)

    def plant(self, pose, currents, steps=5):
        for _ in range(steps):
            matrix, offset = self.core.model.matrix(pose)
            by, bz, fy, fz = matrix@currents+offset
            b = pose[2]
            forward, lateral = np.array([math.cos(b), math.sin(b)]), np.array([-math.sin(b), math.cos(b)])
            pose[:2] += (forward*fz+lateral*fy)/5*.02
            pose[2] += wrap(math.atan2(by, bz))*.02/.15

    def run_plan(self, pose, plan):
        current = np.zeros(10)
        for drive, turn, seconds in plan:
            for _ in range(int(round(seconds/.1))):
                camera = np.array([pose[0], pose[1], pose[2] % math.pi])  # undirected axis
                current, reason = self.core.command(camera, drive, turn, current, .1)
                self.assertIsNotNone(current, reason)
                self.assertLessEqual(np.max(np.abs(current)), 15)
                self.plant(pose, current)
        return current

    def test_aligns_then_holds_with_zero_in_plane_force(self):
        pose = START.copy()
        current = self.run_plan(pose, [(0, 0, 3)])
        self.assertEqual(self.core.phase, 'holding')
        matrix, offset = self.core.model.matrix(pose)
        total = matrix@current+offset
        self.assertLess(np.max(np.abs(total[2:])), .001)        # no in-plane force
        self.assertAlmostEqual(math.hypot(total[0], total[1]), .02, delta=.001)

    def test_reversed_magnet_flips_to_field_instead_of_stopping(self):
        pose = START.copy();pose[2] += math.pi                 # N placed toward -y
        self.run_plan(pose, [(0, 0, 3)])
        self.assertEqual(self.core.phase, 'holding')
        self.assertLess(abs(wrap(pose[2]-START[2])), math.radians(5))

    def test_input_inhibited_while_aligning(self):
        current, reason = self.core.command(START, 1, 1, np.zeros(10), .1)
        self.assertIn('aligning', reason)
        self.assertEqual(self.core.target[3], 0)
        self.assertAlmostEqual(self.core.field_heading, START[2])

    def test_drive_moves_target_at_capped_speed_and_release_stops(self):
        pose = START.copy()
        self.run_plan(pose, [(0, 0, 3)])
        start = pose.copy()
        self.run_plan(pose, [(1, 0, 2)])
        self.assertLessEqual(abs(self.core.target[2]), self.core.hold['max_lateral_force_N'])
        self.assertLessEqual(abs(self.core.target[3]), self.core.hold['max_axis_force_N'])
        forward = np.array([math.cos(start[2]), math.sin(start[2])])
        travelled = forward@(pose[:2]-start[:2])
        self.assertGreater(travelled, .001)
        self.assertLess(travelled, 2*self.core.hold['max_speed_m_s']+self.core.hold['max_target_lead_m'])
        released = pose.copy()
        self.run_plan(pose, [(0, 0, 3)])
        self.assertLess(np.linalg.norm(pose[:2]-released[:2]), .001)   # brakes and holds
        heading = self.core.field_heading
        self.run_plan(pose, [(0, 1, 1)])
        self.assertAlmostEqual(self.core.field_heading-heading,
                               self.config['control']['turn_rate_rad_s']*1.0, delta=1e-9)

    def test_sideways_bias_is_held_by_lateral_integral(self):
        pose = START.copy()
        current = self.run_plan(pose, [(0, 0, 3)])
        hold = pose[:2].copy()
        lateral = np.array([-math.sin(pose[2]), math.cos(pose[2])])
        for _ in range(100):  # 10 s with an unmodeled 1 mN sideways push
            camera = np.array([pose[0], pose[1], pose[2] % math.pi])
            current, reason = self.core.command(camera, 0, 0, current, .1)
            self.assertIsNotNone(current, reason)
            for _ in range(5):
                matrix, offset = self.core.model.matrix(pose)
                fy, fz = (matrix@current+offset)[2:]
                pose[:2] += (lateral*(fy+.001)+np.array([math.cos(pose[2]), math.sin(pose[2])])*fz)/5*.02
        self.assertLess(abs(lateral@(pose[:2]-hold)), .0015)

    def test_holding_does_not_wind_up_breakaway_force(self):
        pose = START.copy()
        current = self.run_plan(pose, [(0, 0, 3)])
        stuck = pose.copy();stuck[:2] -= .0008*np.array([math.cos(pose[2]), math.sin(pose[2])])
        for _ in range(50):   # 0.8 mm inside the deadband, friction holds it
            current, reason = self.core.command(np.array([stuck[0], stuck[1], stuck[2] % math.pi]), 0, 0, current, .1)
        self.assertEqual(self.core.integral, 0)
        self.assertLess(abs(self.core.target[3]), .001)

    def test_slip_ahead_while_driving_is_not_pulled_back(self):
        pose = START.copy()
        current = self.run_plan(pose, [(0, 0, 3)])
        forward = np.array([math.cos(pose[2]), math.sin(pose[2])])
        slipped = pose.copy()
        for k in range(15):   # magnet lurches 4 mm ahead, then sits there
            if k == 3:slipped[:2] += .004*forward
            current, reason = self.core.command(np.array([slipped[0], slipped[1], slipped[2] % math.pi]), 1, 0, current, .1)
            self.assertIsNotNone(current, reason)
            lead = forward@(self.core.target_xy-self.core.filtered_xy)
            self.assertGreaterEqual(lead, -1e-12)          # target never behind the magnet
        # After release it coasts (no spring) and then holds where it stopped.
        for _ in range(10):
            current, reason = self.core.command(np.array([slipped[0], slipped[1], slipped[2] % math.pi]), 0, 0, current, .1)
        self.assertLess(abs(forward@(self.core.target_xy-slipped[:2])), .0006)
        self.assertLessEqual(abs(self.core.target[3]), self.core.hold['max_hold_force_N'])

    def test_magnet_not_following_field_stops(self):
        pose = START.copy()
        current = self.run_plan(pose, [(0, 0, 3)])
        stuck = START.copy();stuck[2] += math.radians(70)
        current, reason = self.core.command(stuck, 0, 0, current, .1)
        self.assertIsNone(current);self.assertIn('no longer follows', reason)

    def test_alignment_timeout_stops(self):
        current = np.zeros(10)
        wrong = START.copy();wrong[2] += math.radians(45)    # never aligns
        for _ in range(200):
            current, reason = self.core.command(wrong, 0, 0, current, .1)
            if current is None:break
        self.assertIsNone(current);self.assertIn('did not align', reason)

    def test_invalid_settings_rejected(self):
        config = copy.deepcopy(self.config);config['field_steer']['max_axis_force_N'] = .2
        with self.assertRaises(ValueError):FieldSteerController(config)
        config = copy.deepcopy(self.config);config['field_steer']['hardware_validated'] = 1
        with self.assertRaises(ValueError):FieldSteerController(config)


if __name__ == '__main__':
    unittest.main()
