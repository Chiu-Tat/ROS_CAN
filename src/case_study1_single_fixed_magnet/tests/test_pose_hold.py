"""Pose-reference and bounded-startup tests. No hardware is opened."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cs1_experiment.core import load_config,wrap
from cs1_experiment.pose_hold import PoseHoldController


class PoseHoldTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.core = PoseHoldController(self.config)
        self.pose = self.core.workspace.seed()
        self.core.begin_hold(self.pose)

    def reach_hold(self):
        current = np.zeros(10)
        for _ in range(300):
            next_current,reason = self.core.command(self.pose,0,0,current,.1)
            self.assertIsNotNone(next_current,reason)
            self.assertLessEqual(np.max(np.abs(next_current-current)),.05000001)
            current = next_current
            if self.core.phase=='holding':return current
        self.fail('Startup did not settle')

    def test_neutral_holds_captured_pose_and_startup_is_slew_limited(self):
        current = self.reach_hold()
        reference = self.core.reference.copy()
        for _ in range(10):
            current,reason = self.core.command(self.pose,0,0,current,.1)
            np.testing.assert_allclose(self.core.reference,reference)
            self.assertEqual(reason,'holding captured pose')

    def test_startup_ignores_input_and_aborts_actual_displacement(self):
        current,reason = self.core.command(self.pose,1,1,np.zeros(10),.1)
        np.testing.assert_allclose(self.core.reference,self.pose)
        self.assertIn('inhibited',reason)
        escaped = self.pose.copy();escaped[0] += .0031
        current,reason = self.core.command(escaped,0,0,current,.1)
        self.assertIsNone(current);self.assertIn('displacement',reason)

    def test_intermediate_startup_currents_preserve_force_balance(self):
        matrix,offset = self.core.model.matrix(self.pose)
        bound = max(np.linalg.norm(offset[2:]),self.core.hold['max_force_N'])+.0001
        current = np.zeros(10)
        for _ in range(300):
            current,reason = self.core.command(self.pose,0,0,current,.1)
            self.assertIsNotNone(current,reason)
            force = (matrix@current+offset)[2:]
            self.assertLessEqual(np.linalg.norm(force),bound)
            if self.core.phase=='holding':break
        self.assertEqual(self.core.phase,'holding')

    def test_position_feedback_force_opposes_displacement_and_is_capped(self):
        current = self.reach_hold()
        moved = self.pose.copy();moved[0] += .002
        for _ in range(15):
            current,reason = self.core.command(moved,0,0,current,.1)
            self.assertIsNotNone(current,reason)
        beta = moved[2]
        lateral,forward = np.array([-math.sin(beta),math.cos(beta)]),np.array([math.cos(beta),math.sin(beta)])
        world_force = lateral*self.core.target[2]+forward*self.core.target[3]
        self.assertLess(world_force[0],0)
        self.assertLessEqual(np.linalg.norm(world_force),self.core.hold['max_force_N']+1e-12)

    def test_field_direction_stays_at_reference_when_measured_axis_jitters(self):
        current = self.reach_hold()
        measured = self.pose.copy();measured[2] += .1
        current,reason = self.core.command(measured,0,0,current,.1)
        self.assertIsNotNone(current,reason)
        relative_field = math.atan2(self.core.target[0],self.core.target[1])
        self.assertAlmostEqual(wrap(measured[2]+relative_field-self.pose[2]),0)

    def test_keys_give_direct_thrust_and_release_freezes_new_reference(self):
        current = self.reach_hold()
        current,reason = self.core.command(self.pose,1,.5,current,.1)
        self.assertIn('thrust 100 mN',reason)
        # Requested forward force = full-key thrust (+ small position correction).
        self.assertAlmostEqual(self.core.target[3],self.core.hold['drive_thrust_N'],delta=self.core.hold['max_force_N'])
        ref = self.core.reference.copy()
        self.assertGreater(ref[2],self.pose[2])
        for _ in range(5):
            current,reason = self.core.command(self.pose,0,0,current,.1)
            self.assertIsNotNone(current,reason)
            self.assertAlmostEqual(self.core.target[3],0,delta=self.core.hold['max_force_N'])
        np.testing.assert_allclose(self.core.reference,ref)

    def test_backward_key_reverses_thrust(self):
        current = self.reach_hold()
        current,reason = self.core.command(self.pose,-.5,0,current,.1)
        self.assertIsNotNone(current,reason)
        self.assertAlmostEqual(self.core.target[3],-.5*self.core.hold['drive_thrust_N'],delta=self.core.hold['max_force_N'])

    def test_target_follows_magnet_forward_but_keeps_lateral_correction(self):
        current = self.reach_hold()
        beta = self.pose[2]
        forward,lateral = np.array([math.cos(beta),math.sin(beta)]),np.array([-math.sin(beta),math.cos(beta)])
        moved = self.pose.copy();moved[:2] += .002*forward+.0015*lateral
        for _ in range(20):
            current,reason = self.core.command(moved,1,0,current,.1)
            self.assertIsNotNone(current,reason)
        offset = self.core.reference[:2]-moved[:2]
        self.assertLess(abs(forward@offset),.0002)       # follows along heading
        self.assertAlmostEqual(lateral@offset,-.0015,delta=.0003)  # still pulls back sideways

    def test_full_thrust_allocates_within_current_limit(self):
        current = self.reach_hold()
        for _ in range(80):  # current ramp is slew-limited
            current,reason = self.core.command(self.pose,1,0,current,.1)
            self.assertIsNotNone(current,reason)
        matrix,offset = self.core.model.matrix(self.pose)
        achieved = matrix@current+offset
        self.assertAlmostEqual(achieved[3],self.core.target[3],delta=.002)
        self.assertLessEqual(np.max(np.abs(current)),15)

    def test_thrust_above_certified_task_is_rejected(self):
        config = copy.deepcopy(self.config);config['pose_hold']['drive_thrust_N'] = .2
        with self.assertRaises(ValueError):PoseHoldController(config)

    def test_startup_settles_when_friction_holds_magnet_off_reference(self):
        # Bench case: magnet slid 1.8 mm at startup, then stayed put (friction),
        # with a few degrees of camera heading noise. Startup must still finish
        # and re-capture the hold target where the magnet rests.
        rest = self.pose.copy();rest[0] -= .0018
        current = np.zeros(10)
        for k in range(300):
            measured = rest.copy();measured[2] += math.radians(3)*(1 if k%3==0 else -1 if k%3==1 else 0)
            current,reason = self.core.command(measured,0,0,current,.1)
            self.assertIsNotNone(current,reason)
            if self.core.phase=='holding':break
        self.assertEqual(self.core.phase,'holding')
        self.assertLess(np.linalg.norm(self.core.reference[:2]-rest[:2]),.0003)

    def test_large_pose_error_and_invalid_gains_stop(self):
        current = self.reach_hold()
        moved = self.pose.copy();moved[0] += .005
        current,reason = self.core.command(moved,0,0,current,.1)
        self.assertIsNone(current);self.assertIn('error exceeded',reason)
        config = copy.deepcopy(self.config);config['pose_hold']['max_force_N'] = -1
        with self.assertRaises(ValueError):PoseHoldController(config)


if __name__=='__main__':unittest.main()
