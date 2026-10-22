"""Hardware-independent regression tests for the experiment's physical safeguards."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cs1_experiment.core import (PACKAGE, MagneticModel, ManualController, Workspace,
                             calibrated_line, dipole_field_gradient, load_config, unwrap_axis)
from calibrate_camera import fit


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.core = ManualController(cls.config)

    def test_matches_independent_paper_implementation(self):
        fixtures = json.loads((PACKAGE/'tests/paper_reference.json').read_text())
        for row in fixtures:
            matrix,offset = self.core.model.matrix(row['pose'])
            np.testing.assert_allclose(np.column_stack([matrix,offset]),row['matrix'],rtol=1e-10,atol=1e-12)

    def test_dipole_gradient_against_finite_difference(self):
        point,source,moment = np.array([.02,-.03,.082]),np.array([.1,0,.07]),np.array([2.,-1.,3.])
        _,gradient = dipole_field_gradient(point,source,moment)
        for j in range(3):
            step = np.eye(3)[j]*1e-7
            numerical = (dipole_field_gradient(point+step,source,moment)[0]
                         -dipole_field_gradient(point-step,source,moment)[0])/2e-7
            np.testing.assert_allclose(gradient[:,j],numerical,rtol=1e-7,atol=1e-10)

    def test_full_task_extremes_at_seed_and_milliamp_rounding(self):
        pose = self.core.workspace.seed()
        for force in [-.1,0,.1]:
            current,error = self.core.model.allocate(pose,np.array([0,.02,0,force]),np.zeros(10))
            self.assertLess(error, .001)
            self.assertLessEqual(np.max(abs(current)),15)
            np.testing.assert_allclose(current*1000,np.round(current*1000))

    def test_tfw_configuration_mismatch_is_rejected(self):
        model = copy.deepcopy(self.config['model'])
        model['fixed_xy_m'][0] += .001
        with self.assertRaises(ValueError):
            Workspace(model)

    def test_box_union_and_uncovered_hole(self):
        workspace = Workspace.__new__(Workspace)
        workspace.lower = np.array([[0,0,0],[.5,0,0]],float)
        workspace.upper = np.array([[.5,1,1],[1,1,1]],float)
        self.assertTrue(workspace._covers(np.array([.1,.1,.1]),np.array([.9,.9,.9])))
        workspace.lower[1,0] = .51
        self.assertFalse(workspace._covers(np.array([.1,.1,.1]),np.array([.9,.9,.9])))
        # Even a gap between consecutive floats must not be hidden by a rounded midpoint.
        workspace.lower[1,0] = np.nextafter(.5,1)
        self.assertFalse(workspace._covers(np.array([.1,.1,.1]),np.array([.9,.9,.9])))

    def test_box_union_matches_independent_integer_grid_reference(self):
        workspace = Workspace.__new__(Workspace)
        rng = np.random.default_rng(42)
        for _ in range(40):
            workspace.lower = rng.integers(0,4,size=(12,3)).astype(float)
            workspace.upper = workspace.lower+rng.integers(1,4,size=(12,3))
            lo = rng.integers(0,4,size=3).astype(float)
            hi = lo+rng.integers(1,4,size=3)
            from itertools import product
            points = product(*(np.arange(a,b)+.5 for a,b in zip(lo,hi)))
            expected = all(np.any(np.all((workspace.lower<=p)&(p<=workspace.upper),axis=1)) for p in points)
            self.assertEqual(workspace._covers(lo,hi),expected)

    def test_periodic_heading_and_nonfinite_pose(self):
        pose = self.core.workspace.seed()
        for shift in [-2*math.pi,0,2*math.pi]:
            self.assertTrue(self.core.workspace.contains(pose+[0,0,shift]))
        self.assertFalse(self.core.workspace.contains([math.nan,0,0]))
        self.assertFalse(self.core.workspace.contains([0.2,0,0]))

    def test_margin_crossing_heading_seam(self):
        workspace = Workspace.__new__(Workspace)
        workspace.lower = np.array([[-1,-1,0],[-1,-1,6]],float)
        workspace.upper = np.array([[1,1,.3],[1,1,2*math.pi]],float)
        self.assertTrue(workspace.covers([0,0,0],.1,.1))
        workspace.upper[1,2] = 2*math.pi-.01
        self.assertFalse(workspace.covers([0,0,0],.1,.1))

    def test_ambiguous_line_unwrap_and_calibrated_axes(self):
        for degrees in [179,181,359,361,539]:
            truth = math.radians(degrees)
            resolved = unwrap_axis(truth % math.pi,truth-.03)
            self.assertAlmostEqual(resolved,truth)
        xy,axis = calibrated_line({'x':10,'y':20,'axis_rad':0},[[0,.001,0],[.001,0,0]])
        np.testing.assert_allclose(xy,[.02,.01])
        self.assertAlmostEqual(axis,math.pi/2)
        with self.assertRaises(ValueError):
            calibrated_line({'x':0,'y':0,'axis_rad':0},[[0,0,0],[0,0,0]])

    def test_boundary_block_and_collision_rejection(self):
        pose = self.core.workspace.seed()
        current,_ = self.core.model.allocate(pose,np.array([0,.02,0,0]),np.zeros(10))
        result,reason = self.core.command(pose,1,1,current,.1,speed_m_s=1)
        self.assertIsNotNone(result)
        self.assertIn('blocked',reason)
        result,_ = self.core.command([.02,0,math.pi/2],0,0,np.zeros(10),.1)
        self.assertIsNone(result)

    def test_camera_fit_and_degenerate_points(self):
        points = [[0,0,0,0],[100,0,.1,0],[0,100,0,.1],[100,100,.1,.1]]
        result = fit(points)
        np.testing.assert_allclose(result['affine'],[[.001,0,0],[0,.001,0]],atol=1e-12)
        with self.assertRaises(ValueError):
            fit([[k,k,k*.001,k*.001] for k in range(4)])

    def test_line_tracker_never_reinitializes_after_detection_loss(self):
        import threading
        import rospy
        from openmv_tracker import LineTracker
        import tempfile
        from tracking_fixture import tracker as make_tracker, frame
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        tracker = make_tracker(folder.name)
        self.addCleanup(tracker.close)
        tracker.config['tracking']['initialization_frames'] = 1
        tracker.config['tracking']['initialization_boundary_margin_rad'] = .001
        with mock.patch.object(rospy,'get_param',return_value=179), \
             mock.patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(1)):
            tracker.process(frame(0,179))
            self.assertIsNone(tracker.heading)
            self.assertTrue(tracker.initialize(None).success)
            tracker.process(frame(1,1))
            self.assertAlmostEqual(math.degrees(tracker.heading),181)
            tracker.last_valid_time -= tracker.config['tracking']['dropout_grace_s']+.05
            tracker.process(frame(2,lines=[]))
            self.assertIsNone(tracker.heading)
            tracker.process(frame(3,2))
            self.assertIsNone(tracker.heading)
            self.assertTrue(tracker.initialize(None).success)
            tracker.process(frame(4,3))
            self.assertAlmostEqual(math.degrees(tracker.heading),3)

    def test_hardware_rejects_simulated_feedback_before_opening_can(self):
        import controller_node
        def param(name, default=None):
            return True if name in ('~hardware','/cs1/simulated_feedback') else default
        with mock.patch.object(controller_node.rospy,'get_param',side_effect=param), \
             mock.patch.object(controller_node,'CurrentOutput') as output:
            with self.assertRaisesRegex(ValueError,'simulated feedback'):
                controller_node.ControllerNode()
            output.assert_not_called()


if __name__ == '__main__':
    unittest.main()
