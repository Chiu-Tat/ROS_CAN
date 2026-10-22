"""Theoretical applied force shown in the UI matches the controller's model."""
import math
from pathlib import Path
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from cs1_experiment.core import load_config,dipole_field_gradient
from cs1_experiment.pose_hold import PoseHoldController
import controller_node


class AppliedForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.core = PoseHoldController(cls.config)
        cls.pose = cls.core.workspace.seed()

    def applied(self, currents):
        node = types.SimpleNamespace(pose=self.pose, core=self.core, config=self.config,
                                     previous=np.asarray(currents, float))
        return controller_node.ControllerNode.applied_model(node)

    def test_zero_current_is_fixed_magnet_only(self):
        value = self.applied(np.zeros(10))
        np.testing.assert_allclose(value['body_coils'], 0, atol=1e-15)
        np.testing.assert_allclose(value['body_total'], value['body_fixed'])
        self.assertGreater(np.linalg.norm(value['force_world_N']),1e-8)
        self.assertEqual(value['torque_coils_z_Nm'],0)
        self.assertEqual(value['torque_z_Nm'],value['torque_fixed_z_Nm'])

    def test_total_force_and_torque_match_direct_dipole_superposition(self):
        currents = np.linspace(-1.3,2.1,10)
        value = self.applied(currents)
        settings = self.config['model'];x,y,beta = self.pose
        position = [x,y,settings['z_m']]
        moment = settings['free_moment_Am2']*np.array([math.cos(beta),math.sin(beta),0])
        coil_field = np.zeros(3);coil_force = np.zeros(3)
        for amps,row in zip(currents,settings['coil_parameters']):
            field,gradient = dipole_field_gradient(position,row[3:],row[:3])
            coil_field += amps*field;coil_force += amps*(gradient.T@moment)
        angle = settings['fixed_heading_rad']
        fixed_moment = settings['fixed_moment_Am2']*np.array([math.cos(angle),math.sin(angle),0])
        fixed_field,gradient = dipole_field_gradient(position,
            list(settings['fixed_xy_m'])+[settings['z_m']],fixed_moment,
            settings['interaction_radius_epsilon_m'])
        fixed_force = gradient.T@moment
        np.testing.assert_allclose(value['force_world_N'],(coil_force+fixed_force)[:2],atol=1e-12)
        np.testing.assert_allclose(value['force_world_N'],
            np.array(value['force_coils_world_N'])+value['force_fixed_world_N'],atol=1e-12)
        self.assertAlmostEqual(value['torque_coils_z_Nm'],np.cross(moment,coil_field)[2])
        self.assertAlmostEqual(value['torque_fixed_z_Nm'],np.cross(moment,fixed_field)[2])
        self.assertAlmostEqual(value['torque_z_Nm'],np.cross(moment,coil_field+fixed_field)[2])
        self.assertAlmostEqual(value['torque_z_Nm'],value['torque_coils_z_Nm']+value['torque_fixed_z_Nm'])

    def test_allocated_forward_force_points_along_heading(self):
        target = np.array([0, .02, 0, .01])  # 10 mN forward, holding field
        currents, residual = self.core.model.allocate(self.pose, target, np.zeros(10))
        self.assertLess(residual, .001)
        value = self.applied(currents)
        np.testing.assert_allclose(value['body_total'], target, atol=2e-4)
        beta = self.pose[2]
        np.testing.assert_allclose(value['force_world_N'],
                                   [.01*math.cos(beta), .01*math.sin(beta)], atol=2e-5)
        self.assertAlmostEqual(value['torque_z_Nm'], 0, delta=1e-5)

    def test_no_pose_gives_no_force(self):
        node = types.SimpleNamespace(pose=None)
        self.assertIsNone(controller_node.ControllerNode.applied_model(node))

    def test_panel_text(self):
        import control_panel
        value = self.applied(np.zeros(10))
        text = control_panel.ControlPanel.format_applied_force(value)
        self.assertIn('THEORETICAL FORCE', text)
        self.assertIn('mN', text)
        self.assertIn('coils + fixed',text)
        self.assertIn('total torque',text)
        self.assertEqual(control_panel.ControlPanel.format_applied_force(None),
                         'Theoretical force: waiting for pose')


if __name__ == '__main__':
    unittest.main()
