"""Multi-fixed-magnet models, matching TFW assets and startup selection."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cs1_experiment.core import (PACKAGE,LAYOUT_LABELS,MagneticModel,Workspace,
                                 fixed_magnets,fixed_separation,load_config)
from main import launch_command


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {name:load_config(layout=name) for name in LAYOUT_LABELS}
        cls.workspaces = {name:Workspace(config['model']) for name,config in cls.configs.items()}

    def test_matching_datasets_counts_and_new_layout_arming_locks(self):
        for name,count in [('single',1),('gate',2),('triangle',3)]:
            config = self.configs[name];w = self.workspaces[name]
            self.assertEqual(len(fixed_magnets(config['model'])),count)
            self.assertEqual(w.provenance['protocol']['layout'],name)
            self.assertEqual(len(w.lower),w.provenance['inside_box_count'])
            self.assertTrue(w.permitted(w.seed(),config['control']))
            if name!='single':
                self.assertFalse(config['field_steer']['hardware_validated'])
                self.assertFalse(config['pose_hold']['hardware_validated'])

    def test_fixed_interaction_matches_independent_dipole_pair_formula(self):
        for name,config in self.configs.items():
            model = config['model'];pose = self.workspaces[name].seed()
            _,offset = MagneticModel(model).matrix(pose)
            beta = pose[2];forward = np.array([math.cos(beta),math.sin(beta),0.])
            lateral = np.array([-math.sin(beta),math.cos(beta),0.])
            free = model['free_moment_Am2']*forward
            force = np.zeros(3);field = np.zeros(3)
            for fixed in fixed_magnets(model):
                r = np.r_[pose[:2]-fixed['xy_m'],0.]
                radius = np.linalg.norm(r)+model['interaction_radius_epsilon_m']
                direction = r/radius
                theta = fixed['heading_rad']
                moment = fixed['moment_Am2']*np.array([math.cos(theta),math.sin(theta),0.])
                field += 1e-7/radius**3*(3*direction*(moment@direction)-moment)
                force += 3e-7/radius**4*((free@direction)*moment+(moment@direction)*free
                    +(free@moment)*direction-5*(free@direction)*(moment@direction)*direction)
            np.testing.assert_allclose(offset,[lateral@field,forward@field,lateral@force,forward@force],rtol=1e-10,atol=1e-12)
            self.assertAlmostEqual(model['free_moment_Am2']*offset[0],np.cross(free,field)[2])

    def test_clearance_applies_to_every_fixed_magnet(self):
        for name,config in self.configs.items():
            w = self.workspaces[name]
            with mock.patch.object(w,'covers',return_value=True):
                for magnet in fixed_magnets(config['model']):
                    pose = [*magnet['xy_m'],1.]
                    self.assertEqual(fixed_separation(pose,config['model']),0)
                    self.assertFalse(w.permitted(pose,config['control']))
                self.assertTrue(w.permitted([-.05,-.05,1.],config['control']))

    def test_tfw_from_another_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'differs'):
            Workspace(self.configs['gate']['model'],PACKAGE/'data/layouts/triangle')

    def test_startup_command_uses_selected_layout_and_never_arms(self):
        for layout in LAYOUT_LABELS:
            command = launch_command(layout,'example.json',simulate=True)
            self.assertIn('layout:='+layout,command)
            self.assertIn('hardware:=false',command)
            self.assertNotIn('arm:=true',command)
        with self.assertRaises(ValueError):launch_command('invalid','example.json')
        with self.assertRaises(ValueError):launch_command('gate','example.json',hardware=True,simulate=True)


if __name__=='__main__':unittest.main()


class CameraPortTests(unittest.TestCase):
    """Two OpenMV boards can be connected; ttyACM numbers follow plug order."""
    def test_every_layout_uses_the_stable_cs1_camera_id(self):
        from cs1_experiment.core import LAYOUT_LABELS, load_config
        ports = {load_config(layout=layout)['camera']['port'] for layout in LAYOUT_LABELS}
        self.assertEqual(len(ports), 1)
        port = ports.pop()
        self.assertTrue(port.startswith('/dev/serial/by-id/'), port)
        self.assertNotIn('ttyACM', port)

    def test_launcher_defers_to_config_port_unless_given(self):
        command = launch_command('gate', 'example.json', simulate=True)
        self.assertIn('port:=', command)
        self.assertIn('port:=/dev/x', launch_command('gate', 'example.json', simulate=True, port='/dev/x'))
