"""Checked import, provenance, rollback data, and runtime integrity regression checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cs1_experiment.core import PACKAGE, Workspace, load_config
from import_tfw import (GENERATOR_FILE, MODEL_FILE, SOURCE_JSON, WORKSPACE_FILE,
                        import_workspace, sha256, validate)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace,self.package = self.root/'workspace',self.root/'package'
        self.model = load_config()['model']
        constants = [('Z_PLANE','z_m'),('FREE_M','free_moment_Am2'),
                     ('B_HOLD','holding_field_T'),('F_MAX','task_force_N')]
        source = '\n'.join('%s = %r' % (name,self.model[key]) for name,key in constants)
        source += '\nI_MIN, I_MAX = -15.0, 15.0\n'
        source += 'LAYOUTS = '+repr({'single':[[*self.model['fixed_xy_m'],
                         self.model['fixed_heading_rad'],self.model['fixed_moment_Am2']]]})+'\n'
        files = {WORKSPACE_FILE:source,MODEL_FILE:'params_list = '+repr(self.model['coil_parameters']),
                 GENERATOR_FILE:'# source generator fixture\n','emns/workspace/affine.py':'# affine fixture\n',
                 'hybrid_region/certificates.py':'# certificate fixture\n'}
        for name,text in files.items():
            path = self.workspace/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        self.data = {'protocol':{'layout':'single','z_m':.082,'free_m':.12,
                                'task':['By=0','Bz=0.02','fy=0','fz=±0.1'],'current_A':[-15.,15.]},
                     'in_workspace':[[-.03,-.02,-.02,-.01,1.,2.]],'out_of_workspace':[],
                     'undetermined':[],'eps':[.00078125,.00078125,.09817477042468103]}
        path = self.workspace/SOURCE_JSON;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(self.data))
        (self.package/'config').mkdir(parents=True)
        (self.package/'data').mkdir()
        (self.package/'config/experiment.json').write_text(json.dumps({'model':self.model}))
        np.savez_compressed(str(self.package/'data/tfw_single_fixed.npz'),boxes=np.array([[0,1,0,1,0,1]]))
        previous = {'source_workspace_file':'legacy.json','source_workspace_sha256':'a'*64,
                    'inside_box_count':1,'model':self.model}
        (self.package/'data/provenance.json').write_text(json.dumps(previous))

    def test_import_preserves_source_boxes_and_previous_data_and_is_repeatable(self):
        old = (self.package/'data/tfw_single_fixed.npz').read_bytes()
        result = import_workspace(self.workspace,self.package,self.root/'archive')
        with np.load(str(self.package/'data/tfw_single_fixed.npz')) as data:
            np.testing.assert_array_equal(data['boxes'],self.data['in_workspace'])
        backup = self.root/'archive'/('cs1_single_fixed_'+'a'*12)
        self.assertEqual((backup/'tfw_single_fixed.npz').read_bytes(),old)
        self.assertEqual(result['asset_sha256'],sha256(self.package/'data/tfw_single_fixed.npz'))
        self.assertEqual(len(Workspace(self.model,self.package/'data').lower),1)
        again = import_workspace(self.workspace,self.package,self.root/'archive')
        self.assertEqual(again['supersedes'],result['supersedes'])
        self.assertEqual(len(list((self.root/'archive').iterdir())),1)

    def test_wrong_task_layout_and_coil_model_are_rejected_before_replacement(self):
        old = sha256(self.package/'data/tfw_single_fixed.npz')
        for key,value in [('layout','gate'),('z_m',0),('current_A',[-10,10])]:
            modified = copy.deepcopy(self.data);modified['protocol'][key]=value
            with self.assertRaises(ValueError):
                validate(modified,self.workspace,self.model)
        (self.workspace/MODEL_FILE).write_text('params_list = [[0]*6]*10')
        with self.assertRaises(ValueError):
            import_workspace(self.workspace,self.package,self.root/'archive')
        self.assertEqual(sha256(self.package/'data/tfw_single_fixed.npz'),old)

    def test_corrupted_imported_asset_is_rejected(self):
        import_workspace(self.workspace,self.package,self.root/'archive')
        path = self.package/'data/tfw_single_fixed.npz'
        path.write_bytes(path.read_bytes()+b'corruption')
        with self.assertRaisesRegex(ValueError,'checksum'):
            Workspace(self.model,self.package/'data')

    def test_gate_and_triangle_import_check_every_fixed_magnet(self):
        path = self.workspace/WORKSPACE_FILE
        prefix = path.read_text().split('LAYOUTS =')[0]
        path.write_text(prefix+'''_TRI_R = .03
LAYOUTS = {
 'gate': ((0.,.028,0.,.1),(0.,-.028,math.pi,.1)),
 'triangle': ((0.,_TRI_R,0.,.075),
  (-_TRI_R*math.sqrt(3)/2,-_TRI_R/2,2*math.pi/3,.075),
  (_TRI_R*math.sqrt(3)/2,-_TRI_R/2,4*math.pi/3,.075))}
''')
        for name in ['gate','triangle']:
            config = load_config(layout=name)
            config_dir = self.package/'config/layouts';config_dir.mkdir(exist_ok=True)
            (config_dir/(name+'.json')).write_text(json.dumps(config))
            source = copy.deepcopy(self.data);source['protocol']['layout'] = name
            (self.workspace/SOURCE_JSON.replace('single.json',name+'.json')).write_text(json.dumps(source))
            result = import_workspace(self.workspace,self.package,layout=name)
            data_dir = self.package/'data/layouts'/name
            self.assertEqual(result['protocol']['layout'],name)
            self.assertEqual(len(Workspace(config['model'],data_dir).lower),1)
            changed = copy.deepcopy(config['model'])
            changed['fixed_magnets'][-1]['heading_rad'] += .1
            with self.assertRaisesRegex(ValueError,'fixed-magnet layout'):
                validate(source,self.workspace,changed,layout=name)

    def test_shipped_fine_dataset_matches_recorded_source_statistics(self):
        metadata = json.loads((PACKAGE/'data/provenance.json').read_text())
        workspace = Workspace(load_config()['model'])
        self.assertEqual(len(workspace.lower),metadata['inside_box_count'])
        volume = np.prod(workspace.upper-workspace.lower,axis=1).sum()/(.1*.1*2*np.pi)
        self.assertAlmostEqual(volume,metadata['source_volume_fraction']['in_workspace'],places=12)


if __name__ == '__main__':
    unittest.main()
