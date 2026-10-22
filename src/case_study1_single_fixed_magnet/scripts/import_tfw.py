#!/usr/bin/env python3
"""Import a checked fine CS1 TFW (single, gate or triangle), without executing source."""
import argparse
import ast
import hashlib
import json
import math
import operator
from pathlib import Path
import shutil
import tempfile

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE_JSON = 'revised_case_studies/results/cs1/tfw3d_fine/tfw3d_single.json'
MODEL_FILE = 'emns/workspace/fast.py'
WORKSPACE_FILE = 'cases/case_6_interaction_obstacle_avoidance/workspace_hybrid.py'
GENERATOR_FILE = 'revised_case_studies/cs1_cylinder_interaction/run_tfw3d_fine.py'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assignment(path, name):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == name for t in node.targets):
            return node.value
        if isinstance(node,ast.Assign) and isinstance(node.value,(ast.Tuple,ast.List)):
            for target in node.targets:
                if isinstance(target,(ast.Tuple,ast.List)):
                    for key,value in zip(target.elts,node.value.elts):
                        if isinstance(key,ast.Name) and key.id == name:
                            return value
    raise ValueError('Missing source model assignment: '+name)


def numeric(node, symbols=None):
    """Read numeric source constants, allowing only basic arithmetic and np.array."""
    if isinstance(node,ast.Constant) and type(node.value) in (int,float):
        return node.value
    if isinstance(node,(ast.List,ast.Tuple)):
        return [numeric(item,symbols) for item in node.elts]
    if isinstance(node,ast.Name) and symbols is not None and node.id in symbols:
        return symbols[node.id]
    if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name):
        if node.value.id == 'math' and node.attr == 'pi':
            return math.pi
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
        return -numeric(node.operand,symbols)
    operations = {ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
    if isinstance(node,ast.BinOp) and type(node.op) in operations:
        return operations[type(node.op)](numeric(node.left,symbols),numeric(node.right,symbols))
    if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id=='math'
            and node.func.attr=='sqrt' and len(node.args)==1 and not node.keywords):
        return math.sqrt(numeric(node.args[0],symbols))
    if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id == 'np'
            and node.func.attr == 'array' and len(node.args) == 1 and not node.keywords):
        return numeric(node.args[0],symbols)
    raise ValueError('Unsupported source model expression: '+ast.dump(node))


def source_layout(workspace_root, layout):
    layouts = assignment(workspace_root/WORKSPACE_FILE,'LAYOUTS')
    value = next(value for key,value in zip(layouts.keys,layouts.values)
                 if isinstance(key,ast.Constant) and key.value==layout)
    symbols = {}
    if layout=='triangle':symbols['_TRI_R'] = numeric(assignment(workspace_root/WORKSPACE_FILE,'_TRI_R'))
    return numeric(value,symbols)


def validate(source, workspace_root, model, layout='single'):
    if model.get('layout','single')!=layout:raise ValueError('Configuration layout does not match requested import')
    protocol = source['protocol']
    expected_task = ['By=0','Bz='+str(model['holding_field_T']),
                     'fy=0','fz=±'+str(model['task_force_N'])]
    if (protocol.get('layout') != layout or protocol.get('z_m') != model['z_m']
            or protocol.get('free_m') != model['free_moment_Am2']
            or protocol.get('task') != expected_task
            or protocol.get('current_A') != [-model['current_limit_A'],model['current_limit_A']]):
        raise ValueError('TFW protocol does not match the '+layout+' experiment')
    coils = numeric(assignment(workspace_root/MODEL_FILE,'params_list'))
    if not np.array_equal(coils,model['coil_parameters']):
        raise ValueError('Source coil model differs from ROS_CAN configuration')
    expected_fixed = ([[*m['xy_m'],m['heading_rad'],m['moment_Am2']] for m in model['fixed_magnets']]
                      if 'fixed_magnets' in model else
                      [[*model['fixed_xy_m'],model['fixed_heading_rad'],model['fixed_moment_Am2']]])
    if source_layout(workspace_root,layout) != expected_fixed:
        raise ValueError('Source fixed-magnet layout differs from ROS_CAN configuration')
    for name,key in [('Z_PLANE','z_m'),('FREE_M','free_moment_Am2'),
                     ('B_HOLD','holding_field_T'),('F_MAX','task_force_N'),('I_MAX','current_limit_A')]:
        if numeric(assignment(workspace_root/WORKSPACE_FILE,name)) != model[key]:
            raise ValueError('Source workspace constant differs: '+name)
    if model.get('interaction_radius_epsilon_m') != 1e-9:
        raise ValueError('Expected the source point-dipole interaction radius convention')
    boxes = np.asarray(source['in_workspace'],dtype=np.float64)
    if (boxes.ndim != 2 or boxes.shape[1] != 6 or not len(boxes)
            or not np.isfinite(boxes).all() or np.any(boxes[:,1::2] <= boxes[:,::2])):
        raise ValueError('Invalid source model-in boxes')
    if (np.min(boxes[:,:4]) < -.05-1e-12 or np.max(boxes[:,:4]) > .05+1e-12
            or np.min(boxes[:,4]) < 0 or np.max(boxes[:,5]) > 2*math.pi+1e-12):
        raise ValueError('Source TFW domain differs from the ROS experiment')
    return boxes


def import_workspace(workspace_root, package=PACKAGE, archive_root=None, layout='single'):
    if layout not in ('single','gate','triangle'):raise ValueError('Unknown layout: '+layout)
    workspace_root,package = Path(workspace_root).resolve(),Path(package).resolve()
    source_file = SOURCE_JSON.replace('tfw3d_single.json','tfw3d_'+layout+'.json')
    source_path = workspace_root/source_file
    source = json.loads(source_path.read_text())
    config_file = package/'config'/('experiment.json' if layout=='single' else 'layouts/'+layout+'.json')
    model = json.loads(config_file.read_text())['model']
    boxes = validate(source,workspace_root,model,layout)
    data = package/'data'/('' if layout=='single' else 'layouts/'+layout)
    data.mkdir(parents=True,exist_ok=True)
    asset_name = 'tfw_single_fixed.npz' if layout=='single' else 'tfw_'+layout+'.npz'
    current_meta = json.loads((data/'provenance.json').read_text()) if (data/'provenance.json').exists() else None
    previous = None if current_meta is None else {'source_workspace_file':current_meta['source_workspace_file'],
                'source_workspace_sha256':current_meta['source_workspace_sha256'],
                'inside_box_count':current_meta['inside_box_count']}
    metadata = {'source_workspace_file':source_file,'source_workspace_sha256':sha256(source_path),
                'asset_file':asset_name,
                'source_model_file':MODEL_FILE,'source_model_sha256':sha256(workspace_root/MODEL_FILE),
                'source_generator_file':GENERATOR_FILE,'source_generator_sha256':sha256(workspace_root/GENERATOR_FILE),
                'source_classifier_sha256':{name:sha256(workspace_root/name) for name in [
                    WORKSPACE_FILE,'emns/workspace/affine.py','hybrid_region/certificates.py']},
                'model':model,'protocol':source['protocol'],'inside_box_count':len(boxes),
                'source_box_counts':{key:len(source[key]) for key in
                                     ['in_workspace','out_of_workspace','undetermined']},
                'stopping_widths':{'x_m':source['eps'][0],'y_m':source['eps'][1],'heading_rad':source['eps'][2]},
                'source_volume_fraction':source.get('volume_fraction'),
                'qualification':'Fine corrected-dBy/dy model-in boxes. Conditional on the coil/dipole model; '
                                'custom affine backend is not outward-rounded throughout. No hardware invariance guarantee.'}
    with tempfile.TemporaryDirectory(prefix='cs1_tfw_import_') as temporary:
        staged = Path(temporary)/asset_name
        np.savez_compressed(str(staged),boxes=boxes)
        metadata['asset_sha256'] = sha256(staged)
        with np.load(str(staged),allow_pickle=False) as check:
            if not np.array_equal(check['boxes'],boxes):
                raise ValueError('Imported boxes differ from source')
        if current_meta is None:
            metadata['supersedes'] = None
        elif current_meta['source_workspace_sha256'] == metadata['source_workspace_sha256']:
            metadata['supersedes'] = current_meta.get('supersedes')
        else:
            # Keep the previous uncommitted data recoverable as well as Git history.
            archive_root = Path(archive_root or package.parents[1]/'archive/tfw')
            backup = archive_root/('cs1_'+('single_fixed' if layout=='single' else layout)+'_'+current_meta['source_workspace_sha256'][:12])
            backup.mkdir(parents=True,exist_ok=True)
            for name in [asset_name,'provenance.json']:
                existing = backup/name
                if existing.exists() and sha256(existing) != sha256(data/name):
                    raise ValueError('Archive already exists with different contents: '+str(existing))
                if not existing.exists():
                    shutil.copy2(str(data/name),str(existing))
            metadata['supersedes'] = previous
        # Stage locally so final replacement does not cross filesystems.
        pending = data/(asset_name+'.pending')
        pending_meta = data/'provenance.json.pending'
        pending_meta.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        shutil.copyfile(str(staged),str(pending))
        pending.replace(data/asset_name)
        pending_meta.replace(data/'provenance.json')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace_root',help='Path to Workspace_eMNS-main')
    parser.add_argument('--layout',choices=['single','gate','triangle'],default='single')
    args = parser.parse_args()
    metadata = import_workspace(args.workspace_root,layout=args.layout)
    print('Imported %d model-in boxes from %s' % (metadata['inside_box_count'],metadata['source_workspace_file']))
    print('Source SHA256:',metadata['source_workspace_sha256'])
