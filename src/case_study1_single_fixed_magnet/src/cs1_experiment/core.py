#!/usr/bin/env python3
"""ROS-independent paper model, exact box-union queries, and current allocation."""
import json
import math
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear


PACKAGE = Path(__file__).resolve().parents[2]
if not (PACKAGE / 'package.xml').is_file():
    import rospkg
    PACKAGE = Path(rospkg.RosPack().get_path('case_study1_single_fixed_magnet'))


LAYOUT_LABELS = {'single':'Single · 1 fixed magnet','gate':'Gate · 2 fixed magnets',
                 'triangle':'Triangle · 3 fixed magnets'}


def config_path(layout='single', package=PACKAGE):
    if layout not in LAYOUT_LABELS:raise ValueError('Unknown magnet layout: '+str(layout))
    return Path(package)/'config'/('experiment.json' if layout=='single' else 'layouts/'+layout+'.json')


def load_config(path=None, layout='single'):
    with open(path or config_path(layout)) as stream:
        return json.load(stream)


def fixed_magnets(settings):
    """Canonical fixed-dipole list, retaining the original single-case schema."""
    magnets = settings.get('fixed_magnets')
    if magnets is None:
        magnets = [{'xy_m':settings['fixed_xy_m'],'heading_rad':settings['fixed_heading_rad'],
                    'moment_Am2':settings['fixed_moment_Am2']}]
    if not magnets:raise ValueError('At least one fixed magnet is required')
    for magnet in magnets:
        if (len(magnet['xy_m'])!=2 or not np.isfinite([*magnet['xy_m'],magnet['heading_rad'],magnet['moment_Am2']]).all()
                or magnet['moment_Am2']<=0):raise ValueError('Invalid fixed-magnet pose or moment')
    return magnets


def fixed_separation(point, settings):
    return min(float(np.linalg.norm(np.asarray(point)[:2]-magnet['xy_m']))
               for magnet in fixed_magnets(settings))


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def unwrap_axis(axis, reference):
    """Choose the directed heading nearest reference from an undirected line."""
    return reference + (axis - reference + math.pi / 2) % math.pi - math.pi / 2


def calibrated_line(blob, affine):
    """Transform the centroid AND axis with the calibrated pixel-to-world map."""
    transform = np.asarray(affine, float)
    if transform.shape != (2, 3) or not np.isfinite(transform).all():
        raise ValueError('Camera affine must be a finite 2x3 pixel-to-meter map')
    if abs(np.linalg.det(transform[:, :2])) < 1e-15:
        raise ValueError('Camera calibration is singular')
    values = np.array([blob['x'], blob['y'], blob['axis_rad']], float)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite camera detection')
    xy = transform @ np.r_[values[:2], 1.0]
    direction = transform[:, :2] @ np.array([math.cos(values[2]), math.sin(values[2])])
    return xy, math.atan2(direction[1], direction[0])


def dipole_field_gradient(point, source, moment, radius_epsilon=0.0):
    """Analytic B [T] and dB_i/dx_j [T/m] for an SI point dipole."""
    r = np.asarray(point, float) - np.asarray(source, float)
    moment = np.asarray(moment, float)
    norm = np.linalg.norm(r)
    if norm < 1e-6:
        raise ValueError('Dipole singularity')
    # The archived interaction model adds 1 nm to its radius in both formulas.
    norm += radius_epsilon
    dot = float(moment @ r)
    field = 1e-7 * (3 * r * dot / norm**5 - moment / norm**3)
    gradient = 3e-7 * ((np.outer(r, moment) + np.outer(moment, r)
                       + dot * np.eye(3)) / norm**5
                      - 5 * dot * np.outer(r, r) / norm**7)
    return field, gradient


class MagneticModel:
    def __init__(self, settings):
        self.settings = settings
        self.coils = np.asarray(settings['coil_parameters'], float)
        if self.coils.shape != (10, 6) or not np.isfinite(self.coils).all():
            raise ValueError('Expected ten finite [mx,my,mz,x,y,z] coil parameters')
        self.row_scale = np.array([settings['holding_field_T']] * 2
                                  + [settings['task_force_N']] * 2)
        self.limit = float(settings['current_limit_A'])
        if self.limit <= 0 or np.any(self.row_scale <= 0):
            raise ValueError('Field, force and current limits must be positive')

    def matrix(self, pose):
        x, y, beta = pose
        s = self.settings
        point = [x, y, s['z_m']]
        forward = np.array([math.cos(beta), math.sin(beta), 0.0])
        lateral = np.array([-math.sin(beta), math.cos(beta), 0.0])
        rotation = np.array([lateral, forward])
        free_moment = s['free_moment_Am2'] * forward
        columns = []
        for coil in self.coils:
            field, gradient = dipole_field_gradient(point, coil[3:], coil[:3])
            columns.append(np.r_[rotation @ field, rotation @ (gradient.T @ free_moment)])
        offset = np.zeros(4)
        for magnet in fixed_magnets(s):
            fb = magnet['heading_rad']
            fixed_moment = magnet['moment_Am2'] * np.array([math.cos(fb),math.sin(fb),0.])
            field,gradient = dipole_field_gradient(point,list(magnet['xy_m'])+[s['z_m']],
                fixed_moment,s['interaction_radius_epsilon_m'])
            offset += np.r_[rotation@field,rotation@(gradient.T@free_moment)]
        return np.column_stack(columns), offset

    def allocate(self, pose, target, previous, step_limit=None):
        matrix, offset = self.matrix(pose)
        lower, upper = np.full(10, -self.limit), np.full(10, self.limit)
        if step_limit is not None:
            lower = np.maximum(lower, previous - step_limit)
            upper = np.minimum(upper, previous + step_limit)
        result = lsq_linear(matrix / self.row_scale[:, None],
                            (target - offset) / self.row_scale,
                            bounds=(lower, upper), tol=1e-10, max_iter=100)
        # Validate the actual rounded CAN commands, not just floating-point I.
        currents = np.round(result.x * 1000) / 1000
        residual = float(np.max(np.abs((matrix @ currents + offset - target) / self.row_scale)))
        return currents, residual


class Workspace:
    """Membership uses original 3-D boxes, never an expanded display raster.

    covers() partitions the requested uncertainty box at original box faces
    and checks complete cell containment. Internal box faces add no gaps.
    """
    def __init__(self, settings, data_dir=None):
        layout = settings.get('layout','single')
        if layout not in LAYOUT_LABELS:raise ValueError('Unknown TFW layout: '+str(layout))
        data_dir = Path(data_dir or PACKAGE/'data'/('' if layout=='single' else 'layouts/'+layout))
        metadata = json.loads((data_dir / 'provenance.json').read_text())
        if metadata.get('protocol',{}).get('layout','single')!=layout:
            raise ValueError('TFW protocol differs from selected layout')
        self.provenance = metadata
        if settings != metadata['model']:
            raise ValueError('Model differs from imported TFW. Recompute/import matching TFW first.')
        asset = data_dir/metadata.get('asset_file','tfw_single_fixed.npz')
        if metadata.get('asset_sha256') and hashlib.sha256(asset.read_bytes()).hexdigest() != metadata['asset_sha256']:
            raise ValueError('TFW asset checksum differs from provenance; reimport the matching dataset')
        with np.load(str(asset), allow_pickle=False) as data:
            boxes = data['boxes']
        if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
            raise ValueError('Invalid TFW boxes')
        if len(boxes) != metadata['inside_box_count']:
            raise ValueError('TFW box count differs from provenance')
        self.lower, self.upper = boxes[:, ::2], boxes[:, 1::2]
        if np.any(self.upper <= self.lower):
            raise ValueError('Degenerate TFW box')
        self.settings = settings
        self.domain_lower = self.lower.min(axis=0)
        self.domain_upper = self.upper.max(axis=0)

    def contains(self, pose):
        pose = np.asarray(pose, float).copy()
        if pose.shape != (3,) or not np.isfinite(pose).all():
            return False
        pose[2] %= 2 * math.pi
        return bool(np.any(np.all((self.lower <= pose) & (pose <= self.upper), axis=1)))

    def _covers(self, lower, upper):
        domain_lower = getattr(self,'domain_lower',None)
        domain_upper = getattr(self,'domain_upper',None)
        if domain_lower is None:
            domain_lower,domain_upper = self.lower.min(axis=0),self.upper.max(axis=0)
        if np.any(lower < domain_lower) or np.any(upper > domain_upper):
            return False
        selected = np.all((self.upper >= lower) & (self.lower <= upper), axis=1)
        lo,hi = self.lower[selected],self.upper[selected]
        if not len(lo):
            return False
        # Exact faces, including adjacent floating-point values, remain distinct.
        # Check cell endpoints rather than rounded midpoints, which could hide a gap.
        starts,ends = [],[]
        for axis in range(3):
            edges = np.unique(np.clip(np.r_[lower[axis],upper[axis],lo[:,axis],hi[:,axis]],
                                      lower[axis],upper[axis]))
            starts.append(edges[:-1] if len(edges)>1 else edges)
            ends.append(edges[1:] if len(edges)>1 else edges)
        shape = tuple(len(values) for values in starts)
        if math.prod(shape) > 250000:
            return False  # Bound control-loop work; uncertain queries stay blocked.
        covered = np.zeros(shape,dtype=bool)
        volumes = np.prod(np.minimum(hi,upper)-np.maximum(lo,lower),axis=1)
        for index in np.argsort(-volumes):
            indices = [np.flatnonzero((starts[axis]>=lo[index,axis]) & (ends[axis]<=hi[index,axis]))
                       for axis in range(3)]
            if all(len(values) for values in indices):
                covered[np.ix_(*indices)] = True
                if covered.all():
                    return True
        return False

    def covers(self, pose, position_margin, heading_margin):
        pose = np.asarray(pose, float)
        if not np.isfinite(pose).all() or min(position_margin, heading_margin) < 0:
            return False
        beta = pose[2] % (2 * math.pi)
        low, high = beta - heading_margin, beta + heading_margin
        spans = [(max(0, low), min(2 * math.pi, high))]
        if low < 0:
            spans.append((low + 2 * math.pi, 2 * math.pi))
        if high > 2 * math.pi:
            spans.append((0, high - 2 * math.pi))
        return all(self._covers(np.array([pose[0]-position_margin, pose[1]-position_margin, a]),
                                np.array([pose[0]+position_margin, pose[1]+position_margin, b]))
                   for a, b in spans)

    def permitted(self, pose, control):
        separation = fixed_separation(pose,self.settings)
        return (separation >= control['min_separation_m'] + control['position_margin_m']
                and self.covers(pose, control['position_margin_m'], control['heading_margin_rad']))

    def seed(self):
        control = load_config()['control']
        candidates = (self.lower + self.upper)[::20] / 2
        preferred = np.array([-0.025, -0.015, math.pi/2])
        scores = np.linalg.norm(candidates[:, :2]-preferred[:2],axis=1)
        scores += 0.005*np.abs([wrap(b-preferred[2]) for b in candidates[:,2]])
        for pose in candidates[np.argsort(scores)]:
            if (self.permitted(pose, control)
                    and self.covers(pose,0.0035,0.22)):
                return pose.copy()
        raise ValueError('No starting pose with configured clearance')


class ManualController:
    def __init__(self, config):
        self.control = config['control']
        positive = ['rate_hz','drive_force_N','turn_rate_rad_s','prediction_speed_m_s',
                    'prediction_horizon_s','position_margin_m','heading_margin_rad',
                    'min_separation_m','pose_timeout_s','command_timeout_s','max_pose_jump_m',
                    'max_heading_jump_rad','allocation_tolerance','current_slew_A_s']
        if any(not math.isfinite(self.control[key]) or self.control[key] <= 0 for key in positive):
            raise ValueError('Control settings must be finite and positive')
        if self.control['heading_margin_rad'] >= math.pi/2:
            raise ValueError('Heading margin must be less than pi/2')
        self.model = MagneticModel(config['model'])
        self.workspace = Workspace(config['model'])

    def command(self, pose, drive, turn, previous, dt, speed_m_s=0.0):
        c, s = self.control, self.model.settings
        if not np.isfinite(np.r_[pose, drive, turn, previous, speed_m_s, dt]).all() or dt <= 0:
            return None, 'invalid input'
        if not self.workspace.permitted(pose, c):
            return None, 'measured pose outside guarded model-in TFW'
        # Check the full swept uncertainty volume, not isolated path samples.
        turn_delta = float(np.clip(turn, -1, 1)) * c['turn_rate_rad_s'] * c['prediction_horizon_s']
        distance = max(abs(float(np.clip(drive, -1, 1))) * c['prediction_speed_m_s'],
                       abs(speed_m_s)) * c['prediction_horizon_s']
        swept_margin = c['position_margin_m'] + distance
        separation = fixed_separation(pose,s)
        if (separation < c['min_separation_m'] + swept_margin
                or not self.workspace.covers(pose, swept_margin, c['heading_margin_rad'] + abs(turn_delta))):
            drive, turn = 0.0, 0.0
            reason = 'boundary blocked; holding'
        else:
            reason = 'driving' if drive or turn else 'holding'
        delta = float(np.clip(turn, -1, 1)) * c['turn_rate_rad_s'] * dt
        field = s['holding_field_T']
        thrust = float(np.clip(drive, -1, 1)) * min(c['drive_force_N'], s['task_force_N'])
        target = np.array([field * math.sin(delta), field * math.cos(delta), 0.0, thrust])
        currents, residual = self.model.allocate(pose, target, previous, c['current_slew_A_s'] * dt)
        if residual > c['allocation_tolerance']:
            return None, 'current allocation failed tolerance (%.4f)' % residual
        return currents, reason
