"""Checked paper references and short nonholonomic connections from camera poses.

Nominal Hybrid A* routes remain in world coordinates. A connection turns in
place, drives forward/reverse, then turns to the nominal start heading. Every
swept interval is checked in the original box union, including uncertainty.
"""
import hashlib
import json
import math

import numpy as np

from cs1_experiment.core import fixed_separation, wrap
from . import PACKAGE


def config_path(layout):
    if layout not in ('single', 'gate', 'triangle'):
        raise ValueError('Unknown layout: '+str(layout))
    return PACKAGE/'config/layouts'/(layout+'.json')


def load_reference(layout):
    manifest = json.loads((PACKAGE/'data/provenance.json').read_text())
    path = PACKAGE/'data/trajectories'/(layout+'.json')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest['trajectories'][layout]['sha256']:
        raise ValueError('Reference checksum differs from provenance')
    data = json.loads(path.read_text())
    samples = np.asarray(data['trajectory'], float)
    if (data['layout'] != layout or data['kind'] != 'planned'
            or data['columns'] != ['x_m', 'y_m', 'beta_rad', 't_s']
            or samples.ndim != 2 or samples.shape[1] != 4 or len(samples) < 2
            or not np.isfinite(samples).all() or samples[0, 3] != 0
            or np.any(np.diff(samples[:, 3]) <= 0)):
        raise ValueError('Invalid reference trajectory')
    samples[:, 2] = np.unwrap(samples[:, 2])
    data['sha256'] = digest
    return samples, data


def check_swept(samples, workspace, control):
    """Enclose each whole interpolated edge, not just its endpoints."""
    for a, b in zip(samples[:-1], samples[1:]):
        distance = float(np.linalg.norm(b[:2]-a[:2]))
        angle = abs(b[2]-a[2])
        # Subdivide long connections to avoid rejecting safe curved corridors.
        count = max(1, int(math.ceil(max(distance/.0005, angle/.035))))
        for k in range(count):
            lo = a[:3]+(b[:3]-a[:3])*k/count
            hi = a[:3]+(b[:3]-a[:3])*(k+1)/count
            mid = (lo+hi)/2
            margin = control['position_margin_m']+np.linalg.norm(hi[:2]-lo[:2])/2
            heading_margin = control['heading_margin_rad']+abs(hi[2]-lo[2])/2
            if (fixed_separation(mid, workspace.settings) < control['min_separation_m']+margin
                    or not workspace.covers(mid, margin, heading_margin)):
                raise ValueError('Path leaves guarded TFW near (%.1f, %.1f) mm, %.1f deg' %
                                 (mid[0]*1000, mid[1]*1000, math.degrees(mid[2])))


def connection(start, goal, settings, workspace, control):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if start.shape != (3,) or not np.isfinite(start).all():
        raise ValueError('Need a finite measured start pose')
    distance = float(np.linalg.norm(goal[:2]-start[:2]))
    if distance > settings['max_start_offset_m']:
        raise ValueError('Place magnet within %.1f mm of the nominal start (now %.1f mm)' %
                         (settings['max_start_offset_m']*1000, distance*1000))
    candidates = []
    for reverse in (False, True):
        poses = [start.copy()]
        if distance > 1e-7:
            heading = math.atan2(goal[1]-start[1], goal[0]-start[0])+(math.pi if reverse else 0)
            turn = start.copy(); turn[2] += wrap(heading-start[2]); poses.append(turn)
            drive = turn.copy(); drive[:2] = goal[:2]; poses.append(drive)
        else:
            # Retain even a sub-micrometre displacement in the checked edge.
            reached = start.copy(); reached[:2] = goal[:2]; poses.append(reached)
        end = poses[-1].copy(); end[2] += wrap(goal[2]-end[2]); poses.append(end)
        route = [np.r_[poses[0], 0.]]
        for a, b in zip(poses[:-1], poses[1:]):
            length = float(np.linalg.norm(b[:2]-a[:2])); angle = abs(b[2]-a[2])
            n = max(1, int(math.ceil(max(length/.0005, angle/.035))))
            duration = max(length/settings['approach_speed_m_s'], angle/settings['turn_rate_rad_s'], 1e-4)
            t0 = route[-1][3]
            route.extend(np.r_[a+(b-a)*k/n, t0+duration*k/n] for k in range(1, n+1))
        route = np.asarray(route)
        try:
            check_swept(route, workspace, control)
        except ValueError:
            continue
        candidates.append(route)
    if not candidates:
        raise ValueError('No checked short approach to the start. Move closer and plan again.')
    return min(candidates, key=lambda r:r[-1, 3])


class RoutePlanner:
    def __init__(self, config, workspace):
        self.config, self.workspace = config, workspace
        self.layout = config['model'].get('layout', 'single')
        self.nominal, self.source = load_reference(self.layout)
        self.validated = False

    def prepare(self, pose):
        if not self.validated:
            check_swept(self.nominal, self.workspace, self.config['control'])
            self.validated = True
        approach = connection(pose, self.nominal[0, :3], self.config['autonomous'],
                              self.workspace, self.config['control'])
        nominal = self.nominal.copy()
        nominal[:, 2] += approach[-1, 2]-nominal[0, 2]
        nominal[:, 3] = nominal[:, 3]/self.config['autonomous']['speed_scale']+approach[-1, 3]
        combined = np.vstack([approach, nominal[1:]])
        return {'samples':combined, 'approach_duration_s':float(approach[-1, 3]),
                'planned_from_pose':np.asarray(pose, float).tolist(),
                'source':self.source, 'nominal_goal':self.nominal[-1, :3].tolist(),
                'requested_goal':self.source['goal']}
