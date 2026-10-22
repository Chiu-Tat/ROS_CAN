"""Feedback trajectory tracking using measured pose and all fixed dipoles.

Paper heading/thrust feed-forward is supplemented by bounded position feedback,
reverse-aware steering, progress gating, and startup drift/recording guards.
Sideways feedback and transient steering fields require pointwise allocation;
the paper's zero-sideways-force TFW alone does not certify these corrections.
"""
import math
import numpy as np
from cs1_experiment.core import ManualController, wrap
from .planning import RoutePlanner


class AutonomousController(ManualController):
    def __init__(self, config):
        super().__init__(config)
        self.settings = config['autonomous']
        if type(self.settings['hardware_validated']) is not bool:
            raise ValueError('Hardware validation flag must be boolean')
        for key, value in self.settings.items():
            if key != 'hardware_validated' and (not math.isfinite(value) or value <= 0):
                raise ValueError('Invalid autonomous setting: '+key)
        if (self.settings['speed_scale'] > 1 or self.settings['max_axis_force_N'] > config['model']['task_force_N']
                or self.settings['max_lateral_force_N'] > self.settings['max_axis_force_N']):
            raise ValueError('Autonomous task exceeds configured limits')
        self.planner = RoutePlanner(config, self.workspace)
        self.plan = None
        self.reset_hold()

    def reset_hold(self):
        self.phase = 'disarmed'
        self.reference = self.initial = self.last_pose = None
        self.field_heading = None
        self.elapsed = self.progress = self.settled = self.stalled = 0.
        self.error = np.zeros(3); self.velocity = np.zeros(2)
        self.integral = 0.
        self.target = None

    def begin_hold(self, pose):
        if self.plan is None:
            raise ValueError('Plan from the measured pose first')
        h = self.settings
        planned = np.asarray(self.plan['planned_from_pose'])
        if (np.linalg.norm(np.asarray(pose)[:2]-planned[:2]) > h['plan_drift_limit_m']
                or abs(wrap(pose[2]-planned[2])) > h['plan_heading_drift_rad']):
            raise ValueError('Magnet moved after planning; plan again')
        if not self.workspace.permitted(pose, self.control):
            raise ValueError('Measured start outside guarded TFW')
        self.reset_hold()
        self.phase = 'aligning'
        self.initial = self.last_pose = np.asarray(pose, float).copy()
        self.reference = self.initial.copy()
        self.field_heading = float(pose[2])

    def reference_at(self, t):
        rows = self.plan['samples']
        return np.array([np.interp(t, rows[:, 3], rows[:, i]) for i in range(3)])

    def snapshot(self):
        return {'controller':'autonomous', 'phase':self.phase,
                'hardware_validated':self.settings['hardware_validated'],
                'plan_ready':self.plan is not None, 'progress_s':self.progress,
                'duration_s':0. if self.plan is None else float(self.plan['samples'][-1, 3]),
                'approach_duration_s':0. if self.plan is None else self.plan['approach_duration_s'],
                'reference_pose':None if self.reference is None else self.reference.tolist(),
                'pose_error':self.error.tolist(), 'velocity_m_s':self.velocity.tolist(),
                'field_heading_rad':self.field_heading,
                'target_task':None if self.target is None else self.target.tolist()}

    def command(self, pose, drive, turn, previous, dt, speed_m_s=0.):
        pose, previous = np.asarray(pose, float), np.asarray(previous, float)
        h, c = self.settings, self.control
        if (pose.shape != (3,) or previous.shape != (10,) or dt <= 0
                or not np.isfinite(np.r_[pose, previous, dt, speed_m_s]).all()):
            return None, 'Invalid autonomous feedback'
        if self.phase == 'disarmed' or self.plan is None:
            return None, 'Autonomous route not started'
        if not self.workspace.permitted(pose, c):
            return None, 'Measured pose left guarded TFW'
        if abs(speed_m_s) > h['max_speed_m_s']:
            return None, 'Measured speed exceeded autonomous limit'
        self.elapsed += dt
        raw_velocity = (pose[:2]-self.last_pose[:2])/dt
        self.velocity += dt/(.15+dt)*(raw_velocity-self.velocity)
        self.last_pose = pose.copy()
        if self.phase == 'aligning':
            if np.linalg.norm(pose[:2]-self.initial[:2]) > h['startup_drift_limit_m']:
                return None, 'Magnet drifted during startup; inspect model and polarity'
            if self.elapsed > h['startup_timeout_s']:
                return None, 'Startup did not settle'
            self.reference = self.initial.copy()
            v_ref = np.zeros(2)
        else:
            old_ref = self.reference_at(self.progress)
            position_error = np.linalg.norm(old_ref[:2]-pose[:2])
            heading_error = abs(wrap(old_ref[2]-pose[2]))
            if position_error > h['max_tracking_error_m']:
                return None, 'Tracking error exceeded limit'
            if heading_error > h['max_heading_error_rad']:
                return None, 'Heading no longer follows the planned route'
            advance = (position_error <= h['progress_position_error_m']
                       and heading_error <= h['progress_heading_error_rad'])
            self.stalled = 0. if advance else self.stalled+dt
            if self.stalled > h['stalled_timeout_s']:
                return None, 'Tracking stalled; inspect friction, camera and gains'
            if self.elapsed > self.plan['samples'][-1, 3]*h['tracking_timeout_factor']+h['startup_timeout_s']:
                return None, 'Trajectory timed out'
            end = float(self.plan['samples'][-1, 3])
            if advance:self.progress = min(end, self.progress+dt)
            self.reference = self.reference_at(self.progress)
            a = self.reference_at(max(0., self.progress-.05))
            b = self.reference_at(min(end, self.progress+.05))
            v_ref = (b[:2]-a[:2])/.1
            if not advance:v_ref *= 0.
            self.phase = 'approaching' if self.progress < self.plan['approach_duration_s'] else 'tracking'
        error = self.reference[:2]-pose[:2]
        beta_r = float(self.reference[2])
        forward_r = np.array([math.cos(beta_r), math.sin(beta_r)])
        normal_r = np.array([-forward_r[1], forward_r[0]])
        direction = -1. if v_ref@forward_r < -1e-7 else 1.
        correction = direction*h['heading_kp_rad_m']*(error@normal_r)
        # During in-place turns the field follows the planned heading directly.
        if np.linalg.norm(v_ref) < 1e-7:correction = 0.
        self.field_heading = beta_r+float(np.clip(correction, -h['max_heading_correction_rad'], h['max_heading_correction_rad']))
        if not self.workspace.permitted([pose[0], pose[1], self.field_heading], c):
            return None, 'Steered heading outside guarded TFW'
        forward = np.array([math.cos(pose[2]), math.sin(pose[2])])
        lateral = np.array([-forward[1], forward[0]])
        axial_error = float(error@forward)
        if self.phase != 'aligning':
            limit = h['max_integral_force_N']/h['axial_ki_N_ms']
            if self.integral*axial_error < 0:self.integral *= .5
            self.integral = float(np.clip(self.integral+axial_error*dt, -limit, limit))
        force = h['drag_Ns_m']*v_ref+h['position_kp_N_m']*error-h['velocity_kd_Ns_m']*self.velocity
        axis = float(np.clip(force@forward+h['axial_ki_N_ms']*self.integral, -h['max_axis_force_N'], h['max_axis_force_N']))
        side = float(np.clip(h['lateral_kp_N_m']*(error@lateral)-h['velocity_kd_Ns_m']*(self.velocity@lateral),
                             -h['max_lateral_force_N'], h['max_lateral_force_N']))
        delta = wrap(self.field_heading-pose[2]); field = self.model.settings['holding_field_T']
        self.target = np.array([field*math.sin(delta), field*math.cos(delta), side, axis])
        self.error = np.r_[error, wrap(self.reference[2]-pose[2])]
        desired, residual = self.model.allocate(pose, self.target, previous)
        if residual > c['allocation_tolerance']:
            return None, 'Autonomous force/field task cannot be allocated'
        rate = h['startup_current_slew_A_s'] if self.phase == 'aligning' else h['tracking_current_slew_A_s']
        steps = math.floor(min(rate, c['current_slew_A_s'])*dt*1000+1e-9)
        if steps < 1:return None, 'Current ramp is below CAN resolution'
        diff = np.round(desired*1000)-np.round(previous*1000)
        fraction = min(1., steps/max(1., float(np.max(np.abs(diff)))))
        currents = (np.round(previous*1000)+np.trunc(diff*fraction))/1000
        if self.phase == 'aligning':
            settled = (np.max(np.abs(currents-desired)) < .05 and abs(delta) < .12
                       and np.linalg.norm(self.velocity) < .002)
            self.settled = self.settled+dt if settled else 0.
            if self.settled >= h['settle_s']:
                self.phase = 'approaching'; self.settled = 0.
        else:
            # Test the rounded, slew-limited vector too; no sustained task loss.
            matrix, fixed = self.model.matrix(pose)
            actual_error = np.max(np.abs((matrix@currents+fixed-self.target)/self.model.row_scale))
            if actual_error > .15:return None, 'Current slew prevented autonomous task allocation'
            done = (self.progress >= self.plan['samples'][-1, 3]
                    and np.linalg.norm(error) < h['finish_position_tolerance_m']
                    and abs(delta) < h['finish_heading_tolerance_rad']
                    and np.linalg.norm(self.velocity) < .002)
            self.settled = self.settled+dt if done else 0.
            if self.settled >= h['settle_s']:self.phase = 'complete'
        return currents, self.phase
