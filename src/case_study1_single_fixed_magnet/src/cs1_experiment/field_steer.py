"""Field-led manual control, following the working Case 6 controller.

Lessons from Case 6 (archive/ros_packages/case6_single_sphere_closed_loop):
  * The camera line has no polarity. The applied 20 mT field defines it: the
    magnet's N aligns with the field, so the measured axis is resolved to the
    branch nearest the field direction. No hand-placed N direction is trusted.
  * Apply the holding field quickly at startup and let the magnet align
    (even flip) with it, instead of treating rotation as a fault.
  * Push mainly along the magnet axis and steer by turning the field. This is
    the imported TFW task family [By, Bz, 0, fz]; a small capped sideways
    force is added (outside that family) because on the bench the lying
    cylinder rolls sideways under unmodeled forces.
  * Close the loop on position with PID + damping (Case 6 used KP/KI/KD toward
    a moving lookahead point). An open-loop push either does nothing (sliding
    friction along the axis) or, once it breaks free, accelerates far too fast.

Keyboard: W/S moves a target point along the heading at a capped speed; the
axial PID (whose integral builds the breakaway force) drives the magnet to
that point and actively brakes/holds it when the key is released. A sideways
PID holds the line. A/D turns the field (heading).
"""
import math

import numpy as np

from .core import ManualController, fixed_separation, unwrap_axis, wrap


class FieldSteerController(ManualController):
    def __init__(self, config):
        super().__init__(config)
        self.hold = config['field_steer']
        if type(self.hold['hardware_validated']) is not bool:
            raise ValueError('hardware_validated must be an explicit boolean')
        for key, value in self.hold.items():
            if key != 'hardware_validated' and (not math.isfinite(value) or value <= 0):
                raise ValueError('Field-steer settings must be finite and positive: '+key)
        if self.hold['max_axis_force_N'] > self.model.settings['task_force_N']:
            raise ValueError('Axial force limit exceeds the imported model task scale')
        if self.hold['max_lateral_force_N'] > self.hold['max_axis_force_N']:
            raise ValueError('Sideways force limit must not exceed the axial force limit')
        if self.hold['integral_unwind_factor'] >= 1:
            raise ValueError('integral_unwind_factor must be below 1')
        if self.hold['max_integral_force_N'] > self.hold['max_axis_force_N']:
            raise ValueError('Integral force limit must not exceed the axial force limit')
        self.reset_hold()

    def reset_hold(self):
        self.reference = self.initial = None
        self.field_heading = None
        self.target_xy = self.filtered_xy = self.last_filtered_xy = None
        self.velocity = np.zeros(2)
        self.integral = self.lateral_integral = 0.
        self.last_drive = 0.
        self.coasting = False
        self.phase = 'disarmed'
        self.elapsed = self.settled = 0.
        self.target = None
        self.error = np.zeros(3)

    def begin_hold(self, pose):
        pose = np.asarray(pose, float)
        if not self.workspace.permitted(pose, self.control):
            raise ValueError('Initial pose outside guarded TFW')
        self.reset_hold()
        self.initial = pose.copy()
        self.field_heading = float(pose[2])
        self.target_xy = pose[:2].copy()
        self.filtered_xy = self.last_filtered_xy = pose[:2].copy()
        self.reference = pose.copy()
        self.phase = 'aligning'

    def snapshot(self):
        return {'phase': self.phase, 'hardware_validated': self.hold['hardware_validated'],
                'controller': 'field_steer',
                'field_heading_rad': self.field_heading,
                'reference_pose': None if self.reference is None else self.reference.tolist(),
                'pose_error': self.error.tolist(), 'velocity_m_s': self.velocity.tolist(),
                'integral_force_N': self.hold['axis_ki_N_ms']*self.integral,
                'target_task': None if self.target is None else self.target.tolist()}

    def axial_force(self, forward, dt, driving=True):
        """PID along the axis toward target_xy (Case 6 style), with anti-windup.

        The integral builds the force needed to break sliding friction. When
        the magnet passes the target (error and integral of opposite sign) the
        integral is unwound quickly so the stored push does not overshoot.
        """
        h = self.hold
        error = float(forward@(self.target_xy-self.filtered_xy))
        speed = float(forward@self.velocity)
        if not driving:
            self.integral = 0.
            if self.coasting:
                # Just released while sliding: brake with damping only and let
                # the target follow; pulling back makes it slide back.
                force = -h['axis_kd_Ns_m']*speed
                return float(np.clip(force, -h['max_hold_force_N'], h['max_hold_force_N'])), error
            # Holding: sliding friction keeps the magnet still. Use only a
            # gentle, capped spring outside a deadband; an integral here builds
            # to the breakaway force from camera noise and makes it hunt.
            excess = math.copysign(max(0., abs(error)-h['hold_deadband_m']), error)
            force = h['axis_kp_N_m']*excess-h['axis_kd_Ns_m']*speed
            return float(np.clip(force, -h['max_hold_force_N'], h['max_hold_force_N'])), error
        if error*self.integral < 0 or abs(speed) > h['breakaway_speed_factor']*h['max_speed_m_s']:
            # Passed the target, or friction has broken and the magnet is
            # slipping fast: release the stored breakaway push quickly.
            self.integral *= h['integral_unwind_factor']
        limit = h['max_integral_force_N']/h['axis_ki_N_ms']
        self.integral = float(np.clip(self.integral+error*dt, -limit, limit))
        force = h['axis_kp_N_m']*error+h['axis_ki_N_ms']*self.integral-h['axis_kd_Ns_m']*speed
        return float(np.clip(force, -h['max_axis_force_N'], h['max_axis_force_N'])), error

    def lateral_force(self, lateral, dt):
        """Small PID across the axis: a lying cylinder rolls sideways easily,
        and unmodeled sideways forces made it drift. The integral learns that
        bias. Capped well below the axial force."""
        h = self.hold
        error = float(lateral@(self.target_xy-self.filtered_xy))
        speed = float(lateral@self.velocity)
        limit = h['max_lateral_integral_force_N']/h['lateral_ki_N_ms']
        self.lateral_integral = float(np.clip(self.lateral_integral+error*dt, -limit, limit))
        force = (h['lateral_kp_N_m']*error+h['lateral_ki_N_ms']*self.lateral_integral
                 -h['lateral_kd_Ns_m']*speed)
        return float(np.clip(force, -h['max_lateral_force_N'], h['max_lateral_force_N'])), error

    def command(self, pose, drive, turn, previous, dt, speed_m_s=0.):
        pose, previous = np.asarray(pose, float), np.asarray(previous, float)
        h, c = self.hold, self.control
        if (pose.shape != (3,) or previous.shape != (10,) or dt <= 0
                or not np.isfinite(np.r_[pose, previous, drive, turn, dt, speed_m_s]).all()):
            return None, 'invalid field-steer input'
        if self.field_heading is None:
            return None, 'Field steering has not been initialized'
        # Polarity comes from the field: take the axis branch nearest it.
        heading = unwrap_axis(pose[2], self.field_heading)
        measured = np.array([pose[0], pose[1], heading])
        guarded = np.array([pose[0], pose[1], self.field_heading])
        if not self.workspace.permitted(guarded, c):
            return None, 'measured pose outside guarded model-in TFW'
        self.elapsed += dt
        alpha = dt/(h['position_filter_tau_s']+dt)
        self.filtered_xy = self.filtered_xy+alpha*(pose[:2]-self.filtered_xy)
        measured_velocity = (self.filtered_xy-self.last_filtered_xy)/dt
        self.velocity = self.velocity+dt/(h['velocity_filter_tau_s']+dt)*(measured_velocity-self.velocity)
        self.last_filtered_xy = self.filtered_xy.copy()
        forward = np.array([math.cos(heading), math.sin(heading)])
        drive, turn = float(np.clip(drive, -1, 1)), float(np.clip(turn, -1, 1))
        axis_force, axis_error = 0., 0.
        lateral_force, lateral_error = 0., 0.
        if self.phase == 'aligning':
            if self.elapsed > h['align_timeout_s']:
                return None, 'Magnet did not align with the holding field'
            # The magnet may rotate/flip into the field: follow it, no push.
            self.target_xy = self.filtered_xy.copy()
            reason = 'aligning magnet with field; input inhibited'
        else:
            if abs(wrap(self.field_heading-heading)) > h['max_heading_error_rad']:
                return None, 'Magnet no longer follows the field heading'
            proposal = self.field_heading+turn*c['turn_rate_rad_s']*dt
            moved = self.target_xy+drive*h['max_speed_m_s']*dt*forward
            distance = max(abs(drive)*h['max_speed_m_s'], abs(speed_m_s), np.linalg.norm(self.velocity))*c['prediction_horizon_s']
            margin = c['position_margin_m']+distance
            angle_margin = c['heading_margin_rad']+abs(turn)*c['turn_rate_rad_s']*c['prediction_horizon_s']
            separation = fixed_separation(pose,self.model.settings)
            safe = (separation >= c['min_separation_m']+margin
                    and self.workspace.permitted(np.array([moved[0], moved[1], proposal]), c)
                    and self.workspace.covers(guarded, margin, angle_margin))
            if safe:
                self.field_heading = proposal
                self.target_xy = moved
            released = bool(self.last_drive) and not drive
            if released:
                # Key released: coast to a stop (damping only), then hold.
                self.coasting = True
                self.integral = 0.
            self.last_drive = drive
            if drive:
                self.coasting = False
            elif self.coasting and abs(forward@self.velocity) < h['coast_stop_speed_m_s']:
                self.coasting = False   # stopped: hold from here
            # The target is a fixed 2-D point. Along the axis it may lead the
            # magnet by at most max_target_lead_m in the driving direction and
            # never lies behind it while driving or coasting: when the magnet
            # slips ahead, the target jumps to it (damping brakes; the spring
            # does not pull it back). Across the axis it holds.
            lateral = np.array([-forward[1], forward[0]])
            offset = self.target_xy-self.filtered_xy
            lead = float(np.clip(forward@offset, -h['max_target_lead_m'], h['max_target_lead_m']))
            moving = drive and safe
            if moving and lead*drive < 0:
                lead = 0.
                self.integral *= h['integral_unwind_factor']
            if self.coasting or released:
                lead = 0.   # stop where the magnet is, not at the lead point
            self.target_xy = self.filtered_xy+forward*lead+lateral*(lateral@offset)
            axis_force, axis_error = self.axial_force(forward, dt, driving=bool(drive) and safe)
            lateral_force, lateral_error = self.lateral_force(lateral, dt)
            if abs(forward@(self.target_xy-pose[:2])) > h['max_axis_error_m']:
                return None, 'Axial position error exceeded limit'
            if abs(lateral@(self.target_xy-pose[:2])) > h['max_lateral_error_m']:
                return None, 'Sideways drift exceeded limit'
            if not safe and (drive or turn):
                reason = 'boundary blocked; holding position'
            elif drive:
                reason = 'driving %.1f mm/s target' % (1e3*drive*h['max_speed_m_s'])
            elif turn:
                reason = 'turning field; holding position'
            else:
                reason = 'holding position'
        delta = wrap(self.field_heading-heading)
        field = self.model.settings['holding_field_T']
        # Body frame of the measured (field-resolved) heading: field toward the
        # commanded heading, no lateral force, PID force along the axis.
        self.target = np.array([field*math.sin(delta), field*math.cos(delta), lateral_force, axis_force])
        self.reference = np.array([self.target_xy[0], self.target_xy[1], self.field_heading])
        self.error = np.array([axis_error*forward[0]-lateral_error*forward[1],
                               axis_error*forward[1]+lateral_error*forward[0], delta])
        desired, residual = self.model.allocate(measured, self.target, previous)
        if residual > c['allocation_tolerance']:
            return None, 'Field-steer task cannot be allocated'
        rate = h['align_current_slew_A_s'] if self.phase == 'aligning' else h['current_slew_A_s']
        step = min(rate, c['current_slew_A_s'])*dt
        steps_mA = math.floor(step*1000+1e-9)
        if steps_mA < 1:
            return None, 'Configured current ramp is below CAN resolution'
        difference_mA = np.round(desired*1000)-np.round(previous*1000)
        fraction = min(1., steps_mA/max(1., np.max(np.abs(difference_mA))))
        # One common interpolation factor preserves the coordinated task.
        current = (np.round(previous*1000)+np.trunc(fraction*difference_mA))/1000
        if self.phase == 'aligning':
            aligned = (np.max(np.abs(current-desired)) <= h['align_current_tolerance_A']
                       and abs(delta) <= h['align_tolerance_rad'])
            self.settled = self.settled+dt if aligned else 0.
            if self.settled >= h['align_settle_s']:
                self.phase = 'holding'
                self.target_xy = self.filtered_xy.copy()
                self.integral = self.lateral_integral = 0.
                reason = 'holding position'
        return current, reason
