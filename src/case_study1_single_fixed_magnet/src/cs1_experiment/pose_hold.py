"""Uncommissioned pose-hold controller; hardware validation is explicit.

The imported TFW remains a geometric exclusion guard. Its forward-force task
family does not certify the added lateral position-correction forces.
"""
import math

import numpy as np

from .core import ManualController, wrap


class PoseHoldController(ManualController):
    def __init__(self, config):
        super().__init__(config)
        self.hold = config['pose_hold']
        if type(self.hold['hardware_validated']) is not bool:
            raise ValueError('hardware_validated must be an explicit boolean')
        for key,value in self.hold.items():
            if key != 'hardware_validated' and (not math.isfinite(value) or value <= 0):
                raise ValueError('Pose-hold settings must be finite and positive: '+key)
        if self.hold['max_force_N'] > self.model.settings['task_force_N']:
            raise ValueError('Pose-hold force exceeds the imported model task scale')
        # Direct W/S thrust is checked against the TFW's certified forward-force
        # family (|fz| <= task_force_N with the holding field).
        if self.hold['drive_thrust_N'] > self.model.settings['task_force_N']:
            raise ValueError('Drive thrust exceeds the imported model task scale')
        self.reset_hold()

    def reset_hold(self):
        self.reference = self.initial = self.filtered_xy = self.last_filtered_xy = None
        self.velocity = np.zeros(2)
        self.phase = 'disarmed'
        self.elapsed = self.settled = 0.
        self.target = None
        self.error = np.zeros(3)
        self.following = False
        self.heading_error_filtered = 0.

    def begin_hold(self, pose):
        pose = np.asarray(pose,float)
        if not self.workspace.permitted(pose,self.control):
            raise ValueError('Initial pose outside guarded TFW')
        self.reset_hold()
        self.reference = self.initial = pose.copy()
        self.filtered_xy = self.last_filtered_xy = pose[:2].copy()
        self.phase = 'starting'

    def snapshot(self):
        return {'phase':self.phase,'hardware_validated':self.hold['hardware_validated'],
                'reference_pose':None if self.reference is None else self.reference.tolist(),
                'pose_error':self.error.tolist(),'velocity_m_s':self.velocity.tolist(),
                'target_task':None if self.target is None else self.target.tolist()}

    def command(self, pose, drive, turn, previous, dt, speed_m_s=0.):
        pose,previous = np.asarray(pose,float),np.asarray(previous,float)
        h,c = self.hold,self.control
        if (pose.shape != (3,) or previous.shape != (10,) or dt<=0
                or not np.isfinite(np.r_[pose,previous,drive,turn,dt]).all()):
            return None,'invalid pose-hold input'
        if self.reference is None:return None,'Pose hold has not been initialized'
        if not self.workspace.permitted(pose,c):return None,'measured pose outside guarded model-in TFW'
        self.elapsed += dt
        thrust = 0.
        alpha = dt/(h['position_filter_tau_s']+dt)
        self.filtered_xy += alpha*(pose[:2]-self.filtered_xy)
        measured_velocity = (self.filtered_xy-self.last_filtered_xy)/dt
        self.velocity += dt/(h['velocity_filter_tau_s']+dt)*(measured_velocity-self.velocity)
        self.last_filtered_xy = self.filtered_xy.copy()
        if self.phase == 'starting':
            if np.linalg.norm(pose[:2]-self.initial[:2]) > h['startup_drift_limit_m']:
                return None,'Startup displacement exceeded limit; inspect force model'
            if abs(wrap(pose[2]-self.initial[2])) > h['startup_heading_limit_rad']:
                return None,'Startup heading changed excessively; inspect polarity/model'
            if self.elapsed > h['startup_timeout_s']:return None,'Pose-hold startup did not settle'
            reason = 'starting pose hold; movement input inhibited'
        else:
            drive = float(np.clip(drive,-1,1))
            proposal = self.reference.copy()
            proposal[2] += float(np.clip(turn,-1,1))*c['turn_rate_rad_s']*dt
            # Forward force the coils are still producing (currents are
            # slew-limited, so thrust fades over a few ticks after release).
            matrix,offset = self.model.matrix(pose)
            applied_forward = float((matrix@previous+offset)[3])
            still_pushing = abs(applied_forward) > h['coast_release_force_N']
            want_follow = bool(drive) or (self.following and still_pushing)
            if want_follow:
                # Direct thrust: while W/S is held, and until the thrust has
                # faded after release, the hold target slides with the magnet
                # along its heading so the position spring does not fight the
                # propulsion. The sideways part of the target is kept, so
                # lateral drift is still corrected. Then the pose is captured.
                lateral = np.array([-math.sin(pose[2]),math.cos(pose[2])])
                proposal[:2] = pose[:2]+lateral*(lateral@(self.reference[:2]-pose[:2]))
            speed = max(c['prediction_speed_m_s']*abs(drive),np.linalg.norm(self.velocity))
            margin = c['position_margin_m']+speed*c['prediction_horizon_s']
            angle_margin = c['heading_margin_rad']+abs(turn)*c['turn_rate_rad_s']*c['prediction_horizon_s']
            safe = (np.linalg.norm(proposal[:2]-self.filtered_xy) <= h['max_reference_lead_m']
                    and self.workspace.permitted(proposal,c)
                    and self.workspace.covers(pose,margin,angle_margin))
            if safe:
                self.reference = proposal
                thrust = drive*h['drive_thrust_N']
                self.following = want_follow
            elif want_follow and (self.following or drive) and still_pushing:
                # The guard cut the thrust request, but the slew-limited coils
                # are still pushing: keep following the magnet's position (not
                # heading) until that push fades, then hold where it stopped.
                self.reference[:2] = proposal[:2]
                self.following = True
            else:
                self.following = False
            reason = 'driving: thrust %.0f mN' % (1e3*thrust) if safe and drive else (
                'coasting: thrust fading (%.0f mN)' % (1e3*applied_forward) if self.following else
                'turning pose target' if safe and turn else 'holding captured pose')
            if not safe and (drive or turn):reason = 'boundary/reference limit blocked; holding pose'
        self.error = np.r_[self.reference[:2]-self.filtered_xy,wrap(self.reference[2]-pose[2])]
        if np.linalg.norm(pose[:2]-self.reference[:2]) > h['max_position_error_m']:
            return None,'Pose-hold position error exceeded limit'
        if abs(self.error[2]) > h['max_heading_error_rad']:
            return None,'Pose-hold heading error exceeded limit'
        distance = np.linalg.norm(self.error[:2])
        position_error = self.error[:2]*max(0.,1-h['position_deadband_m']/max(distance,1e-12))
        force = h['position_kp_N_m']*position_error-h['position_kd_Ns_m']*self.velocity
        # Keep the field direction fixed in EM coordinates, rather than turning
        # it with every noisy measured axis. Projection uses the measured frame.
        heading_error = self.error[2]
        beta = pose[2]
        forward,lateral = np.array([math.cos(beta),math.sin(beta)]),np.array([-math.sin(beta),math.cos(beta)])
        if self.following:
            # Along the heading the magnet is being propelled: correct sideways only.
            force -= forward*(forward@force)
        force *= min(1.,h['max_force_N']/max(np.linalg.norm(force),1e-12))
        field = self.model.settings['holding_field_T']
        # Position correction (capped at max_force_N) plus direct W/S thrust
        # along the measured heading (up to drive_thrust_N).
        self.target = np.array([field*math.sin(heading_error),field*math.cos(heading_error),
                                lateral@force,forward@force+thrust])
        desired,residual = self.model.allocate(pose,self.target,previous)
        if residual > c['allocation_tolerance']:return None,'Pose-hold task cannot be allocated'
        rate = h['startup_current_slew_A_s'] if self.phase=='starting' else h['holding_current_slew_A_s']
        step = min(rate,c['current_slew_A_s'])*dt
        # Truncate the step to milliamp resolution so rounding cannot exceed it.
        steps_mA = math.floor(step*1000+1e-9)
        if steps_mA < 1:return None,'Configured current ramp is below CAN resolution'
        difference_mA = np.round(desired*1000)-np.round(previous*1000)
        fraction = min(1.,steps_mA/max(1.,np.max(np.abs(difference_mA))))
        # One common interpolation factor preserves the coordinated task. Ten
        # independent clipping factors can create a large unintended force.
        current = (np.round(previous*1000)+np.trunc(fraction*difference_mA))/1000
        if self.phase == 'starting':
            # Camera heading noise is a few degrees per frame; judge the
            # heading on a filtered error.
            self.heading_error_filtered += alpha*(self.error[2]-self.heading_error_filtered)
            # On the bench, friction holds the magnet a little away from the
            # arming pose and the small position spring cannot pull it back.
            # Startup succeeds once the currents have reached the holding
            # solution and the magnet is stationary (the drift/heading limits
            # above already bound how far it may have moved). Then the hold
            # target is re-captured where the magnet actually rests.
            stable = (np.max(np.abs(current-desired)) <= h['startup_current_tolerance_A']
                      and np.linalg.norm(self.velocity) <= h['startup_settle_speed_m_s']
                      and abs(self.heading_error_filtered) <= h['heading_deadband_rad'])
            self.settled = self.settled+dt if stable else 0.
            if self.settled >= h['startup_settle_s']:
                self.reference[:2] = self.filtered_xy
                self.phase = 'holding';reason = 'holding captured pose'
        return current,reason
