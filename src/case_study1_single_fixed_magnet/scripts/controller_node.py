#!/usr/bin/env python3
"""ROS/CAN bridge; starts disarmed and never opens CAN in preview mode."""
import json
import math
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Float64MultiArray, String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

from cs1_experiment.core import PACKAGE, calibrated_line, load_config, wrap


from cs1_experiment.can_driver import CurrentOutput, DEVICE_IDS
from cs1_experiment.recording import TrialStore, message_dict


class ControllerNode:
    def __init__(self):
        self.config = load_config(rospy.get_param('~config', str(PACKAGE / 'config/experiment.json')))
        self.layout = self.config['model'].get('layout','single')
        if rospy.get_param('/cs1/layout',self.layout)!=self.layout:
            raise ValueError('Launch layout and controller configuration differ')
        # 'field_steer' (default, Case-6 style) or the older 'pose_hold'.
        self.controller_mode = self.config.get('controller', 'pose_hold')
        self.core = self.create_core(self.config)
        self.hardware = rospy.get_param('~hardware', False)
        self.calibration_path = rospy.get_param('~calibration', str(PACKAGE/'config/camera_calibration.json'))
        if self.hardware:
            if rospy.get_param('/cs1/simulated_feedback', False):
                raise ValueError('Hardware cannot run with simulated feedback')
            calibration_path = rospy.get_param('~calibration', str(PACKAGE / 'config/camera_calibration.json'))
            with open(calibration_path) as stream:
                calibration = json.load(stream)
            if calibration.get('affine') is None:
                raise ValueError('Calibrate camera before enabling hardware')
            calibrated_line({'x':0,'y':0,'axis_rad':0}, calibration['affine'])
        self.can_channel = rospy.get_param('~can_channel','can0')
        self.output = CurrentOutput(self.hardware, self.can_channel)
        self.lock = threading.RLock()
        self.pose = None
        self.speed = 0.0
        self.closed = False
        self.pose_time = self.command_time = 0.0
        self.drive = self.turn = 0.0
        self.input_source = rospy.get_param('/cs1/input_source', 'gamepad')
        if self.input_source not in ('gamepad', 'keyboard'):
            raise ValueError('Input source must be gamepad or keyboard')
        from pathlib import Path
        self.trials = TrialStore(rospy.get_param('~results_dir', str(PACKAGE/'results')))
        self.output.feedback_callback=lambda value:self.trials.record('/can/current_feedback',value)
        self.video_required = rospy.get_param('~record_ui', True)
        self.heading_unwrapped = None
        self.tracking_valid = False
        self.initial_pose_metadata = None
        self.ui_heartbeat = 0.0
        self.previous = np.zeros(10)
        self.armed = False
        self.arming = False
        self.arm_generation = 0
        self.reason = 'disarmed'
        self.last_run_stop = None
        self.pub_current = rospy.Publisher('/cs1/currents', Float64MultiArray, queue_size=1)
        from std_msgs.msg import Float64 as _Float64
        # While armed in field mode, the tracker resolves polarity from this.
        self.pub_field_heading = rospy.Publisher('/cs1/field_heading', _Float64, queue_size=1)
        self._Float64 = _Float64
        self.pub_status = rospy.Publisher('/cs1/status', String, queue_size=1)
        self.pose_sub = rospy.Subscriber('/cs1/pose', PoseStamped, self.on_pose, queue_size=1)
        self.pub_command = rospy.Publisher('/cs1/command', Twist, queue_size=1)
        self.command_sub = rospy.Subscriber('/cs1/gamepad_command', Twist,
                                           lambda msg:self.on_command(msg, 'gamepad'), queue_size=1)
        self.keyboard_sub = rospy.Subscriber('/cs1/keyboard_command', Twist,
                                            lambda msg:self.on_command(msg, 'keyboard'), queue_size=1)
        self.valid_sub = rospy.Subscriber('/cs1/tracking_valid', Bool, self.on_valid, queue_size=1)
        self.service = rospy.Service('/cs1/arm', SetBool, self.on_arm)
        self.driver_services = [rospy.Service('/cs1/'+name, Trigger,
            lambda request, action=name:self.on_driver_action(action))
            for name in ('zero_currents','reset_drivers','prepare_drivers','enable_current_feedback')]
        self.pub_feedback = rospy.Publisher('/cs1/current_feedback', String, queue_size=1)
        from std_msgs.msg import Float64, Float32
        from sensor_msgs.msg import CompressedImage
        self.recording_pub = rospy.Publisher('/cs1/recording/state', String, queue_size=10, latch=True)
        self.recording_services = [
            rospy.Service('/cs1/recording/prepare', Trigger, self.prepare_recording),
            rospy.Service('/cs1/recording/video_ready', Trigger, self.video_ready),
            rospy.Service('/cs1/recording/video_done', Trigger, self.video_done),
            rospy.Service('/cs1/recording/cancel', Trigger, self.cancel_recording),
            rospy.Service('/cs1/recording/resolve', SetBool, self.resolve_recording),
            rospy.Service('/cs1/recording/stop_now', Trigger, self.stop_recording_now)]
        # Seconds of recording kept after control stops (0 = stop at once).
        self.post_stop_s = float(rospy.get_param('~post_stop_record_s', 15.0))
        self.tail_deadline = self.tail_reason = None
        self.recording_subs = [
            rospy.Subscriber('/cs1/heading_unwrapped', Float64, self.on_heading, queue_size=50),
            rospy.Subscriber('/cs1/initial_pose_metadata', String, self.on_initial_metadata, queue_size=1),
            rospy.Subscriber('/cs1/recording/ui_heartbeat', Bool, self.on_ui_heartbeat, queue_size=1),
            rospy.Subscriber('/cs1/camera/image/compressed', CompressedImage, self.record_image,
                             queue_size=5, buff_size=1024*1024)]
        for topic, kind in [('/cs1/tracking_status',String),('/cs1/fps',Float32),('/cs1/initial_pose',PoseStamped)]:
            self.recording_subs.append(rospy.Subscriber(topic,kind,
                lambda msg, name=topic:self.trials.record(name,message_dict(msg)),queue_size=50))
        self.publish_recording()
        rospy.on_shutdown(self.close)

    def create_core(self, config):
        """Allow experiment packages to share the CAN/recording lifecycle."""
        if self.controller_mode == 'field_steer':
            from cs1_experiment.field_steer import FieldSteerController
            return FieldSteerController(config)
        if self.controller_mode == 'pose_hold':
            from cs1_experiment.pose_hold import PoseHoldController
            return PoseHoldController(config)
        raise ValueError('Unknown controller mode: %s' % self.controller_mode)

    def publish_recording(self):
        self.recording_pub.publish(String(data=json.dumps(self.trials.snapshot(), allow_nan=False)))

    def recording_metadata(self):
        from pathlib import Path
        calibration = json.loads(Path(self.calibration_path).read_text())
        provenance = self.core.workspace.provenance
        return {'layout':self.layout,'hardware':self.hardware, 'input_source':self.input_source, 'can_channel':self.can_channel,
                'configuration':self.config, 'calibration':calibration, 'tfw_provenance':provenance,
                'initial_pose':self.initial_pose_metadata, 'pose_at_prepare':None if self.pose is None else self.pose.tolist()}

    def prepare_recording(self, _request):
        with self.lock:
            if self.closed or self.armed or self.arming:
                return TriggerResponse(False,'Disarm before preparing a new recording')
            if getattr(self,'tail_deadline',None) is not None:
                return TriggerResponse(False,'Previous trial is still recording after the stop; '
                                       'wait or press Stop recording now')
            try:
                value = self.trials.prepare(self.recording_metadata(), self.video_required)
                self.publish_recording()
                return TriggerResponse(True,json.dumps(value))
            except Exception as exc:return TriggerResponse(False,str(exc))

    def video_ready(self, _request):
        with self.lock:
            if self.trials.snapshot()['phase']!='prepared':return TriggerResponse(False,'No prepared recording')
            self.trials.video_ready=True
            return TriggerResponse(True,'UI video encoder ready')

    def video_done(self, _request):
        self.publish_recording()
        return TriggerResponse(True,'Video finalized; waiting for all data writers')

    def cancel_recording(self, _request):
        with self.lock:
            try:
                self.trials.cancel_prepared();self.publish_recording()
                return TriggerResponse(True,'Unarmed recording preparation cancelled')
            except Exception as exc:return TriggerResponse(False,str(exc))

    def resolve_recording(self, request):
        with self.lock:
            try:
                note=rospy.get_param('/cs1/recording_note','') if request.data else ''
                destination=self.trials.resolve(request.data,note)
                self.publish_recording()
                return SetBoolResponse(True,'Saved: '+str(destination) if destination else 'Trial video/data deleted')
            except Exception as exc:return SetBoolResponse(False,str(exc))

    def on_ui_heartbeat(self, msg):
        self.ui_heartbeat=time.monotonic() if msg.data else 0.0

    def on_heading(self, msg):
        if math.isfinite(msg.data):self.heading_unwrapped=float(msg.data)
        self.trials.record('/cs1/heading_unwrapped',message_dict(msg))

    def on_initial_metadata(self, msg):
        try:self.initial_pose_metadata=json.loads(msg.data)
        except ValueError:return
        self.trials.record('/cs1/initial_pose_metadata',self.initial_pose_metadata)

    def record_image(self, msg):
        self.trials.record('/cs1/camera/image/compressed',{'header':message_dict(msg.header),
                           'format':msg.format},jpeg=msg.data)

    def applied_model(self):
        """Total theoretical field, force and torque: coils PLUS fixed magnet.

        Uses the measured pose and the COMMANDED currents (not measured ones).
        Body frame: z = forward along the dipole, y = lateral (+90 deg CCW).
        """
        if self.pose is None:
            return None
        try:
            matrix, fixed = self.core.model.matrix(self.pose)
        except ValueError:
            return None
        coils = matrix @ self.previous
        total = coils + fixed
        beta = self.pose[2]
        forward = np.array([math.cos(beta), math.sin(beta)])
        lateral = np.array([-math.sin(beta), math.cos(beta)])

        def world(task):
            return (task[3]*forward + task[2]*lateral).tolist()
        moment = self.config['model']['free_moment_Am2']
        return {'source': 'model; measured pose + commanded currents',
                'body_total': total.tolist(), 'body_coils': coils.tolist(), 'body_fixed': fixed.tolist(),
                'force_world_N': world(total), 'force_coils_world_N': world(coils),
                'force_fixed_world_N': world(fixed),
                # In-plane torque m x B about +z (positive = turns CCW / left).
                'torque_z_Nm': float(moment*total[0]),
                'torque_coils_z_Nm': float(moment*coils[0]),
                'torque_fixed_z_Nm': float(moment*fixed[0]),
                'field_angle_from_heading_rad': float(math.atan2(total[0], total[1]))}

    def telemetry(self):
        now=time.monotonic()
        return {'layout':self.layout,'hardware':self.hardware,'armed':self.armed,'reason':self.reason,
                'input_source':self.input_source,'drive':self.drive,'turn':self.turn,
                'tracking_valid':self.tracking_valid,'heading_unwrapped_rad':self.heading_unwrapped,
                'z_m':self.config['model']['z_m'],
                'pose':None if self.pose is None else self.pose.tolist(),
                'pose_fresh':now-self.pose_time<self.core.control['pose_timeout_s'],
                'currents':self.previous.tolist(),'current_feedback':self.output.feedback_snapshot(now),
                'pose_hold':self.core.snapshot(),'applied_model':self.applied_model()}

    def stop(self, reason, send_zero=True):
        if self.armed or self.arming or (hasattr(self,'trials') and getattr(self,'tail_deadline',None) is None
                                        and self.trials.snapshot()['phase']=='recording'):
            self.last_run_stop = {'reason':reason, 'unix_time_s':time.time(),
                                  'commanded_currents_A':self.previous.tolist()}
        self.arm_generation += 1
        self.armed = False
        self.reason = reason
        self.previous = np.zeros(10)
        if hasattr(self,'core'):self.core.reset_hold()
        try:
            if send_zero:self.output.write(self.previous)
        finally:
            if hasattr(self,'trials') and self.trials.snapshot()['phase']=='prepared':
                if self.video_required:
                    self.trials.meta['phase']='cancelled'
                    self.trials.video_ready=False
                else:self.trials.cancel_prepared()
                self.publish_recording()
            if hasattr(self,'trials') and self.trials.snapshot()['phase']=='recording':
                self.trials.record('controller',self.telemetry())
                post_stop_s = getattr(self,'post_stop_s',0.0)
                recorder_failed = bool(self.trials.error) or reason == 'UI recorder heartbeat lost'
                if post_stop_s <= 0 or recorder_failed or getattr(self,'closed',False):
                    self.finish_recording(reason)
                elif getattr(self,'tail_deadline',None) is None:
                    # Keep recording after control stops (e.g. the free magnet
                    # snapping onto the fixed one) for post_stop_s seconds.
                    self.tail_reason = reason
                    self.tail_deadline = time.monotonic()+post_stop_s
                    self.trials.mark_control_stop(reason, post_stop_s)
                self.publish_recording()

    def finish_recording(self, reason):
        self.tail_deadline = None
        self.tail_reason = None
        self.trials.stop(reason)

    def stop_recording_now(self, _request):
        with self.lock:
            if getattr(self,'tail_deadline',None) is None:
                return TriggerResponse(False,'No post-stop recording in progress')
            self.trials.record('controller',self.telemetry())
            self.finish_recording(self.tail_reason)
            self.publish_recording()
            return TriggerResponse(True,'Recording stopped')

    def on_driver_action(self, action):
        with self.lock:
            if self.closed:
                return TriggerResponse(False, 'Controller is shutting down')
            if action == 'enable_current_feedback':
                if not self.hardware:
                    return TriggerResponse(False, 'CAN disabled; measured current feedback is unavailable')
                try:
                    self.output.enable_feedback()
                    return TriggerResponse(True, 'Feedback rate command sent to all drivers')
                except Exception as exc:
                    return TriggerResponse(False, 'Feedback request failed: '+str(exc))
            # Cancel in-flight arming before any reset/mode/zero action. The
            # generation check prevents that arming request from resuming motion.
            was_active = self.armed or self.arming or bool(np.any(self.previous))
            self.stop('Driver action: '+action, send_zero=False)
            self.drive = self.turn = self.command_time = 0.0
            try:
                if action == 'zero_currents':
                    self.output.zero_all()
                    return TriggerResponse(True, 'Zero commands sent; controller disarmed' if self.hardware
                                           else 'Commanded currents cleared; CAN is disabled')
                if not self.hardware:
                    return TriggerResponse(False, 'CAN disabled; driver action unavailable')
                if action == 'reset_drivers':
                    self.output.reset()
                    self.reason = 'Drivers reset; prepare and rearm when ready'
                elif action == 'prepare_drivers':
                    if was_active:
                        self.output.zero_all()
                    self.output.initialize()
                    self.reason = 'Current mode prepared at zero; controller remains disarmed'
                else:
                    return TriggerResponse(False, 'Unknown driver action')
                return TriggerResponse(True, self.reason)
            except Exception as exc:
                self.reason = 'Driver action failed: '+str(exc)
                return TriggerResponse(False, self.reason)

    def on_pose(self, msg):
        self.trials.record('/cs1/pose',message_dict(msg))
        with self.lock:
            p, q = msg.pose.position, msg.pose.orientation
            norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
            age = (rospy.Time.now() - msg.header.stamp).to_sec()
            valid = (msg.header.frame_id == 'emns_world'
                     and abs(norm-1) < 0.02 and abs(p.z-self.config['model']['z_m']) < 1e-5)
            beta = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            pose = np.array([p.x, p.y, beta])
            if not valid or not np.isfinite(pose).all():
                self.stop('invalid camera pose/frame')
                return
            if not -0.05 <= age < self.core.control['pose_timeout_s']:
                # A late message (e.g. queued while a driver action held the
                # lock) is skipped, not fatal. The tick watchdog still stops
                # the run if no fresh pose arrives within pose_timeout_s.
                return
            if self.armed and self.pose is not None:
                if (np.linalg.norm(pose[:2]-self.pose[:2]) > self.core.control['max_pose_jump_m']
                        or (self.controller_mode != 'field_steer'
                            and abs(wrap(beta-self.pose[2])) > self.core.control['max_heading_jump_rad'])):
                    self.stop('tracking jump; reinitialize heading')
                    return
            now = time.monotonic()
            self.speed = (0.0 if self.pose is None or now-self.pose_time > self.core.control['pose_timeout_s']
                          else np.linalg.norm(pose[:2]-self.pose[:2])/max(1e-3, now-self.pose_time))
            self.pose, self.pose_time = pose, now

    def on_valid(self, msg):
        self.tracking_valid=bool(msg.data)
        self.trials.record('/cs1/tracking_valid',message_dict(msg))
        if not msg.data:
            with self.lock:
                self.pose_time = 0
                # Uninitialized tracking publishes invalid on every camera frame.
                # Zero once when active; do not continuously command idle drivers.
                self.stop('tracking lost; initialize heading again',
                          send_zero=self.armed or self.arming or bool(np.any(self.previous)))

    def sync_input_source(self):
        source = rospy.get_param('/cs1/input_source', 'gamepad')
        if source != self.input_source:
            self.stop('Input changed; release controls and rearm')
            self.drive = self.turn = self.command_time = 0.0
            self.input_source = source

    def on_command(self, msg, source):
        self.trials.record('/cs1/'+source+'_command',message_dict(msg))
        with self.lock:
            if source != self.input_source:
                return
            if not np.isfinite([msg.linear.x, msg.angular.z]).all():
                self.stop('invalid gamepad command')
                return
            self.drive = float(np.clip(msg.linear.x, -1, 1))
            self.turn = float(np.clip(msg.angular.z, -1, 1))
            self.command_time = time.monotonic()
            self.pub_command.publish(msg)

    def arm_check(self):
        if self.hardware:
            mode = self.config.get('controller', 'pose_hold')
            if not self.config.get(mode, {}).get('hardware_validated', False):
                return ('Powered arming is locked for the %s layout until it is hardware-validated '
                        '(set %s.hardware_validated in its config)' % (getattr(self, 'layout', 'single'), mode))
            if self.output.initialized is not True:
                return 'Prepare Current mode / Zero before arming'
            if not all(self.output.feedback_snapshot()['fresh']):
                return 'Need fresh current feedback from all ten drivers; Enable feedback and inspect driver alarms'
        if self.pose is None or time.monotonic()-self.pose_time > self.core.control['pose_timeout_s']:
            return 'Need fresh initialized camera pose'
        if time.monotonic()-self.command_time > self.core.control['command_timeout_s'] or self.drive or self.turn:
            return 'Need fresh neutral %s command; release controls' % self.input_source
        if not self.core.workspace.permitted(self.pose, self.core.control):
            return 'Place free magnet inside guarded TFW first'
        return None

    def on_arm(self, request):
        with self.lock:
            if self.closed:
                return SetBoolResponse(False, 'Controller is shutting down')
            self.sync_input_source()
            if not request.data:
                self.stop('disarmed')
                return SetBoolResponse(True, self.reason)
            if self.armed:
                return SetBoolResponse(True, 'already armed')
            if self.arming:
                return SetBoolResponse(False, 'Initialization already in progress')
            reason = self.arm_check()
            if reason:
                return SetBoolResponse(False, reason)
            if hasattr(self,'trials'):
                phase=self.trials.snapshot()['phase']
                if phase=='idle' and not self.video_required:
                    self.trials.prepare(self.recording_metadata(),False)
                if self.trials.snapshot()['phase']!='prepared' or not self.trials.video_ready:
                    return SetBoolResponse(False,'Recording not ready: use Arm in the main UI; resolve any previous trial first')
                if self.video_required and time.monotonic()-self.ui_heartbeat>1.0:
                    return SetBoolResponse(False,'UI recorder heartbeat missing')
            self.arm_generation += 1
            generation = self.arm_generation
            self.arming = True
        # Allow camera/gamepad callbacks during the driver's mode-set delay.
        # Recheck the newest measurements before applying any holding currents.
        try:
            # Hardware is explicitly prepared by Current mode / Zero. Do not
            # repeat mode switching or broadcast zero when Arm is clicked.
            if self.output.initialized is not True:
                self.output.initialize()
        except Exception as exc:
            with self.lock:
                self.arming = False
                self.stop('CAN initialization failed')
            return SetBoolResponse(False, str(exc))
        with self.lock:
            self.arming = False
            if self.closed:
                return SetBoolResponse(False, 'Controller is shutting down')
            reason = self.arm_check()
            if generation != self.arm_generation or reason:
                self.stop(reason or 'Arming cancelled')
                return SetBoolResponse(False, self.reason)
            target = np.array([0, self.config['model']['holding_field_T'], 0, 0])
            currents, error = self.core.model.allocate(self.pose, target, np.zeros(10))
            if error > self.core.control['allocation_tolerance']:
                self.stop('Holding task cannot be allocated')
                return SetBoolResponse(False, 'Holding task cannot be allocated')
            try:
                # Arm captures the reference at ZERO current. Subsequent ticks
                # ramp toward the holding solution with displacement limits.
                # Both recorders must be ready before the ramp can begin.
                if hasattr(self,'trials'):
                    self.trials.start({'pose_at_arm':self.pose.tolist(),
                        'heading_unwrapped_at_arm_rad':self.heading_unwrapped,
                        'commanded_currents_at_arm_A':[0.0]*10,
                        'requested_steady_holding_currents_A':currents.tolist()})
                    self.trials.record('controller',self.telemetry())
                    self.publish_recording()
                    reason=self.arm_check()
                    if reason:
                        self.stop(reason)
                        return SetBoolResponse(False,reason)
                self.core.begin_hold(self.pose)
                self.output.write(np.zeros(10))
                self.previous, self.armed, self.reason = np.zeros(10), True, 'starting pose hold; movement input inhibited'
                if hasattr(self,'trials'):
                    self.trials.record('controller',self.telemetry())
                    self.publish_recording()
                return SetBoolResponse(True, 'armed')
            except Exception as exc:
                self.stop('CAN initialization failed')
                return SetBoolResponse(False, str(exc))

    def tick(self):
        with self.lock:
            if self.closed:
                return
            self.sync_input_source()
            now = time.monotonic()
            if getattr(self,'tail_deadline',None) is not None and (
                    now >= self.tail_deadline or self.trials.error
                    or (self.video_required and now-self.ui_heartbeat > 1.0)):
                self.trials.record('controller',self.telemetry())
                self.finish_recording(self.tail_reason)
            if self.armed:
                if self.hardware and not all(self.output.feedback_snapshot(now)['fresh']):
                    self.stop('Driver current feedback lost; inspect driver alarms')
                elif self.trials.error or (self.video_required and now-self.ui_heartbeat>1.0):
                    self.stop(self.trials.error or 'UI recorder heartbeat lost')
                elif now-self.pose_time > self.core.control['pose_timeout_s']:
                    self.stop('stale pose')
                elif now-self.command_time > self.core.control['command_timeout_s']:
                    self.stop('input heartbeat lost')
                else:
                    currents, reason = self.core.command(self.pose, self.drive, self.turn,
                                                          self.previous, 1/self.core.control['rate_hz'], self.speed)
                    if currents is None:
                        self.stop(reason)
                    else:
                        self.output.write(currents)
                        self.previous, self.reason = currents, reason
            self.pub_current.publish(Float64MultiArray(data=self.previous.tolist()))
            field_heading = getattr(self.core, 'field_heading', None)
            if self.armed and field_heading is not None:
                self.pub_field_heading.publish(self._Float64(data=float(field_heading)))
            feedback = self.output.feedback_snapshot(now)
            self.pub_feedback.publish(String(data=json.dumps(feedback, allow_nan=False)))
            status = {'layout':self.layout,'hardware':self.hardware,'armed':self.armed,'reason':self.reason,
                      'input_source':self.input_source,
                      'last_run_stop':self.last_run_stop,
                      'pose_hold':self.core.snapshot(),
                      'pose':None if self.pose is None else self.pose.tolist(),
                      'pose_fresh':now-self.pose_time < self.core.control['pose_timeout_s'],
                      'currents':self.previous.tolist(), 'current_feedback':feedback,
                      'applied_model':self.applied_model(),
                      'recording':self.trials.snapshot()}
            self.pub_status.publish(String(data=json.dumps(status)))
            self.trials.record('controller',self.telemetry())
            self.publish_recording()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.arm_generation += 1
            self.armed = False
            self.previous = np.zeros(10)
            try:
                self.output.close()
            finally:
                if hasattr(self,'trials'):
                    self.trials.stop(getattr(self,'tail_reason',None) or 'Controller shutdown')
                    self.tail_deadline = None
                    if self.trials.worker is not None:self.trials.worker.join(timeout=3)


if __name__ == '__main__':
    rospy.init_node('cs1_controller')
    node = ControllerNode()
    try:
        rate = rospy.Rate(node.core.control['rate_hz'])
        while not rospy.is_shutdown():
            node.tick()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    except Exception:
        node.close()
        raise
