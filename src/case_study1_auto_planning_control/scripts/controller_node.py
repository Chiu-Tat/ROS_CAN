#!/usr/bin/env python3
"""Autonomous controller with the shared single CAN owner and trial recorder."""
import json
import time
import numpy as np
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from cs1_experiment.core import wrap
from cs1_autonomy.control import AutonomousController
from cs1_autonomy.shared import shared_node


class AutoNode(shared_node('controller_node.py').ControllerNode):
    def create_core(self, config):
        return AutonomousController(config)

    def __init__(self):
        super().__init__()
        self.input_source = 'autonomous'
        self.planning = False
        self.plan_pub = rospy.Publisher('/cs1/autonomy/plan', String, queue_size=1, latch=True)
        self.plan_service = rospy.Service('/cs1/autonomy/plan_from_pose', Trigger, self.prepare_plan)

    def sync_input_source(self):
        # Autonomy is owned by this node, never by keyboard/gamepad messages.
        self.input_source = 'autonomous'

    def on_command(self, msg, source):
        return

    def prepare_plan(self, _request):
        with self.lock:
            if self.armed or self.arming or self.planning:
                return TriggerResponse(False, 'Stop before planning; one request at a time')
            if self.trials.snapshot()['phase'] != 'idle':
                return TriggerResponse(False, 'Save or Delete the previous trial first')
            self.planning = True
            self.core.plan = None
            self.plan_pub.publish(String(data=json.dumps({'ready':False, 'message':'Checking path…'})))
            generation = self.arm_generation
        try:
            # Expensive nominal path checks do not hold the camera/control lock.
            from cs1_autonomy.planning import check_swept
            if not self.core.planner.validated:
                check_swept(self.core.planner.nominal, self.core.workspace, self.core.control)
                self.core.planner.validated = True
            with self.lock:
                if (self.pose is None or not self.tracking_valid
                        or time.monotonic()-self.pose_time > self.core.control['pose_timeout_s']):
                    raise ValueError('Set the start pose and wait for fresh tracking first')
                pose = self.pose.copy()
            plan = self.core.planner.prepare(pose)
            with self.lock:
                if generation != self.arm_generation or self.armed or self.closed:
                    raise ValueError('Planning cancelled; check tracking and plan again')
                h = self.core.settings
                if (np.linalg.norm(self.pose[:2]-pose[:2]) > h['plan_drift_limit_m']
                        or abs(wrap(self.pose[2]-pose[2])) > h['plan_heading_drift_rad']):
                    raise ValueError('Magnet moved during planning; plan again')
                self.core.plan = plan
                self.reason = 'Plan ready; Start auto to record and run'
                self.publish_plan(plan)
            return TriggerResponse(True, 'Plan ready: approach %.1f s, total %.1f s' %
                                   (plan['approach_duration_s'], plan['samples'][-1, 3]))
        except Exception as exc:
            self.plan_pub.publish(String(data=json.dumps({'ready':False, 'message':str(exc)})))
            return TriggerResponse(False, str(exc))
        finally:
            with self.lock:self.planning = False

    def publish_plan(self, plan):
        self.plan_pub.publish(String(data=json.dumps({'ready':True, 'layout':self.layout,
            'samples':plan['samples'].tolist(), 'approach_duration_s':plan['approach_duration_s'],
            'planned_from_pose':plan['planned_from_pose'], 'requested_goal':plan['requested_goal'],
            'nominal_goal':plan['nominal_goal'], 'source_sha256':plan['source']['sha256']})))

    def arm_check(self):
        self.command_time = time.monotonic()
        reason = super().arm_check()
        if reason:return reason
        if self.planning or self.core.plan is None:return 'Plan from the measured start pose first'
        planned = np.asarray(self.core.plan['planned_from_pose'])
        h = self.core.settings
        if (np.linalg.norm(self.pose[:2]-planned[:2]) > h['plan_drift_limit_m']
                or abs(wrap(self.pose[2]-planned[2])) > h['plan_heading_drift_rad']):
            return 'Magnet moved since planning; plan again'
        return None

    def recording_metadata(self):
        value = super().recording_metadata()
        value['experiment_title'] = 'CS1 autonomous planning control: '+self.layout
        if self.core.plan is not None:
            value['autonomy'] = {k:(v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in self.core.plan.items()}
        return value

    def tick(self):
        with self.lock:
            self.command_time = time.monotonic()
            if self.armed and self.core.phase == 'complete':
                self.stop('Trajectory complete')
                self.core.plan = None
            super().tick()
            if self.trials.snapshot()['phase'] == 'recording':
                self.trials.record('/cs1/autonomy/tracking', self.core.snapshot())


if __name__ == '__main__':
    rospy.init_node('cs1_controller')
    node = AutoNode()
    try:
        rate = rospy.Rate(node.core.control['rate_hz'])
        while not rospy.is_shutdown():
            node.tick(); rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.close()
