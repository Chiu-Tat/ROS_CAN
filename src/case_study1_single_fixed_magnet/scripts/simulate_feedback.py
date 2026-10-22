#!/usr/bin/env python3
"""Preview plant only; no camera, gamepad emulation, or CAN connection."""
import math
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
import json
from std_srvs.srv import Trigger, TriggerResponse

from cs1_experiment.core import ManualController, load_config, wrap


class PreviewPlant:
    def __init__(self):
        self.core = ManualController(load_config(rospy.get_param('~config',None)))
        self.pose = self.core.workspace.seed()
        self.currents = np.zeros(10)
        self.current_time = 0
        self.armed = False
        self.pub = rospy.Publisher('/cs1/pose', PoseStamped, queue_size=1)
        self.heading_pub = rospy.Publisher('/cs1/heading_unwrapped',Float64,queue_size=1)
        self.initial_pub = rospy.Publisher('/cs1/initial_pose',PoseStamped,queue_size=1,latch=True)
        self.initial_metadata_pub = rospy.Publisher('/cs1/initial_pose_metadata',String,queue_size=1,latch=True)
        self.valid = rospy.Publisher('/cs1/tracking_valid', Bool, queue_size=1)
        self.sub = rospy.Subscriber('/cs1/currents', Float64MultiArray, self.on_current, queue_size=1)
        self.status_sub = rospy.Subscriber('/cs1/status', String, self.on_status, queue_size=1)
        self.service = rospy.Service('/cs1/initialize_heading', Trigger,self.initialize)
        self.initialize(None)
        rospy.loginfo('SIMULATED feedback only. Initial pose %s',self.pose)

    def initialize(self, _request):
        msg=PoseStamped();msg.header.stamp=rospy.Time.now();msg.header.frame_id='emns_world'
        msg.pose.position.x,msg.pose.position.y=map(float,self.pose[:2]);msg.pose.position.z=self.core.model.settings['z_m']
        msg.pose.orientation.z,msg.pose.orientation.w=math.sin(self.pose[2]/2),math.cos(self.pose[2]/2)
        self.initial_pub.publish(msg)
        self.initial_metadata_pub.publish(String(data=json.dumps({'source':'simulation',
            'position_m':[float(self.pose[0]),float(self.pose[1]),self.core.model.settings['z_m']],
            'heading_unwrapped_rad':float(self.pose[2]),'initialization_rule':'simulation_known_heading'})))
        return TriggerResponse(True,'Preview initial pose recorded; heading is known from simulation')

    def on_current(self, msg):
        values = np.asarray(msg.data,float)
        if values.shape == (10,) and np.isfinite(values).all():
            self.currents, self.current_time = values, time.monotonic()

    def on_status(self, msg):
        self.armed = bool(json.loads(msg.data).get('armed'))

    def tick(self, dt):
        currents = self.currents if time.monotonic()-self.current_time < 0.3 else np.zeros(10)
        matrix, offset = self.core.model.matrix(self.pose)
        by,bz,fy,fz = matrix @ currents + offset
        beta = self.pose[2]
        forward,lateral = np.array([math.cos(beta),math.sin(beta)]),np.array([-math.sin(beta),math.cos(beta)])
        if self.armed:
            self.pose[:2] += (forward*fz+lateral*fy)/5.0*dt
            self.pose[2] += wrap(math.atan2(by,bz))*dt/0.15
        self.valid.publish(Bool(data=True))
        msg = PoseStamped()
        msg.header.stamp,msg.header.frame_id = rospy.Time.now(),'emns_world'
        msg.pose.position.x,msg.pose.position.y = map(float,self.pose[:2])
        msg.pose.position.z = self.core.model.settings['z_m']
        msg.pose.orientation.z,msg.pose.orientation.w = math.sin(self.pose[2]/2),math.cos(self.pose[2]/2)
        self.pub.publish(msg)
        self.heading_pub.publish(Float64(data=float(self.pose[2])))


if __name__ == '__main__':
    rospy.init_node('cs1_preview_feedback')
    plant = PreviewPlant()
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        plant.tick(0.02)
        rate.sleep()
