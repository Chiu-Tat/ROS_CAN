#!/usr/bin/env python3
"""Shared camera/current/recording UI with autonomous path controls."""
import json
import sys
import threading
import numpy as np
import rospy
from PyQt5 import QtCore, QtWidgets
from std_msgs.msg import String
from std_srvs.srv import Trigger
from cs1_autonomy.planning import load_reference, config_path
from cs1_autonomy.shared import shared_node


class AutoPanel(shared_node('control_panel.py').ControlPanel):
    plan_received = QtCore.pyqtSignal(object)

    def __init__(self, connect_ros=True, layout='single'):
        self.auto_ready = False
        self.auto_route = None
        self.auto_plan_busy = False
        super().__init__(connect_ros, layout)
        if not connect_ros:
            self.config = json.loads(config_path(self.layout).read_text())
        self.nominal, self.reference_source = load_reference(self.layout)
        self.setWindowTitle('Case Study 1 · Autonomous · '+self.layout.title())
        side = self.controls_scroll.widget().layout()
        for label in self.controls_scroll.widget().findChildren(QtWidgets.QLabel):
            if label.text() == 'Case 1 · Manual control':label.setText('Case 1 · Autonomous')
        self.arm_button.setText('3. Start auto')
        self.arm_button.setToolTip('Prepare video/data, ramp holding field, approach the path, then track it.')
        self.plan_button = QtWidgets.QPushButton('2. Plan from camera pose')
        self.plan_button.clicked.connect(self.plan_from_pose)
        self.plan_button.setToolTip('Check the paper path and a short approach from the measured position. No current is applied.')
        side.insertWidget(side.indexOf(self.arm_button), self.plan_button)
        self.auto_status = QtWidgets.QLabel('Plan from the measured start pose.')
        self.auto_status.setWordWrap(True)
        side.insertWidget(side.indexOf(self.arm_button)+1, self.auto_status)
        s, g = self.nominal[0], self.nominal[-1]
        self.instructions.setText('Nominal start: %.1f, %.1f mm · %.0f°\nPath end: %.1f, %.1f mm · %.0f°\nPlace within %.0f mm of start.\nBlue: path · Orange: approach · Esc: stop' %
            (s[0]*1000,s[1]*1000,np.degrees(s[2]),g[0]*1000,g[1]*1000,np.degrees(g[2]),
             self.config['autonomous']['max_start_offset_m']*1000))
        self.instructions.setToolTip('Path stays in world coordinates. Planning connects the actual start without shifting obstacles or the nominal path.\nW/S/A/D do not command motion in this package.')
        self.recorder.label.setText('Starts with Start auto')
        self.plan_received.connect(self.on_plan)
        if connect_ros:
            self.plan_sub = rospy.Subscriber('/cs1/autonomy/plan',String,
                lambda msg:self.plan_received.emit(json.loads(msg.data)),queue_size=1)
        self.refresh()

    def keyboard_vector(self):
        return 0., 0.

    def plan_from_pose(self):
        if self.auto_plan_busy or not self.connect_ros:return
        self.auto_plan_busy = True
        self.auto_ready = False
        self.plan_button.setEnabled(False)
        self.auto_status.setText('Checking route and approach…')
        def worker():
            try:
                rospy.wait_for_service('/cs1/autonomy/plan_from_pose',timeout=2)
                reply = rospy.ServiceProxy('/cs1/autonomy/plan_from_pose',Trigger)()
                self.service_message.emit(reply.message)
                if not reply.success:self.plan_received.emit({'ready':False,'message':reply.message})
            except Exception as exc:
                self.plan_received.emit({'ready':False,'message':str(exc)})
            finally:self.auto_plan_busy = False
        threading.Thread(target=worker,daemon=True).start()

    def on_plan(self, value):
        self.auto_ready = bool(value.get('ready'))
        self.auto_route = np.asarray(value['samples']) if self.auto_ready else None
        self.auto_status.setText('Plan ready · Start auto' if self.auto_ready else value.get('message','Plan again'))

    def refresh(self):
        super().refresh()
        if not hasattr(self,'plan_button'):return
        with self.state_lock:state = dict(self.state)
        armed = state.get('armed',False)
        auto = state.get('pose_hold',{})
        self.plan_button.setEnabled(not armed and not self.auto_plan_busy
                                    and state.get('recording',{}).get('phase','idle') == 'idle')
        self.arm_button.setEnabled(self.auto_ready and not armed
            and (not state.get('hardware') or self.config['autonomous']['hardware_validated']))
        if armed:
            self.auto_status.setText('%s · %.1f / %.1f s\nError: %.2f mm' %
                (auto.get('phase','running').title(),auto.get('progress_s',0),auto.get('duration_s',0),
                 1000*np.linalg.norm(auto.get('pose_error',[0,0])[:2])))
        elif state.get('last_run_stop'):
            self.auto_status.setText(state['last_run_stop']['reason']+' · Plan again to repeat')
        self.ax.plot(self.nominal[:,0]*1000,self.nominal[:,1]*1000,'--',color='#2467a0',lw=1.6,zorder=2)
        if self.auto_route is not None:
            self.ax.plot(self.auto_route[:,0]*1000,self.auto_route[:,1]*1000,color='#de8b25',lw=.8,alpha=.8,zorder=2)
        self.ax.plot(self.nominal[0,0]*1000,self.nominal[0,1]*1000,'o',color='#2467a0',ms=6,zorder=5)
        self.ax.plot(self.nominal[-1,0]*1000,self.nominal[-1,1]*1000,'*',color='#a32b36',ms=12,zorder=5)
        self.canvas.draw_idle()


if __name__ == '__main__':
    rospy.init_node('cs1_control_panel',disable_signals=True)
    app = QtWidgets.QApplication(sys.argv)
    panel = AutoPanel();panel.showMaximized()
    sys.exit(app.exec_())
