"""Qt-side full-window recording and explicit Save/Delete trial review."""
import json
from pathlib import Path
import threading
import time

from PyQt5 import QtCore, QtWidgets
import rospy
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

from .recording import UiVideoRecorder, TrialStore, write_json


class TrialUi(QtCore.QObject):
    state_received = QtCore.pyqtSignal(object)
    prepare_requested = QtCore.pyqtSignal(object)
    finish_requested = QtCore.pyqtSignal(object)
    resolved = QtCore.pyqtSignal(object)

    def __init__(self, panel, side, connect_ros):
        super().__init__(panel)
        self.panel = panel
        self.state = {'phase':'idle'}
        self.video = None
        self.video_id = None
        self.prepare_waiter = None
        self.finish_waiters = []
        self.done_notified = False
        self.arm_token = 0
        self.arm_busy = False
        self.close_requested = False
        self.shutdown_deadline = None
        self.dialog = None
        self.reviewed_id = None
        self.auto_review = True
        self.last_heartbeat = self.last_capture = 0.
        self.require_video = rospy.get_param('/cs1_controller/record_ui',True) if connect_ros else False
        box=QtWidgets.QGroupBox('Recording')
        layout=QtWidgets.QVBoxLayout(box)
        self.label=QtWidgets.QLabel('Starts with holding')
        self.label.setWordWrap(True);layout.addWidget(self.label)
        self.review_button=QtWidgets.QPushButton('Save / Delete trial')
        self.review_button.setEnabled(False);self.review_button.clicked.connect(self.review)
        layout.addWidget(self.review_button)
        # After control stops, recording continues briefly (e.g. to capture the
        # free magnet snapping onto the fixed one). This ends it early.
        self.stop_now_button=QtWidgets.QPushButton('Stop recording now')
        self.stop_now_button.setVisible(False);self.stop_now_button.clicked.connect(self.stop_now)
        layout.addWidget(self.stop_now_button)
        side.insertWidget(3,box)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Fixed)
        self.state_received.connect(self.on_state)
        self.prepare_requested.connect(self.prepare_video)
        self.finish_requested.connect(self.finish_video)
        self.resolved.connect(self.on_resolved)
        self.heartbeat_pub = None
        if connect_ros:
            self.heartbeat_pub=rospy.Publisher('/cs1/recording/ui_heartbeat',Bool,queue_size=1)
            self.sub=rospy.Subscriber('/cs1/recording/state',String,
                lambda msg:self.state_received.emit(json.loads(msg.data)),queue_size=20)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(50)

    @staticmethod
    def call(name, value=None):
        name='/cs1/recording/'+name
        rospy.wait_for_service(name,timeout=2)
        return rospy.ServiceProxy(name,Trigger)() if value is None else rospy.ServiceProxy(name,SetBool)(value)

    def cancel_intent(self):
        self.arm_token+=1

    def request_arm(self, armed, wait=False):
        if not self.panel.connect_ros:return
        if not armed:self.cancel_intent()
        if armed and self.arm_busy:return
        token=self.arm_token
        if armed:self.arm_busy=True
        def request():
            prepared=False
            try:
                if armed and self.require_video:
                    response=self.call('prepare')
                    if not response.success:raise RuntimeError(response.message)
                    prepared=True
                    value=json.loads(response.message)
                    waiter={'state':value,'event':threading.Event(),'error':''}
                    self.prepare_requested.emit(waiter)
                    if not waiter['event'].wait(6):raise RuntimeError('UI video encoder did not become ready')
                    if waiter['error']:raise RuntimeError(waiter['error'])
                    if token!=self.arm_token:raise RuntimeError('Arming cancelled')
                    response=self.call('video_ready')
                    if not response.success:raise RuntimeError(response.message)
                if armed and token!=self.arm_token:raise RuntimeError('Arming cancelled')
                rospy.wait_for_service('/cs1/arm',timeout=1)
                response=rospy.ServiceProxy('/cs1/arm',SetBool)(armed)
                if armed and not response.success:raise RuntimeError(response.message)
                self.panel.service_message.emit(response.message)
            except Exception as exc:
                self.panel.service_message.emit(str(exc))
                if prepared:
                    finished={'event':threading.Event()}
                    self.finish_requested.emit(finished)
                    if finished['event'].wait(5):
                        try:self.call('cancel')
                        except Exception:pass
            finally:
                if armed:self.arm_busy=False
        if wait:request()
        else:threading.Thread(target=request,daemon=True).start()

    def prepare_video(self, waiter):
        try:
            value=waiter['state']
            self.video=UiVideoRecorder(value['path'],(self.panel.width(),self.panel.height()),fps=5)
            self.video_id=value['id'];self.done_notified=False
            self.prepare_waiter=waiter
        except Exception as exc:
            waiter['error']=str(exc);waiter['event'].set()

    def finish_video(self, waiter=None):
        if waiter is not None:self.finish_waiters.append(waiter)
        if self.video is not None and not self.video.done.is_set():
            self.video.stop(self.state.get('stop_ns'))
        else:
            for entry in self.finish_waiters:entry['event'].set()
            self.finish_waiters=[]

    def on_state(self, state):
        self.state=state
        phase=state['phase']
        if phase=='recording' and self.video is not None and self.video_id==state['id']:
            if self.video.start_ns is None:
                self.video.begin(state['start_ns'])
                self.capture()
        if phase in ('stopping','cancelled') and self.video is not None:
            self.finish_video()
        self.review_button.setEnabled(phase=='pending')
        if phase=='pending' and self.auto_review and self.reviewed_id!=state.get('id'):
            self.reviewed_id=state['id']
            QtCore.QTimer.singleShot(0,self.review)
        self.update_label()

    def in_post_stop(self):
        return self.state.get('phase')=='recording' and bool(self.state.get('control_stop_reason'))

    def stop_now(self):
        self.stop_now_button.setEnabled(False)
        def run():
            try:self.call('stop_now')
            except Exception as exc:self.panel.service_message.emit(str(exc))
        threading.Thread(target=run,daemon=True).start()

    def update_label(self):
        phase=self.state['phase']
        post_stop=self.in_post_stop()
        if self.stop_now_button.isVisible()!=post_stop:
            self.stop_now_button.setVisible(post_stop);self.stop_now_button.setEnabled(post_stop)
        if phase=='recording':
            elapsed=max(0.,(time.monotonic_ns()-self.state['start_ns'])/1e9)
            if post_stop:
                left=max(0.,(self.state.get('post_stop_until_ns',0)-time.monotonic_ns())/1e9)
                self.label.setText('● Finishing recording · %.0f s left'%left)
                self.label.setToolTip('Control stopped: '+self.state['control_stop_reason'])
            else:
                self.label.setText('● Recording · %.1f s'%elapsed)
                self.label.setToolTip(self.state['id'])
            self.label.setStyleSheet('color:#a32b36;font-weight:bold')
        else:
            labels={'idle':'Ready · starts with holding', 'prepared':'Preparing…',
                    'cancelled':'Cancelled · finishing…',
                    'stopping':'Saving temporary files…', 'pending':'Unsaved trial · choose Save or Delete'}
            self.label.setText(labels.get(phase,phase));self.label.setStyleSheet('');self.label.setToolTip('')

    def capture(self):
        if self.video is not None and self.video.start_ns is not None and not self.video.finish.is_set():
            captured=time.monotonic_ns()
            self.video.submit(self.panel.grab().toImage(),captured)
            self.last_capture=time.monotonic()

    def poll(self):
        now=time.monotonic()
        if self.heartbeat_pub is not None and now-self.last_heartbeat>.2 and not rospy.is_shutdown():
            healthy = self.state['phase']!='recording' or (self.video is not None and not self.video.error and not self.video.done.is_set())
            self.heartbeat_pub.publish(Bool(data=healthy));self.last_heartbeat=now
        if self.prepare_waiter is not None and self.video is not None and self.video.ready.is_set():
            self.prepare_waiter['error']=self.video.error
            self.prepare_waiter['event'].set();self.prepare_waiter=None
        if self.video is not None:
            if self.state['phase']=='recording' and now-self.last_capture>=.2:self.capture()
            if self.video.done.is_set() and not self.done_notified:
                self.done_notified=True
                for entry in self.finish_waiters:entry['event'].set()
                self.finish_waiters=[]
                if self.video.error and self.state['phase']=='recording':
                    self.panel.service_message.emit(self.video.error)
                    self.request_arm(False)
                threading.Thread(target=self.notify_done,daemon=True).start()
        self.update_label()
        if self.panel.connect_ros and rospy.is_shutdown():self.handle_shutdown()
        if self.close_requested and self.can_close():
            self.close_requested=False;QtCore.QTimer.singleShot(0,self.panel.close)

    def notify_done(self):
        try:self.call('video_done')
        except Exception:pass

    def review(self):
        if self.state['phase']!='pending' or self.dialog is not None:return
        self.reviewed_id=self.state['id']
        dialog=QtWidgets.QDialog(self.panel);dialog.setWindowTitle('Save this control trial?');dialog.setModal(True)
        layout=QtWidgets.QVBoxLayout(dialog)
        duration=((self.state.get('stop_ns') or 0)-(self.state.get('start_ns') or 0))/1e9
        label=QtWidgets.QLabel('Trial: %s\nDuration: %.2f s\nStop: %s\n\nSave the UI video and all recorded data, or delete this trial?'%
            (self.state['id'],max(0,duration),self.state.get('stop_reason','unknown')))
        label.setWordWrap(True);layout.addWidget(label)
        layout.addWidget(QtWidgets.QLabel('Brief introduction / notes (optional):'))
        note=QtWidgets.QPlainTextEdit();note.setPlaceholderText('For example: keyboard forward/backward test, one fixed magnet.')
        note.setMaximumHeight(90);layout.addWidget(note)
        buttons=QtWidgets.QHBoxLayout();layout.addLayout(buttons)
        yes=QtWidgets.QPushButton('Yes — Save video + data')
        no=QtWidgets.QPushButton('No — Delete this trial')
        buttons.addWidget(yes);buttons.addWidget(no)
        yes.clicked.connect(lambda:self.resolve(True,note.toPlainText(),yes,no))
        no.clicked.connect(lambda:self.resolve(False,'',yes,no))
        dialog.finished.connect(self.dialog_finished)
        self.dialog=dialog;dialog.resize(540,300);dialog.open()

    def dialog_finished(self, _result):
        self.dialog=None
        # Closing the dialog is not permission to delete; keep the trial pending.
        self.close_requested=False

    def resolve(self, keep, note='', *buttons):
        for button in buttons:button.setEnabled(False)
        state=dict(self.state)
        def run():
            try:
                if rospy.is_shutdown():raise rospy.ROSException('ROS stopped')
                rospy.set_param('/cs1/recording_note',note)
                response=self.call('resolve',keep)
                if not response.success:raise RuntimeError(response.message)
                result={'success':True,'message':response.message}
            except Exception as exc:
                try:
                    folder=Path(state['path']);meta=json.loads((folder/'session.json').read_text())
                    if not (folder/'data_complete.json').exists():raise RuntimeError('Data writer is not finished')
                    if meta.get('ui_video_required') and not (folder/'video_complete.json').exists():raise RuntimeError('Video writer is not finished')
                    meta['phase']='pending';write_json(folder/'session.json',meta)
                    destination=TrialStore.resolve_folder(folder.parent.parent,folder,keep,note)
                    result={'success':True,'message':'Saved: '+str(destination) if keep else 'Trial video/data deleted','offline':True}
                except Exception as second:result={'success':False,'message':str(exc)+'; '+str(second)}
            result['resolved_id']=state.get('id')
            self.resolved.emit(result)
        threading.Thread(target=run,daemon=True).start()

    def on_resolved(self, value):
        self.panel.message.setText(value['message'])
        if value['success']:
            closing=self.close_requested
            if self.dialog is not None:self.dialog.accept()
            if value.get('offline') or self.state.get('id')==value.get('resolved_id'):self.state={'phase':'idle'}
            self.close_requested=closing
            if self.state['phase']=='pending' and self.state.get('id')!=self.reviewed_id and self.auto_review:self.review()
        elif self.dialog is not None:
            for button in self.dialog.findChildren(QtWidgets.QPushButton):button.setEnabled(True)

    def can_close(self):
        return self.state['phase']=='idle' and not self.arm_busy and (self.video is None or self.video.done.is_set())

    def request_close(self):
        self.panel.pressed_keys.clear();self.panel.publish_keyboard()
        self.close_requested=True
        if self.state['phase']=='pending':self.review()
        else:
            self.request_arm(False)
            # Closing the window ends any post-stop recording immediately.
            threading.Thread(target=self._stop_now_after_disarm,daemon=True).start()

    def _stop_now_after_disarm(self):
        for _ in range(20):
            if self.in_post_stop():
                try:self.call('stop_now')
                except Exception:pass
                return
            if self.state.get('phase')!='recording':return
            time.sleep(.1)

    def handle_shutdown(self):
        if self.shutdown_deadline is None:self.shutdown_deadline=time.monotonic()+5
        if self.video is not None and not self.video.finish.is_set():self.finish_video()
        folder=Path(self.state['path']) if self.state.get('path') else None
        if folder is not None and (folder/'data_complete.json').exists() and (self.video is None or self.video.done.is_set()):
            meta=json.loads((folder/'session.json').read_text());meta['phase']='pending'
            write_json(folder/'session.json',meta);meta['path']=str(folder)
            if self.state['phase']!='pending':self.on_state(meta)
        elif time.monotonic()>self.shutdown_deadline:
            # A crashed writer must never cause automatic deletion or fake Save.
            # Preserve .pending for recovery at the next startup.
            self.state={'phase':'idle'}

    def shutdown(self):
        self.timer.stop()
        # Stop ROS callbacks into this Qt object once the window closes.
        if getattr(self, 'sub', None) is not None:
            self.sub.unregister();self.sub = None
        if self.video is not None:self.video.stop(self.state.get('stop_ns'))
