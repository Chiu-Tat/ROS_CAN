#!/usr/bin/env python3
"""Live heading-dependent TFW display and explicit heading/arming controls."""
import json
import math
import sys
import threading
import time

import numpy as np
import rospy
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from std_msgs.msg import Bool, Float64, String
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool, Trigger

from cs1_experiment.core import PACKAGE, LAYOUT_LABELS, Workspace, fixed_magnets, load_config
from cs1_experiment.trial_ui import TrialUi
from cs1_experiment.current_bars import CurrentBars
from cs1_experiment.magnet_view import draw_magnet


class CameraView(QtWidgets.QLabel):
    """Display the calibrated 100 mm square without stretching its aspect ratio."""
    def __init__(self):
        super().__init__('Waiting for OpenMV camera frames\nUpload the updated camera script to enable video')
        self.original = None
        self.affine = None
        self.display_image = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(300,225)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet('background:#15212a;color:#c9d5dd;border:1px solid #314653;')
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def set_frame(self, image):
        self.original = image.copy()
        if self.affine is None:
            self.display_image = image.transformed(QtGui.QTransform().rotate(-90))
        else:
            side = 600
            self.display_image = QtGui.QImage(side,side,QtGui.QImage.Format_RGB32)
            self.display_image.fill(QtGui.QColor('#15212a'))
            painter = QtGui.QPainter(self.display_image)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform,True)
            painter.setTransform(self.world_transform(side))
            painter.drawImage(0,0,self.original)
            painter.end()
        self.render_frame()

    def set_calibration(self, affine):
        value = np.asarray(affine,dtype=float)
        if value.shape != (2,3) or not np.isfinite(value).all() or abs(np.linalg.det(value[:,:2]))<1e-15:
            raise ValueError('Camera view requires a finite invertible pixel-to-EM affine calibration')
        self.affine = value
        if self.original is not None:self.set_frame(self.original)

    def world_transform(self, side):
        # Raw camera pixels -> EM meters -> square display pixels, +y upward.
        a = self.affine
        scale = (side-1)/.1
        return QtGui.QTransform(scale*a[0,0],-scale*a[1,0],
                               scale*a[0,1],-scale*a[1,1],
                               scale*(a[0,2]+.05),scale*(.05-a[1,2]))

    def render_frame(self):
        if self.display_image is not None:
            self.setPixmap(QtGui.QPixmap.fromImage(self.display_image).scaled(
                self.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The calibrated view is a square (x/y -50..+50 mm). Keep the pane
        # square so the image fills it instead of leaving bars above/below.
        if self.affine is not None and self.maximumHeight() != self.width():
            self.setMaximumHeight(max(self.minimumHeight(), self.width()))
        self.render_frame()


class ControlPanel(QtWidgets.QWidget):
    service_message = QtCore.pyqtSignal(str)

    def __init__(self, connect_ros=True, layout='single'):
        super().__init__()
        self.connect_ros = connect_ros
        self.pressed_keys = set()
        self.command_pub = None
        self.key_lock = threading.Lock()
        self.key_vector = (0.0, 0.0)
        self.ui_alive = time.monotonic()
        self.heartbeat_stop = threading.Event()
        self.slice_cache = {}
        self.config = load_config(rospy.get_param('~config',rospy.get_param('/cs1_controller/config',None))
                                  if connect_ros else None,layout=layout)
        self.layout = self.config['model'].get('layout','single')
        self.workspace = Workspace(self.config['model'])
        self.state = {}
        self.status_arrival = 0.0
        self.driver_busy = False
        self.state_lock = threading.Lock()
        self.feedback_pose = None
        self.feedback_arrival = 0.0
        self.feedback_sequence = 0
        self.tracking_valid = False
        self.continuous_heading = None
        self.heading_arrival = 0.0
        self.tracker_status = {}
        self.tracker_arrival = 0.0
        self.initial_pose = None
        self.latest_image = None
        self.image_version = 0
        self.rendered_image_version = -1
        self.last_image_render_ok = False
        self.simulated_feedback = rospy.get_param('/cs1/simulated_feedback',False) if connect_ros else False
        self.trail = []
        self.setWindowTitle('Case Study 1 · '+LAYOUT_LABELS[self.layout])
        self.setStyleSheet('QWidget { font-size: 13px; } QPushButton { padding: 6px; }')
        self.resize(1450, 780)
        layout = QtWidgets.QHBoxLayout(self)
        views = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        views_column = QtWidgets.QWidget()
        views_layout = QtWidgets.QVBoxLayout(views_column)
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.addWidget(views, 1)
        layout.addWidget(views_column, 4)
        camera_box = QtWidgets.QWidget()
        camera_layout = QtWidgets.QVBoxLayout(camera_box)
        camera_title = QtWidgets.QLabel('OPENMV · REAL CAMERA')
        camera_title.setStyleSheet('font-size:17px;font-weight:bold;color:#173b47')
        camera_layout.addWidget(camera_title)
        self.camera_status = QtWidgets.QLabel('Camera: waiting for frames')
        self.camera_status.setWordWrap(True)
        camera_layout.addWidget(self.camera_status)
        self.camera_view = CameraView()
        calibrated_view = False
        if connect_ros and not self.simulated_feedback:
            calibration_path = rospy.get_param('/cs1_controller/calibration',str(PACKAGE/'config/camera_calibration.json'))
            try:
                with open(calibration_path) as stream:calibration = json.load(stream)
                if calibration.get('frame_id') != 'emns_world':raise ValueError('Expected emns_world calibration')
                self.camera_view.set_calibration(calibration['affine'])
                calibrated_view = True
            except (OSError,ValueError,KeyError,TypeError) as exc:
                rospy.logwarn('Camera square unavailable: %s',exc)
        if self.simulated_feedback:
            self.camera_view.setText('SIMULATION PREVIEW\nNo physical camera stream is generated')
        camera_layout.addWidget(self.camera_view,1)
        camera_note = QtWidgets.QLabel('Orange: green-object candidate axes; host checks size.\n'
                                      + ('EM square: x/y −50 to +50 mm · +y up, +x right.' if calibrated_view
                                         else 'Unrectified camera view · EM square calibration unavailable.'))
        camera_note.setWordWrap(True)
        camera_layout.addWidget(camera_note)
        camera_layout.addStretch(0)   # spare height goes below, not into bars
        views.addWidget(camera_box)
        world_box = QtWidgets.QWidget()
        world_layout = QtWidgets.QVBoxLayout(world_box)
        self.pose_status = QtWidgets.QLabel('TFW · waiting for initialized pose feedback')
        self.pose_status.setWordWrap(True)
        world_layout.addWidget(self.pose_status)
        metadata = self.workspace.provenance
        detail = 'TFW: {:,} model-in boxes'.format(metadata['inside_box_count'])
        if metadata.get('stopping_widths'):
            stop = metadata['stopping_widths']
            detail += ' · stop %.3f mm / %.3f°' % (stop['x_m']*1000,math.degrees(stop['heading_rad']))
        self.dataset_status = QtWidgets.QLabel('TFW · {:,} cells'.format(metadata['inside_box_count']))
        self.dataset_status.setToolTip(detail)
        self.dataset_status.setWordWrap(True)
        world_layout.addWidget(self.dataset_status)
        figure = Figure(figsize=(7, 6))
        # Fixed margins: tight_layout on every redraw cost ~80 ms of Qt time.
        figure.subplots_adjust(left=.12, right=.97, bottom=.09, top=.94)
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.ax = figure.add_subplot(111)
        world_layout.addWidget(self.canvas,1)
        views.addWidget(world_box)
        views.setSizes([440,620])
        currents_group = QtWidgets.QGroupBox('COIL CURRENTS · command and measured CAN feedback (A)')
        currents_layout = QtWidgets.QVBoxLayout(currents_group)
        self.current_summary = QtWidgets.QLabel('Measured feedback: waiting for controller')
        self.current_summary.setWordWrap(True)
        currents_layout.addWidget(self.current_summary)
        self.current_bars = CurrentBars(self.config['model']['current_limit_A'])
        currents_layout.addWidget(self.current_bars)
        details = QtWidgets.QToolButton()
        details.setText('Numeric details · bars = measured, black ticks = commanded')
        details.setCheckable(True)
        details.setArrowType(QtCore.Qt.RightArrow)
        details.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        details.setFocusPolicy(QtCore.Qt.NoFocus)
        currents_layout.addWidget(details)
        self.current_table = QtWidgets.QTableWidget(5, 10)
        self.current_table.setHorizontalHeaderLabels(['Coil','Cmd (A)','Measured (A)','Age (s)','State']*2)
        self.current_table.verticalHeader().hide()
        self.current_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.current_table.setFocusPolicy(QtCore.Qt.NoFocus)
        self.current_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.current_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.current_table.verticalHeader().setDefaultSectionSize(24)
        self.current_table.setFixedHeight(150)
        for coil in range(10):
            row, offset = coil%5, (coil//5)*5
            for col, value in enumerate([str(coil+1), '—', '—', '—', 'NO DATA']):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setToolTip('Coil %d: command CAN 0x%03X; feedback CAN 0x%03X' %
                                (coil+1, 0x013+0x10*coil, 0x01B+0x10*coil))
                self.current_table.setItem(row, offset+col, item)
        currents_layout.addWidget(self.current_table)
        self.current_table.setVisible(False)
        details.toggled.connect(self.current_table.setVisible)
        details.toggled.connect(lambda checked:details.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow))
        views_layout.addWidget(currents_group)
        side_widget = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(side_widget)
        side.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setMinimumWidth(310)
        self.controls_scroll.setWidget(side_widget)
        layout.addWidget(self.controls_scroll, 1)
        title = QtWidgets.QLabel('Case 1 · Manual control')
        title.setStyleSheet('font-size:21px;font-weight:bold;color:#173b47')
        side.addWidget(title)
        self.layout_label = QtWidgets.QLabel(LAYOUT_LABELS[self.layout])
        self.layout_label.setToolTip('Choose the layout in main.py before starting. Close the experiment to change layouts.')
        side.addWidget(self.layout_label)
        self.mode = QtWidgets.QLabel('Waiting for controller…')
        self.mode.setWordWrap(True)
        side.addWidget(self.mode)
        # Keyboard is the only movement input in this UI. Make sure the
        # controller listens to it (it ignores commands from other sources).
        if connect_ros:
            rospy.set_param('/cs1/input_source', 'keyboard')
        initial_help = QtWidgets.QLabel('Start: N points into +y (0°–180°).')
        initial_help.setToolTip('The camera measures the initial angle. Keep N at least %.0f° away from horizontal.' %
                               math.degrees(self.config['tracking']['initialization_boundary_margin_rad']))
        initial_help.setWordWrap(True)
        side.addWidget(initial_help)
        self.candidate_status = QtWidgets.QLabel('Camera start pose: waiting for green magnet')
        self.candidate_status.setWordWrap(True)
        side.addWidget(self.candidate_status)
        self.init_button = QtWidgets.QPushButton('1. Set start pose')
        self.init_button.setToolTip('Save the measured initial position and N-pole heading.')
        self.init_button.clicked.connect(self.initialize_heading)
        self.init_button.setEnabled(False)
        side.addWidget(self.init_button)
        arm = self.arm_button = QtWidgets.QPushButton('2. Start holding')
        arm.setToolTip('Arm pose holding and start recording. Requires validated hardware setup.')
        arm.clicked.connect(lambda: self.set_arm(True))
        side.addWidget(arm)
        stop = QtWidgets.QPushButton('ZERO / Stop')
        stop.setToolTip('Disarm and command zero current to all ten coils.')
        stop.setStyleSheet('background:#a32b36;color:white;padding:12px;font-weight:bold')
        stop.clicked.connect(lambda: self.driver_action('zero_currents'))
        side.addWidget(stop)
        driver_grid = QtWidgets.QGridLayout()
        self.driver_buttons = {}
        for index, (action, label, hint) in enumerate([
                ('reset_drivers','Reset','Disarm, send zero currents, then reset all ten CAN drivers.'),
                ('prepare_drivers','Current mode','Prepare current mode at zero. Remain disarmed.'),
                ('enable_current_feedback','Feedback','Request measured current reports from all ten drivers.')]):
            button = QtWidgets.QPushButton(label)
            button.setToolTip(hint)
            button.clicked.connect(lambda checked=False, name=action:self.driver_action(name))
            button.setEnabled(False)
            self.driver_buttons[action] = button
            driver_grid.addWidget(button, index//2, index%2)
        side.addLayout(driver_grid)
        self.instructions = QtWidgets.QLabel()
        self.instructions.setWordWrap(True)
        side.addWidget(self.instructions)
        self.update_input_help()
        self.tracking_summary = QtWidgets.QLabel('Tracking: waiting')
        side.addWidget(self.tracking_summary)
        self.detection_status = QtWidgets.QLabel('Green magnet: waiting for detection')
        self.detection_status.setWordWrap(True)
        self.initial_status = QtWidgets.QLabel('Initial pose: not recorded')
        self.initial_status.setWordWrap(True)
        self.values = QtWidgets.QLabel('No initialized camera pose')
        self.values.setWordWrap(True)
        side.addWidget(self.values)
        self.force_label = QtWidgets.QLabel('Total force / torque (model): —')
        self.force_label.setWordWrap(True)
        self.force_label.setStyleSheet('font-family:monospace')
        self.force_label.setToolTip('Total in-plane force = coils + fixed magnet interaction.\n'
                                    'Dipole-model estimate from pose and COMMANDED currents, not a force measurement.\n'
                                    'Forward/side follow the free magnet heading; torque is about the plane normal.')
        side.addWidget(self.force_label)
        details_button = QtWidgets.QToolButton()
        details_button.setText('Details');details_button.setCheckable(True)
        details_button.setArrowType(QtCore.Qt.RightArrow)
        details_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        side.addWidget(details_button)
        self.diagnostics = QtWidgets.QWidget()
        diagnostics_layout = QtWidgets.QVBoxLayout(self.diagnostics)
        diagnostics_layout.setContentsMargins(0,0,0,0)
        self.force_details = QtWidgets.QLabel('Theoretical force: waiting for pose')
        self.force_details.setWordWrap(True)
        self.target_details = QtWidgets.QLabel()
        self.target_details.setWordWrap(True)
        for label in (self.detection_status,self.initial_status,self.target_details,self.force_details):
            diagnostics_layout.addWidget(label)
        side.addWidget(self.diagnostics);self.diagnostics.hide()
        details_button.toggled.connect(self.diagnostics.setVisible)
        details_button.toggled.connect(lambda checked:details_button.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow))
        self.message = QtWidgets.QLabel('')
        self.message.setWordWrap(True)
        self.service_message.connect(self.message.setText)
        side.addWidget(self.message)
        side.addStretch()
        for label in side_widget.findChildren(QtWidgets.QLabel):
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.recorder = TrialUi(self,side,connect_ros)
        self.sub = None
        if connect_ros:
            self.sub = rospy.Subscriber('/cs1/status', String, self.on_status, queue_size=1)
            self.heading_sub = rospy.Subscriber('/cs1/heading_unwrapped',Float64,self.on_continuous_heading,queue_size=1)
            self.initial_sub = rospy.Subscriber('/cs1/initial_pose',PoseStamped,self.on_initial_pose,queue_size=1)
            self.tracker_sub = rospy.Subscriber('/cs1/tracking_status',String,self.on_tracker_status,queue_size=1)
            self.pose_sub = rospy.Subscriber('/cs1/pose',PoseStamped,self.on_pose,queue_size=1)
            self.valid_sub = rospy.Subscriber('/cs1/tracking_valid',Bool,self.on_tracking_valid,queue_size=1)
            self.image_sub = rospy.Subscriber('/cs1/camera/image/compressed',CompressedImage,
                                              self.on_image,queue_size=1,buff_size=1024*1024)
            self.command_pub = rospy.Publisher('/cs1/keyboard_command', Twist, queue_size=1)
            threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.command_timer = QtCore.QTimer(self)
        self.command_timer.timeout.connect(self.publish_keyboard)
        self.command_timer.start(50)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)
        self.refresh()

    def update_input_help(self):
        self.instructions.setText('W/S: move · A/D: turn\nRelease: hold · Esc: zero/stop')
        self.instructions.setToolTip('Click the plot to focus. Hold movement keys; release or Space freezes the target.\n'
                                     'Losing focus clears held keys. Red cylinder: fixed; green cylinder: free magnet.')

    def keyboard_vector(self):
        keys = self.pressed_keys
        return (float((QtCore.Qt.Key_W in keys)-(QtCore.Qt.Key_S in keys)),
                float((QtCore.Qt.Key_A in keys)-(QtCore.Qt.Key_D in keys)))

    def publish_keyboard(self):
        focused = QtWidgets.QApplication.focusWidget()
        if (not self.isActiveWindow()
                or isinstance(focused,(QtWidgets.QLineEdit,QtWidgets.QAbstractSpinBox,QtWidgets.QComboBox))):
            self.pressed_keys.clear()
        drive,turn = self.keyboard_vector()
        with self.key_lock:
            self.key_vector = (drive, turn)
            self.ui_alive = time.monotonic()
        if self.command_pub is not None:
            msg = Twist()
            msg.linear.x,msg.angular.z = drive,turn
            self.command_pub.publish(msg)

    HEARTBEAT_PERIOD_S = 0.05
    UI_STALL_HOLD_S = 1.0

    def heartbeat_loop(self):
        """Keyboard heartbeat that does not depend on the Qt thread.

        Plot redraws can block the Qt thread for >100 ms, which starved the
        50 ms timer and tripped the controller's 0.3 s input timeout. If the
        Qt thread itself stalls for over UI_STALL_HOLD_S, send neutral (hold)
        rather than repeating a possibly stale held key.
        """
        while not self.heartbeat_stop.wait(self.HEARTBEAT_PERIOD_S):
            if rospy.is_shutdown() or self.command_pub is None:
                break
            with self.key_lock:
                drive, turn = self.key_vector
                stalled = time.monotonic()-self.ui_alive > self.UI_STALL_HOLD_S
            msg = Twist()
            if not stalled:
                msg.linear.x, msg.angular.z = drive, turn
            self.command_pub.publish(msg)

    def eventFilter(self, watched, event):
        if not hasattr(self, 'recorder'):  # still constructing
            return super().eventFilter(watched,event)
        belongs = watched is self or (isinstance(watched,QtWidgets.QWidget) and self.isAncestorOf(watched))
        if belongs and event.type() in (QtCore.QEvent.WindowDeactivate,QtCore.QEvent.FocusOut):
            self.pressed_keys.clear()
            self.publish_keyboard()
        if (belongs and event.type() in
                (QtCore.QEvent.KeyPress,QtCore.QEvent.KeyRelease)):
            key = event.key()
            if key == QtCore.Qt.Key_Escape:
                self.pressed_keys.clear()
                self.publish_keyboard()
                if event.type() == QtCore.QEvent.KeyPress and not event.isAutoRepeat():
                    self.set_arm(False)
                return True
            if isinstance(watched,(QtWidgets.QLineEdit,QtWidgets.QAbstractSpinBox,QtWidgets.QComboBox)):
                return super().eventFilter(watched,event)
            if key in (QtCore.Qt.Key_W,QtCore.Qt.Key_S,QtCore.Qt.Key_A,QtCore.Qt.Key_D,QtCore.Qt.Key_Space):
                if event.isAutoRepeat():
                    return True
                if key == QtCore.Qt.Key_Space:
                    self.pressed_keys.clear()
                elif event.type() == QtCore.QEvent.KeyPress:
                    self.pressed_keys.add(key)
                else:
                    self.pressed_keys.discard(key)
                self.publish_keyboard()
                return True
        return super().eventFilter(watched,event)

    def on_status(self, msg):
        try:
            state = json.loads(msg.data)
            with self.state_lock:
                self.state = state
                self.status_arrival = time.monotonic()
        except ValueError:
            pass

    def on_pose(self, msg):
        p,q = msg.pose.position,msg.pose.orientation
        pose = [p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]
        if msg.header.frame_id != 'emns_world' or not np.isfinite(pose).all():
            return
        with self.state_lock:
            self.feedback_pose = pose
            self.feedback_arrival = time.monotonic()
            self.feedback_sequence = msg.header.seq

    def on_continuous_heading(self, msg):
        if math.isfinite(msg.data):
            with self.state_lock:
                self.continuous_heading = float(msg.data)
                self.heading_arrival = time.monotonic()

    def on_initial_pose(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        if msg.header.frame_id != 'emns_world':
            return
        heading = math.atan2(2*q.w*q.z, 1-2*q.z*q.z)
        with self.state_lock:
            self.initial_pose = (p.x, p.y, heading)

    def on_tracker_status(self, msg):
        try:
            value = json.loads(msg.data)
            if isinstance(value, dict):
                with self.state_lock:
                    self.tracker_status = value
                    self.tracker_arrival = time.monotonic()
        except ValueError:
            pass

    def on_tracking_valid(self, msg):
        with self.state_lock:
            self.tracking_valid = bool(msg.data)

    def on_image(self, msg):
        # Store transport data only here. Qt image decoding/rendering runs on the UI thread.
        from cs1_experiment.camera_stream import MAX_JPEG_BYTES
        if not msg.format.lower().endswith('jpeg') or not 0 < len(msg.data) <= MAX_JPEG_BYTES:
            return
        with self.state_lock:
            self.latest_image = (bytes(msg.data),time.monotonic(),msg.header.seq)
            self.image_version += 1

    def refresh_camera(self, latest, version):
        if latest is None:
            self.camera_status.setText('Camera: no physical stream in simulation' if self.simulated_feedback
                                       else 'Camera: awaiting frames · upload updated OpenMV code')
            return
        data,arrival,sequence = latest
        if version != self.rendered_image_version:
            image = QtGui.QImage.fromData(data,'JPEG')
            self.last_image_render_ok = not image.isNull()
            if self.last_image_render_ok:
                self.camera_view.set_frame(image)
            self.rendered_image_version = version
        age = time.monotonic()-arrival
        if not self.last_image_render_ok:
            self.camera_status.setText('Camera: invalid JPEG · previous image is not current')
            self.camera_status.setStyleSheet('color:#a32b36;font-weight:bold')
        elif age > 1.0:
            self.camera_status.setText('Camera: STALE · last image received %.1f s ago · image msg #%d' % (age,sequence))
            self.camera_status.setStyleSheet('color:#a32b36;font-weight:bold')
        else:
            self.camera_status.setText('Camera: live · received %.0f ms ago · image msg #%d' % (age*1000,sequence))
            self.camera_status.setStyleSheet('color:#246545')

    def update_initialization_controls(self, *_args):
        with self.state_lock:
            status, arrival, armed = dict(self.tracker_status), self.tracker_arrival, self.state.get('armed', False)
        candidate = status.get('candidate_pose')
        fresh = time.monotonic()-arrival <= self.config['control']['pose_timeout_s']
        margin = self.config['tracking']['initialization_boundary_margin_rad']
        measurable = (fresh and candidate is not None
                      and status.get('stable_frames', 0) >= self.config['tracking']['initialization_frames']
                      and margin < candidate[2] < math.pi-margin)
        self.init_button.setEnabled(not armed and (self.simulated_feedback or measurable))

    def initialize_heading(self):
        if not self.init_button.isEnabled():
            self.message.setText('Wait for a stable camera estimate with 0° < β₀ < 180° while disarmed.')
            return
        if not self.connect_ros:
            return
        try:
            rospy.wait_for_service('/cs1/initialize_heading', timeout=1)
            response = rospy.ServiceProxy('/cs1/initialize_heading', Trigger)()
            self.message.setText(response.message)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self.message.setText(str(exc))

    def driver_action(self, action):
        if action!='enable_current_feedback':self.recorder.cancel_intent()
        self.pressed_keys.clear()
        self.publish_keyboard()
        if not self.connect_ros:
            return
        if action == 'zero_currents':
            # Emergency zero must never be dropped because another driver
            # action is in flight. It does not take or release driver_busy.
            def zero_request():
                try:
                    rospy.wait_for_service('/cs1/zero_currents', timeout=1)
                    response = rospy.ServiceProxy('/cs1/zero_currents', Trigger)()
                    self.service_message.emit(response.message)
                except (rospy.ROSException, rospy.ServiceException) as exc:
                    self.service_message.emit('ZERO FAILED: '+str(exc))
            threading.Thread(target=zero_request, daemon=True).start()
            return
        if self.driver_busy:
            return
        self.driver_busy = True
        for button in self.driver_buttons.values():
            button.setEnabled(False)
        def request():
            try:
                name = '/cs1/'+action
                rospy.wait_for_service(name, timeout=1)
                response = rospy.ServiceProxy(name, Trigger)()
                self.service_message.emit(response.message)
            except (rospy.ROSException, rospy.ServiceException) as exc:
                self.service_message.emit(str(exc))
            finally:
                self.driver_busy = False
        threading.Thread(target=request, daemon=True).start()

    def refresh_currents(self, state, arrival):
        feedback = state.get('current_feedback', {})
        values = feedback.get('values_A', [None]*10)
        ages = feedback.get('age_s', [None]*10)
        commands = state.get('currents', [])
        elapsed = max(0., time.monotonic()-arrival) if arrival else float('inf')
        available = feedback.get('available', False)
        fresh_count = 0
        bar_samples = []
        for coil in range(10):
            command = commands[coil] if coil<len(commands) else None
            value = values[coil] if coil<len(values) else None
            age = ages[coil] if coil<len(ages) else None
            effective_age = None if age is None else age+elapsed
            fresh = available and value is not None and effective_age is not None and effective_age<=1.
            fresh_count += int(fresh)
            label = 'NO CAN' if not available else 'NO DATA' if value is None else 'FRESH' if fresh else 'STALE'
            bar_samples.append({'value':value if available else None,'command':command,'state':label})
            row, offset = coil%5, (coil//5)*5
            texts = ['—' if command is None else '%+.3f' % command,
                     '—' if value is None else '%+.3f' % value,
                     '—' if effective_age is None or not math.isfinite(effective_age) else '%.2f' % effective_age,
                     label]
            for col, text in enumerate(texts, 1):
                item = self.current_table.item(row, offset+col)
                item.setText(text)
                item.setForeground(QtGui.QColor('#246b3a' if fresh else '#a32b36' if value is not None else '#687985'))
        self.current_bars.set_samples(bar_samples)
        self.current_summary.setText(('Measured CAN feedback: %d/10 fresh' % fresh_count if available
                                      else 'CAN disabled · measured currents unavailable')
                                     + (' · '+feedback['error'] if feedback.get('error') else ''))
        for button in self.driver_buttons.values():
            button.setEnabled(bool(state.get('hardware')) and elapsed<=1 and not self.driver_busy)


    @staticmethod
    def format_applied_force(applied):
        if not applied:
            return 'Theoretical force: waiting for pose'
        by, bz, fy, fz = applied['body_total']
        fx_w, fy_w = applied['force_world_N']
        cy, cz = applied['body_coils'][2:]
        my, mz = applied['body_fixed'][2:]
        text = ('THEORETICAL FORCE (coils + fixed)\n'
                'Model estimate · commanded currents\n'
                '|F| = %.2f mN\n'
                'forward %+.2f  lateral %+.2f mN\n'
                'world   x %+.2f  y %+.2f mN\n'
                ' coils  fwd %+.2f  lat %+.2f mN\n'
                ' fixed  fwd %+.2f  lat %+.2f mN\n'
                'B = %.2f mT at %+.1f° from heading\n'
                'total torque %+.3f mN·m (+ = turn left)'
                % (1e3*math.hypot(fx_w, fy_w), 1e3*fz, 1e3*fy, 1e3*fx_w, 1e3*fy_w,
                   1e3*cz, 1e3*cy, 1e3*mz, 1e3*my,
                   1e3*math.hypot(by, bz), math.degrees(applied['field_angle_from_heading_rad']),
                   1e3*applied['torque_z_Nm']))
        if 'torque_coils_z_Nm' in applied and 'torque_fixed_z_Nm' in applied:
            text += '\n coils %+.3f · fixed %+.3f mN·m' % (
                1000*applied['torque_coils_z_Nm'],1000*applied['torque_fixed_z_Nm'])
        return text

    SLICE_RESOLUTION_M = 0.00025
    # Display dimensions only; do not change the calibrated detector or TFW model.
    MAGNET_LENGTH_MM = 10.0
    MAGNET_DIAMETER_MM = 4.0

    def draw_magnet(self, *args, **kwargs):
        return draw_magnet(self.ax, *args, **kwargs)

    def slice_image(self, beta):
        """RGBA raster of the TFW slice at this heading, cached per heading cell.

        Display only; the controller guard always uses the exact boxes. The
        selected box set only changes when beta crosses a box heading edge.
        """
        lower, upper = self.workspace.lower, self.workspace.upper
        b = beta % (2*math.pi)
        if not hasattr(self, 'heading_edges'):
            self.heading_edges = np.unique(np.r_[lower[:, 2], upper[:, 2]])
        key = int(np.searchsorted(self.heading_edges, b, side='right'))
        if key not in self.slice_cache:
            n = int(round(.1/self.SLICE_RESOLUTION_M))
            mask = np.zeros((n, n), bool)
            selected = (lower[:, 2] <= b) & (b <= upper[:, 2])
            lo = np.clip(np.round((lower[selected, :2]+.05)/self.SLICE_RESOLUTION_M).astype(int), 0, n)
            hi = np.clip(np.round((upper[selected, :2]+.05)/self.SLICE_RESOLUTION_M).astype(int), 0, n)
            for (x0, y0), (x1, y1) in zip(lo, hi):
                mask[y0:y1, x0:x1] = True
            image = np.zeros((n, n, 4))
            image[mask] = (0xb6/255, 0xde/255, 0xc3/255, 1.0)
            if len(self.slice_cache) > 128:
                self.slice_cache.clear()
            self.slice_cache[key] = image
        return self.slice_cache[key]

    def set_arm(self, armed, wait=False):
        self.recorder.request_arm(armed,wait)

    def refresh(self):
        if self.connect_ros and rospy.is_shutdown():
            self.close()
            return
        with self.state_lock:
            state = dict(self.state)
            status_arrival = self.status_arrival
            feedback_pose = self.feedback_pose
            feedback_arrival = self.feedback_arrival
            feedback_sequence = self.feedback_sequence
            latest_image,image_version = self.latest_image,self.image_version
            tracking_valid = self.tracking_valid
            continuous_heading, heading_arrival = self.continuous_heading, self.heading_arrival
            tracker_status, initial_pose = dict(self.tracker_status), self.initial_pose
            tracker_arrival = self.tracker_arrival
        self.refresh_currents(state, status_arrival)
        self.refresh_camera(latest_image,image_version)
        pose = feedback_pose if feedback_pose is not None else state.get('pose')
        age = time.monotonic()-feedback_arrival
        if feedback_pose is not None:
            label = 'SIMULATED feedback' if self.simulated_feedback else 'CAMERA pose feedback'
            freshness = (' · TRACKING LOST' if not tracking_valid else
                         ' · STALE' if age > self.config['control']['pose_timeout_s'] else '')
            self.pose_status.setText('TFW · %s · pose msg #%d · received %.0f ms ago%s'
                                     % (label,feedback_sequence,age*1000,freshness))
        else:
            self.pose_status.setText('TFW · waiting for initialized pose feedback')
        if tracker_status:
            size = tracker_status.get('size')
            dimensions = ('' if size is None else '\nMeasured %.1f × %.1f mm' %
                          (size['length_m']*1000, size['diameter_m']*1000))
            loss = tracker_status.get('last_loss')
            if loss:
                dimensions += '\nLAST TRACKING LOSS (%s): %s' % (
                    time.strftime('%H:%M:%S', time.localtime(loss['wall_time_s'])), loss['reason'])
            self.detection_status.setText('Green magnet: '+tracker_status.get('reason', '')+dimensions)
        if initial_pose is not None:
            self.initial_status.setText('Saved initial pose: x %.1f, y %.1f mm; N %.1f°' %
                                        (initial_pose[0]*1000, initial_pose[1]*1000, math.degrees(initial_pose[2])))
        candidate = tracker_status.get('candidate_pose')
        candidate_fresh = time.monotonic()-tracker_arrival <= self.config['control']['pose_timeout_s']
        if candidate is not None:
            self.candidate_status.setText('%s%s: %.1f, %.1f mm · %.1f°' %
                ('Start pose' if initial_pose is None else 'Camera axis',
                 '' if candidate_fresh else ' (stale)', candidate[0]*1000, candidate[1]*1000, math.degrees(candidate[2])))
        else:
            self.candidate_status.setText('Start pose: waiting for camera')
        self.candidate_status.setVisible(initial_pose is None)
        tracking_ready = tracking_valid and age<=self.config['control']['pose_timeout_s']
        self.tracking_summary.setText('Tracking: '+('ready' if tracking_ready else
            'set start pose' if initial_pose is None and candidate_fresh else
            'lost' if feedback_pose is not None else 'waiting'))
        self.tracking_summary.setStyleSheet('color:'+('#246b3a' if tracking_ready else '#9a6b20'))
        self.tracking_summary.setToolTip(self.detection_status.text())
        self.update_initialization_controls()
        beta = (candidate[2] if candidate is not None and candidate_fresh else 0.0) if pose is None else pose[2]
        self.ax.clear()
        self.ax.set_facecolor('#edf0f2')
        self.ax.imshow(self.slice_image(beta), extent=(-50,50,-50,50), origin='lower',
                       interpolation='nearest', zorder=0)
        for magnet in fixed_magnets(self.config['model']):
            fx,fy = np.asarray(magnet['xy_m'])*1000
            self.draw_magnet(fx,fy,magnet['heading_rad'])
        if pose is None and candidate is not None and candidate_fresh:
            self.draw_magnet(candidate[0]*1000,candidate[1]*1000,candidate[2],
                             free=True,directed=False,alpha=.5)
        if pose is not None:
            x,y = np.array(pose[:2])*1000
            self.draw_magnet(x,y,beta,free=True,alpha=1. if tracking_ready else .45)
            self.trail.append([x,y]); self.trail = self.trail[-600:]
            trail = np.asarray(self.trail)
            self.ax.plot(trail[:,0],trail[:,1],color='#285e78',linewidth=1)
            heading_label = 'wrapped'
            if continuous_heading is not None and time.monotonic()-heading_arrival <= self.config['control']['pose_timeout_s']:
                beta = continuous_heading
                heading_label = 'continuous'
            applied = state.get('applied_model')
            self.force_details.setText(self.format_applied_force(applied))
            if applied:
                force_world = applied['force_world_N']
                self.force_label.setText('Total force (model): %.2f mN\nForward %+.2f · Side %+.2f mN\nTorque %+.3f mN·m' %
                    (1000*math.hypot(*force_world),1000*applied['body_total'][3],1000*applied['body_total'][2],
                     1000*applied['torque_z_Nm']))
            else:self.force_label.setText('Total force / torque (model): —')
            self.values.setText('x = %.1f mm    y = %.1f mm\nβ = %.1f° (%s)'
                                % (x,y,math.degrees(beta),heading_label))
        mode = 'CAN' if state.get('hardware') else 'Preview'
        reason = state.get('reason','Waiting for controller')
        brief = {'disarmed':'','holding captured pose':'Holding pose',
                 'moving pose target':'Moving','starting pose hold; movement input inhibited':'Starting · wait',
                 'tracking lost; initialize heading again':'Tracking lost · set start pose again'}
        self.mode.setText(mode+' · '+('ARMED' if state.get('armed') else 'DISARMED')+
                          ('\n'+brief.get(reason,reason) if brief.get(reason,reason) else ''))
        self.mode.setToolTip(reason)
        if not state.get('armed') and state.get('last_run_stop') and state['last_run_stop']['reason']!=reason:
            last_reason = state['last_run_stop']['reason']
            self.mode.setText(self.mode.text()+'\nLast stop: '+brief.get(last_reason,last_reason))
        hold = state.get('pose_hold',{})
        locked = bool(state.get('hardware')) and not hold.get('hardware_validated',False)
        self.arm_button.setEnabled(not locked)
        if locked:self.mode.setText(self.mode.text()+'\nMotion locked · validation required')
        self.target_details.setText('')
        if hold.get('reference_pose') is not None:
            ref = hold['reference_pose']
            self.ax.plot(ref[0]*1000,ref[1]*1000,'x',color='#275eaa',markersize=9,zorder=6)
            self.target_details.setText('Pose hold: '+hold['phase']+
                              '\nTarget: %.1f, %.1f mm; %.1f°' %
                              (ref[0]*1000,ref[1]*1000,math.degrees(ref[2])))
        self.ax.set(xlim=(-50,50),ylim=(-50,50),aspect='equal',xlabel='World x (mm)',
                    ylabel='World y (mm)',title=('Model-in TFW at heading %.1f°' % math.degrees(beta) if pose is not None
                    else 'TFW preview: camera β₀ %.1f° (not initialized)' % math.degrees(beta)
                    if candidate is not None and candidate_fresh else 'TFW preview β = 0°; waiting for camera'))
        self.ax.grid(alpha=0.2)
        self.canvas.draw_idle()

    def closeEvent(self, event):
        if not self.recorder.can_close():
            self.recorder.request_close();event.ignore();return
        self.recorder.shutdown()
        self.pressed_keys.clear()
        self.publish_keyboard()
        # Closing the panel stops the heartbeat, so the controller disarms.
        self.heartbeat_stop.set()
        self.command_timer.stop()
        QtWidgets.QApplication.instance().removeEventFilter(self)
        self.timer.stop()
        if self.connect_ros:
            self.set_arm(False,wait=True)
        event.accept()


if __name__ == '__main__':
    rospy.init_node('cs1_control_panel', disable_signals=True)
    app = QtWidgets.QApplication(sys.argv)
    panel = ControlPanel()
    panel.showMaximized()
    sys.exit(app.exec_())
