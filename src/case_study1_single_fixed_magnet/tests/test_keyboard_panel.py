"""Exercise actual Qt key events without ROS, camera, gamepad, or CAN."""
import os
from pathlib import Path
import sys
import unittest
import threading
from unittest import mock
import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
from control_panel import ControlPanel


class KeyboardPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.panel = ControlPanel(connect_ros=False)
        self.panel.timer.stop(); self.panel.command_timer.stop()
        self.panel.command_pub = mock.Mock()
        self.panel.show(); self.panel.activateWindow()
        self.panel.canvas.setFocus()
        self.app.processEvents()

    def tearDown(self):
        self.panel.close()
        self.app.processEvents()

    def last_vector(self):
        msg = self.panel.command_pub.publish.call_args[0][0]
        return msg.linear.x,msg.angular.z

    def test_camera_square_maps_world_corners_with_rotation_and_shear(self):
        affine = np.array([[0,.0005,-.06],[.0006,.00001,-.09]])
        camera = self.panel.camera_view
        camera.set_calibration(affine)
        transform = camera.world_transform(301)
        for world,expected in [([-.05,-.05],[0,300]),([.05,.05],[300,0]),
                               ([-.05,.05],[0,0]),([.05,-.05],[300,300]),([0,0],[150,150])]:
            pixel = np.linalg.solve(affine[:,:2],np.array(world)-affine[:,2])
            point = transform.map(QtCore.QPointF(*pixel))
            np.testing.assert_allclose([point.x(),point.y()],expected,atol=1e-9)

    def test_camera_square_rectifies_image_and_preserves_square_when_resized(self):
        camera = self.panel.camera_view
        camera.set_calibration([[.001,0,-.05],[0,.001,-.05]])
        image = QtGui.QImage(100,100,QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor('black'))
        painter = QtGui.QPainter(image)
        painter.fillRect(49,24,3,3,QtGui.QColor('red'));painter.end()
        camera.set_frame(image)
        self.assertEqual(camera.display_image.size(),QtCore.QSize(600,600))
        color = camera.display_image.pixelColor(300,449)
        self.assertGreater(color.red(),200);self.assertLess(color.green(),40)
        camera.resize(430,280);camera.render_frame()
        self.assertEqual(camera.pixmap().width(),camera.pixmap().height())

    def test_hold_combinations_and_release(self):
        QtTest.QTest.keyPress(self.panel.canvas,QtCore.Qt.Key_W)
        QtTest.QTest.keyPress(self.panel.canvas,QtCore.Qt.Key_A)
        self.assertEqual(self.last_vector(),(1,1))
        QtTest.QTest.keyRelease(self.panel.canvas,QtCore.Qt.Key_W)
        self.assertEqual(self.last_vector(),(0,1))
        QtTest.QTest.keyRelease(self.panel.canvas,QtCore.Qt.Key_A)
        self.assertEqual(self.last_vector(),(0,0))
        QtTest.QTest.keyPress(self.panel.canvas,QtCore.Qt.Key_S)
        QtTest.QTest.keyPress(self.panel.canvas,QtCore.Qt.Key_D)
        self.assertEqual(self.last_vector(),(-1,-1))

    def test_autorepeat_release_does_not_interrupt_hold(self):
        QtTest.QTest.keyPress(self.panel.canvas,QtCore.Qt.Key_W)
        event = QtGui.QKeyEvent(QtCore.QEvent.KeyRelease,QtCore.Qt.Key_W,QtCore.Qt.NoModifier,'w',True)
        self.app.sendEvent(self.panel.canvas,event)
        self.panel.publish_keyboard()
        self.assertEqual(self.last_vector(),(1,0))

    def test_focus_loss_and_space_clear_movement(self):
        QtTest.QTest.keyPress(self.panel.canvas,QtCore.Qt.Key_W)
        self.app.sendEvent(self.panel.canvas,QtGui.QFocusEvent(QtCore.QEvent.FocusOut))
        self.assertEqual(self.last_vector(),(0,0))
        QtTest.QTest.keyPress(self.panel.canvas,QtCore.Qt.Key_A)
        QtTest.QTest.keyPress(self.panel.canvas,QtCore.Qt.Key_Space)
        self.assertEqual(self.last_vector(),(0,0))

    def test_numeric_editing_does_not_drive_and_escape_disarms(self):
        numeric = QtWidgets.QDoubleSpinBox(self.panel)
        numeric.show(); numeric.setFocus(); self.app.processEvents()
        QtTest.QTest.keyPress(numeric,QtCore.Qt.Key_W)
        self.panel.publish_keyboard()
        self.assertEqual(self.last_vector(),(0,0))
        self.panel.set_arm = mock.Mock()
        QtTest.QTest.keyPress(numeric,QtCore.Qt.Key_Escape)
        self.panel.set_arm.assert_called_once_with(False)

    def test_calibrated_camera_view_is_square_without_bars(self):
        from PyQt5 import QtGui
        view = self.panel.camera_view
        view.set_calibration([[0,.0005,-.06],[.0006,.00001,-.09]])
        self.panel.resize(1600, 1000);self.app.processEvents();QtTest.QTest.qWait(50)
        self.assertLessEqual(view.height(), view.width()+1)
        image = QtGui.QImage(320, 240, QtGui.QImage.Format_RGB32);image.fill(QtGui.QColor('white'))
        view.set_frame(image)
        pixmap = view.pixmap()
        self.assertGreaterEqual(pixmap.height(), view.height()-2)   # fills the pane: no top/bottom bars
        self.assertGreaterEqual(pixmap.width(), view.width()-2)

    def test_panel_is_keyboard_only(self):
        self.assertFalse(self.panel.findChildren(QtWidgets.QComboBox))
        self.assertNotIn('GAMEPAD',self.panel.instructions.text())
        QtTest.QTest.keyPress(self.panel.canvas,QtCore.Qt.Key_W)
        self.assertEqual(self.panel.keyboard_vector(),(1,0))
        QtTest.QTest.keyRelease(self.panel.canvas,QtCore.Qt.Key_W)

    def test_panel_selects_keyboard_input_in_ros(self):
        import control_panel
        with mock.patch.object(control_panel.rospy,'set_param') as set_param, \
             mock.patch.object(control_panel.rospy,'get_param',side_effect=lambda name,default=None:default), \
             mock.patch.object(control_panel.rospy,'Subscriber'), \
             mock.patch.object(control_panel.rospy,'Publisher'):
            panel = ControlPanel(connect_ros=True)
            try:
                set_param.assert_any_call('/cs1/input_source','keyboard')
            finally:
                panel.timer.stop();panel.command_timer.stop();panel.recorder.timer.stop()
                panel.connect_ros=False;panel.recorder.can_close=lambda:True;panel.close()

    def test_arming_service_does_not_block_keyboard_heartbeat(self):
        import control_panel
        entered,finish = threading.Event(),threading.Event()
        def service(_armed):
            entered.set()
            finish.wait(2)
            return mock.Mock(message='armed')
        self.panel.connect_ros = True
        self.panel.command_timer.start(20)
        try:
            with mock.patch.object(control_panel.rospy,'wait_for_service'), \
                 mock.patch.object(control_panel.rospy,'ServiceProxy',return_value=service):
                self.panel.set_arm(True)
                self.assertTrue(entered.wait(1))
                previous = self.panel.command_pub.publish.call_count
                QtTest.QTest.qWait(120)
                self.assertGreater(self.panel.command_pub.publish.call_count,previous)
                finish.set()
                QtTest.QTest.qWait(30)
        finally:
            finish.set()
            self.panel.connect_ros = False

    def test_camera_render_and_stale_image_indicator(self):
        import io,time
        from PIL import Image
        from sensor_msgs.msg import CompressedImage
        stream = io.BytesIO()
        Image.new('RGB',(32,24),(100,50,10)).save(stream,format='JPEG')
        message = CompressedImage(format='jpeg',data=stream.getvalue())
        message.header.seq = 23
        self.panel.on_image(message)
        self.panel.refresh()
        self.assertEqual(self.panel.camera_view.original.size(),QtCore.QSize(32,24))
        self.assertIn('image msg #23',self.panel.camera_status.text())
        self.assertIn('live',self.panel.camera_status.text())
        data,_,sequence = self.panel.latest_image
        self.panel.latest_image = (data,time.monotonic()-2,sequence)
        self.panel.refresh()
        self.assertIn('STALE',self.panel.camera_status.text())

    def test_all_coil_feedback_distinguishes_missing_stale_and_commands(self):
        import json, time
        from std_msgs.msg import String
        values = [float(k)/10 for k in range(10)]
        values[-1] = None
        ages = [.05]*10; ages[1] = 2.; ages[-1] = None
        self.panel.on_status(String(data=json.dumps({'hardware':True, 'currents':[-.5]*10,
            'current_feedback':{'available':True,'values_A':values,'age_s':ages,'error':''}})))
        self.panel.refresh()
        self.assertIn('8/10 fresh',self.panel.current_summary.text())
        self.assertEqual(self.panel.current_table.item(0,1).text(),'-0.500')
        self.assertEqual(self.panel.current_table.item(0,2).text(),'+0.000')
        self.assertEqual(self.panel.current_table.item(0,7).text(),'+0.500')
        self.assertEqual(self.panel.current_table.item(1,4).text(),'STALE')
        self.assertEqual(self.panel.current_table.item(4,7).text(),'—')
        self.assertEqual(self.panel.current_table.item(4,9).text(),'NO DATA')
        self.assertEqual(self.panel.current_bars.samples[1]['state'],'STALE')
        self.assertIsNone(self.panel.current_bars.samples[9]['value'])
        self.assertEqual(self.panel.current_bars.samples[5]['command'],-.5)
        self.panel.status_arrival = time.monotonic()-2
        self.panel.refresh()
        self.assertIn('0/10 fresh',self.panel.current_summary.text())
        self.assertTrue(all(not button.isEnabled() for button in self.panel.driver_buttons.values()))
        self.panel.state = {'currents':[0.]*10}
        self.panel.refresh()
        self.assertIn('measured currents unavailable',self.panel.current_summary.text())
        self.assertEqual(self.panel.current_table.item(0,2).text(),'—')
        self.assertTrue(all(sample['value'] is None for sample in self.panel.current_bars.samples))

    def test_zero_service_runs_without_blocking_keyboard_release(self):
        import time
        import control_panel
        entered, release = threading.Event(), threading.Event()
        def request():
            entered.set();release.wait(timeout=2)
            return mock.Mock(success=True,message='zero command sent')
        self.panel.connect_ros=True
        self.panel.pressed_keys={QtCore.Qt.Key_W}
        try:
            with mock.patch.object(control_panel.rospy,'wait_for_service'), \
                 mock.patch.object(control_panel.rospy,'ServiceProxy',return_value=request) as proxy:
                started=time.monotonic()
                self.panel.driver_action('zero_currents')
                self.assertLess(time.monotonic()-started,.2)
                self.assertTrue(entered.wait(timeout=1))
                self.assertEqual(self.last_vector(),(0,0))
                proxy.assert_called_once_with('/cs1/zero_currents',control_panel.Trigger)
                release.set()
                end=time.monotonic()+1
                while self.panel.driver_busy and time.monotonic()<end:
                    QtTest.QTest.qWait(10)
                self.assertFalse(self.panel.driver_busy)
        finally:
            release.set(); self.panel.connect_ros=False

    def test_zero_button_is_not_dropped_while_driver_action_busy(self):
        import control_panel
        called = threading.Event()
        def request():
            called.set();return mock.Mock(success=True,message='zero command sent')
        self.panel.connect_ros=True
        self.panel.driver_busy=True  # e.g. Current mode / Zero still running
        try:
            with mock.patch.object(control_panel.rospy,'wait_for_service'), \
                 mock.patch.object(control_panel.rospy,'ServiceProxy',return_value=request) as proxy:
                self.panel.driver_action('zero_currents')
                self.assertTrue(called.wait(timeout=1))
                proxy.assert_called_once_with('/cs1/zero_currents',control_panel.Trigger)
                # Other driver actions stay gated while busy.
                self.panel.driver_action('reset_drivers')
                self.assertEqual(proxy.call_count,1)
                self.assertTrue(self.panel.driver_busy)
        finally:
            self.panel.driver_busy=False;self.panel.connect_ros=False

    def test_measured_start_pose_boundary_gate_without_checkbox(self):
        import json, time, math
        from std_msgs.msg import String
        self.assertFalse(self.panel.findChildren(QtWidgets.QCheckBox))
        self.panel.update_initialization_controls()
        self.assertFalse(self.panel.init_button.isEnabled())  # no camera estimate yet
        self.panel.on_tracker_status(String(data=json.dumps({
            'candidate_pose':[.01,.02,math.radians(35)], 'stable_frames':1})))
        self.panel.update_initialization_controls()
        self.assertFalse(self.panel.init_button.isEnabled())  # not yet stable
        self.panel.on_tracker_status(String(data=json.dumps({
            'candidate_pose':[.01,.02,math.radians(35)], 'stable_frames':5})))
        self.panel.refresh()
        self.assertIn('35.0°', self.panel.candidate_status.text())
        self.assertTrue(self.panel.init_button.isEnabled())
        self.panel.tracker_status['candidate_pose'][2] = math.radians(1)
        self.panel.update_initialization_controls()
        self.assertFalse(self.panel.init_button.isEnabled())
        self.panel.tracker_status['candidate_pose'][2] = math.radians(35)
        self.panel.tracker_arrival = time.monotonic()-1
        self.panel.update_initialization_controls()
        self.assertFalse(self.panel.init_button.isEnabled())

    def test_continuous_heading_and_saved_initial_pose_are_visible(self):
        import math
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Float64, String
        pose = PoseStamped()
        pose.header.frame_id = 'emns_world'
        pose.pose.position.x, pose.pose.position.y = .005, .025
        pose.pose.orientation.z = math.sin(math.pi/4)
        pose.pose.orientation.w = math.cos(math.pi/4)
        self.panel.on_pose(pose)
        self.panel.on_initial_pose(pose)
        self.panel.on_continuous_heading(Float64(data=math.radians(450)))
        self.panel.on_tracker_status(String(data=__import__('json').dumps({
            'reason':'Tracking green magnet', 'size':{'length_m':.011,'diameter_m':.004}})))
        self.panel.refresh()
        self.assertIn('450.0° (continuous)', self.panel.values.text())
        self.assertIn('11.0 × 4.0 mm', self.panel.detection_status.text())
        self.assertIn('N 90.0°', self.panel.initial_status.text())

    def test_world_pose_uses_feedback_instead_of_controller_or_command(self):
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Bool
        pose = PoseStamped()
        pose.header.frame_id = 'emns_world'
        pose.pose.position.x,pose.pose.position.y = -.012,.023
        pose.pose.orientation.w = 1
        self.panel.state = {'pose':[.04,.04,1.57],'currents':[0]*10}
        self.panel.on_pose(pose)
        self.panel.on_tracking_valid(Bool(data=True))
        self.panel.refresh()
        self.assertIn('x = -12.0 mm',self.panel.values.text())
        self.assertIn('y = 23.0 mm',self.panel.values.text())
        self.assertIn('CAMERA pose feedback',self.panel.pose_status.text())
        self.panel.on_tracking_valid(Bool(data=False))
        self.panel.refresh()
        self.assertIn('TRACKING LOST',self.panel.pose_status.text())


if __name__ == '__main__':
    unittest.main()
