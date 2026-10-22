"""Calibration geometry, actual Qt clicks, serial preview, and saved tracker input."""
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
from camera_calibration_ui import CalibrationWindow, mapped_heading
from cs1_experiment.core import calibrated_line


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = CalibrationWindow(position_only=False)
        self.window.timer.stop()
        image = QtGui.QImage(320, 240, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor('gray'))
        self.window.set_image(image)
        self.window.show(); self.app.processEvents()

    def tearDown(self):
        self.window.dirty = False
        self.window.close(); self.app.processEvents()

    def record_points(self):
        for u, v in [(30, 30), (290, 30), (30, 210), (290, 210), (160, 120)]:
            self.window.select_pixel(u, v)
            self.window.x_input.setValue((u-160)*0.25)
            self.window.y_input.setValue(-(v-120)*0.25)
            self.window.add_point()
        self.window.fit_mapping()

    def test_click_scaling_letterbox_and_freeze(self):
        view = self.window.view
        rect = view.image_rect()
        # Integer pixel center maps exactly through letterboxing/scaling.
        pos = QtCore.QPointF(rect.left()+60.5*rect.width()/240,
                            rect.top()+(320-0.5-80)*rect.height()/320)
        u, v = view.pixel_at(pos)
        self.assertAlmostEqual(u, 80); self.assertAlmostEqual(v, 60)
        if rect.top() > 1:
            self.assertIsNone(view.pixel_at(QtCore.QPointF(1, 0)))
        QtTest.QTest.mouseClick(view, QtCore.Qt.LeftButton, pos=pos.toPoint())
        self.assertTrue(self.window.freeze.isChecked())
        self.assertIsNotNone(self.window.selected)
        self.window.freeze.setChecked(False)
        self.assertIsNone(self.window.selected)

    def test_default_window_focuses_on_xy_only(self):
        window = CalibrationWindow()
        window.timer.stop(); window.show(); self.app.processEvents()
        self.assertFalse(window.mode.isVisible())
        self.assertFalse(window.heading_input.isVisible())
        self.assertFalse(window.angle_status.isVisible())
        self.assertTrue(window.x_input.isVisible())
        self.assertTrue(window.y_input.isVisible())
        window.close()

    def test_ccw_rotation_preserves_raw_clicks_and_marks(self):
        view = self.window.view
        # Original top-right appears at the display top-left after a CCW turn.
        raw_corners = [(319, 0), (319, 239), (0, 0), (0, 239)]
        positions = [view.pixel_position(*point) for point in raw_corners]
        self.assertLess(positions[0].x(), positions[1].x())
        self.assertLess(positions[0].y(), positions[2].y())
        for raw, position in zip(raw_corners, positions):
            u, v = view.pixel_at(position)
            self.assertAlmostEqual(u, raw[0]); self.assertAlmostEqual(v, raw[1])
        for raw in [(80, 60), (240, 180)]:
            position = view.pixel_position(*raw)
            u, v = view.pixel_at(position)
            self.window.select_pixel(u, v)
            self.assertAlmostEqual(self.window.selected[0], raw[0])
            self.assertAlmostEqual(self.window.selected[1], raw[1])

    def test_fit_mm_units_angle_and_save_compatible_with_tracker(self):
        self.record_points()
        self.assertLess(self.window.result['max_error_m'], 1e-12)
        self.assertAlmostEqual(self.window.points[0][2], -0.0325)
        self.window.mode.setCurrentIndex(1)
        self.window.select_pixel(160, 180); self.window.select_pixel(160, 60)
        self.window.heading_input.setValue(90); self.window.check_heading()
        self.assertAlmostEqual(self.window.heading_checks[0]['error_deg'], 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'camera.json'
            self.window.save_calibration(path)
            self.assertFalse(self.window.dirty)
            data = json.loads(path.read_text())
            self.assertEqual(data['image_size_px'], [320, 240])
            xy, axis = calibrated_line({'x':160, 'y':120, 'axis_rad':0}, data['affine'])
            self.assertAlmostEqual(float(xy[0]), 0)
            self.assertAlmostEqual(float(xy[1]), 0)
            with self.assertRaises(FileExistsError):
                self.window.save_calibration(path)
        self.window.mode.setCurrentIndex(0)
        self.window.select_pixel(200, 100); self.window.add_point()
        self.assertIsNone(self.window.result)
        self.assertEqual(self.window.heading_checks, [])

    def test_degenerate_points_and_zero_axis_rejected(self):
        for u in [20, 60, 120, 200]:
            self.window.select_pixel(u, 60); self.window.x_input.setValue(u)
            self.window.add_point()
        self.window.fit_mapping()
        self.assertIsNone(self.window.result)
        with self.assertRaises(ValueError):
            mapped_heading([[1, 0, 0], [0, 1, 0]], (1, 1), (1, 1))

    def test_camera_preview_without_mapping_and_frozen_frame(self):
        image = QtGui.QImage(320, 240, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor('red'))
        data = QtCore.QByteArray(); buffer = QtCore.QBuffer(data)
        buffer.open(QtCore.QIODevice.WriteOnly); image.save(buffer, 'JPEG')
        packet = (json.dumps({'sequence':1, 'jpeg_b64':base64.b64encode(bytes(data)).decode()})+'\n').encode()
        fake = mock.Mock(); fake.in_waiting = len(packet)
        fake.read.side_effect = [packet[:100], packet[100:], b'']
        self.window.ser = fake
        self.window.poll_camera()
        self.assertGreater(self.window.view.image.pixelColor(0, 0).red(), 240)
        self.window.freeze.setChecked(True)
        self.window.view.image.fill(QtGui.QColor('blue'))
        fake.read.side_effect = [packet, b'']
        self.window.poll_camera()
        self.assertGreater(self.window.view.image.pixelColor(0, 0).blue(), 240)
        self.window.disconnect_camera(); fake.close.assert_called_once()

    def test_resolution_change_cannot_mix_calibration_points(self):
        self.record_points()
        with self.assertRaises(ValueError):
            self.window.set_image(QtGui.QImage(640, 480, QtGui.QImage.Format_RGB32))


if __name__ == '__main__':
    unittest.main()
