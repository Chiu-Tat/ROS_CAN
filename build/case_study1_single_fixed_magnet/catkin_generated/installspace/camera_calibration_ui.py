#!/usr/bin/env python3
"""Click-to-record OpenMV calibration. Standalone; no ROS master or CAN required."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import serial

# Also support running this file directly before sourcing the catkin workspace.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
from calibrate_camera import fit
from cs1_experiment.camera_stream import JsonFrameBuffer, decode_preview
from cs1_experiment.core import PACKAGE


def mapped_heading(affine, start, end):
    direction = np.asarray(affine, float)[:, :2] @ (np.asarray(end)-start)
    if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12:
        raise ValueError('Choose two different points along the cylinder axis')
    return math.degrees(math.atan2(direction[1], direction[0]))


class PixelView(QtWidgets.QWidget):
    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.image = QtGui.QImage()
        self.marks = []
        self.setMinimumSize(480, 360)
        self.setCursor(QtCore.Qt.CrossCursor)

    def image_rect(self):
        if self.image.isNull():
            return QtCore.QRectF()
        # Display a 90-degree counterclockwise rotation, retaining raw pixels.
        scale = min(self.width()/self.image.height(), self.height()/self.image.width())
        w, h = self.image.height()*scale, self.image.width()*scale
        return QtCore.QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)

    def pixel_at(self, pos):
        rect = self.image_rect()
        if rect.isEmpty() or not rect.contains(pos):
            return None
        # Pixel centers are integer coordinates; Qt draws the image over pixel edges.
        rotated_u = (pos.x()-rect.left())*self.image.height()/rect.width()-0.5
        rotated_v = (pos.y()-rect.top())*self.image.width()/rect.height()-0.5
        u = self.image.width()-1-rotated_v
        v = rotated_u
        return (min(self.image.width()-1, max(0, u)),
                min(self.image.height()-1, max(0, v)))

    def pixel_position(self, u, v):
        """Raw camera pixel center → position on the rotated display."""
        rect = self.image_rect()
        return QtCore.QPointF(rect.left()+(v+0.5)*rect.width()/self.image.height(),
                             rect.top()+(self.image.width()-0.5-u)*rect.height()/self.image.width())

    def mousePressEvent(self, event):
        point = self.pixel_at(event.localPos())
        if point is not None and event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(*point)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor('#101b29'))
        if self.image.isNull():
            painter.setPen(QtGui.QColor('white'))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             'Connect OpenMV to see the camera\nNo calibration is needed to start')
            return
        rect = self.image_rect()
        painter.drawImage(rect, self.image.transformed(QtGui.QTransform().rotate(-90)))
        painter.setPen(QtGui.QPen(QtGui.QColor('#ffcc33'), 2))
        points = []
        for i, (u, v) in enumerate(self.marks):
            position = self.pixel_position(u, v)
            x, y = position.x(), position.y()
            points.append(QtCore.QPointF(x, y))
            painter.drawLine(QtCore.QPointF(x-8, y), QtCore.QPointF(x+8, y))
            painter.drawLine(QtCore.QPointF(x, y-8), QtCore.QPointF(x, y+8))
            painter.drawText(QtCore.QPointF(x+10, y-8), str(i+1))
        if len(points) == 2:
            painter.drawLine(points[0], points[1])


class CalibrationWindow(QtWidgets.QWidget):
    def __init__(self, port='/dev/ttyACM0', position_only=True):
        super().__init__()
        self.setWindowTitle('CS1 | Camera → EM coordinate calibration')
        self.resize(1240, 810)
        self.ser = None
        self.buffer = JsonFrameBuffer(skip_invalid_lines=True)
        self.points, self.heading_checks = [], []
        self.result = None
        self.selected = None
        self.dimensions = None
        self.last_frame = 0
        self.offline = False
        self.dirty = False
        self.view = PixelView()
        self.view.clicked.connect(self.select_pixel)
        self.port = QtWidgets.QLineEdit(port)
        self.connect_button = QtWidgets.QPushButton('Connect camera')
        self.connect_button.clicked.connect(self.toggle_connection)
        self.freeze = QtWidgets.QCheckBox('Freeze frame (clicking also freezes)')
        self.freeze.toggled.connect(self.freeze_changed)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Position: click a reference point', 'Angle: click tail, then head'])
        self.mode.currentIndexChanged.connect(self.clear_selection)
        self.pixel_label = QtWidgets.QLabel('Pixel: —')
        self.world_label = QtWidgets.QLabel('Mapped EM coordinate: fit points first')
        self.x_input, self.y_input = self.number(), self.number()
        self.heading_input = self.number(-360, 360)
        self.heading_input.setValue(90)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Pixel u', 'Pixel v', 'EM x (mm)', 'EM y (mm)', 'Error (mm)'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setMaximumHeight(210)
        self.quality = QtWidgets.QLabel('Record ≥4 points spread across the working plane, then fit.')
        self.quality.setWordWrap(True)
        self.angle_status = QtWidgets.QLabel('Angle uses the same affine map. Known magnetic direction resolves 180° ambiguity.')
        self.angle_status.setWordWrap(True)
        self.status = QtWidgets.QLabel('Disconnected. Stop the ROS tracker / IDE / serial monitor before connecting.')
        self.status.setWordWrap(True)
        self.save_status = QtWidgets.QLabel('')
        self.save_status.setWordWrap(True)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel('USB port')); top.addWidget(self.port)
        top.addWidget(self.connect_button)
        top.addWidget(self.button('Load image (offline)', self.load_image))
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.view, 1); left.addWidget(self.freeze); left.addWidget(self.status)
        right = QtWidgets.QVBoxLayout()
        intro = QtWidgets.QLabel('View: 90° CCW; pixels stay in original camera coordinates\n1  Click → enter known EM x/y → Add point\n2  Fit mapping and check unused reference points\n3  Check magnet angle → Save calibration')
        intro.setWordWrap(True)
        right.addWidget(intro); right.addWidget(self.mode)
        right.addWidget(self.pixel_label); right.addWidget(self.world_label)
        form = QtWidgets.QFormLayout()
        form.addRow('Known EM x (mm)', self.x_input)
        form.addRow('Known EM y (mm)', self.y_input)
        right.addLayout(form)
        right.addWidget(self.button('Add position point', self.add_point))
        right.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.button('Remove selected', self.remove_point))
        row.addWidget(self.button('Fit mapping', self.fit_mapping))
        right.addLayout(row); right.addWidget(self.quality)
        heading_form = QtWidgets.QFormLayout()
        heading_form.addRow('Known magnetic heading (°)', self.heading_input)
        right.addLayout(heading_form)
        angle_button = self.button('Record angle check', self.check_heading)
        right.addWidget(angle_button)
        right.addWidget(self.angle_status)
        right.addWidget(self.button('Save calibration + recorded points…', self.save_dialog))
        right.addWidget(self.save_status); right.addStretch()
        if position_only:
            self.mode.hide()
            self.heading_input.hide()
            heading_form.labelForField(self.heading_input).hide()
            angle_button.hide()
            self.angle_status.hide()
            intro.setText('View: 90° CCW; pixels stay in original camera coordinates\n'
                          '1  Click → enter known EM x/y → Add point\n'
                          '2  Fit mapping and check unused reference points\n'
                          '3  Save x/y calibration')
        body = QtWidgets.QHBoxLayout(); body.addLayout(left, 3); body.addLayout(right, 2)
        layout = QtWidgets.QVBoxLayout(self); layout.addLayout(top); layout.addLayout(body)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll_camera)
        self.timer.start(20)

    @staticmethod
    def number(low=-1000, high=1000):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(low, high); widget.setDecimals(3)
        return widget

    @staticmethod
    def button(label, callback):
        widget = QtWidgets.QPushButton(label); widget.clicked.connect(callback)
        return widget

    def message(self, text):
        self.status.setText(text)

    def disconnect_camera(self):
        if self.ser is not None:
            self.ser.close(); self.ser = None
        self.buffer.clear()
        self.connect_button.setText('Connect camera')

    def toggle_connection(self):
        if self.ser is not None:
            self.disconnect_camera(); self.message('Disconnected; USB port released.'); return
        try:
            self.ser = serial.Serial(self.port.text(), 115200, timeout=0, exclusive=True)
            self.buffer.clear(); self.last_frame = time.monotonic(); self.offline = False
            self.connect_button.setText('Disconnect camera')
            self.message('Connected; waiting for OpenMV JPEG preview…')
        except (OSError, serial.SerialException) as exc:
            self.message('Cannot connect: '+str(exc))

    def poll_camera(self):
        if self.ser is None:
            return
        try:
            frames = []
            for _ in range(4):
                chunk = self.ser.read(min(65536, self.ser.in_waiting))
                if not chunk:
                    break
                frames.extend(self.buffer.feed(chunk))
            for frame in reversed(frames):
                jpeg = decode_preview(frame)
                if jpeg is not None:
                    image = QtGui.QImage.fromData(jpeg)
                    if image.isNull():
                        raise ValueError('Cannot decode camera JPEG')
                    self.last_frame = time.monotonic()
                    if not self.freeze.isChecked():
                        self.set_image(image)
                    self.message(('FROZEN' if self.freeze.isChecked() else 'LIVE')+
                                 ' | %d × %d pixels | Camera sequence %s' %
                                 (image.width(), image.height(), frame.get('sequence', '?')))
                    break
            if time.monotonic()-self.last_frame > 2:
                self.message('No fresh JPEG for >2 s. Check main.py has SEND_PREVIEW=True; reconnect after reset.')
        except (OSError, serial.SerialException) as exc:
            self.disconnect_camera(); self.message('Camera disconnected: '+str(exc))
        except ValueError as exc:
            self.message('Invalid camera packet: '+str(exc))

    def set_image(self, image):
        dimensions = (image.width(), image.height())
        if self.dimensions is not None and dimensions != self.dimensions and self.points:
            raise ValueError('Image size changed. Start a new calibration window to avoid mixing pixel coordinates.')
        self.dimensions = dimensions
        self.view.image = image.copy(); self.view.update()

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Offline calibration image', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if not path:
            return
        image = QtGui.QImage(path)
        if image.isNull():
            self.message('Cannot read image'); return
        try:
            self.set_image(image)
        except ValueError as exc:
            self.message(str(exc)); return
        self.disconnect_camera(); self.offline = True; self.freeze.setChecked(True)
        self.message('OFFLINE IMAGE | '+Path(path).name)

    def clear_selection(self, *_args):
        self.selected = None; self.view.marks = []; self.view.update()
        self.pixel_label.setText('Pixel: —')

    def freeze_changed(self, frozen):
        if not frozen:
            self.clear_selection()

    def select_pixel(self, u, v):
        self.freeze.setChecked(True)
        self.selected = (u, v)
        if self.mode.currentIndex() == 0 or len(self.view.marks) == 2:
            self.view.marks = []
        self.view.marks.append((u, v)); self.view.update()
        self.pixel_label.setText('Pixel u = %.2f, v = %.2f (original image)' % (u, v))
        if self.result is not None:
            xy = np.asarray(self.result['affine']) @ [u, v, 1]
            self.world_label.setText('Mapped EM: x = %.3f, y = %.3f mm' % tuple(1000*xy))

    def invalidate_fit(self):
        self.result = None; self.heading_checks = []; self.dirty = True
        self.save_status.clear()
        self.quality.setText('Points changed: fit mapping again before saving.')
        self.angle_status.setText('Angle checks cleared: fit mapping again.')
        self.world_label.setText('Mapped EM coordinate: fit points first')
        for row in range(self.table.rowCount()):
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem('—'))

    def add_point(self):
        if self.selected is None or self.mode.currentIndex() != 0:
            self.message('Choose Position mode and click a reference point first.'); return
        u, v = self.selected
        if any(math.hypot(u-p[0], v-p[1]) < 1 for p in self.points):
            self.message('This pixel has already been recorded; choose another reference point.'); return
        self.points.append([u, v, self.x_input.value()/1000, self.y_input.value()/1000])
        row = self.table.rowCount(); self.table.insertRow(row)
        for col, value in enumerate([u, v, self.x_input.value(), self.y_input.value()]):
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem('%.3f' % value))
        self.invalidate_fit(); self.clear_selection()
        self.message('Point recorded. Resume live video for the next position, or select another frozen reference.')

    def remove_point(self):
        row = self.table.currentRow()
        if row >= 0:
            del self.points[row]; self.table.removeRow(row); self.invalidate_fit()

    def fit_mapping(self):
        try:
            self.result = fit(self.points)
        except ValueError as exc:
            self.message(str(exc)); return
        data = np.asarray(self.points)
        predicted = np.column_stack([data[:, :2], np.ones(len(data))]) @ np.asarray(self.result['affine']).T
        for row, error in enumerate(np.linalg.norm(predicted-data[:, 2:], axis=1)*1000):
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem('%.4f' % error))
        self.quality.setText('RMS %.4f mm | Maximum %.4f mm\nCheck unused reference points; fit residuals alone do not validate accuracy.' %
                             (1000*self.result['rms_error_m'], 1000*self.result['max_error_m']))
        self.message('Mapping fitted. Click an unused reference to read its mapped EM coordinate.')

    def check_heading(self):
        if self.result is None or self.mode.currentIndex() != 1 or len(self.view.marks) != 2:
            self.message('Fit mapping, choose Angle mode, then click magnetic tail → head along the axis.'); return
        try:
            angle = mapped_heading(self.result['affine'], *self.view.marks)
        except ValueError as exc:
            self.message(str(exc)); return
        known = self.heading_input.value()
        error = (angle-known+180) % 360-180
        self.dirty = True
        self.save_status.clear()
        self.heading_checks.append({'pixel_tail': list(self.view.marks[0]),
                                    'pixel_head': list(self.view.marks[1]),
                                    'known_heading_deg': known, 'mapped_heading_deg': angle,
                                    'error_deg': error})
        self.angle_status.setText('Mapped %.2f° | Known %.2f° | Error %+.2f°\nIn the experiment, place N in the +y half-plane and save the measured initial pose.' %
                                  (angle, known, error))
        self.message('Angle check recorded. No separate angle offset is needed: the affine map transforms the line axis.')

    def save_calibration(self, path):
        if self.result is None:
            raise ValueError('Fit mapping before saving')
        data = dict(self.result)
        data.update({'image_size_px': list(self.dimensions), 'heading_checks': self.heading_checks,
                     'created_at': datetime.now().astimezone().isoformat(),
                     'angle_convention': 'EM +x = 0 deg, +y = 90 deg; magnetic direction initialized manually',
                     'source': 'offline_image' if self.offline else 'openmv_usb'})
        # points are embedded in this JSON, ready for the existing CLI fitter too.
        with open(path, 'x') as stream:
            json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')
        self.dirty = False
        self.save_status.setText('Saved: '+str(path)+'\nUse this file as experiment.launch calibration:=…')
        return data

    def save_dialog(self):
        if self.result is None:
            self.message('Fit mapping before saving.'); return
        folder = PACKAGE / 'config' / 'calibrations'
        folder.mkdir(parents=True, exist_ok=True)
        default = folder / datetime.now().strftime('camera_%Y%m%d_%H%M%S.json')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save new measured calibration', str(default), 'JSON (*.json)')
        if path:
            try:
                self.save_calibration(path)
            except (OSError, ValueError) as exc:
                self.message('Cannot save (choose a new filename): '+str(exc))

    def closeEvent(self, event):
        if self.dirty:
            # Keep recorded measurements from being discarded accidentally.
            reply = QtWidgets.QMessageBox.question(self, 'Unsaved calibration', 'Close and discard unsaved calibration changes?',
                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                event.ignore(); return
        self.timer.stop(); self.disconnect_camera(); event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--connect', action='store_true', help='Connect to the camera when the window opens')
    parser.add_argument('--check-angle', action='store_true', help='Also show optional angle validation controls')
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    window = CalibrationWindow(args.port, position_only=not args.check_angle); window.show()
    if args.connect:
        QtCore.QTimer.singleShot(0, window.toggle_connection)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
