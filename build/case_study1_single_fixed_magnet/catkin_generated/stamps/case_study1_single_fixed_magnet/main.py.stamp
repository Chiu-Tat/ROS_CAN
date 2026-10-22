#!/usr/bin/env python3
"""Choose a CS1 layout before starting the shared camera/control UI."""
import argparse
import math
from pathlib import Path
import signal
import subprocess
import sys

from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from cs1_experiment.core import PACKAGE, LAYOUT_LABELS, fixed_magnets, load_config
from cs1_experiment.magnet_view import draw_magnet


def default_calibration():
    files = sorted((PACKAGE/'config/calibrations').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    return str(files[0] if files else PACKAGE/'config/camera_calibration.json')


def launch_command(layout, calibration, hardware=False, simulate=False, port='/dev/ttyACM0'):
    if layout not in LAYOUT_LABELS:raise ValueError('Unknown layout: '+layout)
    if hardware and simulate:raise ValueError('Simulation cannot enable CAN output')
    return ['roslaunch','case_study1_single_fixed_magnet','experiment.launch',
            'layout:='+layout,'calibration:='+str(Path(calibration).resolve()),'port:='+port,
            'hardware:='+str(hardware).lower(),'simulate:='+str(simulate).lower(),
            'gamepad:=false','input_source:=keyboard']


class LayoutDialog(QtWidgets.QDialog):

    def __init__(self, layout='single', calibration=None, hardware=False, simulate=False):
        super().__init__()
        self.setWindowTitle('Case Study 1 · Choose layout')
        self.setStyleSheet('QWidget {font-size:13px;} QPushButton {padding:7px;}')
        self.resize(580,630)
        box = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel('Choose fixed-magnet layout')
        title.setStyleSheet('font-size:20px;font-weight:bold;color:#173b47')
        box.addWidget(title)
        self.selector = QtWidgets.QComboBox()
        for name,label in LAYOUT_LABELS.items():self.selector.addItem(label,name)
        self.selector.setCurrentIndex(list(LAYOUT_LABELS).index(layout));box.addWidget(self.selector)
        self.figure = Figure(figsize=(5,3.6));self.figure.subplots_adjust(left=.14,right=.96,bottom=.15,top=.94)
        self.canvas = FigureCanvasQTAgg(self.figure);self.ax = self.figure.add_subplot(111)
        box.addWidget(self.canvas,1)
        self.layout_note = QtWidgets.QLabel();self.layout_note.setWordWrap(True);box.addWidget(self.layout_note)
        row = QtWidgets.QHBoxLayout()
        self.calibration = QtWidgets.QLineEdit(calibration or default_calibration())
        self.calibration.setToolTip('Saved pixel-to-EM calibration, shared by all layouts.')
        browse = QtWidgets.QPushButton('Calibration…');browse.clicked.connect(self.browse)
        row.addWidget(self.calibration,1);row.addWidget(browse);box.addLayout(row)
        self.simulate = QtWidgets.QCheckBox('Simulation (no devices)')
        self.hardware = QtWidgets.QCheckBox('Connect CAN · starts disarmed')
        self.hardware.setChecked(hardware);self.simulate.setChecked(simulate)
        box.addWidget(self.simulate);box.addWidget(self.hardware)
        self.simulate.toggled.connect(self.set_simulation);self.set_simulation(simulate)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Open|QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Open).setText('Open main UI')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);box.addWidget(buttons)
        self.selector.currentIndexChanged.connect(self.update_layout)
        self.update_layout()

    def set_simulation(self, enabled):
        if enabled:self.hardware.setChecked(False)
        self.hardware.setEnabled(not enabled)

    def browse(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self,'Camera calibration',self.calibration.text(),'JSON (*.json)')
        if path:self.calibration.setText(path)

    def update_layout(self, *_):
        layout = self.selector.currentData()
        config = load_config(layout=layout)
        self.ax.clear();self.ax.set_facecolor('#edf0f2')
        magnets = fixed_magnets(config['model'])
        for magnet in magnets:
            draw_magnet(self.ax,1000*magnet['xy_m'][0],1000*magnet['xy_m'][1],magnet['heading_rad'])
        self.ax.set(xlim=(-50,50),ylim=(-50,50),aspect='equal',xlabel='x (mm)',ylabel='y (mm)')
        self.ax.grid(alpha=.2);self.canvas.draw_idle()
        poses = '; '.join('(%.1f, %.1f) mm · N %.0f°' %
            (1000*m['xy_m'][0],1000*m['xy_m'][1],math.degrees(m['heading_rad'])) for m in magnets)
        moments = ', '.join('%.3f'%m['moment_Am2'] for m in magnets)
        note = ('Match layout and magnet strengths. Red = N, blue = S.\n'+poses+
                '\nFixed moments: '+moments+' A·m²; free: %.3f A·m².'%config['model']['free_moment_Am2'])
        mode = config.get('controller','pose_hold')
        if not config[mode].get('hardware_validated',False):note += '\nPowered arming is locked pending layout validation.'
        self.layout_note.setText(note)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout',choices=list(LAYOUT_LABELS),help='Skip layout dialog and use this layout')
    parser.add_argument('--calibration',default=default_calibration())
    parser.add_argument('--port',default='/dev/ttyACM0')
    parser.add_argument('--hardware',action='store_true',help='Connect CAN; still starts disarmed')
    parser.add_argument('--preview',action='store_true',help='Simulated feedback, no camera/CAN')
    args = parser.parse_args()
    if args.preview and args.hardware:parser.error('--preview cannot be combined with --hardware')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # One experiment owns /cs1 and the CAN/camera connections at a time.
    try:
        nodes = subprocess.run(['rosnode','list'],capture_output=True,text=True,timeout=3)
        if '/cs1_controller' in nodes.stdout.splitlines():
            QtWidgets.QMessageBox.warning(None,'Experiment already open',
                'Close the current experiment before choosing another layout.')
            return 1
    except subprocess.TimeoutExpired:
        QtWidgets.QMessageBox.warning(None,'ROS unavailable','Cannot check the current ROS experiment.');return 1
    if not args.layout:
        dialog = LayoutDialog(calibration=args.calibration,hardware=args.hardware,simulate=args.preview)
        if dialog.exec_()!=QtWidgets.QDialog.Accepted:return 0
        args.layout = dialog.selector.currentData();args.calibration = dialog.calibration.text()
        args.hardware = dialog.hardware.isChecked();args.preview = dialog.simulate.isChecked()
    command = launch_command(args.layout,args.calibration,args.hardware,args.preview,args.port)
    process = subprocess.Popen(command)
    try:return process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        try:return process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.terminate();return process.wait()


if __name__=='__main__':sys.exit(main())
