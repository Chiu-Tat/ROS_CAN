#!/usr/bin/env python3
"""Choose an autonomous CS1 layout and launch camera or simulation, disarmed."""
import argparse
import json
import signal
import subprocess
import sys
from PyQt5 import QtWidgets
from cs1_experiment.core import LAYOUT_LABELS
from cs1_autonomy.planning import config_path
from cs1_autonomy.shared import shared_node

shared_main = shared_node('main.py')


class AutoLayoutDialog(shared_main.LayoutDialog):
    def update_layout(self, *_):
        super().update_layout()
        self.setWindowTitle('Case Study 1 · Autonomous planning')
        self.layout_note.setText(self.layout_note.text().split('\nPowered arming')[0]+
            '\nAutonomous: approach the paper path from the camera pose.\nPowered arming is locked pending autonomous hardware validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout',choices=list(LAYOUT_LABELS))
    parser.add_argument('--preview',action='store_true',help='Simulated plant with a small start offset; no devices')
    parser.add_argument('--hardware',action='store_true',help='Connect CAN, remain disarmed')
    parser.add_argument('--port',help='OpenMV port; defaults to the selected layout camera config')
    parser.add_argument('--calibration',default=shared_main.default_calibration())
    args = parser.parse_args()
    if args.hardware and args.preview:parser.error('Preview cannot connect CAN')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        result = subprocess.run(['rosnode','list'],capture_output=True,text=True,timeout=5)
        if '/cs1_controller' in result.stdout.splitlines():
            QtWidgets.QMessageBox.warning(None,'Experiment already open',
                'Close the manual or autonomous experiment, resolving its trial, before starting another.')
            return 1
    except subprocess.TimeoutExpired:
        QtWidgets.QMessageBox.warning(None,'ROS unavailable','Cannot check the existing experiment.');return 1
    if not args.layout:
        dialog = AutoLayoutDialog(calibration=args.calibration,hardware=args.hardware,simulate=args.preview)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:return 0
        args.layout = dialog.selector.currentData(); args.calibration = dialog.calibration.text()
        args.hardware = dialog.hardware.isChecked(); args.preview = dialog.simulate.isChecked()
    config = json.loads(config_path(args.layout).read_text())
    command = ['roslaunch','case_study1_auto_planning_control','experiment.launch',
               'layout:='+args.layout,'hardware:='+str(args.hardware).lower(),
               'simulate:='+str(args.preview).lower(),'calibration:='+args.calibration,
               'port:='+(args.port or config['camera']['port'])]
    child = subprocess.Popen(command)
    try:return child.wait()
    except KeyboardInterrupt:
        child.send_signal(signal.SIGINT)
        try:return child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.terminate();return child.wait()


if __name__ == '__main__':sys.exit(main())
