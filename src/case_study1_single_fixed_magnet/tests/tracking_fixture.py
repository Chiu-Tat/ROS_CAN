"""Synthetic camera observations; never touch USB/CAN or real experiment files."""
import math
from pathlib import Path
import threading
from unittest import mock

import numpy as np
from cs1_experiment.core import load_config
from cs1_experiment.tracking import DETECTOR


def blob(degrees=90, center=(20, 20), length_px=11, width_px=4, kind=DETECTOR):
    angle = math.radians(degrees)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    corners = np.array([[-length_px/2,-width_px/2], [length_px/2,-width_px/2],
                        [length_px/2,width_px/2], [-length_px/2,width_px/2]]) @ rotation.T + center
    return {'x':center[0], 'y':center[1], 'axis_rad':angle % math.pi,
            'pixels':40, 'roundness':.15, 'kind':kind, 'corners_px':corners.tolist()}


def frame(sequence, degrees=90, lines=None):
    return {'sequence':sequence, 'lines':[blob(degrees)] if lines is None else lines,
            'fps':30, 'detector':DETECTOR, 'image_size_px':[320,240]}


def tracker(folder):
    from openmv_tracker import LineTracker
    value = LineTracker.__new__(LineTracker)
    value.config = load_config()
    value.affine = [[.001,0,-.05],[0,.001,-.05]]
    value.image_size = [320,240]
    value.calibration_path = 'synthetic_camera.json'
    value.results_dir = Path(folder)
    value.lock = threading.RLock()
    value.raw = value.previous_xy = value.heading = value.acquisition_xy = None
    value.raw_time = value.last_valid_time = 0
    value.last_sequence = None
    value.stable_frames = 0
    value.acquisition_axis = None
    value.last_size = value.initial_heading = None
    value.session_log = value.session_writer = value.ser = None
    for name in ['pub','valid','fps','image_pub','heading_pub','initial_pose_pub','status_pub']:
        setattr(value, name, mock.Mock())
    return value
