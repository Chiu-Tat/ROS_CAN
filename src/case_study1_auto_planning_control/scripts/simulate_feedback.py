#!/usr/bin/env python3
"""Shared dipole plant seeded near the paper mission start, never hardware."""
import numpy as np
import rospy
from cs1_autonomy.planning import load_reference
from cs1_autonomy.shared import shared_node


class AutoPreviewPlant(shared_node('simulate_feedback.py').PreviewPlant):
    def __init__(self):
        super().__init__()
        layout = self.core.model.settings.get('layout', 'single')
        route, _ = load_reference(layout)
        offset = np.asarray(rospy.get_param('~start_offset', [.001, .001, .08]), float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('Expected finite [dx_m, dy_m, dbeta_rad]')
        self.pose = route[0, :3].copy()+offset
        self.initialize(None)


if __name__ == '__main__':
    rospy.init_node('cs1_preview_feedback')
    plant = AutoPreviewPlant()
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        plant.tick(.02); rate.sleep()
