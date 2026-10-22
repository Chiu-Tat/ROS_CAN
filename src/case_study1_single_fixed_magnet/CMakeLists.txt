cmake_minimum_required(VERSION 3.0.2)
project(case_study1_single_fixed_magnet)
find_package(catkin REQUIRED COMPONENTS rospy geometry_msgs std_msgs std_srvs sensor_msgs)
catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS
  scripts/main.py
  scripts/controller_node.py
  scripts/gamepad_node.py
  scripts/openmv_tracker.py
  scripts/control_panel.py
  scripts/simulate_feedback.py
  scripts/camera_calibration_ui.py
  scripts/calibrate_camera.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
# Helpers/camera code and package-relative assets must remain together for install-space use.
install(DIRECTORY scripts config data launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "__pycache__" EXCLUDE
)
