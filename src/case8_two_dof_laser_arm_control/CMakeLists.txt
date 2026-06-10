cmake_minimum_required(VERSION 3.0.2)
project(case8_two_dof_laser_arm_control)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/collect_data.py
  scripts/train_model.py
  scripts/evaluate_model.py
  scripts/nn_closed_loop_control.py
  scripts/mlp_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
