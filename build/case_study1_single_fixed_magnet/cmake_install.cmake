# Install script for directory: /home/dz/Documents/CUHK/ROS_CAN/src/case_study1_single_fixed_magnet

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/dz/Documents/CUHK/ROS_CAN/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/safe_execute_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/case_study1_single_fixed_magnet.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/case_study1_single_fixed_magnet/cmake" TYPE FILE FILES
    "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/case_study1_single_fixed_magnetConfig.cmake"
    "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/case_study1_single_fixed_magnetConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/case_study1_single_fixed_magnet" TYPE FILE FILES "/home/dz/Documents/CUHK/ROS_CAN/src/case_study1_single_fixed_magnet/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/case_study1_single_fixed_magnet" TYPE PROGRAM FILES "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/main.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/case_study1_single_fixed_magnet" TYPE PROGRAM FILES "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/controller_node.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/case_study1_single_fixed_magnet" TYPE PROGRAM FILES "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/gamepad_node.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/case_study1_single_fixed_magnet" TYPE PROGRAM FILES "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/openmv_tracker.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/case_study1_single_fixed_magnet" TYPE PROGRAM FILES "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/control_panel.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/case_study1_single_fixed_magnet" TYPE PROGRAM FILES "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/simulate_feedback.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/case_study1_single_fixed_magnet" TYPE PROGRAM FILES "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/camera_calibration_ui.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/case_study1_single_fixed_magnet" TYPE PROGRAM FILES "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/installspace/calibrate_camera.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/case_study1_single_fixed_magnet" TYPE DIRECTORY FILES
    "/home/dz/Documents/CUHK/ROS_CAN/src/case_study1_single_fixed_magnet/scripts"
    "/home/dz/Documents/CUHK/ROS_CAN/src/case_study1_single_fixed_magnet/config"
    "/home/dz/Documents/CUHK/ROS_CAN/src/case_study1_single_fixed_magnet/data"
    "/home/dz/Documents/CUHK/ROS_CAN/src/case_study1_single_fixed_magnet/launch"
    REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE)
endif()

