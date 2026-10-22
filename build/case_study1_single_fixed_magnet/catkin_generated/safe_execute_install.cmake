execute_process(COMMAND "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
