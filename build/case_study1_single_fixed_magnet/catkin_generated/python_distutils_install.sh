#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/dz/Documents/CUHK/ROS_CAN/src/case_study1_single_fixed_magnet"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/dz/Documents/CUHK/ROS_CAN/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/dz/Documents/CUHK/ROS_CAN/install/lib/python3/dist-packages:/home/dz/Documents/CUHK/ROS_CAN/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/dz/Documents/CUHK/ROS_CAN/build" \
    "/usr/bin/python3" \
    "/home/dz/Documents/CUHK/ROS_CAN/src/case_study1_single_fixed_magnet/setup.py" \
     \
    build --build-base "/home/dz/Documents/CUHK/ROS_CAN/build/case_study1_single_fixed_magnet" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/dz/Documents/CUHK/ROS_CAN/install" --install-scripts="/home/dz/Documents/CUHK/ROS_CAN/install/bin"
