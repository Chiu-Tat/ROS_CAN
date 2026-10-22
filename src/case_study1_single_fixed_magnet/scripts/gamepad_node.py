#!/usr/bin/env python3
"""Linux /dev/input/js gamepad input with a held deadman button and heartbeat."""
import glob
import fcntl
import os
import select
import struct
import time

import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool


EVENT = struct.Struct('=IhBB')


def deadzone(value, threshold):
    return 0.0 if abs(value) <= threshold else (abs(value)-threshold)/(1-threshold) * (1 if value > 0 else -1)


def find_device(requested):
    for path in ([requested] if requested else sorted(glob.glob('/dev/input/js*'))):
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            axes, buttons = bytearray(1), bytearray(1)
            fcntl.ioctl(fd, 0x80016A11, axes)
            fcntl.ioctl(fd, 0x80016A12, buttons)
            if axes[0] >= 2 and buttons[0] >= 1:
                return path
        except OSError:
            pass
        finally:
            if fd is not None:
                os.close(fd)
    return None


def main():
    rospy.init_node('cs1_gamepad')
    pub = rospy.Publisher('/cs1/gamepad_command', Twist, queue_size=1)
    axis_drive = int(rospy.get_param('~axis_drive', 1))
    axis_turn = int(rospy.get_param('~axis_turn', 0))
    deadman = int(rospy.get_param('~deadman_button', 4))
    stop_button = int(rospy.get_param('~stop_button', 1))
    threshold = float(rospy.get_param('~deadzone', 0.15))
    if not 0 <= threshold < 1:
        raise ValueError('deadzone must be in [0,1)')
    drive_sign = float(rospy.get_param('~drive_sign', -1))
    turn_sign = float(rospy.get_param('~turn_sign', -1))
    fd, axes, buttons, buffer = None, {}, {}, b''
    released = False
    next_connect = 0.0
    rate = rospy.Rate(20)
    try:
        while not rospy.is_shutdown():
            try:
                if fd is None and time.monotonic() >= next_connect:
                    path = find_device(rospy.get_param('~device', ''))
                    if path:
                        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                        axes, buttons, buffer, released = {}, {}, b'', False
                        rospy.loginfo('Gamepad %s; drive axis %d, turn axis %d, deadman %d',
                                      path, axis_drive, axis_turn, deadman)
                    next_connect = time.monotonic()+1
                if fd is not None:
                    while select.select([fd], [], [], 0)[0]:
                        data = os.read(fd, EVENT.size*64)
                        if not data:
                            raise OSError('Gamepad disconnected')
                        buffer += data
                        while len(buffer) >= EVENT.size:
                            _, value, kind, number = EVENT.unpack(buffer[:EVENT.size])
                            buffer = buffer[EVENT.size:]
                            initial, kind = bool(kind & 0x80), kind & 0x7f
                            if kind == 2:
                                axes[number] = max(-1, min(1, value/32767.0))
                            elif kind == 1:
                                buttons[number] = value
                                if number == deadman and not value:
                                    released = True
                                if number == stop_button and value and not initial:
                                    try:
                                        rospy.ServiceProxy('/cs1/arm', SetBool)(False)
                                    except rospy.ServiceException as exc:
                                        rospy.logwarn('Disarm service unavailable: %s', exc)
                    msg = Twist()
                    if released and buttons.get(deadman, 0):
                        msg.linear.x = drive_sign*deadzone(axes.get(axis_drive, 0), threshold)
                        msg.angular.z = turn_sign*deadzone(axes.get(axis_turn, 0), threshold)
                    # Send at a steady rate even while a stick is held still.
                    pub.publish(msg)
            except OSError as exc:
                rospy.logwarn_throttle(2, 'Gamepad unavailable: %s', exc)
                if fd is not None:
                    os.close(fd)
                fd, released = None, False
                # Stop heartbeats: controller disarms on timeout, rather than resuming on reconnect.
            rate.sleep()
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == '__main__':
    main()
