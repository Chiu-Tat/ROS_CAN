#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-dependency Xbox/gamepad reader for Case 7 joystick teleoperation.

Reads the Linux joystick character device (/dev/input/jsX) directly using the
fixed 8-byte js_event protocol, so it needs no pygame / ros-joy / evdev package.

It auto-detects the gamepad by probing each /dev/input/js* device and skipping
ones that are not controllers (e.g. the laptop accelerometer, which reports 0
buttons). The detected device, axis/button indices, deadzone and axis inversion
are all configurable.

Two callbacks drive the controller:
  - on_vector(x, y): normalised right-stick vector in [-1, 1], deadzone applied.
  - on_cycle(): the "switch target" button (A by default) was pressed.
"""

import fcntl
import glob
import math
import os
import select
import struct
import threading
import time


# struct js_event { __u32 time; __s16 value; __u8 type; __u8 number; }
EVENT_STRUCT = struct.Struct("=IhBB")
EVENT_SIZE = EVENT_STRUCT.size  # 8 bytes

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80  # OR'd into type for the synthetic initial-state events

# joystick ioctls (see linux/joystick.h)
JSIOCGAXES = 0x80016A11      # get number of axes (__u8)
JSIOCGBUTTONS = 0x80016A12   # get number of buttons (__u8)

AXIS_MAX = 32767.0

GAMEPAD_NAME_HINTS = (
    "x-box", "xbox", "x box", "microsoft", "gamepad", "controller", "wireless",
    "sony", "playstation", "dualshock", "dualsense", "8bitdo", "logitech",
    "steam", "nintendo", "pro controller", "joystick",
)


def _jsiocgname(length):
    """Build the JSIOCGNAME(len) ioctl request for a given buffer length."""
    return 0x80006A13 + (length << 16)


def probe_device(device_path):
    """Return (name, num_axes, num_buttons) for a js device, or (None, 0, 0)."""
    fd = None
    try:
        fd = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
        axes_buf = bytearray(1)
        fcntl.ioctl(fd, JSIOCGAXES, axes_buf)
        buttons_buf = bytearray(1)
        fcntl.ioctl(fd, JSIOCGBUTTONS, buttons_buf)
        name_buf = bytearray(128)
        fcntl.ioctl(fd, _jsiocgname(len(name_buf)), name_buf)
        name = name_buf.split(b"\x00")[0].decode(errors="ignore")
        return name, int(axes_buf[0]), int(buttons_buf[0])
    except OSError:
        return None, 0, 0
    finally:
        if fd is not None:
            os.close(fd)


class JoystickReader(object):
    def __init__(
        self,
        axis_x=3,
        axis_y=4,
        button_cycle=0,
        deadzone=0.15,
        invert_x=False,
        invert_y=True,
        device=None,
        on_vector=None,
        on_cycle=None,
        logger=print,
        reconnect_delay_s=2.0,
    ):
        self.axis_x = int(axis_x)
        self.axis_y = int(axis_y)
        self.button_cycle = int(button_cycle)
        self.deadzone = min(0.95, max(0.0, float(deadzone)))
        self.invert_x = bool(invert_x)
        self.invert_y = bool(invert_y)
        self.device_param = device or None
        self.on_vector = on_vector
        self.on_cycle = on_cycle
        self.logger = logger
        self.reconnect_delay_s = float(reconnect_delay_s)

        self._fd = None
        self._device_path = None
        self._axes = {}
        self._running = False
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._close()

    def _close(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd = None
        self._device_path = None
        self._axes = {}

    def _detect_device(self):
        if self.device_param:
            return self.device_param
        fallback = None
        for device_path in sorted(glob.glob("/dev/input/js*")):
            name, num_axes, num_buttons = probe_device(device_path)
            if num_buttons < 1 or num_axes < 2:
                self.logger(
                    "[joystick] skipping %s (%s: %d axes, %d buttons)"
                    % (device_path, name, num_axes, num_buttons)
                )
                continue
            lname = (name or "").lower()
            if any(hint in lname for hint in GAMEPAD_NAME_HINTS):
                return device_path
            if fallback is None:
                fallback = device_path
        return fallback

    def _open(self):
        device_path = self._detect_device()
        if device_path is None:
            self.logger("[joystick] no gamepad found; plug in the Xbox controller (waiting)...")
            return False
        try:
            self._fd = os.open(device_path, os.O_RDONLY)
        except OSError as exc:
            self.logger("[joystick] failed to open %s: %s" % (device_path, exc))
            self._fd = None
            return False
        self._device_path = device_path
        self._axes = {}
        name, num_axes, num_buttons = probe_device(device_path)
        self.logger(
            "[joystick] using %s (%s: %d axes, %d buttons); right stick axes=(%d, %d), cycle button=%d"
            % (device_path, name, num_axes, num_buttons, self.axis_x, self.axis_y, self.button_cycle)
        )
        return True

    def _run(self):
        buffer = b""
        while self._running:
            if self._fd is None:
                if not self._open():
                    time.sleep(self.reconnect_delay_s)
                    continue
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.5)
                if not ready:
                    continue
                data = os.read(self._fd, EVENT_SIZE * 32)
                if not data:
                    self.logger("[joystick] device closed; will try to reconnect")
                    self._close()
                    continue
                buffer += data
                while len(buffer) >= EVENT_SIZE:
                    chunk = buffer[:EVENT_SIZE]
                    buffer = buffer[EVENT_SIZE:]
                    _, value, etype, number = EVENT_STRUCT.unpack(chunk)
                    self._handle_event(value, etype, number)
            except OSError as exc:
                self.logger("[joystick] read error (%s); reconnecting" % exc)
                self._close()
                time.sleep(self.reconnect_delay_s)

    def _handle_event(self, value, etype, number):
        etype &= ~JS_EVENT_INIT
        if etype == JS_EVENT_AXIS:
            self._axes[number] = value
            self._emit_vector()
        elif etype == JS_EVENT_BUTTON and number == self.button_cycle and value == 1:
            if self.on_cycle is not None:
                self.on_cycle()

    def _emit_vector(self):
        raw_x = max(-1.0, min(1.0, self._axes.get(self.axis_x, 0) / AXIS_MAX))
        raw_y = max(-1.0, min(1.0, self._axes.get(self.axis_y, 0) / AXIS_MAX))
        if self.invert_x:
            raw_x = -raw_x
        if self.invert_y:
            raw_y = -raw_y

        magnitude = math.hypot(raw_x, raw_y)
        if magnitude <= self.deadzone:
            x = 0.0
            y = 0.0
        else:
            scale = min(1.0, (magnitude - self.deadzone) / (1.0 - self.deadzone))
            x = raw_x / magnitude * scale
            y = raw_y / magnitude * scale

        if self.on_vector is not None:
            self.on_vector(x, y)


if __name__ == "__main__":
    # Standalone smoke test: print detected device and live stick vectors.
    def _print_vector(x, y):
        if x or y:
            print("vector=(% .3f, % .3f)" % (x, y))

    def _print_cycle():
        print("cycle button pressed")

    reader = JoystickReader(on_vector=_print_vector, on_cycle=_print_cycle)
    reader.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        reader.stop()
