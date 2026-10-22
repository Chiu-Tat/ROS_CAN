"""Existing ten-driver CAN protocol, with independent measured-current reception."""
import threading
import time

import numpy as np

DEVICE_IDS = [0x013 + 0x10*k for k in range(10)]
FEEDBACK_IDS = [0x01B + 0x10*k for k in range(10)]


class CurrentOutput:
    def __init__(self, hardware, channel):
        self.bus = None
        self.initialized = False
        self.feedback_enabled = False
        self.io_lock = threading.RLock()
        self.feedback_lock = threading.Lock()
        self.values = [None]*10
        self.arrivals = [None]*10
        self.feedback_error = ''
        self.feedback_callback = None
        self.stop_receiver = threading.Event()
        self.receiver = None
        if hardware:
            import can
            self.can = can
            self.bus = can.interface.Bus(channel=channel, bustype='socketcan')
            self.receiver = threading.Thread(target=self.receive_feedback, daemon=True)
            self.receiver.start()

    def send(self, identifier, data):
        with self.io_lock:
            self.bus.send(self.can.Message(arbitration_id=identifier, data=data,
                                          is_extended_id=False), timeout=.05)

    def initialize(self):
        with self.io_lock:
            self.initialized = False
            if self.bus:
                self.send(0x001, [0x02]+[0x55]*7)
                time.sleep(.5)
                self.send(0x003, [0x13,0x88,0,0]+[0x55]*4)
            self.initialized = True

    def write(self, currents):
        currents = np.asarray(currents, float)
        if currents.shape != (10,) or not np.isfinite(currents).all() or np.max(np.abs(currents)) > 15:
            raise ValueError('Expected ten finite currents within +/-15 A')
        with self.io_lock:
            if self.bus and self.initialized:
                for identifier, amps in zip(DEVICE_IDS, currents):
                    value = int(round(float(amps)*1000)) & 0xffff
                    self.send(identifier, [0x13,0x88,value>>8,value&255]+[0x55]*4)

    def zero_all(self):
        # Broadcast zero also works before this host has initialized the drivers.
        with self.io_lock:
            if self.bus:
                errors = []
                for identifier in [0x003]+DEVICE_IDS:
                    try:
                        self.send(identifier, [0x13,0x88,0,0]+[0x55]*4)
                    except Exception as exc:
                        errors.append('0x%03X: %s' % (identifier, exc))
                if errors:
                    raise RuntimeError('Some zero commands failed: '+'; '.join(errors))

    def reset(self):
        with self.io_lock:
            self.zero_all()
            self.initialized = False
            if self.bus:
                self.send(0x000, [0x55]*8)
            self.feedback_enabled = False
            with self.feedback_lock:
                self.values, self.arrivals = [None]*10, [None]*10

    def enable_feedback(self):
        with self.io_lock:
            if self.bus:
                self.send(0x00A, [0x0A,0]+[0x55]*6)
                self.feedback_enabled = True

    def accept_feedback(self, message, now=None):
        if (message.is_extended_id or message.is_remote_frame or message.is_error_frame
                or message.arbitration_id not in FEEDBACK_IDS or len(message.data)<2):
            return False
        raw = (message.data[0]<<8) | message.data[1]
        if raw & 0x8000:
            raw -= 0x10000
        index = FEEDBACK_IDS.index(message.arbitration_id)
        with self.feedback_lock:
            self.values[index] = raw/1000.0
            self.arrivals[index] = time.monotonic() if now is None else now
            self.feedback_error = ''
        if self.feedback_callback is not None:
            self.feedback_callback({'can_id':message.arbitration_id,'current_A':raw/1000.,'data':list(message.data)})
        return True

    def receive_feedback(self):
        while not self.stop_receiver.is_set():
            try:
                message = self.bus.recv(timeout=.05)
                if message is not None:
                    self.accept_feedback(message)
            except Exception as exc:
                with self.feedback_lock:
                    self.feedback_error = str(exc)
                self.stop_receiver.wait(.1)

    def feedback_snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.feedback_lock:
            ages = [None if arrival is None else max(0., now-arrival) for arrival in self.arrivals]
            return {'available':self.bus is not None, 'enabled':self.feedback_enabled,
                    'values_A':list(self.values), 'age_s':ages,
                    'fresh':[age is not None and age<=1.0 for age in ages],
                    'can_ids':FEEDBACK_IDS, 'error':self.feedback_error}

    def close(self):
        if self.bus:
            try:
                self.zero_all()
            finally:
                self.initialized = False
                self.stop_receiver.set()
                if self.receiver is not None:
                    self.receiver.join(timeout=1)
                self.bus.shutdown()
