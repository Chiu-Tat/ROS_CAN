#!/usr/bin/env python
"""
CAN Library Module
Contains low-level CAN message definitions, conversions, and utilities.
"""

import can
import rospy


# CAN Message Definitions
class CANMessageDefs:
    """CAN message definitions and constants."""
    
    # Control CAN IDs
    RESET_ID = 0x000
    MODE_SET_ID = 0x001
    INIT_ID = 0x003
    FEEDBACK_RATE_ID = 0x00A
    
    # Device control CAN IDs (for setting values)
    DEVICE_IDS = [0x013, 0x023, 0x033, 0x043, 0x053, 0x063, 0x073, 0x083, 0x093, 0x0A3]
    
    # Feedback CAN IDs
    FEEDBACK_IDS = [0x01B, 0x02B, 0x03B, 0x04B, 0x05B, 0x06B, 0x07B, 0x08B, 0x09B, 0x0AB]
    
    # Control message data patterns
    RESET_DATA = [0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55]
    MODE_SET_DATA = [0x02, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55]
    INIT_DATA = [0x13, 0x88, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55]
    FEEDBACK_RATE_DATA = [0x0A, 0x00, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55]
    
    # Current limits
    MIN_CURRENT = -15000
    MAX_CURRENT = 15000


def can_to_current(data):
    """
    Convert CAN message data (2 bytes) to current value.
    
    Args:
        data: List of bytes (must be at least 2 bytes)
    
    Returns:
        int: Current value in milliamperes
    
    Raises:
        ValueError: If data is not exactly 2 bytes
    """
    if len(data) < 2:
        raise ValueError("Data must be at least 2 bytes")
    
    # Combine the two bytes into a single raw value
    raw_value = (data[0] << 8) | data[1]
    
    # If the raw value is negative in two's complement (16 bits), correct it
    if raw_value & 0x8000:  # Check the sign bit
        raw_value -= 0x10000
    
    return raw_value


def current_to_can(current):
    """
    Convert current value to CAN message data (2 bytes).
    
    Args:
        current: Current value in milliamperes
    
    Returns:
        list: Two-byte list [high_byte, low_byte]
    """
    # Clamp the current value to the valid range
    current = max(CANMessageDefs.MIN_CURRENT, min(CANMessageDefs.MAX_CURRENT, current))
    current &= 0xFFFF  # Mask to 16 bits
    data_0 = (current >> 8) & 0xFF  # Get the high byte
    data_1 = current & 0xFF  # Get the low byte
    
    return [data_0, data_1]


def build_device_current_message(current_mA):
    """
    Build a CAN message data array for setting device current.
    
    Args:
        current_mA: Current value in milliamperes
    
    Returns:
        list: 8-byte CAN message data
    """
    [data_0, data_1] = current_to_can(current_mA)
    return [0x13, 0x88, data_0, data_1, 0x55, 0x55, 0x55, 0x55]


class CANBusHandler:
    """Handler for CAN bus operations."""
    
    def __init__(self, channel="can0", bustype="socketcan"):
        """
        Initialize CAN bus handler.
        
        Args:
            channel: CAN channel name (default: "can0")
            bustype: CAN bus type (default: "socketcan")
        """
        self.bus = can.interface.Bus(channel=channel, bustype=bustype)
    
    def send_message(self, can_id, data):
        """
        Send a CAN message.
        
        Args:
            can_id: CAN message ID
            data: List of bytes (up to 8 bytes)
        """
        try:
            message = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
            self.bus.send(message)
            rospy.loginfo(f"Sent CAN message: ID={hex(can_id)}, Data={data}")
        except can.CanError as e:
            rospy.logerr(f"CAN error: {e}")
        except Exception as e:
            rospy.logerr(f"Failed to send CAN message: {e}")
    
    def receive_message(self, timeout=1):
        """
        Receive a CAN message.
        
        Args:
            timeout: Timeout in seconds (default: 1)
        
        Returns:
            can.Message or None: Received message or None if timeout
        """
        try:
            return self.bus.recv(timeout=timeout)
        except Exception as e:
            rospy.logerr(f"Error receiving CAN message: {e}")
            return None
    
    def close(self):
        """Close the CAN bus."""
        if self.bus:
            self.bus.shutdown()


class CANMessageBuilder:
    """Builder class for creating CAN messages."""
    
    @staticmethod
    def reset_all():
        """Build reset message for all devices."""
        return CANMessageDefs.RESET_ID, CANMessageDefs.RESET_DATA
    
    @staticmethod
    def mode_set():
        """Build mode set message."""
        return CANMessageDefs.MODE_SET_ID, CANMessageDefs.MODE_SET_DATA
    
    @staticmethod
    def initialize():
        """Build initialization message (set all currents to 0)."""
        return CANMessageDefs.INIT_ID, CANMessageDefs.INIT_DATA
    
    @staticmethod
    def set_feedback_rate():
        """Build feedback rate setting message."""
        return CANMessageDefs.FEEDBACK_RATE_ID, CANMessageDefs.FEEDBACK_RATE_DATA
    
    @staticmethod
    def set_device_current(device_index, current_mA):
        """
        Build message to set device current.
        
        Args:
            device_index: Device index (0-9)
            current_mA: Current value in milliamperes
        
        Returns:
            tuple: (can_id, data) for the message
        """
        if device_index < 0 or device_index >= len(CANMessageDefs.DEVICE_IDS):
            raise ValueError(f"Device index must be between 0 and {len(CANMessageDefs.DEVICE_IDS)-1}")
        
        can_id = CANMessageDefs.DEVICE_IDS[device_index]
        data = build_device_current_message(current_mA)
        return can_id, data

