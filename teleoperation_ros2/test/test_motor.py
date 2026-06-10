import time
from threading import Event, Lock, Thread
from typing import Protocol, Sequence, Any
from pynput import keyboard

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

# Constants
ADDR_BAUDRATE = 8
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4
TORQUE_DISABLE = 0
TORQUE_ENABLE = 1
CURRENT_MODE = 0
POSITION_MODE = 3
CURRENT_POSITION_MODE = 5

signal = 0


port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VPII-if00-port0"
baudrate = 57600
# baudrate = 1e6

listener = keyboard.Listener(on_press=lambda key: on_press(key, signal))
listener.start()

portHandler = PortHandler(port)
packetHandler = PacketHandler(2.0)

if not portHandler.openPort():
    raise RuntimeError("Failed to open the port")

if not portHandler.setBaudRate(baudrate):
    raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

def on_press(key, signal):
    try:
        if key.char == '1':
            signal = 1
        elif key.char == '2':
            signal = 2
        elif key.char == '3':
            signal = 3
        elif key.char == '4':
            signal = 4
    except AttributeError:
        pass

def read1Byte(portHandler, dxl_id, addr):
    dxl_read_value, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(
        portHandler,
        dxl_id,
        addr
    )
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeError(f"Failed to read 1 byte: {dxl_comm_result}, {dxl_error}")
    return np.array(dxl_read_value, dtype=np.uint8)

def read2Byte(portHandler, dxl_id, addr):
    dxl_read_value, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(
        portHandler,
        dxl_id,
        addr
    )
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeError(f"Failed to read 2 bytes: {dxl_comm_result}, {dxl_error}")
    return np.array(dxl_read_value, dtype=np.int16)

def read4Byte(portHandler, dxl_id, addr):
    dxl_read_value, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
        portHandler,
        dxl_id,
        addr
    )
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeError(f"Failed to read 4 bytes: {dxl_comm_result}, {dxl_error}")
    return np.array(dxl_read_value, dtype=np.int32)

def write1Byte(portHandler, dxl_id, addr, value):
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
        portHandler,
        dxl_id,
        addr,
        value
    )
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeError(f"Failed to write 1 byte: {dxl_comm_result}, {dxl_error}")
    
def write2Byte(portHandler, dxl_id, addr, value):
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
        portHandler,
        dxl_id,
        addr,
        value
    )
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeError(f"Failed to write 2 bytes: {dxl_comm_result}, {dxl_error}")
    
def write4Byte(portHandler, dxl_id, addr, value):
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
        portHandler,
        dxl_id,
        addr,
        value
    )
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeError(f"Failed to write 4 bytes: {dxl_comm_result}, {dxl_error}")
    
# mode_7 = read1Byte(portHandler, 7, ADDR_OPERATING_MODE)
# print(f"Motor 7 mode: {mode_7}")
# write1Byte(portHandler, 7, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

mode_8 = read1Byte(portHandler, 8, ADDR_OPERATING_MODE)
print(f"Motor 8 mode: {mode_8}")
# write1Byte(portHandler, 8, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
# baudrate_8 = read1Byte(portHandler, 8, ADDR_BAUDRATE)
# print(f"Motor 8 baudrate: {baudrate_8}")
# write1Byte(portHandler, 8, ADDR_BAUDRATE, 1)
# if not portHandler.setBaudRate(57600):
#     raise RuntimeError(f"Failed to change the baudrate, {57600}")
# baudrate_8 = read1Byte(portHandler, 8, ADDR_BAUDRATE)
# print(f"Motor 8 baudrate: {baudrate_8}")
# write1Byte(portHandler, 8, ADDR_OPERATING_MODE, CURRENT_MODE)
# write1Byte(portHandler, 8, ADDR_OPERATING_MODE, POSITION_MODE)
write1Byte(portHandler, 8, ADDR_OPERATING_MODE, CURRENT_POSITION_MODE)
mode_8 = read1Byte(portHandler, 8, ADDR_OPERATING_MODE)
print(f"Motor 8 mode: {mode_8}")

# baudrate_1 = read1Byte(portHandler, 1, ADDR_BAUDRATE)
# print(f"Motor 8 baudrate: {baudrate_1}")

# try:
#     while True:
#         begin = time.time()
#         current_7 = read2Byte(portHandler, 7, ADDR_PRESENT_CURRENT)
#         velocity_7 = read4Byte(portHandler, 7, ADDR_PRESENT_VELOCITY)
#         print(f"Current: {current_7}, Velocity: {velocity_7}")
#         end = time.time()
#         if end - begin < 0.1:
#             time.sleep(0.1 - (end - begin))
#         else:
#             continue
# except KeyboardInterrupt:
#     pass
# finally:
#     time.sleep(1)
#     write1Byte(portHandler, 7, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
#     portHandler.closePort()
#     print("Port closed")

write1Byte(portHandler, 8, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
# try: 
#     while True:
#         write2Byte(portHandler, 8, ADDR_GOAL_CURRENT, int(np.array(0).astype(np.uint16)))
# except KeyboardInterrupt:
#     pass
# finally:
#     time.sleep(1)
#     write2Byte(portHandler, 8, ADDR_GOAL_CURRENT, int(np.array(0).astype(np.uint16)))
#     write1Byte(portHandler, 8, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
#     portHandler.closePort()
#     print("Port closed")
while signal == 0:
    write2Byte(portHandler, 8, ADDR_GOAL_CURRENT, int(np.array(10).astype(np.uint16)))

write1Byte(portHandler, 8, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
# current_8 = read2Byte(portHandler, 8, ADDR_PRESENT_CURRENT)
# print(f"Motor 8 current: {current_8}")
portHandler.closePort()
print("Port closed")

listener.stop()