#!/usr/bin/env python3

import time
from threading import Event, Lock, Thread
from typing import Protocol, Sequence, Optional

import rclpy
from rclpy.node import Node

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
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# LEN_GOAL_POSITION = 4
# LEN_PRESENT_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

class Gripper:
    def __init__(self,
                 port: Optional[str] = None,
                 baudrate: Optional[int] = None,
                 motor_ids: Optional[Sequence[int]] = None 
                 ):
        self.port = port
        self.baudrate = baudrate
        self.ids = motor_ids

    def open_port(self):
        self.portHandler = PortHandler(self.port)
        self._packetHandler = PacketHandler(2.0)
        if not self.portHandler.openPort():
            raise RuntimeError(f"Failed to open port: {self.port}")
        if not self.portHandler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate: {self.baudrate}")

    def run_init(self):
        self.set_torque(self.ids[0], False)
        self.set_torque(self.ids[1], True)
        self.leader_init = self.read_angle(self.ids[0])
        self.follow_init = self.read_angle(self.ids[1])


    def set_torque(self, id: int, enable: bool):
        dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
            self.portHandler, id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE if enable else TORQUE_DISABLE
        )
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(f"Failed to set torque: {id}, {enable}")

    def read_angle(self, id: int):
        last_position, dxl_comm_result, dxl_error = self._packetHandler.read4ByteTxRx(
            self.portHandler, id, ADDR_PRESENT_POSITION
        )
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(f"Failed to read angle: {id}")
        return last_position

    def set_angle(self, id: int, angle: int):
        dxl_comm_result, dxl_error = self._packetHandler.write4ByteTxRx(
            self.portHandler, id, ADDR_GOAL_POSITION, angle
        )
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(f"Failed to set angle: {id}, {angle}")

class GripperNode(Node):
    def __init__(self, gripper: Gripper):
        super().__init__("gripper_node")

        self.declare_parameter("u2d2_prot", "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VPII-if00-port0")
        self.declare_parameter("baudrate", 57600)
        self.declare_parameter("motor_ids", [7, 8])
        self.declare_parameter("leader_dis", 500)
        self.declare_parameter("follower_dis", 516)
        self.port = self.get_parameter("u2d2_prot").get_parameter_value().string_value
        self.baudrate = self.get_parameter("baudrate").get_parameter_value().integer_value
        self.ids = self.get_parameter("motor_ids").get_parameter_value().integer_array_value
        self.leader_dis = self.get_parameter("leader_dis").get_parameter_value().integer_value
        self.follower_dis = self.get_parameter("follower_dis").get_parameter_value().integer_value

        self.gripper = gripper
        self.gripper.port = self.port
        self.gripper.baudrate = self.baudrate
        self.gripper.ids = self.ids

        self.gripper.open_port()
        self.gripper.run_init()

        timer_period = 0.02
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.connect = True
    
    def timer_callback(self):
        if self.connect:
            leader_angle = self.gripper.read_angle(self.ids[0])
            purpose_angle = int(self.gripper.follow_init + (leader_angle - self.gripper.leader_init) * self.follower_dis / self.leader_dis)
            if purpose_angle < self.gripper.follow_init-self.follower_dis:
                self.gripper.set_angle(self.ids[1], self.gripper.follow_init-self.follower_dis)
            elif purpose_angle > self.gripper.follow_init:
                self.gripper.set_angle(self.ids[1], self.gripper.follow_init)
            else:
                self.gripper.set_angle(self.ids[1], purpose_angle)

    def cleanup(self):
        # self._packetHandler.write4ByteTxRx(
        #     self._portHandler, self.ids[1], ADDR_GOAL_POSITION, self.follow_init
        # )
        self.connect = False
        time.sleep(0.5)
        self.gripper.portHandler.closePort()
        time.sleep(0.5)
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def main():
    rclpy.init()
    gripper = Gripper()
    gripper_node = GripperNode(gripper)
    port = gripper_node.port
    baudrate = gripper_node.baudrate
    ids = gripper_node.ids
    try:
        # print(f"Leader: {gripper_node.leader_init}, Follower: {gripper_node.follow_init}")
        rclpy.spin(gripper_node)
    except KeyboardInterrupt:
        pass
    finally:
        gripper_node.cleanup()
        gripper_ = Gripper(port, baudrate)
        gripper_.open_port()
        gripper_.set_torque(ids[1], False)
        gripper_.portHandler.closePort()

if __name__ == "__main__":
    main()
