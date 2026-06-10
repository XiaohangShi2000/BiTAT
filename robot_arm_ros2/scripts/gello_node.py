#!/usr/bin/env python3

import numpy as np
import yaml
from typing import Optional, Union, Sequence
import time
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import COMM_SUCCESS

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
# from std_msgs.msg import String

from leap_hand_utils.gello_agent import GelloAgent

ADDR_TORQUE_ENABLE = 64

# class Gripper:
#     def __init__(self,
#                  port: Optional[str] = None,
#                  baudrate: Optional[int] = 57600,
#                  ):
#         self.port = port
#         self.baudrate = baudrate

#     def open_port(self):
#         self.portHandler = PortHandler(self.port)
#         self._packetHandler = PacketHandler(2.0)
#         if not self.portHandler.openPort():
#             raise RuntimeError(f"Failed to open port: {self.port}")
#         if not self.portHandler.setBaudRate(self.baudrate):
#             raise RuntimeError(f"Failed to set baudrate: {self.baudrate}")

#     def set_torque(self, id: int, enable: bool):
#         dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
#             self.portHandler, id, ADDR_TORQUE_ENABLE, 1 if enable else 0
#         )
#         if dxl_comm_result != COMM_SUCCESS:
#             raise RuntimeError(f"Failed to set torque: {id}, {enable}")

class GelloNode(Node):
    give_pos: Optional[np.ndarray]
    def __init__(self):
        super().__init__('gello_node')

        self.declare_parameter('u2d2_prot','/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VPII-if00-port0')
        a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.declare_parameter('start_pos', a)
        self.u2d2_prot = self.get_parameter('u2d2_prot').get_parameter_value().string_value
        start_pos = self.get_parameter('start_pos').get_parameter_value().double_array_value
        if start_pos == a:
            give_pose = None
        else:
            give_pose = np.array(start_pos)
        self.agent = GelloAgent(port=self.u2d2_prot, start_joints=give_pose)

        self.publisher_ = self.create_publisher(JointState, 'gello_joint', 1)
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = JointState()
        msg.name = [f'joint{i+1}' for i in range(8)]
        msg.header.stamp = self.get_clock().now().to_msg()
        joint_data = self.agent.act().tolist()
        if len(msg.name) != len(joint_data):
            self.get_logger().error(
                f"Length mismatch between joint names and positions. "
                f"Names length: {len(msg.name)}, Positions length: {len(joint_data)}"
            )
            return
        msg.position = joint_data
        self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing: "{msg}"')

def main(args=None):
    rclpy.init(args=args)
    gello_node = GelloNode()
    port = gello_node.u2d2_prot
    try:
        rclpy.spin(gello_node)
    except KeyboardInterrupt:
        pass
    finally:
        gello_node.agent.closeport()
        gello_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # time.sleep(0.5)
        # gripper = Gripper(port)
        # gripper.open_port()
        # gripper.set_torque(8, False)
        # gripper.portHandler.closePort()

if __name__ == '__main__':
    main()