from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import COMM_SUCCESS
import hydra
import time
import numpy as np
from typing import Optional
from pynput import keyboard
import concurrent.futures

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from tele_vtrm.gello_utils.dynamixel_driver import DynamixelDriver
from xarm.wrapper import XArmAPI

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
        
def on_press(key, control_node):
    try:
        if key.char == '1':
            control_node.signal = 1
            control_node.start_ep = False
            control_node.reset_ep = True
        elif key.char == '2':
            control_node.signal = 2
            control_node.start_ep = True
        elif key.char == '3':
            control_node.signal = 3
            control_node.start_ep = False
            # control_node.end_ep = True
        elif key.char == '4':
            control_node.signal = 4
            # control_node.rerec_ep = True
            # control_node.save_ep = True
    except AttributeError:
        pass
        
class ControlNode(Node):
    def __init__(self, cfg):
        super().__init__('control_node')

        # cb_group_1 = ReentrantCallbackGroup()
        cb_group_1 = MutuallyExclusiveCallbackGroup()
        cb_group_2 = MutuallyExclusiveCallbackGroup()
        self.timer_period_1 = 0.01
        timer_period_2 = 0.1
        self.timer_1 = self.create_timer(self.timer_period_1, self.timer_callback_1, callback_group = cb_group_1)
        self.obs = self.create_publisher(JointState, 'obs', 1, callback_group = cb_group_1)
        self.timer_2 = self.create_timer(timer_period_2, self.timer_callback_2, callback_group = cb_group_2)
        self.control = self.create_publisher(Int32, 'control', 1, callback_group = cb_group_2)
        self.act = self.create_subscription(JointState, 'act', self.act_callback, 1)
        self.act
        self.gello = DynamixelDriver(port=cfg.u2d2_port)
        self.xarm = XArmAPI(port=cfg.xarm_ip, is_radian=True)
        self.set_init_pose()
        self.signal = 0
        self.start_ep = False
        self.reset_ep = False
        self.goal_pos: Optional[np.ndarray] = None
        self.gripper: Optional[int] = None
        self.min_norm = 0.05
        self.base_velocity_limit = 0.157
        self.max_velocity_limit = 0.471

    def timer_callback_1(self):
        code, joint_states = self.xarm.get_servo_angle(is_radian=True)
        robot_joint = list(joint_states)[:6]
        gripper_joint = self.gello.read_position(8)
        robot_joint = np.append(robot_joint, gripper_joint)
        robot_joint = robot_joint.astype(np.float32)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = robot_joint.tolist()
        self.obs.publish(msg)
        if self.start_ep:
            if self.goal_pos is not None and self.gripper is not None:
                now_pos = np.array(joint_states[:6])
                delta = self.goal_pos - now_pos
                norm = np.linalg.norm(delta)
                max_delta = np.max(np.abs(delta))
                if norm < self.min_norm and max_delta < self.min_norm / 2:
                    scaled_velocity_limit = self.base_velocity_limit
                else:
                    scaled_velocity_limit = min((5 * norm / self.min_norm - 4) * self.base_velocity_limit, self.max_velocity_limit)
                motion_scale = norm / (scaled_velocity_limit * self.timer_period_1)
                final_goal = now_pos + delta / motion_scale
                self.xarm.set_servo_angle_j(angles=final_goal, is_radian=True)
                self.gello.write_position(8, self.gripper)

        if self.reset_ep:
            self.set_init_pose()
            self.goal_pos = None
            self.gripper = None
            self.reset_ep = False

    def timer_callback_2(self):
        msg = Int32()
        msg.data = self.signal
        self.control.publish(msg)
        # self.get_logger().info(f'Publishing: "{msg}"')
        self.signal = 0

    def act_callback(self, msg):
        self.goal_pos = np.array(msg.position[:6])
        self.gripper = int(msg.position[6])

    def set_init_pose(self):
        self.xarm.clean_error()
        self.xarm.clean_warn()
        self.xarm.motion_enable(True)
        time.sleep(0.2)
        self.xarm.set_mode(0)
        time.sleep(0.2)
        self.xarm.set_collision_sensitivity(0)
        time.sleep(0.2)
        self.xarm.set_state(state=0)
        time.sleep(0.2)
        self.xarm.set_servo_angle(angle=[0, 0, -1.57, 0, 1.57, 0], speed=0.8, mvacc=10, is_radian=True)
        time.sleep(3)
        self.xarm.set_mode(1)
        time.sleep(0.2)
        self.xarm.set_state(state=0)
        self.gello.write_position(8)
        self.get_logger().info("Init pose set.")

    def cleanup(self):
        self.xarm.disconnect()
        self.gello.close()
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

@hydra.main(version_base=None, config_path="../config", config_name="eval_cfg")
def main(cfg):
    rclpy.init()
    control_node = ControlNode(cfg)
    listener = keyboard.Listener(on_press=lambda key: on_press(key, control_node))
    listener.start()
    executor = MultiThreadedExecutor()
    executor.add_node(control_node)
    try:
        # rclpy.spin(control_node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        executor.shutdown()
        executor.remove_node(control_node)
        control_node.cleanup()
        # time.sleep(0.5)
        # gripper = Gripper(port=cfg.u2d2_port)
        # gripper.open_port()
        # gripper.set_torque(8, False)
        # gripper.portHandler.closePort()
