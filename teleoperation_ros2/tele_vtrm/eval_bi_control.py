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
        self.gello_r = DynamixelDriver(port=cfg.u2d2_port_r)
        self.gello_l = DynamixelDriver(port=cfg.u2d2_port_l)
        self.xarm_r = XArmAPI(port=cfg.xarm_ip_r, is_radian=True)
        self.xarm_l = XArmAPI(port=cfg.xarm_ip_l, is_radian=True)
        self.set_init_pose(self.xarm_l, self.gello_l)
        self.get_logger().info("Init left arm")
        self.set_init_pose(self.xarm_r, self.gello_r)
        self.get_logger().info("Init right arm")
        self.signal = 0
        self.start_ep = False
        self.reset_ep = False
        self.goal_pos_r: Optional[np.ndarray] = None
        self.goal_pos_l: Optional[np.ndarray] = None
        self.gripper_r: Optional[int] = None
        self.gripper_l: Optional[int] = None
        self.min_norm = 0.05
        self.base_velocity_limit = 0.157
        # self.max_velocity_limit = 0.314
        self.max_velocity_limit = 0.471

    def timer_callback_1(self):
        joint_states_r, robot_joint_r = self.get_robot_joint(self.xarm_r, self.gello_r)
        joint_states_l, robot_joint_l = self.get_robot_joint(self.xarm_l, self.gello_l)
        robot_joint = np.append(robot_joint_r, robot_joint_l)
        robot_joint = robot_joint.astype(np.float32)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = robot_joint.tolist()
        self.obs.publish(msg)
        if self.start_ep:
            if self.goal_pos_r is not None and self.gripper_r is not None and self.goal_pos_l is not None and self.gripper_l is not None:
                # self.move_robot(self.xarm_r, self.gello_r, self.goal_pos_r, self.gripper_r, joint_states_r)
                # self.move_robot(self.xarm_l, self.gello_l, self.goal_pos_l, self.gripper_l, joint_states_l)
                return
            else:
                self.get_logger().warn("Goal position or gripper state is None, skipping movement")

        if self.reset_ep:
            self.set_init_pose(self.xarm_r, self.gello_r)
            self.get_logger().info("Reset right arm")
            self.set_init_pose(self.xarm_l, self.gello_l)
            self.get_logger().info("Reset left arm")
            self.goal_pos_r = None
            self.gripper_r = None
            self.goal_pos_l = None
            self.gripper_l = None
            self.reset_ep = False

    def timer_callback_2(self):
        msg = Int32()
        msg.data = self.signal
        self.control.publish(msg)
        # self.get_logger().info(f'Publishing: "{msg}"')
        self.signal = 0

    def act_callback(self, msg):
        self.goal_pos_r = np.array(msg.position[:6])
        self.gripper_r = np.int32(msg.position[6])
        self.goal_pos_l = np.array(msg.position[7:13])
        self.gripper_l = np.int32(msg.position[13])
        self.get_logger().info(f"{self.gripper_l}")

    def set_init_pose(self, xarm: XArmAPI, gello: DynamixelDriver):
        xarm.clean_error()
        xarm.clean_warn()
        xarm.motion_enable(True)
        time.sleep(0.2)
        xarm.set_mode(0)
        time.sleep(0.2)
        xarm.set_collision_sensitivity(0)
        time.sleep(0.2)
        xarm.set_state(state=0)
        time.sleep(0.2)
        xarm.set_servo_angle(angle=[0, 0, -1.57, 0, 1.57, 0], speed=0.8, mvacc=10, is_radian=True)
        time.sleep(3)
        xarm.set_mode(1)
        time.sleep(0.2)
        xarm.set_state(state=0)
        gello.write_position(8)
    
    def get_robot_joint(self, xarm: XArmAPI, gello: DynamixelDriver):
        code, joint_states = xarm.get_servo_angle(is_radian=True)
        robot_joint = list(joint_states)[:6]
        gripper_joint = gello.read_position(8)
        robot_joint = np.append(robot_joint, gripper_joint)
        return joint_states, robot_joint
    
    def move_robot(self, xarm: XArmAPI, gello: DynamixelDriver, goal_pos: np.ndarray, gripper: int, now_pos):
        now_pos = np.array(now_pos[:6])
        delta = goal_pos - now_pos
        norm = np.linalg.norm(delta)
        max_delta = np.max(np.abs(delta))
        if norm < self.min_norm and max_delta < self.min_norm / 2:
            scaled_velocity_limit = self.base_velocity_limit
        else:
            scaled_velocity_limit = min((5 * norm / self.min_norm - 4) * self.base_velocity_limit, self.max_velocity_limit)
        motion_scale = norm / (scaled_velocity_limit * self.timer_period_1)
        final_goal = now_pos + delta / motion_scale
        xarm.set_servo_angle_j(angles=final_goal, is_radian=True)
        gello.write_position(8, gripper)

    def cleanup(self):
        self.xarm_r.disconnect()
        self.xarm_l.disconnect()
        self.gello_r.close()
        self.gello_l.close()
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

@hydra.main(version_base=None, config_path="../config", config_name="eval_bi_cfg")
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
        # gripper_r = Gripper(port=cfg.u2d2_port_r)
        # gripper_r.open_port()
        # gripper_r.set_torque(8, False)
        # gripper_r.portHandler.closePort()
        # gripper_l = Gripper(port=cfg.u2d2_port_l)
        # gripper_l.open_port()
        # gripper_l.set_torque(8, False)
        # gripper_l.portHandler.closePort()
