#!/usr/bin/env python3

import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32

from xarm.wrapper import XArmAPI

class XArm6Node(Node):
    def __init__(self):
        super().__init__('xarm6_node')

        self.declare_parameter('robot_ip','10.18.18.101')
        robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self.robot = XArmAPI(port=robot_ip, is_radian=True)
        self.set_init_pose()
        # self.publisher_ = self.create_publisher(JointState, 'xarm6_joint', 1)
        # self.observation = self.create_publisher(JointState, 'obs', 1)
        # self.action = self.create_publisher(JointState, 'act', 1)
        self.obs_act = self.create_publisher(JointState, 'obs_act', 1)
        self.timer_period = 0.01
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.control = self.create_subscription(Int32, '/control', self.control_callback, 1)
        self.control
        self.gello = self.create_subscription(JointState, 'gello_joint', self.gello_callback, 1)
        self.gello
        self.enable = False
        self.xarm_joint = np.zeros(6)
        self.gello_joint = np.zeros(6)
        self.min_norm = 0.0785
        # 最大角速度为3.14rad/s
        self.base_velocity_limit = 0.314
        self.max_velocity_limit = 1.57

    def timer_callback(self):
        code, joint_states = self.robot.get_servo_angle(is_radian=True)
        self.xarm_joint = np.array(joint_states[:6])
        if self.enable:
            goal_pos,gripper = self.move_xarm()
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            obs = np.append(self.xarm_joint, gripper[0])
            act = np.append(goal_pos, gripper[1])
            msg.position = np.append(obs, act).tolist()
            self.obs_act.publish(msg)
            # obs = JointState()
            # act = JointState()
            # obs.header.stamp = self.get_clock().now().to_msg()
            # act.header.stamp = self.get_clock().now().to_msg()
            # obs.position = np.append(self.xarm_joint, gripper[0]).tolist()
            # act.position = np.append(goal_pos, gripper[1]).tolist()
            # self.observation.publish(obs)
            # self.action.publish(act)
        else:
            return
        # msg = JointState()
        # msg.header.stamp = self.get_clock().now().to_msg()
        # msg.name = [f'joint{i+1}' for i in range(6)]
        # msg.position = joint_states[:6]
        # self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing: "{msg}"')

    def control_callback(self, msg):
        if msg.data == 1:
            self.enable = False
            self.set_init_pose()
        elif msg.data == 2:
            self.enable = True
        elif msg.data == 3:
            self.enable = False
        else:
            return
        
    def gello_callback(self, msg):
        self._gello_joint = np.array(msg.position)

    def move_xarm(self):
        self.gello_joint = self._gello_joint[:6]
        gripper = self._gello_joint[6:]
        joint_delta = self.gello_joint - self.xarm_joint
        max_delta = np.max(np.abs(joint_delta))
        if max_delta > 0.52:
            self.get_logger().info("Action is too big")
            # print which joints are too big
            joint_index = np.where(joint_delta > 0.52)[0]
            for j in joint_index:
                self.get_logger().warn(
                    f"Joint [{j}], leader: {self.gello_joint[j]}, follower: {self.xarm_joint[j]}, diff: {self.gello_joint[j] - self.xarm_joint[j]}"
                )
            return
        norm = np.linalg.norm(joint_delta)
        if norm < self.min_norm and max_delta < self.min_norm / 2:
            scaled_velocity_limit = self.base_velocity_limit
        else:
            scaled_velocity_limit = min((5 * norm / self.min_norm - 4) * self.base_velocity_limit, self.max_velocity_limit)
        motion_scale = max_delta / (scaled_velocity_limit * self.timer_period)
        final_goal = self.xarm_joint + joint_delta / motion_scale
        self.robot.set_servo_angle_j(angles=final_goal, is_radian=True)
        return final_goal, gripper

    def set_init_pose(self):
        self.robot.clean_error()
        self.robot.clean_warn()
        self.robot.motion_enable(True)
        time.sleep(0.2)
        self.robot.set_mode(0)
        time.sleep(0.2)
        self.robot.set_collision_sensitivity(0)
        time.sleep(0.2)
        self.robot.set_state(state=0)
        time.sleep(0.2)
        self.robot.set_servo_angle(angle=[0, 0, -1.57, 0, 1.57, 0], speed=0.8, mvacc=10, is_radian=True)
        time.sleep(3)
        self.robot.set_mode(1)
        time.sleep(0.2)
        self.robot.set_state(state=0)
        self.get_logger().info("Robot initialized")

    def cleanup(self):
        self.robot.disconnect()
        self.get_logger().info("Robot disconnected")
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    xarm6_node = XArm6Node()
    try:
        rclpy.spin(xarm6_node)
    except KeyboardInterrupt:
        pass
    finally:
        xarm6_node.cleanup()

if __name__ == '__main__':
    main()