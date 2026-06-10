#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from leap_hand_utils.xarm_robot import XArmRobot

class BridgeNode(Node):

    def __init__(self):
        super().__init__('bridge_node')

        self.declare_parameter('robot_ip', '10.18.18.101')
        self.declare_parameter('ns', 'right')
        robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self.ns = self.get_parameter('ns').get_parameter_value().string_value

        self.xarm_joint = np.zeros(6)
        self.gello_joint = np.ones(6)
        # Subscribe to the /robot_command topic to receive joint commands
        self.xarm = self.create_subscription(
            JointState,
            '/'+self.ns+'/xarm6_joint',
            self.xarm_callback,
            1
        )
        self.xarm  # prevent unused variable warning            

        self.gello = self.create_subscription(
            JointState,
            '/'+self.ns+'/gello_joint',
            self.gello_callback,
            1
        )
        self.gello

        # self.bridge_publisher = self.create_publisher(
        #     Float64MultiArray,
        #     'joint_delta',
        #     1
        # )

        self.robot = XArmRobot(ip=robot_ip)
        timer_period = 0.02
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def xarm_callback(self, msg):
        self.xarm_joint = np.array(msg.position)

    def gello_callback(self, msg):
        self.gello_joint = np.array(msg.position)

    def timer_callback(self):
        if (self.gello_joint - self.xarm_joint > 0.5).any():
            self.get_logger().info("Action is too big")
            # print which joints are too big
            joint_index = np.where(self.gello_joint - self.xarm_joint > 0.5)[0]
            for j in joint_index:
                self.get_logger().info(
                    f"Joint [{j}], leader: {self.gello_joint[j]}, follower: {self.xarm_joint[j]}, diff: {self.gello_joint[j] - self.xarm_joint[j]}"
                )
            return
        self.robot.command_joint_state(self.gello_joint)


def main(args=None):
    rclpy.init(args=args)

    bridge_node = BridgeNode()

    rclpy.spin(bridge_node)

    bridge_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()