#!/usr/bin/env python3

import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32

from xarm.wrapper import XArmAPI

class BridgeNode(Node):

    def __init__(self):
        super().__init__('bridge_node')

        self.declare_parameter('robot_ip', '10.18.18.101')
        self.declare_parameter('ns', 'right')
        robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        ns = self.get_parameter('ns').get_parameter_value().string_value

        self.xarm_joint = np.zeros(6)
        self.gello_joint = np.ones(6)

        self.xarm = self.create_subscription(
            JointState,
            '/'+ns+'/xarm6_joint',
            self.xarm_callback,
            1
        )
        self.xarm  

        self.gello = self.create_subscription(
            JointState,
            '/'+ns+'/gello_joint',
            self.gello_callback,
            1
        )
        self.gello

        # self.bridge_publisher = self.create_publisher(
        #     Float64MultiArray,
        #     'joint_delta',
        #     1
        # )

        self.control = self.create_subscription(
            Int32,
            '/control',
            self.control_callback,
            1
        )
        self.control
        self.enable = False
        self.max_delta = 0.05
        self.robot = XArmAPI(port=robot_ip, is_radian=True)
        self.enable_robot()
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def xarm_callback(self, msg):
        self.xarm_joint = np.array(msg.position)

    def gello_callback(self, msg):
        self.gello_joint = np.array(msg.position)

    def control_callback(self, msg):
        if msg.data == 2:
            self.enable = True
        elif msg.data == 3:
            self.enable = False
        else:
            return

    def timer_callback(self):
        if self.enable:
            if (self.gello_joint - self.xarm_joint > 0.35).any():
                self.get_logger().info("Action is too big")
                # print which joints are too big
                joint_index = np.where(self.gello_joint - self.xarm_joint > 0.35)[0]
                for j in joint_index:
                    self.get_logger().info(
                        f"Joint [{j}], leader: {self.gello_joint[j]}, follower: {self.xarm_joint[j]}, diff: {self.gello_joint[j] - self.xarm_joint[j]}"
                    )
                return
            joint_delta = self.gello_joint - self.xarm_joint
            norm = np.linalg.norm(joint_delta)
            if norm > self.max_delta:
                delta = joint_delta / norm * self.max_delta
            else:
                delta = joint_delta
            final_delta = self.xarm_joint+delta
            self.robot.set_servo_angle_j(angles=final_delta, wait=False, is_radian=True)
        else:
            return

    def enable_robot(self):
        self.robot.clean_error()
        self.robot.clean_warn()
        self.robot.motion_enable(True)
        time.sleep(0.5)
        self.robot.set_mode(1)
        time.sleep(0.5)
        self.robot.set_collision_sensitivity(0)
        time.sleep(0.5)
        self.robot.set_state(state=0)
        time.sleep(0.5)

    def cleanup(self):
        self.robot.disconnect()
        print('bridge node disconnected')
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    bridge_node = BridgeNode()
    # time.sleep(5.0)
    try:
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        bridge_node.cleanup()

if __name__ == '__main__':
    main()