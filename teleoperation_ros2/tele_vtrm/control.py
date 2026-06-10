import subprocess
import psutil
import signal
import time
import os
import threading
import shutil
import json
from tqdm import tqdm
from pynput import keyboard
import hydra
from pathlib import Path
import numpy as np
from cv_bridge import CvBridge
import concurrent.futures
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState, Image
from tele_vtrm.video_utils import encode_video_frames

def on_press(key, control_node):
    try:
        if key.char == '1':
            control_node.signal = 1
        elif key.char == '2':
            control_node.signal = 2
        elif key.char == '3':
            control_node.signal = 3
        elif key.char == '4':
            control_node.signal = 4
    except AttributeError:
        pass

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        self.publisher_ = self.create_publisher(Int32, 'control', 1)
        timer_period_1 = 0.1
        self.timer_1 = self.create_timer(timer_period_1, self.timer_callback_1)
        self.signal = 0

    def timer_callback_1(self):
        msg = Int32()
        msg.data = self.signal
        self.publisher_.publish(msg)
        self.signal = 0

    def cleanup(self):
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def main(args=None):

    rclpy.init(args=args)

    control_node = ControlNode()
    listener = keyboard.Listener(on_press=lambda key: on_press(key, control_node))
    listener.start()
    try:
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        pass
    finally:
        control_node.cleanup()
        listener.stop()

if __name__ == '__main__':
    main()