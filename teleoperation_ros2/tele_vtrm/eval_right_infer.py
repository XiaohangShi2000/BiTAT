# import subprocess
# import psutil
# import signal
# import os
# import threading
# import shutil
# from tqdm import tqdm
# from pathlib import Path
# import concurrent.futures
# from PIL import Image as PILImage
import json
import time
# from pynput import keyboard
import hydra
import numpy as np
from cv_bridge import CvBridge
import sys
# from typing import Optional
# from dynamixel_sdk.packet_handler import PacketHandler
# from dynamixel_sdk.port_handler import PortHandler
# from dynamixel_sdk.robotis_def import COMM_SUCCESS

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState, Image
from tele_vtrm.video_utils import encode_video_frames
# from tele_vtrm.gello_utils.gello_agent import GelloAgent
# from tele_vtrm.gello_utils.dynamixel_driver import DynamixelDriver
# from xarm.wrapper import XArmAPI
# from tele_vtrm.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
sys.path.append("/home/xiaohang_shi/Project/lerobot")
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType, PolicyFeature
sys.path.append('/home/xiaohang_shi/miniconda3/envs/lerobot/lib/python3.10/site-packages')
import torch

def config_to_policy_feature(config):
    features = [config["input_features"], config["output_features"]]
    for i in features:
        for key, ft in i.items():
            shape = tuple(ft["shape"])
            if "images" in key:
                type = FeatureType.VISUAL
                if len(shape) != 3:
                    raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")

            elif "state" in key:
                type = FeatureType.STATE
            elif key == "action":
                type = FeatureType.ACTION
            else:
                continue

            i[key] = PolicyFeature(
                type=type,
                shape=shape,
            )

    return config

class InferNode(Node):

    def __init__(self, cfg):
        super().__init__('infer_node')

        cb_group_1 = ReentrantCallbackGroup()
        cb_group_2 = MutuallyExclusiveCallbackGroup()
        self.timer_period_1 = 1 / cfg.fps
        # self.timer_period_2 = 0.01
        # self.timer_1 = self.create_timer(timer_period_1, self.timer_callback_1)
        # self.timer_period_2 = 0.05
        # self.timer_2 = self.create_timer(self.timer_period_2, self.timer_callback_2)
        self.act = self.create_publisher(JointState, 'act', 1, callback_group=cb_group_2)
        self.timer_1 = self.create_timer(self.timer_period_1, self.timer_callback_1, callback_group=cb_group_2)
        # self.timer_2 = self.create_timer(self.timer_period_2, self.timer_callback_2, callback_group=cb_group_2)
        # self.timer_3 = self.create_timer(0.04, self.timer_callback_3, callback_group=cb_group_2)
        self.observation = self.create_subscription(JointState, 'obs', self.observation_callback, 1, callback_group=cb_group_1)
        self.observation
        self.control = self.create_subscription(Int32, 'control', self.control_callback, 1)
        self.control
        self.head_cam_sub = self.create_subscription(Image, '/head/d435i/color/image_raw', self.head_cam_callback, 1, callback_group=cb_group_1)
        self.head_cam_sub
        self.tac_01_sub = self.create_subscription(Image, '/sensor_01/DotView_pp_gray_img', self.tac_01_callback, 1, callback_group=cb_group_1)
        self.tac_01_sub
        self.tac_02_sub = self.create_subscription(Image, '/sensor_02/DotView_pp_gray_img', self.tac_02_callback, 1, callback_group=cb_group_1)
        self.tac_02_sub
        # self.signal = 0
        self.start_ep = False
        # self.end_ep = False
        # self.rerec_ep = False
        # self.save_ep = False
        self.reset_ep = False
        self.obs_dict = {}
        self.bridge = CvBridge()
        # self.thread_pool = thread_pool
        # self.min_norm = 0.1
        # self.base_velocity_limit = 0.314
        # self.max_velocity_limit = 1.57
        # self.xarm_joint: list | tuple | None = None
        # self.gripper_joint: Optional[int] = None
        # # self.np_action: Optional[np.ndarray] = None
        # self.cam_img: Optional[torch.Tensor] = None
        # self.tac_01_img: Optional[torch.Tensor] = None
        # self.tac_02_img: Optional[torch.Tensor] = None
        # self.delta: Optional[np.ndarray] = None
        # self.gripper_pos: Optional[int] = None
        # self.callback1_times = []
        # self.callback2_times = []
        # self.callback1_last_time = None
        # self.callback2_last_time = None
        # self.measure_frequency = True 
        # self.create_timer(3.0, self.print_callback_frequencies, callback_group=cb_group_1)
        self.cfg = cfg
        self.device = cfg.device
        pretrained_policy_path = cfg.pretrained_policy_path
        # with open(pretrained_policy_path+"/config.json", "r", encoding="utf-8") as f:
        #     content = json.load(f)
        #     del content["type"]
        #     del content["normalization_mapping"]
        # config_to_policy_feature(content)
        # config = DiffusionConfig(**content)
        # self.policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, config=config, map_location=self.device)
        self.policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, map_location=self.device)
        torch._dynamo.reset()
        torch.set_float32_matmul_precision("high")
        self.policy.diffusion.unet = torch.compile(self.policy.diffusion.unet, mode = "reduce-overhead")
        # self.policy = torch.compile(self.policy, mode = "max-autotune")
        self.get_logger().info("Policy loaded")

    def timer_callback_1(self):
        # current_time = time.time()
        # if self.callback1_last_time is not None and self.measure_frequency:
        #     interval = current_time - self.callback1_last_time
        #     self.callback1_times.append(interval)
        #     # 保持列表在合理大小
        #     if len(self.callback1_times) > 1 / self.timer_period_1:
        #         self.callback1_times.pop(0)
        # self.callback1_last_time = current_time

        if self.start_ep:
            if self.obs_dict is not None:
                if len(self.obs_dict) == self.cfg.obs_dim:
                    # begin = time.time()
                    with torch.inference_mode():
                        action = self.policy.select_action(self.obs_dict)
                    # end = time.time()
                    # self.get_logger().info(f"Time: {end-begin}")
                    np_action = action.squeeze(0).to("cpu").numpy()
                    msg = JointState()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.position = np_action.tolist()
                    self.act.publish(msg)
            # self.get_logger().info(f"Action: {np_action}")

        if self.reset_ep:
            self.policy.reset()
            self.reset_ep = False

    # def timer_callback_2(self):
    #     current_time = time.time()
    #     if self.callback2_last_time is not None and self.measure_frequency:
    #         interval = current_time - self.callback2_last_time
    #         self.callback2_times.append(interval)
    #         # 保持列表在合理大小
    #         if len(self.callback2_times) > 1 / self.timer_period_2:
    #             self.callback2_times.pop(0)
    #     self.callback2_last_time = current_time

    # def print_callback_frequencies(self):
    #     """定期打印回调函数的实际频率"""
    #     if not self.measure_frequency:
    #         return
            
    #     if self.callback1_times:
    #         avg_interval1 = sum(self.callback1_times) / len(self.callback1_times)
    #         freq1 = 1.0 / avg_interval1 if avg_interval1 > 0 else 0
    #         self.get_logger().info(f"Timer 1 实际频率: {freq1:.2f} Hz (目标: {1.0/self.timer_period_1:.2f} Hz), 间隔: {avg_interval1*1000:.2f} ms")
        
    #     if self.callback2_times:
    #         avg_interval2 = sum(self.callback2_times) / len(self.callback2_times)
    #         freq2 = 1.0 / avg_interval2 if avg_interval2 > 0 else 0
    #         self.get_logger().info(f"Timer 2 实际频率: {freq2:.2f} Hz (目标: {1.0/self.timer_period_2:.2f} Hz), 间隔: {avg_interval2*1000:.2f} ms")
    
    def observation_callback(self, msg):
        robot_state= np.array(msg.position)
        self.obs_dict["observation.state"] = self.np_to_torch(robot_state)

    def control_callback(self, msg):
        if msg.data == 1:
            self.start_ep = False
            self.reset_ep = True
            self.get_logger().info("Episode reset")
        elif msg.data == 2:
            self.start_ep = True
            self.get_logger().info("Episode started")
        elif msg.data == 3:
            self.start_ep = False
            self.get_logger().info("Episode terminated")
        else:
            return

    def head_cam_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.obs_dict["observation.images.head_cam"] = self.np_to_torch(img)

    def tac_01_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.obs_dict["observation.images.tac_01"] = self.np_to_torch(img)

    def tac_02_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.obs_dict["observation.images.tac_02"] = self.np_to_torch(img)

    def np_to_torch(self, array):
        array = torch.from_numpy(array)
        if array.ndim == 3:
            array = array.to(torch.float32) / 255
            array = array.permute(2, 0, 1)
        elif array.ndim == 2:
            array = array.to(torch.float32) / 255
            array = array.unsqueeze(0)
        else:
            array = array.to(torch.float32)
        array = array.unsqueeze(0)
        array = array.to(self.device, non_blocking=True)
        return array

    def cleanup(self):
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

@hydra.main(version_base=None, config_path="../config", config_name="eval_cfg")
def main(cfg):

    rclpy.init()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # device = torch.device(cfg.device)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_image_writers) as executor:
    #     control_node = ControlNode(cfg, data_dir, meta_dir, video_dir)
    # with torch.inference_mode(), torch.autocast(device_type=device.type):
    infer_node = InferNode(cfg)
    # executor = MultiThreadedExecutor(num_threads=5)
    executor = MultiThreadedExecutor()
    executor.add_node(infer_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        executor.remove_node(infer_node)
        infer_node.cleanup()

if __name__ == '__main__':
    main()