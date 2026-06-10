import subprocess
import psutil
import signal
import os
import threading
import shutil
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from PIL import Image as PILImage
import json
import time
from pynput import keyboard
import hydra
import numpy as np
from cv_bridge import CvBridge
import sys
from typing import Optional
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import COMM_SUCCESS

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState, Image
from tele_vtrm.video_utils import encode_video_frames
# from tele_vtrm.gello_utils.gello_agent import GelloAgent
from tele_vtrm.gello_utils.dynamixel_driver import DynamixelDriver
from xarm.wrapper import XArmAPI
# from tele_vtrm.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
sys.path.append("/home/xiaohang_shi/Project/lerobot")
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType, PolicyFeature
sys.path.append('/home/xiaohang_shi/miniconda3/envs/lerobot/lib/python3.10/site-packages')
import torch

ADDR_TORQUE_ENABLE = 64

class Gripper:
    def __init__(self,
                 port: Optional[str] = None,
                 baudrate: Optional[int] = 57600,
                 ):
        self.port = port
        self.baudrate = baudrate

    def open_port(self):
        self.portHandler = PortHandler(self.port)
        self._packetHandler = PacketHandler(2.0)
        if not self.portHandler.openPort():
            raise RuntimeError(f"Failed to open port: {self.port}")
        if not self.portHandler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate: {self.baudrate}")

    def set_torque(self, id: int, enable: bool):
        dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
            self.portHandler, id, ADDR_TORQUE_ENABLE, 1 if enable else 0
        )
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(f"Failed to set torque: {id}, {enable}")

def on_press(key, control_node):
    try:
        if key.char == '1':
            # control_node.signal = 1
            control_node.start_ep = False
            control_node.reset_ep = True
        elif key.char == '2':
            # control_node.signal = 2
            control_node.start_ep = True
        elif key.char == '3':
            # control_node.signal = 3
            control_node.start_ep = False
            # control_node.end_ep = True
        elif key.char == '4':
            # control_node.signal = 4
            # control_node.rerec_ep = True
            control_node.save_ep = True
    except AttributeError:
        pass

# def on_release(key,control_node):
#     try:
#         if key.char =='2':
#             control_node.start_ep = True
#     except AttributeError:
#         pass

# def save_image(img_array, key, frame_index, episode_index, videos_dir):
#     # img = Image.fromarray(img_tensor.numpy()) # if img_tensor is pytorch.Tensor
#     img = PILImage.fromarray(img_array)
#     path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
#     path.parent.mkdir(parents=True, exist_ok=True)
#     img.save(str(path), quality=100)

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

class ControlNode(Node):
    # def __init__(self, cfg, data_dir, meta_dir, video_dir):
    def __init__(self, cfg):
        super().__init__('control_node')

        cb_group_1 = ReentrantCallbackGroup()
        cb_group_2 = MutuallyExclusiveCallbackGroup()
        # self.publisher_ = self.create_publisher(Int32, 'control', 1)
        self.timer_period_1 = 0.1
        self.timer_period_2 = 0.01
        # self.timer_1 = self.create_timer(timer_period_1, self.timer_callback_1)
        # self.timer_period_2 = 0.05
        # self.timer_2 = self.create_timer(self.timer_period_2, self.timer_callback_2)
        self.timer_1 = self.create_timer(self.timer_period_1, self.timer_callback_1, callback_group=cb_group_2)
        self.timer_2 = self.create_timer(self.timer_period_2, self.timer_callback_2, callback_group=cb_group_2)
        self.timer_3 = self.create_timer(0.04, self.timer_callback_3, callback_group=cb_group_2)
        # self.observation = self.create_subscription(JointState, '/right/obs', self.observation_callback, 1)
        # self.observation
        # self.action = self.create_subscription(JointState, '/right/act', self.action_callback, 1)
        # self.action
        # self.obs_act = self.create_subscription(JointState, '/right/obs_act', self.obs_act_callback, 1)
        # self.obs_act
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
        self.save_ep = False
        self.reset_ep = False
        # self.obs_dict, self.act_dict, self.ep_dict = {},{},{}
        self.bridge = CvBridge()
        # self.thread_pool = thread_pool
        # self.data_dir = data_dir
        # self.meta_dir = meta_dir
        # self.video_dir = video_dir
        # self.episode_index = 0
        # self.frame_index = 0
        # self.total_frames = 0
        # self.image_keys, self.low_dim_keys, self.futures = [],[],[]
        # self.enable_record = False
        self.min_norm = 0.1
        self.base_velocity_limit = 0.314
        self.max_velocity_limit = 1.57
        self.xarm_joint: list | tuple | None = None
        self.gripper_joint: Optional[int] = None
        self.np_action: Optional[np.ndarray] = None
        self.cam_img: Optional[torch.Tensor] = None
        self.tac_01_img: Optional[torch.Tensor] = None
        self.tac_02_img: Optional[torch.Tensor] = None
        # self.delta: Optional[np.ndarray] = None
        # self.gripper_pos: Optional[int] = None
        self.callback1_times = []
        self.callback2_times = []
        self.callback1_last_time = None
        self.callback2_last_time = None
        self.measure_frequency = True 
        self.create_timer(3.0, self.print_callback_frequencies, callback_group=cb_group_1)

        self.device = cfg.device
        pretrained_policy_path = cfg.pretrained_policy_path
        self.xarm = XArmAPI(port=cfg.xarm_ip, is_radian=True)
        self.gello = DynamixelDriver(port=cfg.u2d2_port)
        self.set_init_pose()
        with open(pretrained_policy_path+"/config.json", "r", encoding="utf-8") as f:
            content = json.load(f)
            del content["type"]
            del content["normalization_mapping"]
        config_to_policy_feature(content)
        config = DiffusionConfig(**content)
        self.policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, config=config, map_location=self.device)

    def timer_callback_1(self):
        current_time = time.time()
        if self.callback1_last_time is not None and self.measure_frequency:
            interval = current_time - self.callback1_last_time
            self.callback1_times.append(interval)
            # 保持列表在合理大小
            if len(self.callback1_times) > 1 / self.timer_period_1:
                self.callback1_times.pop(0)
        self.callback1_last_time = current_time
        # code, joint_states = self.xarm.get_servo_angle(is_radian=True)
        # # joint_states = list(joint_states)
        # xarm_joint = list(joint_states)[:6]
        # gripper_joint = self.gello.read_position(8)
        
        # if self.xarm_joint is not None and self.cam_img is not None and self.tac_01_img is not None and self.tac_02_img is not None:
        #     if len(self.xarm_joint) == 7:
        #         obs_state = np.array(self.xarm_joint)
        #         obs_state = obs_state.astype(np.float32)
        #         state = self.np_to_torch(obs_state)
        #         observation = {
        #             "observation.state": state,
        #             "observation.images.head_cam": self.cam_img,
        #             "observation.images.tac_01": self.tac_01_img,
        #             "observation.images.tac_02": self.tac_02_img,
        #         }

        if self.xarm_joint is not None and self.gripper_joint is not None and self.cam_img is not None and self.tac_01_img is not None and self.tac_02_img is not None:
            joint_states = list(self.xarm_joint)[:6]
            obs_state = np.append(joint_states, self.gripper_joint)
            obs_state = obs_state.astype(np.float32)
            state = self.np_to_torch(obs_state)
            observation = {
                "observation.state": state,
                "observation.images.head_cam": self.cam_img,
                "observation.images.tac_01": self.tac_01_img,
                "observation.images.tac_02": self.tac_02_img,
            }
        if self.start_ep:
            with torch.inference_mode():
                action = self.policy.select_action(observation)

            self.np_action = action.squeeze(0).to("cpu").numpy()
            # self.get_logger().info(f"Action: {np_action}")

        if self.reset_ep:
            self.set_init_pose()
            self.reset_ep = False
            self.np_action = None
            self.policy.reset()

    # def timer_callback_2(self):
    #     current_time = time.time()
    #     if self.callback2_last_time is not None and self.measure_frequency:
    #         interval = current_time - self.callback2_last_time
    #         self.callback2_times.append(interval)
    #         # 保持列表在合理大小
    #         if len(self.callback2_times) > 1 / self.timer_period_2:
    #             self.callback2_times.pop(0)
    #     self.callback2_last_time = current_time

    #     code, joint_states = self.xarm.get_servo_angle(is_radian=True)
    #     now_pos = np.array(joint_states[:6])
    #     self.xarm_joint = list(joint_states)[:6]
    #     # gripper_joint = self.gello.read_position(8)
    #     if self.gripper_joint is not None:
    #         self.xarm_joint.append(self.gripper_joint)
    #     if self.start_ep:
    #         if self.np_action is not None:
    #             goal_pos = self.np_action[:6]
    #             # gripper_pos = self.np_action[6].astype(int)
    #             # self.gello.write_position(8, gripper_pos)
    #             delta = goal_pos - now_pos
    #             norm = np.linalg.norm(delta)
    #             max_delta = np.max(np.abs(delta))
    #             if norm < self.min_norm and max_delta < self.min_norm / 2:
    #                 scaled_velocity_limit = self.base_velocity_limit
    #             else:
    #                 scaled_velocity_limit = min((5 * norm / self.min_norm - 4) * self.base_velocity_limit, self.max_velocity_limit)
    #             # motion_scale = max_delta / (scaled_velocity_limit * self.timer_period_2)
    #             motion_scale = norm / (scaled_velocity_limit * self.timer_period_2)
    #             final_goal = now_pos + delta / motion_scale
    #             self.xarm.set_servo_angle_j(angles=final_goal, is_radian=True)

    def timer_callback_2(self):
        current_time = time.time()
        if self.callback2_last_time is not None and self.measure_frequency:
            interval = current_time - self.callback2_last_time
            self.callback2_times.append(interval)
            # 保持列表在合理大小
            if len(self.callback2_times) > 1 / self.timer_period_2:
                self.callback2_times.pop(0)
        self.callback2_last_time = current_time

        if self.start_ep:
            if self.np_action is not None:
                goal_pos = self.np_action[:6]
                gripper_pos = self.np_action[6].astype(int)
                now_pos = np.array(self.xarm_joint[:6])
                delta = goal_pos - now_pos
                norm = np.linalg.norm(delta)
                max_delta = np.max(np.abs(delta))
                if norm < self.min_norm and max_delta < self.min_norm / 2:
                    scaled_velocity_limit = self.base_velocity_limit
                else:
                    scaled_velocity_limit = min((5 * norm / self.min_norm - 4) * self.base_velocity_limit, self.max_velocity_limit)
                motion_scale = norm / (scaled_velocity_limit * self.timer_period_2)
                final_goal = now_pos + delta / motion_scale
                self.xarm.set_servo_angle_j(angles=final_goal, is_radian=True)
                self.gello.write_position(8, gripper_pos)

    # def timer_callback_3(self):
    #     self.gripper_joint = self.gello.read_position(8)
    #     if self.start_ep:
    #         if self.np_action is not None:
    #             gripper_pos = self.np_action[6].astype(int)
    #             self.gello.write_position(8, gripper_pos)

    def timer_callback_3(self):
        self.gripper_joint = self.gello.read_position(8)
        code, self.xarm_joint = self.xarm.get_servo_angle(is_radian=True)
        

    def print_callback_frequencies(self):
        """定期打印回调函数的实际频率"""
        if not self.measure_frequency:
            return
            
        if self.callback1_times:
            avg_interval1 = sum(self.callback1_times) / len(self.callback1_times)
            freq1 = 1.0 / avg_interval1 if avg_interval1 > 0 else 0
            self.get_logger().info(f"Timer 1 实际频率: {freq1:.2f} Hz (目标: {1.0/self.timer_period_1:.2f} Hz), 间隔: {avg_interval1*1000:.2f} ms")
        
        if self.callback2_times:
            avg_interval2 = sum(self.callback2_times) / len(self.callback2_times)
            freq2 = 1.0 / avg_interval2 if avg_interval2 > 0 else 0
            self.get_logger().info(f"Timer 2 实际频率: {freq2:.2f} Hz (目标: {1.0/self.timer_period_2:.2f} Hz), 间隔: {avg_interval2*1000:.2f} ms")
    # def timer_callback_1(self):
    #     msg = Int32()
    #     msg.data = self.signal
    #     self.publisher_.publish(msg)
    #     self.signal = 0

    # def timer_callback_2(self):
    #     if self.enable_record:
    #         if self.start_ep:
    #             self.obs_dict["observation.state"] = self.obs_act_data[:7]
    #             self.act_dict["action"] = self.obs_act_data[7:]
    #             self.obs_dict["observation.images.head_cam"] = self.head_cam
    #             self.obs_dict["observation.images.tac_01"] = self.tac_01
    #             self.obs_dict["observation.images.tac_02"] = self.tac_02
    #             if not self.image_keys or not self.low_dim_keys:
    #                 self.image_keys = [key for key in self.obs_dict if "images" in key]
    #                 self.low_dim_keys = [key for key in self.obs_dict if "images" not in key]

    #             for key in self.image_keys:
    #                 self.futures.append(self.thread_pool.submit(save_image, self.obs_dict[key], key, self.frame_index, self.episode_index, self.video_dir))

    #             for key in self.low_dim_keys:
    #                 if key not in self.ep_dict:
    #                     self.ep_dict[key] = []
    #                 self.ep_dict[key].append(self.obs_dict[key])

    #             for key in self.act_dict:
    #                 if key not in self.ep_dict:
    #                     self.ep_dict[key] = []
    #                 self.ep_dict[key].append(self.act_dict[key])

    #             if self.frame_index == 0:
    #                 self.begin_time = time.perf_counter()
    #                 timestamp = 0.0
    #             else:
    #                 timestamp = time.perf_counter() - self.begin_time
    #             if "timestamp" not in self.ep_dict:
    #                 self.ep_dict["timestamp"] = []
    #             self.ep_dict["timestamp"].append(timestamp)

    #             self.frame_index += 1

    #         if self.save_ep:
    #             # diffs = np.diff(self.ep_dict["timestamp"])
    #             invalid_diffs = [i for i, ts in enumerate(self.ep_dict["timestamp"]) if abs(ts - i * self.timer_period_2) > 0.001]
    #             if invalid_diffs:
    #                 self.get_logger().error(f"Invalid time differences detected: {invalid_diffs}")
    #             else:
    #                 for key in self.image_keys:
    #                     video_name = f"{key}_episode_{self.episode_index:06d}.mp4"
    #                     video_path = self.video_dir / video_name
    #                     self.ep_dict[key] = []
    #                     for i in range(self.frame_index):
    #                         # self.ep_dict[key].append({"path":f"video/{video_name}", "timestamp":i*self.timer_period_2})
    #                         self.ep_dict[key].append(i*self.timer_period_2)

    #                 for key in self.low_dim_keys:
    #                     self.ep_dict[key] = np.array(self.ep_dict[key])

    #                 for key in self.act_dict:
    #                     self.ep_dict[key] = np.array(self.ep_dict[key])
                    
    #                 self.ep_dict["timestamp"] = np.array(self.ep_dict["timestamp"])
    #                 self.ep_dict["episode_index"] = np.array([self.episode_index]*self.frame_index)
    #                 self.ep_dict["frame_index"] = np.arange(self.frame_index)
    #                 done = np.zeros(self.frame_index, dtype=bool)
    #                 done[-1] = True
    #                 self.ep_dict["next.done"] = done
    #                 ep_path = self.data_dir / f"episode_{self.episode_index:06d}.npy"
    #                 np.save(ep_path, self.ep_dict, allow_pickle=True)
    #                 self.episode_index += 1
    #                 self.total_frames += self.frame_index

    #                 self.get_logger().info(f"Episode {self.episode_index} saved.")
    #             # self.frame_index = 0
    #             # self.obs_dict, self.act_dict, self.ep_dict = {},{},{}
    #             self.save_ep = False
                
    #         if self.reset_ep:
    #             self.frame_index = 0
    #             self.obs_dict, self.act_dict, self.ep_dict = {},{},{}
    #             for key in self.image_keys:
    #                 image_path = self.video_dir / f"{key}_episode_{self.episode_index:06d}"
    #                 if image_path.exists():
    #                     shutil.rmtree(image_path)
    #             self.reset_ep = False

    #         # if "timestamp" not in self.ep_dict:
    #         #     self.ep_dict["timestamp"] = []
    # # def observation_callback(self, msg):
    # #     self.obs_dict["observation.state"] = np.array(msg.position)

    # # def action_callback(self, msg):
    # #     self.act_dict["action"] = np.array(msg.position)

    # def obs_act_callback(self, msg):
    #     self.obs_act_data = np.array(msg.position)

    def head_cam_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.cam_img = self.np_to_torch(img)

    def tac_01_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.tac_01_img = self.np_to_torch(img)

    def tac_02_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.tac_02_img = self.np_to_torch(img)

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

    # def encode_video(self):
    #     for ep_index in tqdm(range(self.episode_index)):
    #         for key in self.image_keys:
    #             video_name = f"{key}_episode_{ep_index:06d}.mp4"
    #             video_path = self.video_dir / video_name
    #             image_path = self.video_dir / f"{key}_episode_{ep_index:06d}"
    #             if not video_path.exists():
    #                 encode_video_frames(
    #                     image_path, video_path, int(1 / self.timer_period_2), vcodec="libx264", overwrite=True
    #                     )
    #             shutil.rmtree(image_path)
    #     self.get_logger().info("All episodes have been encoded.")
    
    # def create_meta_info(self):
    #     info = {"total_episodes": self.episode_index, "fps": int(1/self.timer_period_2), "total_frames": self.total_frames}
    #     with open(self.meta_dir / "info.json", "w") as f:
    #         for k, v in info.items():
    #             f.write(f"{k}: {v}\n")
    #     self.get_logger().info("Meta info has been created.")

    def cleanup(self):
        # for _ in tqdm(
        #     concurrent.futures.as_completed(self.futures),
        #     total=len(self.futures),
        #     desc="Writing images to disk",
        #     ):
        #     pass
        # if self.enable_record:
        #     self.encode_video()
        #     self.create_meta_info()
        self.xarm.disconnect()
        self.get_logger().info("XArm disconnected.")
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

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

@hydra.main(version_base=None, config_path="../config", config_name="eval_cfg")
def main(cfg):
    # repo_dir = Path(cfg.root) / cfg.repo_id
    # data_dir = repo_dir / "data"
    # meta_dir = repo_dir / "meta"
    # video_dir = repo_dir / "videos"
    # num_image_writers = cfg.num_writers_per_image * (cfg.num_cam + cfg.num_tac)
    # # repo_dir.mkdir(parents=True, exist_ok=True)
    # data_dir.mkdir(parents=True, exist_ok=True)
    # meta_dir.mkdir(parents=True, exist_ok=True)
    # video_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_image_writers) as executor:
    #     control_node = ControlNode(cfg, data_dir, meta_dir, video_dir)
    control_node = ControlNode(cfg)
    # executor = MultiThreadedExecutor(num_threads=5)
    executor = MultiThreadedExecutor()
    executor.add_node(control_node)
    listener = keyboard.Listener(on_press=lambda key: on_press(key, control_node))
                                    # on_release=lambda key: on_release(key, control_node))
    listener.start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        control_node.gello.close()
        executor.shutdown()
        executor.remove_node(control_node)
        control_node.cleanup()
        time.sleep(0.5)
        gripper = Gripper(cfg.u2d2_port)
        gripper.open_port()
        gripper.set_torque(8, False)
        gripper.portHandler.closePort()

    # control_node = ControlNode()
    # listener = keyboard.Listener(on_press=lambda key: on_press(key, control_node))
    # listener.start()

    # try:
    #     rclpy.spin(control_node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     control_node.cleanup()
    #     listener.stop()
    #     # print("Exiting program due to keyboard interrupt.")
    #     # stop_all_processes()
    #     # os._exit(0)

if __name__ == '__main__':
    main()