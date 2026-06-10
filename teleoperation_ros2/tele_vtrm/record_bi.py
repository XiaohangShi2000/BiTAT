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
import sys
import logging

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState, Image
from tele_vtrm.video_utils import encode_video_frames
sys.path.append("/home/xiaohang_shi/Project/lerobot")
from lerobot.common.datasets.compute_stats import sample_indices, get_feature_stats, auto_downsample_height_width
from lerobot.common.datasets.utils import load_image_as_numpy, write_episode_stats
sys.path.append('/home/xiaohang_shi/miniconda3/envs/lerobot/lib/python3.10/site-packages')
from datasets import Dataset

def on_press(key, control_node):
    try:
        if key.char == '1':
            control_node.signal = 1
            control_node.start_ep = False
            control_node.reset_ep = True
        elif key.char == '2':
            control_node.signal = 2
            # control_node.start_ep = True
        elif key.char == '3':
            control_node.signal = 3
            control_node.start_ep = False
            # control_node.end_ep = True
        elif key.char == '4':
            control_node.signal = 4
            # control_node.rerec_ep = True
            if control_node.start_ep:
                control_node.start_ep = False
            else:
                control_node.save_ep = True
    except AttributeError:
        pass

def on_release(key,control_node):
    try:
        if key.char =='2':
            control_node.start_ep = True
    except AttributeError:
        pass

def save_image(img_array, key, frame_index, episode_index, videos_dir):
    # img = Image.fromarray(img_tensor.numpy()) # if img_tensor is pytorch.Tensor
    img = PILImage.fromarray(img_array)
    path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)

class ControlNode(Node):
    def __init__(self, thread_pool, repo_dir, cfg):
        super().__init__('control_node')

        self.publisher_ = self.create_publisher(Int32, 'control', 1)
        timer_period_1 = 0.1
        self.timer_1 = self.create_timer(timer_period_1, self.timer_callback_1)
        self.timer_period_2 = 1 / cfg.fps
        self.timer_2 = self.create_timer(self.timer_period_2, self.timer_callback_2)
        self.obs_act_r = self.create_subscription(JointState, '/right/obs_act', self.obs_act_r_callback, 1)
        self.obs_act_r
        self.obs_act_l = self.create_subscription(JointState, '/left/obs_act', self.obs_act_l_callback, 1)
        self.obs_act_l
        self.head_cam_sub = self.create_subscription(Image, '/head/d435i/color/image_raw', self.head_cam_callback, 1)
        self.head_cam_sub
        self.tac_01_sub = self.create_subscription(Image, '/sensor_01/DotView_pp_gray_img', self.tac_01_callback, 1)
        self.tac_01_sub
        self.tac_02_sub = self.create_subscription(Image, '/sensor_02/DotView_pp_gray_img', self.tac_02_callback, 1)
        self.tac_02_sub
        self.tac_03_sub = self.create_subscription(Image, '/sensor_03/DotView_pp_gray_img', self.tac_03_callback, 1)
        self.tac_03_sub
        self.tac_04_sub = self.create_subscription(Image, '/sensor_04/DotView_pp_gray_img', self.tac_04_callback, 1)
        self.tac_04_sub
        self.signal = 0
        self.start_ep = False
        # self.end_ep = False
        # self.rerec_ep = False
        self.save_ep = False
        self.reset_ep = False
        self.bridge = CvBridge()
        self.thread_pool = thread_pool
        # self.repo_path = Path(cfg.root) / cfg.repo_id
        self.cfg = cfg
        self.repo_dir = repo_dir
        self.data_dir = repo_dir / "data"
        self.meta_dir = repo_dir / "meta"
        self.video_dir = repo_dir / "videos"
        # self.resume = cfg.resume
        if not cfg.resume:
            self.pre_episodes = 0
            self.episode_index = 0
            self.total_frames = 0
        else:
            try:
                with open(self.meta_dir / "info.json", "r") as f:
                    info = json.load(f)
            except FileNotFoundError:
                self.get_logger().error("Resume is true but info.json file not found. Starting from scratch.")
            if cfg.fps != info["fps"]:
                self.get_logger().error(f"FPS mismatch: {cfg.fps} != {info['fps']}.")
            self.pre_episodes = info["total_episodes"]
            self.episode_index = info["total_episodes"]
            self.total_frames = info["total_frames"]
        self.frame_index = 0
        self.obs_dict, self.act_dict, self.ep_dict = {},{},{}
        self.image_keys, self.low_dim_keys, self.futures = [],[],[]
        self.get_logger().info(f"initialization done, be ready to record")

    def timer_callback_1(self):
        msg = Int32()
        msg.data = self.signal
        self.publisher_.publish(msg)
        self.signal = 0

    def timer_callback_2(self):
        if self.start_ep:
            self.obs_dict["observation.state"] = np.append(self.obs_act_r_data[:self.cfg.single_freedom], self.obs_act_l_data[:self.cfg.single_freedom])
            self.act_dict["action"] = np.append(self.obs_act_r_data[self.cfg.single_freedom:], self.obs_act_l_data[self.cfg.single_freedom:])
            self.obs_dict["observation.images.head_cam"] = self.head_cam
            self.obs_dict["observation.images.tac_01"] = self.tac_01
            self.obs_dict["observation.images.tac_02"] = self.tac_02
            self.obs_dict["observation.images.tac_03"] = self.tac_03
            self.obs_dict["observation.images.tac_04"] = self.tac_04
            if not self.image_keys or not self.low_dim_keys:
                self.image_keys = [key for key in self.obs_dict if "images" in key]
                self.low_dim_keys = [key for key in self.obs_dict if "images" not in key]

            for key in self.image_keys:
                self.futures.append(self.thread_pool.submit(save_image, self.obs_dict[key], key, self.frame_index, self.episode_index, self.video_dir))

            for key in self.low_dim_keys:
                if key not in self.ep_dict:
                    self.ep_dict[key] = []
                self.ep_dict[key].append(self.obs_dict[key])

            for key in self.act_dict:
                if key not in self.ep_dict:
                    self.ep_dict[key] = []
                self.ep_dict[key].append(self.act_dict[key])

            if self.frame_index == 0:
                self.begin_time = time.perf_counter()
                timestamp = 0.0
            else:
                timestamp = time.perf_counter() - self.begin_time
            # self.get_logger().info(f"timestamp: {timestamp}, frame_index: {self.frame_index}")
            if "timestamp" not in self.ep_dict:
                self.ep_dict["timestamp"] = []
            self.ep_dict["timestamp"].append(timestamp)

            self.frame_index += 1

        if self.save_ep:
            diffs = np.diff(self.ep_dict["timestamp"])
            invalid_diffs_1 = [i for i, ts in enumerate(self.ep_dict["timestamp"]) if abs(ts - i * self.timer_period_2) > 0.05*self.timer_period_2]
            invalid_diffs_2 = [i for i, diff in enumerate(diffs) if abs(diff - self.timer_period_2) > 0.05*self.timer_period_2]
            invalid_diffs = list(set(invalid_diffs_1 + invalid_diffs_2))
            if invalid_diffs:
                self.get_logger().error(f"Invalid time differences detected: {invalid_diffs}")
            else:
                # for key in self.image_keys:
                #     # video_name = f"{key}_episode_{self.episode_index:06d}.mp4"
                #     # video_path = self.video_dir / video_name
                #     self.ep_dict[key] = []
                #     for i in range(self.frame_index):
                #         # self.ep_dict[key].append({"path":f"video/{video_name}", "timestamp":i*self.timer_period_2})
                #         self.ep_dict[key].append(i*self.timer_period_2)

                for key in self.low_dim_keys:
                    self.ep_dict[key] = np.array(self.ep_dict[key]).astype(np.float32)

                for key in self.act_dict:
                    self.ep_dict[key] = np.array(self.ep_dict[key]).astype(np.float32)
                
                self.ep_dict["timestamp"] = np.array(self.ep_dict["timestamp"]).astype(np.float32)
                self.ep_dict["episode_index"] = np.array([self.episode_index]*self.frame_index).astype(np.int64)
                self.ep_dict["frame_index"] = np.arange(self.frame_index).astype(np.int64)
                done = np.zeros(self.frame_index, dtype=bool)
                done[-1] = True
                self.ep_dict["next.done"] = done
                ep_stats = self.compute_episode_stats()
                write_episode_stats(self.episode_index, ep_stats, self.repo_dir)
                # ep_path = self.data_dir / f"episode_{self.episode_index:06d}.npy"
                # np.save(ep_path, self.ep_dict, allow_pickle=True)
                ep_path = self.data_dir / f"episode_{self.episode_index:06d}.parquet"
                for key in self.ep_dict:
                    self.ep_dict[key] = self.ep_dict[key].tolist()
                df = Dataset.from_dict(self.ep_dict)
                df.to_parquet(ep_path)

                with open(self.meta_dir / "episodes.jsonl", "a") as f:
                    f.write(json.dumps({"episode_index": self.episode_index, "length": self.frame_index}) + "\n")
                self.episode_index += 1
                self.total_frames += self.frame_index

                self.get_logger().info(f"Episode {self.episode_index} saved.")

            self.save_ep = False
            
        if self.reset_ep:
            self.frame_index = 0
            self.obs_dict, self.act_dict, self.ep_dict = {},{},{}
            for key in self.image_keys:
                image_path = self.video_dir / f"{key}_episode_{self.episode_index:06d}"
                if image_path.exists():
                    shutil.rmtree(image_path)
            self.reset_ep = False
            self.get_logger().info(f"robot reset done")
            

    def obs_act_r_callback(self, msg):
        self.obs_act_r_data = np.array(msg.position)

    def obs_act_l_callback(self, msg):
        self.obs_act_l_data = np.array(msg.position)

    def head_cam_callback(self, msg):
        self.head_cam = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def tac_01_callback(self, msg):
        self.tac_01 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def tac_02_callback(self, msg):
        self.tac_02 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def tac_03_callback(self, msg):
        self.tac_03 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def tac_04_callback(self, msg):
        self.tac_04 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def encode_video(self):
        for ep_index in tqdm(range(self.episode_index)):
            for key in self.image_keys:
                image_path = self.video_dir / f"{key}_episode_{ep_index:06d}"
                if not image_path.exists():
                    continue
                video_name = f"{key}_episode_{ep_index:06d}.mp4"
                video_path = self.video_dir / video_name
                if not video_path.exists():
                    if self.cfg.video_codec == "h264":
                        codec = "libx264"
                    elif self.cfg.video_codec == "h265":
                        codec = "libx265"
                    elif self.cfg.video_codec == "av1":
                        codec = "libsvtav1"
                    else:
                        raise ValueError(f"Unsupported video codec: {self.cfg.video_codec}")
                    encode_video_frames(
                        image_path, video_path, self.cfg.fps, vcodec=codec, pix_fmt=self.cfg.video_pix_fmt, overwrite=True
                        )
                if self.cfg.remove_images:
                    # remove the images after encoding
                    shutil.rmtree(image_path)
                    self.get_logger().info(f"Removing images for {video_name}")
        self.get_logger().info("All episodes have been encoded.")
    
    def create_meta_info(self):
        info = {
            "codebase_version": str(self.cfg.codebase_version),
            "robot_type": str(self.cfg.robot_type),
            "total_episodes": self.episode_index,
            "total_frames": self.total_frames,
            "fps": int(self.cfg.fps),
            "data_path": "data/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{key}_episode_{episode_index:06d}.mp4",
            "features": {}
            }
        common_video_info = {
            "video.fps": int(self.cfg.fps),
            "video.codec": str(self.cfg.video_codec),
            "video.pix_fmt": str(self.cfg.video_pix_fmt),
            "video.is_depth_map": False,
            "has_audio": False
            }
        info["features"]["observation.images.head_cam"] = self.create_image_feature(
            dtype="video",
            shape=list(self.cfg.cam_shape),
            names=["height", "width", "channel"],
            video_info=common_video_info
            )
        for tac_id in range(1, self.cfg.num_tac + 1):
            if tac_id < 10:
                tac_id = f"0{tac_id}"
            info["features"][f"observation.images.tac_{tac_id}"] = self.create_image_feature(
                dtype="video",
                shape=[192, 192, 1],
                names=["height", "width", "channel"],
                video_info=common_video_info
                )
        simple_features = {
            "observation.state": {"dtype": "float32", "shape": [int(self.cfg.total_freedom)], "names": None},
            "action": {"dtype": "float32", "shape": [int(self.cfg.total_freedom)], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None}
            }
        info["features"].update(simple_features)
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        if not self.cfg.resume:
            with open(self.meta_dir / "tasks.jsonl", "a") as f:
                f.write(json.dumps({"task_index": 0, "task": self.cfg.task_description}) + "\n")
        self.get_logger().info("Meta info has been created.")

    def create_image_feature(self, dtype: str, shape: list, names: list, video_info: dict):
        return {
            "dtype": dtype,
            "shape": shape,
            "names": names,
            "video_info": video_info
            }
    
    def compute_episode_stats(self) -> dict:
        ep_stats = {}
        # ep_stats['episode_index'] = self.episode_index
        sampled_indices = sample_indices(self.frame_index)
        for key in self.image_keys:
            images = None
            for i, idx in enumerate(sampled_indices):
                path = self.video_dir / f"{key}_episode_{self.episode_index:06d}" / f"frame_{idx:06d}.png"
                # we load as uint8 to reduce memory usage
                img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
                img = auto_downsample_height_width(img)

                if images is None:
                    images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

                images[i] = img
        
            ep_stats[key] = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
            ep_stats[key] = {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()}
        ep_stats["observation.state"] = get_feature_stats(self.ep_dict["observation.state"], axis=0, keepdims=False)
        ep_stats["action"] = get_feature_stats(self.ep_dict["action"], axis=0, keepdims=False)

        return ep_stats

    def cleanup(self):
        # for _ in tqdm(
        #     concurrent.futures.as_completed(self.futures),
        #     total=len(self.futures),
        #     desc="Writing images to disk",
        #     ):
        #     pass
        if self.episode_index > self.pre_episodes:
            self.create_meta_info()
            self.encode_video()
        for key in self.image_keys:
            image_path = self.video_dir / f"{key}_episode_{self.episode_index:06d}"
            if image_path.exists():
                shutil.rmtree(image_path)
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

@hydra.main(version_base=None, config_path="../config", config_name="record_bi_teleop")
def main(cfg):
    repo_dir = Path(cfg.root) / cfg.repo_id
    data_dir = repo_dir / "data"
    meta_dir = repo_dir / "meta"
    video_dir = repo_dir / "videos"
    if not cfg.resume and repo_dir.exists():
        print(f"\033[33mRepository {repo_dir} already exists. Please comfirm if you want to delete it!!!\033[0m")
        confirm = input("Type 'd' to delete the repository and continue, or any other key to exit!!!")
        if confirm.lower() != 'd':
            print("\033[32mExiting without deleting the repository.\033[0m")
            return
        else:
            shutil.rmtree(repo_dir)
            print(f"\033[32mRepository {repo_dir} has been deleted.\033[0m")
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    num_image_writers = cfg.num_writers_per_image * (cfg.num_cam + cfg.num_tac)

    rclpy.init()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_image_writers) as executor:
        control_node = ControlNode(executor, repo_dir, cfg)
        listener = keyboard.Listener(on_press=lambda key: on_press(key, control_node),
                                     on_release=lambda key: on_release(key, control_node))
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