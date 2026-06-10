from pathlib import Path
import os
import paramiko
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
# from ament_index_python.packages import get_package_share_directory

from dotview_driver.mysocket import MySocket

class SensorStream(Node):
    def __init__(self):
        super().__init__('sensor_stream')
        self.declare_parameter('ip', '10.18.18.202')
        self.declare_parameter('channel', 3)
        self.declare_parameter('serial', '02')
        self.declare_parameter('calibrate', True)
        self.declare_parameter('pub_gray_img', True)
        self.declare_parameter('pub_binary_img', True)
        self.declare_parameter('frame_rate', 27)

        self.ip = self.get_parameter('ip').get_parameter_value().string_value
        self.channel = self.get_parameter('channel').get_parameter_value().integer_value
        serial = self.get_parameter('serial').get_parameter_value().string_value
        self.calibrate = self.get_parameter('calibrate').get_parameter_value().bool_value
        self.pub_gray_img = self.get_parameter('pub_gray_img').get_parameter_value().bool_value
        self.pub_binary_img = self.get_parameter('pub_binary_img').get_parameter_value().bool_value
        frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.bg_image_path = Path(f'/home/xiaohang_shi/ros2_ws/humble_ws/src/dotview_ros2/dotview_driver/etc/bg/{serial}.png')

        if not self.calibrate and not self.bg_image_path.exists():
            self.get_logger().error(f"Background image {self.bg_image_path} does not exist! Please check the serial number. If it is correct, please do calibration first.")
            exit(1)

        self.br = CvBridge()
        self._index = 0

        self.raw_img = self.create_publisher(Image, 'DotView_raw_img', 1)
        if self.pub_gray_img:
            self.pp_gray_img = self.create_publisher(Image, 'DotView_pp_gray_img', 1)
        if self.pub_binary_img:
            self.pp_binary_img = self.create_publisher(Image, 'DotView_pp_binary_img', 1)
        timer_period = 1 / frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.open_tcp()
        
    def open_tcp(self):
        channel = self.channel
        ret = self.connect_ssh()
        if ret:
            self.get_logger().info('SSH connection SUCCESS')
            host = ''
            if self.ip == '10.18.18.201':
                port = 20111 + channel
            else:
                port = 20222 + channel
            msg_len = 192 * 192 + 1
            self.server_socket = MySocket(msg_len=msg_len)
            self.server_socket.bind(host, port)
            self.server_socket.accept()     # Wait for tcp on Pi to connect
            self.get_logger().info(f"Connected by {self.server_socket.addr}")
            return True
        else:
            self.get_logger().error('SSH connection FAILED')
            return False
            
    def connect_ssh(self, port=22):
        channel = self.channel
        username = "logm"
        pwd = "meiyoumima"
        command = 'cd Documents/; python3 tst_socket_c.py ' + str(channel) + ' ' + str(1020)
        self.sshclient = paramiko.SSHClient()
        self.sshclient.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.sshclient.connect(hostname=self.ip, port=port, username=username, password=pwd)
            self.sshclient.exec_command(command)
        except:
            return 0
        else:
            return 1
        
    def read_img(self):
        self.server_socket.mysend_one(b'\x01')
        # start = time.time()
        data = self.server_socket.myreceive()
        # end = time.time()
        # self.get_logger().info(f"Time elapsed: {end - start}")
        img = np.frombuffer(data, dtype=np.uint8)
        if img[-1] == 0:
            ret = False
        else:
            ret = True
        img = img[:-1].reshape(192, 192)
        return ret, img
    
    def timer_callback(self):
        ret, img = self.read_img()
        if not ret:
            self.get_logger().error("Failed to read image")
            return
        raw_img = self.br.cv2_to_imgmsg(img)
        self.raw_img.publish(raw_img)
        self._index += 1
        if self.calibrate and not self.bg_image_path.exists():
            return
        gray_img, binary_img = self.preprocessing(img)
        if self.pub_gray_img:
            gray_img = self.br.cv2_to_imgmsg(gray_img)
            self.pp_gray_img.publish(gray_img)
        if self.pub_binary_img:
            binary_img = self.br.cv2_to_imgmsg(binary_img)
            self.pp_binary_img.publish(binary_img)

    def preprocessing(self, img):
        bg_img = cv.imread(str(self.bg_image_path), cv.IMREAD_GRAYSCALE)

        def gray_stretch(data, stretch):
            if stretch:
                if np.max(data) == np.min(data):
                    res = data
                else:
                    res = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0
            else: 
                res = (data - np.min(data)) / (255 - np.min(data)) * 255.0
				# res = (data - np.min(data)) / (255-np.min(bg_image) - np.min(data)) * 255.0
            res = res.astype(np.uint8)
            return res
        
        sub_img = cv.subtract(img, bg_img)
		# clip pixel values that lower than 10 to 0
        sub_img[sub_img < 10] = 0   #最简单的方法是针对不同批次的膜调整这个的值
        dst = gray_stretch(sub_img, stretch=True)
        blur_img = cv.GaussianBlur(dst, (5, 5), 0, 0)
        _, binary_img = cv.threshold(blur_img, 120, 255, cv.THRESH_BINARY)
        return blur_img, binary_img
    
    def shutdown_connect(self):
        self.server_socket.mysend_one(b'\x03')
        self.server_socket.close()
        self.sshclient.close()

    def cleanup(self):
        self.shutdown_connect()
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    sensor_stream = SensorStream()
    try:
        rclpy.spin(sensor_stream)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_stream.cleanup()

if __name__ == '__main__':
    main()