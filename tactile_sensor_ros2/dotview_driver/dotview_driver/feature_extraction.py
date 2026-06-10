import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
from pathlib import Path
import threading
import keyboard
import sys
import tty
import termios
import getch
import time

from dotview_msgs.msg import DotFeature
from dotview_msgs.msg import PointCord

# stop_event = threading.Event()

class FeatureExtraction(Node):
    def __init__(self):
        super().__init__('feature_extraction')
        self.declare_parameter('serial', '02')
        self.declare_parameter('line_num', 7)
        self.declare_parameter('calibrate', True)
        self.declare_parameter('frame_rate', 27)

        serial = self.get_parameter('serial').get_parameter_value().string_value
        self.line_num = self.get_parameter('line_num').get_parameter_value().integer_value
        self.calibrate = self.get_parameter('calibrate').get_parameter_value().bool_value
        frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value

        self.bg_image_path = Path(f'/home/xiaohang_shi/ros2_ws/humble_ws/src/dotview_ros2/dotview_driver/etc/bg/{serial}.png')
        self.init_point_path = Path(f'/home/xiaohang_shi/ros2_ws/humble_ws/src/dotview_ros2/dotview_driver/etc/init_points/{serial}.npy')
        
        self.br = CvBridge()
        self.binary_img_sub = self.create_subscription(Image, 'DotView_pp_binary_img', self.binary_img_callback, 1)
        self.binary_img_sub
        self.binary_img = None
        self._dot_num = self.line_num ** 2

        if self.calibrate:
            self.raw_img_sub = self.create_subscription(Image, 'DotView_raw_img', self.raw_img_callback, 1)
            self.raw_img_sub
            self.keyboard_thread = threading.Thread(target=self.calibrate_init)
            self.keyboard_thread.start()
            # self.calibrate_init()
        else:
            if not self.init_point_path.exists():
                self.get_logger().error(f"init_points file {self.init_point_path} does not exist! Please check the serial number. If it is correct, please do calibration first.")
                exit(1)
            self.set_init_point()            
        
        self.contour_img_pub = self.create_publisher(Image, 'DotView_contour_img', 1)
        self.feature_img_pub = self.create_publisher(DotFeature, 'DotView_feature_img', 1)
        self.depth_pub = self.create_publisher(PointCord, 'DotView_depth', 1)
        timer_period = 1 / frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def set_init_point(self):
        self.init_point_pos = np.load(self.init_point_path)
        self.pos = np.zeros((self._dot_num, 3))
        if self.init_point_pos.shape[0] != self._dot_num:
            self.get_logger().error(f"init_points file {self.init_point_path} does not have enough points! Please check the serial number. If it is correct, please do calibration first.")
            exit(1)
        self.pos[:, :2] = self.init_point_pos
        self.pos = np.append(self.pos, np.arange(self._dot_num).reshape(self._dot_num,1), axis=1)
        self.prv_pos = self.pos.copy()
        self.info = np.zeros((self._dot_num, 1))
            
    def raw_img_callback(self, msg):
        self.raw_img = msg

    def binary_img_callback(self, msg):
        self.binary_img = msg

    def timer_callback(self):
        # time.sleep(0.02)
        if self.calibrate:
            return
        if self.binary_img is None:
            self.get_logger().warn("Binary image is not received yet!")
            return
        self.extract_feature(self.binary_img)

    def calibrate_init(self):
        print("--->>> Step 1: Save backgound image <<<---")
        if Path(self.bg_image_path).exists():
            self.get_logger().warn(f"Background image {self.bg_image_path} already exists! If you want override the file, please press y then Enter to continue. Otherwise, press Enter to skip.")
            choice = input()
            # choice = self.get_single_keypress()
            # choice = getch.getch()
            if choice == 'y' or choice == 'Y':
                bg_cali = True
            else:
                print("Skip saving background image.")
                bg_cali = False
        else:
            bg_cali = True
        if bg_cali:
            print("Make sure DotView sensor contact nothing at present. Please press Enter to continue")
            input()
            # self.get_single_keypress()
            # getch.getch()
            # print(f'{self.raw_img}')
            bg = self.br.imgmsg_to_cv2(self.raw_img)
            # cv.imshow('Background Image', bg)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            ret = cv.imwrite(str(self.bg_image_path), bg)
            # print(ret)
            print(f"Background image saved to {self.bg_image_path}")

        print("--->>> Step 2: Save initial points <<<---")
        if Path(self.init_point_path).exists():
            self.get_logger().warn(f"init_points file {self.init_point_path} already exists! If you want override the file, please press y then Enter to continue. Otherwise, press Enter to skip.")
            choice = input()
            # choice = self.get_single_keypress()
            # choice = getch.getch()
            if choice == 'y' or choice == 'Y':
                pos_cali = True
            else:
                print("Skip saving initial points.")
                pos_cali = False
        else:
            pos_cali = True
        if pos_cali:
            while True: 
                print(f"Press the sensor lightly. Make sure each of ALL {self._dot_num} dots is visible. Please press Enter to continue")
                input()
                # self.get_single_keypress()
                # getch.getch()
                img = self.br.imgmsg_to_cv2(self.binary_img)
                dots_info, _ = self.get_info(img)
                cali_pos = self.calibrate_position(dots_info)
                if cali_pos is not None: 
                    np.save(self.init_point_path, cali_pos)
                    print(f"Calibration file saved to {self.init_point_path}")
                    break
                else:
                    self.get_logger().warn("Calibration failed. Please try again.")
        print("Calibration finished.")
        self.calibrate = False
        self.set_init_point()
        # stop_event.set()


    def get_info(self, img):
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
        dots_info = [] 
        effective_contours = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area:
				# # illustrate contours
				# cv.drawContours(bgr_img, [contour], 0, color=(0,0,255), thickness=1)
				# 筛选轮廓
                effective_contours.append(contour)
				# 提取轮廓信息
                M = cv.moments(contour)
                x, y = M['m10']/M['m00'], M['m01']/M['m00']
                dots_info.append([x, y, area])
        return dots_info, effective_contours
    
    def calibrate_position(self, dots_info):
        if len(dots_info) != self._dot_num:
            self.get_logger().error(f"Calibration failed: {len(dots_info)} dots are detected, but {self._dot_num} dots are needed.")
            return None
        dots_info = np.array(dots_info, copy=True)
        pos_mat = dots_info[:, 0:2]
		# 对dots进行逐行排序编号
        pos_mat = pos_mat[pos_mat[:,1].argsort()]
        for i in range(self.line_num):
            roll_vals= pos_mat[i*self.line_num:(i+1)*self.line_num, :]
            roll_vals = roll_vals[roll_vals[:,0].argsort()]
            pos_mat[i*self.line_num:(i+1)*self.line_num, :] = roll_vals
        return pos_mat
    
    def extract_feature(self, img):
        img = self.br.imgmsg_to_cv2(img)
        dots_info, contours = self.get_info(img)
        self.match_dots(dots_info)
        offsets = self.pos[:, :2] - self.init_point_pos
        # 画箭头可视化位移场
        rgb_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        for j in range(self.pos.shape[0]):
            cv.arrowedLine(
                rgb_img, 
                # (int(self.init_pos[j][0]), int(self.init_pos[j][1])), 
                (int(self.pos[j][0]), int(self.pos[j][1])), 
                (int(self.pos[j][0] + 1.5 * offsets[j][0]), int(self.pos[j][1] + 1.5 * offsets[j][1])), 
                (205, 149, 12), 
                tipLength=0.2
            )
        # 可视化轮廓
        for contour in contours:			
            # illustrate contours
            cv.drawContours(rgb_img, [contour], 0, color=(65, 105, 225), thickness=1)
        # disps = self.pos[:, 0:2]
        areas = self.info.copy()
        pos = self.pos[:, 0:2].copy()
        cord = np.zeros((pos.shape[0], 3))
        cord[:,0] = pos[:,0]
        cord[:,1] = pos[::-1,1]
        cord[:,2] = areas[:,0] * 0.004
        areas = areas.flatten().tolist()
        # # 将偏移场和面积合并成特征矩阵
        # features = np.append(areas, offsets, axis=1)

        # publish features
        msg = DotFeature()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.area = list(areas)
        msg.delta_x = list(offsets[:, 0])
        msg.delta_y = list(offsets[:, 1])
        self.feature_img_pub.publish(msg)
        self.contour_img_pub.publish(self.br.cv2_to_imgmsg(rgb_img, encoding='rgb8'))
        depth_msg = PointCord()
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        depth_msg.x = cord[:, 0].tolist()
        depth_msg.y = cord[:, 1].tolist()
        depth_msg.z = cord[:, 2].tolist()
        self.depth_pub.publish(depth_msg)

    def match_dots(self, dots_info):
        if dots_info != []:
            data_base = np.asarray(dots_info)
            new_pos = data_base[:, 0:2]
            new_area = data_base[:, 2]
            self.prv_pos = self.pos.copy()
            self.pos[:, :2] = self.init_point_pos.copy()
            self.pos[:, 2] = 0
            self.info[:, :] = 0
            # 遍历所有当前帧的激活点
            for j in range(new_pos.shape[0]):
                cord = new_pos[j]
                feature = new_area[j]
                # print("cord: ", cord)
                # 与上一帧点阵所有点计算距离，并找出距离最短的
                distances = (cord[0]-self.prv_pos[:,0])**2 + (cord[1]-self.prv_pos[:,1])**2
                min_index = np.argmin(distances)
                # index_sorted = np.argsort(distances)
                if self.prv_pos[min_index, 2] == 0:
                    # 如果上一帧中距离最近的点，面积是0（未激活），则寻找已激活点中最近的  
                    changed_pts = self.prv_pos[np.where(self.prv_pos[:,2]==1)]
                    if len(changed_pts) == 0: 
                        # 如果上一帧全部点均未激活，那么直接用现在的min_index
                        pass
                    else:
                        # 如果上一帧有激活的点，那么找出距离最近的
                        dises = (cord[0]-changed_pts[:,0])**2 + (cord[1]-changed_pts[:,1])**2
                        key_index = np.argmin(dises)
                        if dises[key_index] > 40*40:
                            # 如果距离太远，则选择未激活点
                            # print("1111")
                            pass
                        else:
                            # 计算该激活点的偏移量
                            dx = changed_pts[key_index, 0] - self.init_point_pos[int(changed_pts[key_index, 3]), 0]
                            dy = changed_pts[key_index, 1] - self.init_point_pos[int(changed_pts[key_index, 3]), 1]
                            # 当前帧的点减去该偏移量
                            cord_modify = [cord[0]-1.0*dx, cord[1]-1.0*dy]
                            unchanged_pts = self.prv_pos[np.where(self.prv_pos[:,2]==0)]
                            # 修改后的当前坐标与上一帧未激活的点对比，找出最近的
                            # TODO 这里unchanged是否必要，可以尝试改成prv_pos
                            dises_u = (cord_modify[0]-unchanged_pts[:,0])**2 + (cord_modify[1]-unchanged_pts[:,1])**2
                            key_index = np.argmin(dises_u)
                            # print(dises_u[key_index])
                            min_index = int(unchanged_pts[key_index, 3])

                # 上一帧中距离最近的点，并且是已激活的
                else:  
                    if distances[min_index] > 50:
                        # 如果距离太大，说明误判，直接下一个点
                        print(f"Over-threshold offset to active dot on previous frame: {distances[min_index]}. Maybe due to fast change of deformation")
                        continue
                self.pos[min_index, 0:2] = cord
                self.pos[min_index, 2] = 1
                self.info[min_index, :] = feature
        # 如果当前帧没有激活点        
        else:
            # 直接初始化pos和info
            self.pos[:, 0:2] = self.init_point_pos.copy()
            self.pos[:, 2] = 0
            self.info[:, :] = 0
    
    def get_single_keypress(self):
        """
        读取单个按键输入，不回显并且不需要按回车确认。
        """
        fd = sys.stdin.fileno()  # 获取标准输入文件描述符
        old_settings = termios.tcgetattr(fd)  # 保存当前终端设置
        try:
            # 设置终端为原始模式
            tty.setraw(sys.stdin.fileno())
            # 读取一个字符
            ch = sys.stdin.read(1)
        finally:
            # 恢复之前的终端设置
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def main(args=None):
    rclpy.init(args=args)
    feature_extraction = FeatureExtraction()
    time.sleep(1)
    try:
        # if feature_extraction.calibrate:
        #     keyboard_thread = threading.Thread(target=feature_extraction.calibrate_init)
        #     keyboard_thread.start()
        rclpy.spin(feature_extraction)
    except KeyboardInterrupt:
        pass
    finally:
        feature_extraction.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

        