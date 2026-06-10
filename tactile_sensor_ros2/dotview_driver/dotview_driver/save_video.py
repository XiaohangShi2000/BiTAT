import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class SaveRgbVideoNode(Node):
    def __init__(self):
        super().__init__('save_rgb_video_node')
        self.subscription = self.create_subscription(
            Image, 
            '/head/d435i/color/image_raw', # 确保主题名称正确
            self.listener_callback, 
            10)
        self.subscription  # prevent unused variable warning

        self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480)) # 调整大小以匹配你的摄像头分辨率
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.out.write(cv_image)
        self.get_logger().info("Writing frame...")

    def __del__(self):
        self.out.release()
        print("Video saved and released.")

def main(args=None):
    rclpy.init(args=args)
    node = SaveRgbVideoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()