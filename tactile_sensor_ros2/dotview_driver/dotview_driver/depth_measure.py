import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
from matplotlib import cm
# import pandas as pd
import csv
import threading
import os
from dotview_msgs.msg import PointCord

class DepthMeasureNode(Node):
    def __init__(self):
        super().__init__('depth_measure_node')
        self.subscriber = self.create_subscription(PointCord, '/sensor_03/DotView_depth', self.listener_callback, 1)
        self.subscriber
        plt.ion()
        self.fig = plt.figure()

    def listener_callback(self, msg):
        # self.get_logger().info(f'Received point coordinates: {msg.x}, {msg.y}, {msg.z}')
        x = np.array(msg.x)
        y = np.array(msg.y)
        z = np.array(msg.z)
        z[z < np.max(z) * 0.3] = 0
        xi = np.linspace(0,191,192*2)
        yi = np.linspace(0,191,192*2)
        X,Y = np.meshgrid(xi,yi)
        Z = griddata((x,y),z,(X,Y),method='cubic')
        if np.max(z) == 0:
            colors = cm.Reds(0)
        else:
            colors = cm.Reds(z / np.max(z))
        plt.clf()
        ax1 = self.fig.add_subplot(131, projection='3d')
        ax2 = self.fig.add_subplot(132, projection='3d')
        ax3 = self.fig.add_subplot(133, projection='3d')
        ax2.axis('off')
        ax3.axis('off')

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])

        ax2.view_init(90, -90)
        ax3.view_init(90, -90)

        ax1.set_xlim([0, 191])
        ax1.set_ylim([0, 191])
        ax1.set_zlim([0, 3])

        ax2.set_xlim([0, 191])
        ax2.set_ylim([0, 191])
        ax2.set_zlim([0, 3])

        ax3.set_xlim([0, 191])
        ax3.set_ylim([0, 191])
        ax3.set_zlim([0, 3])

        ax1.bar3d(x, y, 0, 15, 15, z, color=colors)
        ax2.bar3d(x, y, 0, 15, 15, z, color=colors)
        ax3.contourf(X, Y, Z, cmap='Reds')
        plt.pause(0.04)

    def cleanup(self):
        plt.ioff()
        plt.close(self.fig)
        plt.show()
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def main():
    rclpy.init()
    depth_measure_node = DepthMeasureNode()
    try:
        rclpy.spin(depth_measure_node)
    except KeyboardInterrupt:
        pass
    finally:
        depth_measure_node.cleanup()

if __name__ == '__main__':
    main()