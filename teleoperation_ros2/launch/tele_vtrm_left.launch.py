from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import PushRosNamespace
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    gello = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('leap_hand'), 'launch'), '/gello.launch.py']),
        launch_arguments={
            'u2d2_port': '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9HDF13-if00-port0',
            'start_pos': '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
            'gello_ns': 'left',
        }.items()
    )
    xarm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('leap_hand'), 'launch'), '/xarm6.launch.py']),
        launch_arguments={
            'robot_ip': '10.18.18.102',
            'xarm6_ns': 'left',
        }.items()
    )
    bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('leap_hand'), 'launch'), '/bridge.launch.py']),
        launch_arguments={
            'robot_ip': '10.18.18.102',
            'ns': 'left',
        }.items()
    )
    # return LaunchDescription([GroupAction(actions=[PushRosNamespace('right'), gello, xarm])])
    return LaunchDescription([gello,
                              xarm,
                            #   TimerAction(period=5.0, actions=[bridge]),
                              ])
        