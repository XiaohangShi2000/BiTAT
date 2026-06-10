from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import PushRosNamespace
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    right = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('leap_hand'), 'launch'), '/bridge.launch.py']),
        launch_arguments={
            'robot_ip': '10.18.18.101',
            'ns': 'right',
        }.items()
    )
    left = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('leap_hand'), 'launch'), '/bridge.launch.py']),
        launch_arguments={
            'robot_ip': '10.18.18.102',
            'ns': 'left',
        }.items()
    )
    return LaunchDescription([left, right])