from ament_index_python.packages import get_package_share_directory
import os
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription

def generate_launch_description():
    left = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('tele_vtrm'), 'launch'), '/tele_vtrm_solo.launch.py']),
        launch_arguments={
            'u2d2_port': '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9HDF13-if00-port0',
            'start_pos': '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
            'robot_ip': '10.18.18.102',
            'ns': 'left',
        }.items()
    )
    right = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('tele_vtrm'), 'launch'), '/tele_vtrm_solo.launch.py']),
        launch_arguments={
            'u2d2_port': '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VPII-if00-port0',
            'start_pos': '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
            'robot_ip': '10.18.18.101',
            'ns': 'right',
        }.items()
    )
    # return LaunchDescription([left, TimerAction(period=3.0, actions=[right])])
    return LaunchDescription([left, right])