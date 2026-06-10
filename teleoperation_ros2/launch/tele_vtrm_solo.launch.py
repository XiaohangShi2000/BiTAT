from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import PushRosNamespace
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    declare_arguments = []
    declare_arguments.append(
        DeclareLaunchArgument(
            'u2d2_port',
            default_value='/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VPII-if00-port0',
            description='Port of U2D2',
            )
        )
    declare_arguments.append(
        DeclareLaunchArgument(
            'start_pos',
            default_value='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
            description='Start position of gello',
            )
        )
    declare_arguments.append(
        DeclareLaunchArgument(
            'robot_ip',
            default_value='10.18.18.101',
            description='IP of robot',
            )
        )
    declare_arguments.append(
        DeclareLaunchArgument(
            'ns',
            default_value='right',
            description='Namespace of system',
            )
        )
    port = LaunchConfiguration('u2d2_port')
    pos = LaunchConfiguration('start_pos')
    ip = LaunchConfiguration('robot_ip')
    ns = LaunchConfiguration('ns')

    gello = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('leap_hand'), 'launch'), '/gello.launch.py']),
        launch_arguments={
            'u2d2_port': port,
            'start_pos': pos,
            'gello_ns': ns,
            }.items()
        )
    xarm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('leap_hand'), 'launch'), '/xarm6.launch.py']),
        launch_arguments={
            'robot_ip': ip,
            'xarm6_ns': ns,
            }.items()
        )
    # bridge = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('leap_hand'), 'launch'), '/bridge.launch.py']),
    #     launch_arguments={
    #         'robot_ip': ip,
    #         'ns': ns,
    #     }.items()
    # )
    # return LaunchDescription([GroupAction(actions=[PushRosNamespace('right'), gello, xarm])])
    return LaunchDescription(declare_arguments+[
                            gello,
                            xarm,
                            # TimerAction(period=5.0, actions=[bridge]),
                            ])
        