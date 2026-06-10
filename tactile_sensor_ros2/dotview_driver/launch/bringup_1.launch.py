from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import DeclareLaunchArgument, GroupAction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    declared_arguments = []
    declared_arguments.extend([
        DeclareLaunchArgument(
            'config_file',
            default_value=os.path.join(get_package_share_directory('dotview_driver'), 'config', 'sensor_01.yaml'),
            description='Full path to the sensor configuration file.',
        ),
        DeclareLaunchArgument(
            'sensor_ns',
            default_value='sensor_01',
            description='Namespace of the sensor node',
        ),
        DeclareLaunchArgument(
            'enable_feature',
            default_value='true',
            description='Enable feature extraction.',
        )
    ])
    config = LaunchConfiguration('config_file')
    sensor_ns = LaunchConfiguration('sensor_ns')
    enable_feature = LaunchConfiguration('enable_feature')
    sensor_node = Node(
        package='dotview_driver',
        executable='sensor_stream',
        namespace=sensor_ns,
        name='sensor_stream',
        emulate_tty=True,
        output='screen',
        parameters=[config],
    )
    feature_node = Node(
        package='dotview_driver',
        executable='feature_extraction',
        namespace=sensor_ns,
        name='feature_extraction',
        emulate_tty=True,
        output='screen',
        parameters=[config],
        condition=IfCondition(enable_feature),
    )

    return LaunchDescription(declared_arguments+[sensor_node,
        TimerAction(period=2.0, actions=[feature_node]),
    ])
