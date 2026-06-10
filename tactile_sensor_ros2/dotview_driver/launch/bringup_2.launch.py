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
            'config_file_01',
            default_value=os.path.join(get_package_share_directory('dotview_driver'), 'config', 'sensor_01.yaml'),
            description='Full path to the sensor configuration file.',
        ),
        DeclareLaunchArgument(
            'sensor_ns_01',
            default_value='sensor_01',
            description='Namespace of the sensor node',
        ),
        DeclareLaunchArgument(
            'config_file_02',
            default_value=os.path.join(get_package_share_directory('dotview_driver'), 'config', 'sensor_02.yaml'),
            description='Full path to the sensor configuration file.',
        ),
        DeclareLaunchArgument(
            'sensor_ns_02',
            default_value='sensor_02',
            description='Namespace of the sensor node',
        ),
        DeclareLaunchArgument(
            'enable_feature',
            default_value='false',
            description='Enable feature extraction.',
        ),
    ])
    
    config_1 = LaunchConfiguration('config_file_01')
    sensor_ns_1 = LaunchConfiguration('sensor_ns_01')
    config_2 = LaunchConfiguration('config_file_02')
    sensor_ns_2 = LaunchConfiguration('sensor_ns_02')
    enable_feature = LaunchConfiguration('enable_feature')

    sensor_node_1 = Node(
        package='dotview_driver',
        executable='sensor_stream',
        namespace=sensor_ns_1,
        name='sensor_stream',
        emulate_tty=True,
        output='screen',
        parameters=[config_1],
    )
    feature_node_1 = Node(
        package='dotview_driver',
        executable='feature_extraction',
        namespace=sensor_ns_1,
        name='feature_extraction',
        emulate_tty=True,
        output='screen',
        parameters=[config_1],
        condition=IfCondition(enable_feature),
    )
    sensor_node_2 = Node(
        package='dotview_driver',
        executable='sensor_stream',
        namespace=sensor_ns_2,
        name='sensor_stream',
        emulate_tty=True,
        output='screen',
        parameters=[config_2],
    )
    feature_node_2 = Node(
        package='dotview_driver',
        executable='feature_extraction',
        namespace=sensor_ns_2,
        name='feature_extraction',
        emulate_tty=True,
        output='screen',
        parameters=[config_2],
        condition=IfCondition(enable_feature),
    )

    return LaunchDescription(declared_arguments + [sensor_node_1, sensor_node_2, TimerAction(period=2.0, actions=[feature_node_1, feature_node_2])])