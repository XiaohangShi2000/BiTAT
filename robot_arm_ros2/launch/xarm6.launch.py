from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution

def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_ip',
            default_value='10.18.18.101',
            description='IP address by which the robot can be reached.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'xarm6_ns',
            default_value='right',
            description='Namespace of the xarm node',
        )
    )
    robot_ip = LaunchConfiguration('robot_ip')
    xarm6_ns = LaunchConfiguration('xarm6_ns')
    return LaunchDescription(declared_arguments+[
        Node(
            package='leap_hand',
            executable='xarm6_node.py',
            name='xarm6_node',
            emulate_tty=True,
            output='screen',
            namespace=xarm6_ns,
            parameters=[
                {
                    'robot_ip': robot_ip,
                }
            ]
        )
    ])