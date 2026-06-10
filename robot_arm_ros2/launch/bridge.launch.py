from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

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
            'ns',
            default_value='right',
            description='Namespace of system',
        )
    )
    robot_ip = LaunchConfiguration('robot_ip')
    ns = LaunchConfiguration('ns')
    return LaunchDescription(declared_arguments+[
        Node(
            package='leap_hand',
            executable='gello_to_xarm6.py',
            name='bridge_node',
            emulate_tty=True,
            output='screen',
            namespace=ns,
            parameters=[
                {
                    'robot_ip': robot_ip,
                    'ns': ns,
                }
            ]
        )
    ])