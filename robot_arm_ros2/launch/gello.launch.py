from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution

def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'u2d2_port',
            default_value='/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VPII-if00-port0',
            description='Port of the U2D2',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'start_pos',
            default_value='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
            description='Starting position of the gello arm',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'gello_ns',
            default_value='right',
            description='Namespace of the gello node',
        )
    )
    u2d2_port = LaunchConfiguration('u2d2_port')
    start_pos = LaunchConfiguration('start_pos')
    gello_ns = LaunchConfiguration('gello_ns')
    return LaunchDescription(declared_arguments+[
        Node(
            package='leap_hand',
            executable='gello_node.py',
            name='gello_node',
            emulate_tty=True,
            output='screen',
            namespace=gello_ns,
            parameters=[
                {
                    'u2d2_prot': u2d2_port,
                    'start_pos': start_pos,
                }
            ]
        )
    ])