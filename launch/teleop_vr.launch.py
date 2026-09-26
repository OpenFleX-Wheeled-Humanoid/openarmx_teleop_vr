#!/usr/bin/env python3
"""Launch file for OpenArmX VR teleoperation with Pinocchio IK."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description."""

    # Declare arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('openarmx_teleop_vr'),
            'config',
            'teleop_params.yaml'
        ]),
        description='Path to configuration file'
    )

    urdf_path_arg = DeclareLaunchArgument(
        'urdf_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('openarmx_description'),
            'urdf',
            'robot',
            'openarmx_robot.urdf'
        ]),
        description='Path to robot URDF file'
    )

    control_rate_arg = DeclareLaunchArgument('control_rate', default_value='100.0')
    grip_threshold_arg = DeclareLaunchArgument('grip_threshold', default_value='0.5')
    resync_threshold_deg_arg = DeclareLaunchArgument('resync_threshold_deg', default_value='5.0')
    ik_iterations_arg = DeclareLaunchArgument('ik_iterations', default_value='3')
    sync_joint_states_each_cycle_arg = DeclareLaunchArgument(
        'sync_joint_states_each_cycle', default_value='true'
    )
    max_step_deg_joint1_2_arg = DeclareLaunchArgument('max_step_deg_joint1_2', default_value='8.0')
    max_step_deg_joint3_4_arg = DeclareLaunchArgument('max_step_deg_joint3_4', default_value='5.0')
    max_step_deg_joint5_7_arg = DeclareLaunchArgument('max_step_deg_joint5_7', default_value='5.0')
    threshold_deg_joint1_2_arg = DeclareLaunchArgument('threshold_deg_joint1_2', default_value='4.0')
    threshold_deg_joint3_4_arg = DeclareLaunchArgument('threshold_deg_joint3_4', default_value='12.0')
    threshold_deg_joint5_7_arg = DeclareLaunchArgument('threshold_deg_joint5_7', default_value='12.0')
    home_step_deg_arg = DeclareLaunchArgument('home_step_deg', default_value='8.0')

    left_pose_topic_arg = DeclareLaunchArgument(
        'left_pose_topic', default_value='/pico_left_controller/pose'
    )
    right_pose_topic_arg = DeclareLaunchArgument(
        'right_pose_topic', default_value='/pico_right_controller/pose'
    )
    left_grip_topic_arg = DeclareLaunchArgument(
        'left_grip_topic', default_value='/pico_left_controller/grip'
    )
    right_grip_topic_arg = DeclareLaunchArgument(
        'right_grip_topic', default_value='/pico_right_controller/grip'
    )
    left_trigger_topic_arg = DeclareLaunchArgument(
        'left_trigger_topic', default_value='/pico_left_controller/trigger'
    )
    right_trigger_topic_arg = DeclareLaunchArgument(
        'right_trigger_topic', default_value='/pico_right_controller/trigger'
    )
    # Teleoperation node
    teleop_node = Node(
        package='openarmx_teleop_vr',
        executable='openarmx_teleop_vr_node',
        name='openarmx_teleop_vr_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'urdf_path': LaunchConfiguration('urdf_path'),
                'control_rate': ParameterValue(LaunchConfiguration('control_rate'), value_type=float),
                'grip_threshold': ParameterValue(LaunchConfiguration('grip_threshold'), value_type=float),
                'resync_threshold_deg': ParameterValue(
                    LaunchConfiguration('resync_threshold_deg'), value_type=float
                ),
                'ik_iterations': ParameterValue(LaunchConfiguration('ik_iterations'), value_type=int),
                'sync_joint_states_each_cycle': ParameterValue(
                    LaunchConfiguration('sync_joint_states_each_cycle'), value_type=bool
                ),
                'max_step_deg_joint1_2': ParameterValue(
                    LaunchConfiguration('max_step_deg_joint1_2'), value_type=float
                ),
                'max_step_deg_joint3_4': ParameterValue(
                    LaunchConfiguration('max_step_deg_joint3_4'), value_type=float
                ),
                'max_step_deg_joint5_7': ParameterValue(
                    LaunchConfiguration('max_step_deg_joint5_7'), value_type=float
                ),
                'threshold_deg_joint1_2': ParameterValue(
                    LaunchConfiguration('threshold_deg_joint1_2'), value_type=float
                ),
                'threshold_deg_joint3_4': ParameterValue(
                    LaunchConfiguration('threshold_deg_joint3_4'), value_type=float
                ),
                'threshold_deg_joint5_7': ParameterValue(
                    LaunchConfiguration('threshold_deg_joint5_7'), value_type=float
                ),
                'home_step_deg': ParameterValue(
                    LaunchConfiguration('home_step_deg'), value_type=float
                ),
                'left_pose_topic': LaunchConfiguration('left_pose_topic'),
                'right_pose_topic': LaunchConfiguration('right_pose_topic'),
                'left_grip_topic': LaunchConfiguration('left_grip_topic'),
                'right_grip_topic': LaunchConfiguration('right_grip_topic'),
                'left_trigger_topic': LaunchConfiguration('left_trigger_topic'),
                'right_trigger_topic': LaunchConfiguration('right_trigger_topic'),
            }
        ]
    )

    return LaunchDescription([
        config_file_arg,
        urdf_path_arg,
        control_rate_arg,
        grip_threshold_arg,
        resync_threshold_deg_arg,
        ik_iterations_arg,
        sync_joint_states_each_cycle_arg,
        max_step_deg_joint1_2_arg,
        max_step_deg_joint3_4_arg,
        max_step_deg_joint5_7_arg,
        threshold_deg_joint1_2_arg,
        threshold_deg_joint3_4_arg,
        threshold_deg_joint5_7_arg,
        home_step_deg_arg,
        left_pose_topic_arg,
        right_pose_topic_arg,
        left_grip_topic_arg,
        right_grip_topic_arg,
        left_trigger_topic_arg,
        right_trigger_topic_arg,
        teleop_node
    ])
