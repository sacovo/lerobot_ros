import launch.actions
import launch.substitutions
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    logger = launch.substitutions.LaunchConfiguration("log_level")

    config = launch.substitutions.LaunchConfiguration("config")
    # Params file is in config/params.yml

    return LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="Set the logging level for the nodes",
            ),
            launch.actions.DeclareLaunchArgument(
                "config",
                default_value="config/so101/params.yml",
                description="Path to the so101 configuration file",
            ),
            Node(
                package="so101",
                executable="leader_node",
                name="so101_leader_node",
                output="screen",
                arguments=[
                    "--ros-args",
                    "--params-file",
                    config,
                    "--log-level",
                    logger,
                ],
            ),
            Node(
                package="so101",
                executable="follower_node",
                name="so101_follower_node",
                output="screen",
                arguments=[
                    "--ros-args",
                    "--params-file",
                    config,
                    "--log-level",
                    logger,
                ],
            ),
        ]
    )
