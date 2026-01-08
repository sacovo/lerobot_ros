from pathlib import Path
from typing import cast

import rclpy
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ROSSO101Follower:
    def __init__(self, node: Node) -> None:
        self.node = node
        port = (
            self.node.declare_parameter("port", "/dev/ttyUSB1")
            .get_parameter_value()
            .string_value
        )
        calibration_dir = Path(
            self.node.declare_parameter("calibration_dir", "./calibration")
            .get_parameter_value()
            .string_value
        )

        calibrate = (
            self.node.declare_parameter("calibrate", False)
            .get_parameter_value()
            .bool_value
        )

        self.joint_names = cast(
            list[str],
            self.node.declare_parameter(
                "joint_names",
                [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ],
            )
            .get_parameter_value()
            .string_array_value,
        )

        robot_id = (
            self.node.declare_parameter("robot_id", "so101_follower")
            .get_parameter_value()
            .string_value
        )

        config = SO101FollowerConfig(
            port=port,
            use_degrees=False,
            calibration_dir=calibration_dir,
            id=robot_id,
        )

        self.follower = SO101Follower(config=config)

        self.node.get_logger().info(f"Connecting to SO101 Follower on port {port}...")

        self.follower.connect(calibrate=calibrate)

        self.node.get_logger().info("SO101 Follower connected.")

        self.joint_state_subscription = self.node.create_subscription(
            JointState,
            "so101_leader/joint_states",
            self.joint_state_callback,
            10,
        )
        self.joint_state_publisher = self.node.create_publisher(
            JointState,
            "~/joint_states",
            10,
        )
        freq = (
            self.node.declare_parameter("frequency", 30.0)
            .get_parameter_value()
            .double_value
        )
        self.joint_publisher_timer = self.node.create_timer(
            1.0 / freq, self.publish_joint_states
        )

    def publish_joint_states(self):
        observation = self.follower.get_observation()
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list([key.replace(".pos", "") for key in observation.keys()])
        msg.position = list(observation.values())

        self.node.get_logger().info(f"Publishing joint states: {msg}")
        self.joint_state_publisher.publish(msg)

    def joint_state_callback(self, msg: JointState):
        joint_action = {}
        for name, position in zip(msg.name, msg.position):
            if name in self.follower.bus.motors:
                joint_action[f"{name}.pos"] = position
            else:
                self.node.get_logger().warning(f"Unknown joint name: {name}")

        self.send_action(joint_action)

    def send_action(self, action):
        self.follower.send_action(action)

    def close(self):
        self.follower.disconnect()


def main():
    rclpy.init()
    node = rclpy.create_node("so101_follower_node")
    so101_follower = ROSSO101Follower(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    so101_follower.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
