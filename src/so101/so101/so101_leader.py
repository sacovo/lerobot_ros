from pathlib import Path

import rclpy
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


class ROSSO101Leader:
    def __init__(self, node: Node) -> None:
        self.node = node
        port = (
            self.node.declare_parameter("port", "/dev/ttyUSB0")
            .get_parameter_value()
            .string_value
        )
        calibration_dir = Path(
            self.node.declare_parameter("calibration_dir", "config/calibrations")
            .get_parameter_value()
            .string_value
        )
        calibrate = (
            self.node.declare_parameter("calibrate", False)
            .get_parameter_value()
            .bool_value
        )

        teleoperator_id = (
            self.node.declare_parameter("teleoperator_id", "so101_leader")
            .get_parameter_value()
            .string_value
        )

        config = SO101LeaderConfig(
            port=port,
            use_degrees=False,
            id=teleoperator_id,
            calibration_dir=calibration_dir,
        )

        self.leader = SO101Leader(config=config)

        self.node.get_logger().info(f"Connecting to SO101 Leader on port {port}...")
        self.leader.connect(calibrate=calibrate)

        self.node.get_logger().info("SO101 Leader connected.")

        self.frequency = (
            self.node.declare_parameter("frequency", 10.0)
            .get_parameter_value()
            .double_value
        )
        self._timer = self.node.create_timer(1.0 / self.frequency, self.action_callback)

        self._active = True

        self.deactivate_service = self.node.create_service(
            Trigger,
            "deactivate_so101_leader",
            self.deactivate,
        )

        self.auto_sub = self.node.create_subscription(
            String, "/autonomy_mode", self.autonomy_mode_callback, 10
        )

        self.activate_service = self.node.create_service(
            Trigger,
            "activate_so101_leader",
            self.activate,
        )
        self.joint_state_publisher = self.node.create_publisher(
            JointState, "/so101_leader/joint_states", 10
        )
        self.auto_mode = "manual"

    def autonomy_mode_callback(self, msg: String):
        self.auto_mode = msg.data
        self.node.get_logger().warning(f"Autonomy mode set to: {self.auto_mode}")
        if self.auto_mode == "auto":
            self._active = False
        else:
            self._active = True

    def deactivate(self, request: Trigger.Request, response: Trigger.Response):
        self._active = False
        response.success = True
        response.message = "Deactivated"
        return response

    def activate(self, request: Trigger.Request, response: Trigger.Response):
        self._active = True
        response.success = True
        response.message = "Activated"
        return response

    def action_callback(self):
        if not self._active:
            return
        action = self.leader.get_action()
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list([key.replace(".pos", "") for key in action.keys()])
        msg.position = list(action.values())

        self.node.get_logger().debug(f"Publishing joint states: {msg}")
        self.joint_state_publisher.publish(msg)

    def close(self):
        self.leader.disconnect()


def main():
    rclpy.init()
    node = rclpy.create_node("so101_leader_node")
    so101_leader = ROSSO101Leader(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    so101_leader.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
