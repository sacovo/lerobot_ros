import rclpy
from rclpy.qos import QoSReliabilityPolicy
from std_msgs.msg import Empty

from lerobot_ros.estop import ESTOP_QOS, EStopMonitor


def _spin_until(node, predicate, timeout_s=2.0):
    deadline = node.get_clock().now().nanoseconds / 1e9 + timeout_s
    while node.get_clock().now().nanoseconds / 1e9 < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False


def setup_module(_module):
    rclpy.init()


def teardown_module(_module):
    rclpy.shutdown()


def test_qos_is_reliable_depth_1_volatile():
    assert ESTOP_QOS.depth == 1
    assert ESTOP_QOS.reliability == QoSReliabilityPolicy.RELIABLE


def test_trip_and_clear_programmatic():
    node = rclpy.create_node('test_estop_programmatic')
    try:
        monitor = EStopMonitor(node)
        assert not monitor.tripped
        monitor.trip()
        assert monitor.tripped
        monitor.clear()
        assert not monitor.tripped
    finally:
        node.destroy_node()


def test_on_estop_fires_once_per_trip_not_per_message():
    node = rclpy.create_node('test_estop_callback')
    try:
        calls = []
        monitor = EStopMonitor(node, on_estop=lambda: calls.append(1))
        monitor.trip()
        monitor.trip()
        monitor.trip()
        assert len(calls) == 1
        monitor.clear()
        monitor.trip()
        assert len(calls) == 2
    finally:
        node.destroy_node()


def test_trip_and_reset_over_topics():
    node = rclpy.create_node('test_estop_topics')
    try:
        monitor = EStopMonitor(node, topic='/test_e_stop', reset_topic='/test_e_stop/reset')
        pub_trip = node.create_publisher(Empty, '/test_e_stop', ESTOP_QOS)
        pub_reset = node.create_publisher(Empty, '/test_e_stop/reset', ESTOP_QOS)

        pub_trip.publish(Empty())
        assert _spin_until(node, lambda: monitor.tripped)

        pub_reset.publish(Empty())
        assert _spin_until(node, lambda: not monitor.tripped)
    finally:
        node.destroy_node()
