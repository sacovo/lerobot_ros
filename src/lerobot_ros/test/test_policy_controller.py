"""Tests for policy_controller.py -- the node that actuates the manipulator.

Everything here runs on CPU against a stub policy. The point is not inference
(that is what test_trt_policies.py is for) but the logic wrapped around it: the
goal lifecycle, and above all the two gates that decide whether a joint command
ever reaches the arm -- the operator heartbeat and the latching e-stop. Those
are the parts where a regression is a moving robot rather than a failed test,
and until this file they had no coverage at all.

Each test gets its own ROS namespace so the controllers, whose topic and action
names are otherwise fixed, cannot see each other's traffic across tests.
"""

import threading
import time

import pytest
import rclpy
import torch
from lerobot_interfaces.action import RunPolicy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from std_msgs.msg import Empty, Float64MultiArray

from lerobot_ros.config import PolicyConfig, parse_config
from lerobot_ros.policy_controller import HEARTBEAT_QOS, PolicyController

ACTION_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
CHUNK = 20
POLICY = "stub"


def setup_module(_module):
    rclpy.init()


def teardown_module(_module):
    rclpy.shutdown()


class _StubPolicy:
    """Stands in for a PreTrainedPolicy. Returns a distinguishable ramp so a
    test can tell published actions apart from zero-filled padding."""

    def __init__(self):
        self.calls = 0

    def predict_action_chunk(self, _observation):
        self.calls += 1
        base = torch.arange(CHUNK, dtype=torch.float32).unsqueeze(1)
        return (base + torch.arange(len(ACTION_NAMES), dtype=torch.float32)).unsqueeze(0)


def _config(ns: str):
    """Minimal ROSFeatureConfig: one observation topic, one action topic."""
    return parse_config(
        {
            "fps": 20,
            "topics": {
                f"{ns}/joint_states": {
                    "msg_type": "JointState",
                    "tag": "observation",
                    "joints": ACTION_NAMES,
                    "position": True,
                    "velocity": False,
                    "effort": False,
                },
                f"{ns}/position_controller/commands": {
                    "msg_type": "Float64MultiArray",
                    "tag": "action",
                    "names": ACTION_NAMES,
                },
            },
            "policies": {},
        }
    )


class _Harness:
    """A spun-up PolicyController plus the client side of its interfaces."""

    def __init__(self, ns: str, max_episode_length_s: float = 5.0):
        self.ns = ns
        overrides = [
            # Keep the e-stop off the global topic: a real /e_stop from anything
            # else on the machine would otherwise reach into these tests.
            Parameter("e_stop_topic", value=f"{ns}/e_stop"),
            Parameter("e_stop_reset_topic", value=f"{ns}/e_stop/reset"),
            Parameter("activate_controller", value=""),
        ]
        self.node = rclpy.create_node(
            "policy_controller", namespace=ns, parameter_overrides=overrides
        )
        self.sub_node = rclpy.create_node("subscriber_node", namespace=ns)
        self.config = _config(ns)
        self.controller = PolicyController(self.node, self.config, self.sub_node)

        # Register the stub as a loaded policy. Identity pre/post-processors:
        # normalization is lerobot's business, not this node's.
        self.policy = _StubPolicy()
        self.controller.policies[POLICY] = (lambda x: x, self.policy, lambda x: x)
        self.config.policies[POLICY] = PolicyConfig(
            pretrained_name_or_path="",
            device="cpu",
            action_queue_size=25,
            action_smoothing_beta=0.5,
            max_episode_length_s=max_episode_length_s,
        )

        self.client_node = rclpy.create_node("test_client", namespace=ns)
        self.commands = []
        self.client_node.create_subscription(
            Float64MultiArray,
            f"{ns}/position_controller/commands",
            lambda msg: self.commands.append(list(msg.data)),
            10,
        )
        self.heartbeat_pub = self.client_node.create_publisher(
            Empty, f"{ns}/policy_control/heartbeat", HEARTBEAT_QOS
        )
        self.estop_pub = self.client_node.create_publisher(
            Empty, f"{ns}/e_stop", 10
        )
        self.action_client = ActionClient(self.client_node, RunPolicy, f"{ns}/run_policy")

        self.executor = MultiThreadedExecutor()
        for node in (self.node, self.sub_node, self.client_node):
            self.executor.add_node(node)
        self._spin = threading.Thread(target=self.executor.spin, daemon=True)
        self._spin.start()

    # -- driving --------------------------------------------------------
    def feed_observations(self, count=3):
        """Push frames straight onto the controller's queue.

        Deliberately not published as ROS messages: the subscriber and its
        frame collector have their own timing, and this test is about what the
        controller does with a frame once it has one.
        """
        from lerobot_ros.core.frame_assembler import FrameAssembler

        assembler = FrameAssembler(self.config.topics)
        for _ in range(count):
            frame = assembler.assemble({})
            frame["task"] = ""
            frame["robot_type"] = ""
            self.controller.observation_queue.put((frame, time.time()))

    def beat(self, duration_s, rate_hz=20.0):
        deadline = time.time() + duration_s
        while time.time() < deadline:
            self.heartbeat_pub.publish(Empty())
            time.sleep(1.0 / rate_hz)

    def send_goal(self, policy_name=POLICY, task="t", timeout_s=5.0):
        assert self.action_client.wait_for_server(timeout_sec=timeout_s), "no action server"
        future = self.action_client.send_goal_async(
            RunPolicy.Goal(policy_name=policy_name, task=task)
        )
        deadline = time.time() + timeout_s
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        assert future.done(), "goal request never resolved"
        return future.result()

    def result_of(self, handle, timeout_s=20.0):
        future = handle.get_result_async()
        deadline = time.time() + timeout_s
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        assert future.done(), "goal never produced a result"
        return future.result()

    def close(self):
        # Order matters: stop the controller's own threads, then stop the
        # executor and *wait for the spin thread to leave* before destroying
        # any node. Destroying a node out from under a live spin raises
        # InvalidHandle from inside that thread -- harmless, but it buries the
        # real test output in tracebacks.
        # Drain any still-running goal first. execute_callback only returns on
        # cancel, e-stop or timeout, and tearing down underneath a live goal
        # leaves its result future to be cancelled against handles that are
        # already going away. The e-stop is the node's own documented abort
        # path, so use it rather than reaching into the goal handle.
        if self.controller._goal_active:
            self.controller._estop.trip()
            deadline = time.time() + 3.0
            while self.controller._goal_active and time.time() < deadline:
                time.sleep(0.05)

        self.controller.running = False
        self.controller.collect_frames = False
        self.controller.convertor.running = False
        self.action_client.destroy()
        self.executor.shutdown(timeout_sec=2.0)
        self._spin.join(timeout=5.0)
        for node in (self.client_node, self.sub_node, self.node):
            node.destroy_node()


@pytest.fixture
def harness(request):
    made = []

    def _make(max_episode_length_s=5.0):
        h = _Harness(f"/{request.node.name.replace('[', '_').replace(']', '')}",
                     max_episode_length_s)
        made.append(h)
        return h

    yield _make
    for h in made:
        h.close()


# ---------------------------------------------------------------------------
# The safety gates
# ---------------------------------------------------------------------------

def test_no_actions_published_until_the_heartbeat_arrives(harness):
    """The heartbeat is a deadman: a running policy with a full action queue
    must stay silent until an operator publishes it, then actuate."""
    h = harness()
    handle = h.send_goal()
    assert handle.accepted

    h.feed_observations()
    time.sleep(1.0)  # >> heartbeat_timeout_s (0.5), well under pause_flush (5.0)
    assert h.commands == [], f"published {len(h.commands)} commands with no heartbeat"
    assert len(h.controller.action_queue) > 0, "nothing was queued, so this proved nothing"

    h.beat(1.0)
    time.sleep(0.2)
    assert h.commands, "no commands published even with a fresh heartbeat"


def test_estop_during_a_run_aborts_the_goal_and_drops_queued_actions(harness):
    h = harness()
    handle = h.send_goal()
    assert handle.accepted

    h.feed_observations()
    h.beat(0.5)
    assert h.commands, "precondition: should be actuating before the e-stop"

    h.estop_pub.publish(Empty())
    result = h.result_of(handle)
    assert result.result.message == "e-stop"
    assert result.result.success is False
    assert len(h.controller.action_queue) == 0, "queued actions survived an e-stop"

    # And it latches: a new goal is refused until reset.
    assert h.controller._estop.tripped
    assert not h.send_goal().accepted


def test_goal_rejected_while_estop_latched(harness):
    h = harness()
    h.controller._estop.trip()
    assert not h.send_goal().accepted


def test_second_goal_rejected_while_one_is_running(harness):
    h = harness()
    assert h.send_goal().accepted
    assert not h.send_goal().accepted, "controller accepted a concurrent goal"


# ---------------------------------------------------------------------------
# Goal lifecycle
# ---------------------------------------------------------------------------

def test_unknown_policy_aborts_with_a_named_reason(harness):
    h = harness()
    handle = h.send_goal(policy_name="does-not-exist")
    assert handle.accepted  # accepted, then aborted by execute_callback
    result = h.result_of(handle)
    assert result.result.success is False
    assert "unknown policy" in result.result.message


def test_goal_succeeds_with_timeout_at_max_episode_length(harness):
    """No progress regressor is configured, so max_episode_length_s is the only
    thing that ends the goal -- and it must end it as a success, not a fault."""
    h = harness(max_episode_length_s=1.0)
    start = time.time()
    handle = h.send_goal()
    assert handle.accepted
    result = h.result_of(handle)
    elapsed = time.time() - start

    assert result.result.success is True
    assert result.result.message == "timeout"
    assert 0.9 <= elapsed <= 4.0, f"timeout fired after {elapsed:.2f}s"
    assert h.controller.running is False
    assert h.controller.active_policy_name is None


def test_status_is_published_and_lists_the_loaded_policy(harness):
    h = harness()
    deadline = time.time() + 3.0
    while time.time() < deadline and not h.controller.status_pub:
        time.sleep(0.05)
    h.controller.publish_status()
    time.sleep(0.3)
    assert POLICY in h.controller.policies


def test_stop_clears_the_queue_so_a_resume_cannot_replay_it(harness):
    """_stop() must leave nothing queued -- a resume that replayed a stale
    trajectory would move the arm along a plan made for an older observation.

    Also pins the behaviour of stopping *underneath* a live goal: the goal has
    to end (as "stopped"), because a goal that never resolves leaves
    _goal_active True and every later goal is rejected as busy.
    """
    h = harness()
    handle = h.send_goal()
    assert handle.accepted
    h.feed_observations()
    time.sleep(0.5)
    assert len(h.controller.action_queue) > 0

    h.controller._stop("test")
    assert len(h.controller.action_queue) == 0
    assert h.controller.running is False

    result = h.result_of(handle)
    assert result.result.success is False
    assert result.result.message == "stopped"
    assert h.controller._goal_active is False


# ---------------------------------------------------------------------------
# Pure logic -- no spinning required
# ---------------------------------------------------------------------------

def test_needs_frame_skips_ticks_no_inference_will_consume(harness):
    """The gate exists to keep three JPEG decodes off ticks whose frame would
    be assembled and then dropped; if it stops returning False when the queue
    is deep, that saving silently disappears."""
    h = harness()
    controller = h.controller

    assert controller.needs_frame() is False, "no active policy: must not convert"

    controller._active_policy = POLICY
    controller.collect_frames = True
    controller.action_queue.clear()
    assert controller.needs_frame() is True, "empty queue: the next frame is needed"

    size = h.config.policies[POLICY].action_queue_size
    for _ in range(size + 1):
        controller.action_queue.append((torch.zeros(len(ACTION_NAMES)), 0.0))
    assert controller.needs_frame() is False, "deep queue: this tick is wasted work"

    controller.collect_frames = False
    assert controller.needs_frame() is False


def test_action_weights_fade_from_old_to_new(harness):
    """Blending weights the *old* trajectory most at the head of the queue and
    least at the tail, so a fresh chunk takes over smoothly instead of stepping."""
    h = harness()
    h.controller.active_policy_name = POLICY
    weights = h.controller.action_weights

    assert len(weights) == h.config.policies[POLICY].action_queue_size
    assert weights[0] == pytest.approx(1.0)
    assert weights[-1] < weights[0]
    assert all(b <= a + 1e-9 for a, b in zip(weights, weights[1:])), "not monotonic"
    assert all(0.0 <= w <= 1.0 for w in weights)
