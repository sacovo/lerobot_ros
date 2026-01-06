import threading
import time
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Any, Dict, List, Optional

import rerun as rr
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from rust_py_timer import FrameCollector

    HAS_RUST_COLLECTOR = True
except ImportError:
    HAS_RUST_COLLECTOR = False


from .config import empty_frame
from .convert import BaseTopic
from .convert.image import ImageTopic


class Ros2Feature:
    """ROS2 Feature subscriber using Rust for precise timing.

    Uses a Rust-based FrameCollector for precise frame timing while
    keeping message conversion and processing in Python.
    """

    def __init__(
        self,
        node: Node,
        topics: Dict[str, BaseTopic],
        fps: int = 30,
        rerun_remote: Optional[str] = None,
        visualize: bool = False,
        use_rust_collector: bool = True,
    ):
        self.node = node
        self.frame_callback = None

        self.rerun_remote = rerun_remote
        self.visualize = visualize
        if visualize:
            self._init_rr()

        self.msg_callback = None
        self.topics = topics
        self.subscribers = {}
        self.fps = fps

        self.t_queue: Queue[float] = Queue(maxsize=100)
        self.t_thread = threading.Thread(target=self.write_timestamps, daemon=True)

        # Use Rust collector if available and requested
        self.use_rust_collector = use_rust_collector and HAS_RUST_COLLECTOR

        if self.use_rust_collector:
            self._init_rust_collector()
        else:
            self.node.get_logger().warn(
                "Rust FrameCollector not available, falling back to Python timer."
            )
            self._init_python_collector()
        self.t_thread.start()

    def _init_rust_collector(self):
        """Initialize Rust-based frame collector for precise timing."""
        topic_names = list(self.topics.keys())
        self.collector = FrameCollector(topic_names, float(self.fps))

        # Processing queue for frames from Rust
        self.proc_queue: Queue[tuple[Dict[str, List[Any]], float]] = Queue(maxsize=20)

        self.running_lock = threading.Lock()
        self._running = True

        # Register callback from Rust to put frames on queue
        self.collector.register_callback(self._rust_frame_callback)

        # Processing thread (timing not critical)
        self.proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.proc_thread.start()

        self.node.get_logger().info("Using Rust FrameCollector for precise timing")

    def _init_python_collector(self):
        """Initialize Python-based frame collector (fallback)."""
        self.frame = empty_frame(self.topics)
        self.frame_lock = threading.Lock()
        self.proc_queue: Queue[tuple[Dict[str, List[Any]], float]] = Queue(maxsize=10)

        self.running_lock = threading.Lock()
        self._running = True

        self.proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.proc_thread.start()

        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()

        self.node.get_logger().info("Using Python timer for frame collection")

    def _rust_frame_callback(self, frame: Dict[str, List[Any]], timestamp: float):
        """Callback from Rust collector - just puts frame on queue."""
        try:
            self.proc_queue.put_nowait((frame, timestamp))
        except Exception as ex:
            self.node.get_logger().warn(f"Exception in Rust frame callback: {ex}")
            # Queue full, drop frame
            self.node.get_logger().warn("Frame queue full, dropping frame")

    def _init_rr(self, session_name: str = "lerobot_control_loop") -> None:
        rr.init(session_name)
        self.node.get_logger().info(
            f"Rerun initialized with session name: {session_name}"
        )
        if self.rerun_remote is not None:
            self.node.get_logger().info(
                f"Connecting to Rerun remote: {self.rerun_remote}"
            )
            rr.connect_grpc(self.rerun_remote)
        else:
            memory_limit = "10%"
            rr.spawn(memory_limit=memory_limit)

    @property
    def running(self):
        with self.running_lock:
            return self._running

    @running.setter
    def running(self, value: bool):
        with self.running_lock:
            self._running = value
        if not value and self.use_rust_collector:
            self.collector.stop()

    def setup_subscribers(self):
        for topic_name, topic in self.topics.items():
            self.subscribers[topic_name] = self.node.create_subscription(
                topic.msg_type(),
                topic_name,
                lambda msg, topic_name=topic_name: self._msg_callback(msg, topic_name),
                topic.qos,
            )
        self.node.get_logger().info(
            f"Subscribed to topics: {', '.join(self.topics.keys())}"
        )

    def _visualize(self, topic_name: str, topic_type: BaseTopic, tensor: torch.Tensor):
        """Visualize the message using Rerun."""

        names = topic_type.feature_description().get("names", [])
        if tensor.ndim == 1 and len(names) == tensor.shape[0]:
            for i, name in enumerate(names):
                rr.log(f"{topic_name}.{name}", rr.Scalars(tensor[i]))
        elif tensor.ndim == 3:
            rr.log(topic_name, rr.Image(tensor.numpy()))
        else:
            self.node.get_logger().warn(
                f"Could not visualize tensor: {tensor.shape} for topic {topic_name}"
            )

    def _msg_callback(self, msg, topic_name):
        self.node.get_logger().debug(f"Received message: {topic_name}")

        if self.msg_callback:
            self.msg_callback(msg, topic_name)

        if self.use_rust_collector:
            # Pass raw message to Rust collector
            self.collector.add_message(topic_name, msg)
        else:
            # Python path: convert to tensor immediately
            topic = self.topics[topic_name]

            try:
                tensor = topic.to_tensor(msg)
            except Exception as e:
                self.node.get_logger().error(
                    f"Error converting message to tensor for topic {topic_name}: {e}"
                )
                return

            if self.visualize:
                self._visualize(topic_name, topic, tensor)

            with self.frame_lock:
                self.frame[topic_name].append(tensor)

    def _get_timestamp(self):
        """
        Get the current timestamp in seconds.
        This is used to timestamp the frames in the dataset.
        """
        return time.time()

    def new_frame(self, last_frame) -> Dict[str, List[torch.Tensor]]:
        """Create an empty frame with the configured topics."""
        # keep the last value, so we always have at least one entry
        return {topic_name: values[-1:] for topic_name, values in last_frame.items()}

    def _timer_loop(self):
        """Python timer loop (only used when Rust collector is not available)."""
        timer_period = 1.0 / self.fps
        next_call = time.time() + timer_period
        while self.running:
            self._timer_callback()
            next_call += timer_period
            time.sleep(max(0, next_call - time.time()))

    def _timer_callback(self):
        """Python timer callback (only used when Rust collector is not available)."""
        now = self._get_timestamp()

        with self.frame_lock:
            frame = self.frame
            self.frame = self.new_frame(frame)
            self.proc_queue.put((frame, now))

    def write_timestamps(self):
        while self.running:
            try:
                t = self.t_queue.get(timeout=1.0)
            except QueueEmpty:
                continue
            with open("timestamps.txt", "a") as f:
                f.write(f"{t}\n")
            self.t_queue.task_done()

    def _process_loop(self):
        while self.running:
            try:
                item = self.proc_queue.get(timeout=1.0)
            except QueueEmpty:
                continue

            if not self.running:
                continue

            frame, t = item

            if self.use_rust_collector:
                # Convert raw messages to tensors
                converted_msgs = self._convert_raw_frame(frame)
            else:
                # Already tensors from Python path
                converted_msgs = frame

            converted_frame = self._convert_frame(converted_msgs)

            if self.frame_callback:
                self.frame_callback(converted_frame, t)

            self.t_queue.put(t)

            self.proc_queue.task_done()

    def _convert_raw_frame(
        self, frame: Dict[str, List[Any]]
    ) -> Dict[str, List[torch.Tensor]]:
        """Convert raw ROS messages to tensors (used with Rust collector)."""
        converted = {}
        for topic_name, messages in frame.items():
            topic = self.topics[topic_name]
            tensors = []
            for msg in messages:
                try:
                    tensor = topic.to_tensor(msg)
                    tensors.append(tensor)

                    if self.visualize:
                        self._visualize(topic_name, topic, tensor)
                except Exception as e:
                    self.node.get_logger().error(
                        f"Error converting message to tensor for topic {topic_name}: {e}"
                    )
            converted[topic_name] = tensors
        return converted

    def _convert_frame(self, frame: Dict[str, List[torch.Tensor]]):
        out_frame = {
            "action": [],
            "observation.state": [],
        }

        # average out the high frequency measurements
        for topic_name, tensors in frame.items():
            topic = self.topics[topic_name]
            if len(tensors) == 0:
                tensor = torch.zeros(
                    topic.feature_description().get("shape", (1,)),
                    dtype=torch.uint8 if topic.msg_type() == Image else torch.float32,
                )
                self.node.get_logger().info(
                    f"Topic {topic_name} has no data, using zero tensor: {tensor.shape}"
                )
            else:
                tensor = tensors[-1]

            if isinstance(topic, ImageTopic):
                out_frame[f"observation.images.{topic.key}"] = tensor
            else:
                key = "action" if topic.is_action else "observation.state"
                out_frame[key].append(tensor)

        out_frame["action"] = torch.cat(out_frame["action"], dim=0)
        if len(out_frame["observation.state"]) != 0:
            out_frame["observation.state"] = torch.cat(
                out_frame["observation.state"], dim=0
            )
        else:
            del out_frame["observation.state"]

        return out_frame

    def register_frame_callback(self, callback):
        """
        Register a callback to be called with the current frame.
        The callback should accept a single argument: the current frame.
        """
        self.frame_callback = callback

    def register_msg_callback(self, callback):
        """
        Register a callback to be called with each message received.
        The callback should accept two arguments: the message and the topic name.
        """
        self.msg_callback = callback

    def get_feature_description(self) -> Dict[str, Dict[str, str]]:
        """
        Get the feature description for the dataset.
        Returns a dictionary mapping topic names to their feature descriptions.
        """
        feature_description = {
            "action": {"dtype": "float32", "shape": (0,), "names": []},
            "observation.state": {"dtype": "float32", "shape": (0,), "names": []},
        }
        for topic in self.topics.values():
            if isinstance(topic, ImageTopic):
                feature_description[f"observation.images.{topic.key}"] = {
                    "dtype": "video",
                    "shape": (topic.height, topic.width, topic.channels),
                    "names": ["height", "width", "channels"],
                }
                continue
            key = "action" if topic.is_action else "observation.state"
            combined_feature = feature_description[key]

            feature = topic.feature_description()
            size = combined_feature.get("shape", (0,))[0]
            feature = topic.feature_description()
            size += feature.get("shape", (0,))[0]

            combined_feature["shape"] = (size,)
            combined_feature["names"] = combined_feature["names"] + feature["names"]

        if feature_description["observation.state"]["shape"] == (0,):
            del feature_description["observation.state"]
        if feature_description["action"]["shape"] == (0,):
            del feature_description["action"]

        return feature_description
