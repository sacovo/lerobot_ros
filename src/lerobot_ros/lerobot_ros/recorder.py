import os
import sys
import threading
import traceback
from typing import Optional

import rclpy
import rclpy.executors
import rclpy.logging
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot_interfaces.srv import EndEpisode, NewDataset, StartEpisode
from rclpy.node import Node
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from tqdm import tqdm

from .config import ROSFeatureConfig, load_toml_dict, parse_config
from .subscriber import Ros2Feature


class Recorder:
    def __init__(self, node: Node, config: ROSFeatureConfig, subscriber_node: Node):
        self.visualize = config.visualize

        self.config = config
        self.node = node

        self.recording_lock = threading.Lock()
        self._recording = False

        self.dataset: Optional[LeRobotDataset] = None

        self.episodes_lock = threading.Lock()
        self.episodes = []

        self.out_frames_lock = threading.Lock()
        self.out_frames = []

        self._last_call = None

        self.dataset_root = config.dataset_root
        self.fps = config.fps
        self.tolerance_s = config.tolerance_s

        self.node.create_service(NewDataset, "new_dataset", self.new_dataset_service)
        self.node.create_service(
            StartEpisode, "start_episode", self.start_episode_service
        )
        self.node.create_service(EndEpisode, "end_episode", self.end_episode_service)
        self.node.create_service(Trigger, "store_episodes", self.store_episodes_service)

        self.frame_publisher = node.create_publisher(Int32, "frame", 10)
        self.episode_publisher = node.create_publisher(Int32, "episode", 10)

        self.convertor = Ros2Feature(
            subscriber_node,
            config.topics,
            fps=config.fps,
            rerun_remote=config.rerrun_remote,
            visualize=self.visualize,
        )
        self.convertor.register_frame_callback(self._timer_callback)
        self.convertor.setup_subscribers()
        self.convertor.running = True
        self.last_t = 0.0

        self._background_threads = []

    @property
    def recording(self):
        with self.recording_lock:
            return self._recording

    @recording.setter
    def recording(self, value: bool):
        with self.recording_lock:
            self._recording = value

    def _timer_callback(self, frame, t):
        if self.last_t != 0.0:
            delta = t - self.last_t
            error = 1.0 / self.fps - delta
            if abs(error) > 1e-5:
                self.node.get_logger().warning(
                    f"Received frame at time {t - self.last_t:.6f}s"
                )
        self.last_t = t
        if not self.recording:
            return

        with self.out_frames_lock:
            assert self.out_frames is not None
            self.out_frames.append((frame, self.task, t))

    def _convert_feature_name(self, topic_name: str) -> str:
        return topic_name.replace("/", "", 1).replace("/", ".")

    def new_dataset_service(
        self, request: NewDataset.Request, response: NewDataset.Response
    ):
        name = request.repo_id
        try:
            self.new_dataset(name)
            response.success = True
            self.node.get_logger().info(f"Created new dataset: {name}")
        except Exception as e:
            response.success = False
            response.msg = str(e)
            self.node.get_logger().error(f"Failed to create dataset {name}: {e}")
        return response

    def start_episode_service(
        self, request: StartEpisode.Request, response: StartEpisode.Response
    ):
        response.episode_id = self.start_episode(request.task)
        return response

    def start_episode(self, task: str):
        """
        Start a new episode with the given task.
        This will reset the recording state and prepare the dataset for a new episode.
        """
        if self.dataset is None:
            raise RuntimeError("Dataset is not initialized. Call new_dataset first.")

        self.task = task

        with self.out_frames_lock:
            self.out_frames = []

        self.recording = True

        self.node.get_logger().info(f"Started episode for task: {self.task}")
        if self.dataset is not None:
            stored = self.dataset.num_episodes
            return stored + len(self.episodes) + 1
        # else:
        return len(self.episodes) + 1  # Return the episode ID as the next index

    def end_episode_service(
        self, request: EndEpisode.Request, response: EndEpisode.Response
    ):
        response.frames = self.end_episode(request.discard)
        return response

    def end_episode(self, discard: bool = False) -> int:
        if not self.recording:
            self.node.get_logger().warn("No episode is currently being recorded.")
            return 0
        self.recording = False

        with self.out_frames_lock:
            out_frames = self.out_frames
            self.out_frames = None

        l = len(out_frames)
        if not discard:
            with self.episodes_lock:
                self.episodes.append(out_frames)
            self.node.get_logger().info(
                f"Ended episode with {l} frames (total: {len(self.episodes)})."
            )
        else:
            self.node.get_logger().info(f"Discarded episode with {l} frames.")
        return l

    def store_episodes_service(
        self, request: Trigger.Request, response: Trigger.Response
    ):
        try:
            self.store_episodes()
            response.success = True
            response.message = "Episodes stored successfully."
        except Exception as e:
            response.message = str(e)
            response.success = False
        return response

    def store_episodes(self):
        if self.dataset is None:
            self.node.get_logger().info(
                "No dataset initialized, skipping episode storage."
            )
            return

        with self.episodes_lock:
            episodes = self.episodes
            self.episodes = []

        thread = threading.Thread(
            target=self.store_thread, args=(episodes,), daemon=False
        )
        thread.start()
        self._background_threads.append(thread)

    def store_thread(self, episodes):
        if self.dataset is None:
            self.node.get_logger().info(
                "No dataset initialized, skipping episode storage."
            )
            return
        while len(episodes) > 0:
            episode = episodes.pop(0)
            self.node.get_logger().info(f"Storing episode with {len(episode)} frames.")
            frame0 = episode[0]
            t0 = frame0[2]
            for i, frame in enumerate(tqdm(episode, desc="Frame")):
                try:
                    frame, task, t = frame
                    frame["task"] = task
                    self.node.get_logger().debug(
                        f"Storing frame {i} at time {t - t0:.5f}s"
                    )

                    self.dataset.add_frame(frame)
                except Exception as e:
                    self.node.get_logger().error(f"Failed to add frame: {e}")
                    continue
            try:
                self.dataset.save_episode()
            except ValueError as e:
                self.node.get_logger().error(f"Failed to save episode: {e}")
                traceback.print_exc()
                continue
            self.node.get_logger().info(f"Stored episode with {len(episode)} frames.")
            del episode
        self.dataset.finalize()

    def new_dataset(self, dataset_name: str):
        """
        Create a new dataset with the given name.
        """
        path = os.path.join(self.dataset_root, dataset_name)

        if self.dataset and self.dataset.root == path:
            self.node.get_logger(f"Dataset {dataset_name} is already loaded, skipping")
            return False

        if os.path.exists(path):
            dataset = LeRobotDataset(
                dataset_name, root=path, tolerance_s=self.tolerance_s
            )
            ds_features = set(dataset.features) - set(DEFAULT_FEATURES)
            features = set(self.convertor.get_feature_description())
            if ds_features != features:
                raise ValueError(
                    f"Dataset {dataset_name} already exists with different features. "
                    f"Expected: {features.keys()}, Found: {ds_features.keys()}"
                )
            self.dataset = dataset
            self.node.get_logger().info(f"Using existing dataset: {dataset_name}")
        else:
            self.dataset = LeRobotDataset.create(
                dataset_name,
                fps=self.fps,
                features=self.convertor.get_feature_description(),
                root=os.path.abspath(os.path.join(self.dataset_root, dataset_name)),
                tolerance_s=self.tolerance_s,
            )


def main():
    rclpy.init(args=sys.argv)
    executor = rclpy.executors.SingleThreadedExecutor()
    node = Node("recorder_node")
    subscriber_node = Node("subscriber_node")

    config_path = (
        node.declare_parameter(
            "config", os.getenv("CONFIG_PATH", "config/hufi_arm.toml")
        )
        .get_parameter_value()
        .string_value
    )
    config = parse_config(load_toml_dict(config_path))
    node.get_logger().info(f"Loaded config from {config_path}")
    node.get_logger().info(f"Config: {config}")

    # node.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

    recorder = Recorder(node, config, subscriber_node)

    executor.add_node(subscriber_node)
    executor.add_node(node)

    try:
        node.get_logger().info("Starting recorder...")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down recorder...")
        recorder.store_episodes()
        node.get_logger().info("Recorder shutdown complete.")
    except Exception as e:
        node.get_logger().error(f"Error occurred: {e}")

    node.destroy_node()
    recorder.convertor.node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
