# LeRobot ROS Package Overview

This package (`lerobot_ros`) bridges the [LeRobot](https://github.com/huggingface/lerobot) imitation learning framework and **ROS 2** (Jazzy/Ubuntu 24.04). It provides nodes and tools to record training datasets from ROS topics, train and evaluate policies, package them with TensorRT for target deployment, and run real-time inference on the robot.

---

## File Structure & Map

* **`lerobot_ros` Package Root**: [`/workspaces/ros-fhnw-autonomy/src/lerobot_ros`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros)
  * **Configuration Files**: [`config/`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/config/) — Contains TOML and YAML parameters (e.g., [so101.toml](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/config/so101/so101.toml)).
  * **Documentation**: [`docs/`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/docs/) — Additional guides like [bag_to_dataset_tool.md](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/docs/bag_to_dataset_tool.md).
  * **ROS 2 Custom Interfaces**: [`src/lerobot_interfaces/`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_interfaces/) — Action, service, and message definitions.
  * **Main Python Package**: [`src/lerobot_ros/lerobot_ros/`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/)
    * [`config.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/config.py) — Config parser mapping ROS message formats.
    * [`subscriber.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/subscriber.py) — Subscribes to inputs and compiles frames.
    * [`recorder.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/recorder.py) — Real-time dataset recorder node.
    * [`policy_controller.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/policy_controller.py) — Real-time model inference and publisher node.
    * [`episode_tracker.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/episode_tracker.py) — Progress tracking network (HL-Gauss).
    * [`replay.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/replay.py) — Plays back recorded actions back onto ROS topics.
    * [`bag_to_dataset.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/bag_to_dataset.py) — Offline ROS bag to LeRobot dataset converter.
    * [`annotation_server.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/annotation_server.py) — FastAPI backend for manual bag annotation.
    * [`convert_policy.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/convert_policy.py) — ONNX/TensorRT conversion script.
    * [`package_engines.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/package_engines.py) — Engines packager for deployment.
    * **`convert/` sub-package**: [`convert/`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/convert/) — Handlers for converting ROS messages (geometry, sensor, image, std) to/from PyTorch Tensors.
    * **`core/` sub-package**: [`core/`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/core/) — Dataset writers, frame assemblers, and ROS publishers.
    * **`trt/` sub-package**: [`trt/`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/trt/) — Engine execution and validation files.
    * **`gui/` Web Folder**: [`gui/`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/gui/) — Frontend code for manual bag annotation.
  * **SO101 Manipulator Package**: [`src/so101/`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/so101/) — Leader and follower robot drivers.

---

## ROS 2 Nodes (Console Scripts)

### 1. `dataset_recorder`
* **File**: [`recorder.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/recorder.py)
* **Description**: Subscribes to observation topics (images, joint states, sensors) and action topics. It collects synchronized steps at the target FPS.
* **Key Services & Actions**:
  * `/new_dataset` (`lerobot_interfaces/srv/NewDataset`): Initializes a new or resumed dataset directory.
  * `/start_episode` (`lerobot_interfaces/srv/StartEpisode`): Starts recording incoming frames for a task description.
  * `/end_episode` (`lerobot_interfaces/srv/EndEpisode`): Ends current episode with option to discard.
  * `/store_episodes` (`std_srvs/srv/Trigger`): Triggers asynchronous thread to save episodes to disk.
  * `store_episodes_action` (`lerobot_interfaces/action/StoreEpisodes`): Action alternative showing progress.
  * `/finalize_dataset` (`lerobot_interfaces/srv/FinalizeDataset`): Computes statistics and finishes writing files.
  * `/push_to_hub` (`lerobot_interfaces/srv/PushToHub`): Uploads dataset to Hugging Face Hub.

### 2. `policy_controller`
* **File**: [`policy_controller.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/policy_controller.py)
* **Description**: Loads trained policies (supporting PyTorch compilation and TensorRT acceleration) and runs inference.
* **Key Services, Topics & Actions**:
  * `run_policy` (`lerobot_interfaces/action/RunPolicy`): Runs a policy for a specified task. Cancel to stop.
  * `policy_control/heartbeat` (`std_msgs/msg/Empty`): Safety deadman switch topic. Inference stops if not published within `heartbeat_timeout_s`.
  * `policy_control/status` (`lerobot_interfaces/msg/PolicyStatus`): Latched topic publishing current running status, active policy, task name, and progress.
  * `policy_control/metrics` (`std_msgs/msg/String`): Timing statistics (only in benchmark mode).
  * **Benchmark Mode (`benchmark:=True`)**: Replaces real camera subscribers with zero-filled synthetic inputs, publishing actions to a `benchmark/` prefix without moving the physical robot.

### 3. `replay`
* **File**: [`replay.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/replay.py)
* **Description**: Queries a dataset locally and replays recorded action frames back onto the configured ROS topics at the original FPS.

---

## Offline Utilities

### 1. `bag_to_dataset`
* **File**: [`bag_to_dataset.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/bag_to_dataset.py)
* **Description**: CLI tool to convert recorded ROS 2 Bags (`.mcap` or SQLite3 `.db3`) into a LeRobot dataset. Can segment episodes automatically using a string or boolean `/task` topic.

### 2. `annotation_gui`
* **File**: [`annotation_server.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/annotation_server.py)
* **Description**: Launches a FastAPI server hosting a responsive web GUI. It lets users scrub through a ROS bag, visualize camera streams and decoded telemetry, mark episode ranges, and run background compilation into a LeRobot dataset.

### 3. `convert_policy`
* **File**: [`convert_policy.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/convert_policy.py)
* **Description**: Exposes ONNX and TensorRT builders. Converts PyTorch weights (`act`, `smolvla`, or `smolvla_recap`) to TensorRT engines, verifying outputs against original PyTorch results.

### 4. `package_engines`
* **File**: [`package_engines.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/lerobot_ros/lerobot_ros/package_engines.py)
* **Description**: Bundles exported ONNX engines, config metadata, and creates rebuild shell/python scripts into a `.tar.gz` archive for deployment on target robot hardware (e.g., NVIDIA Jetson).

---

## Interfaces Summary (`lerobot_interfaces`)

### Messages (`msg/`)
* **`PolicyStatus.msg`**: Running status, active policy name, active task description, list of available policies, task progress, and heartbeat liveness.
* **`TaskProgress.msg`**: Progress estimation percentage for the running policy.

### Services (`srv/`)
* **`NewDataset.srv`**: Repo ID string, resume boolean -> success boolean, message.
* **`StartEpisode.srv`**: Task string -> episode ID.
* **`EndEpisode.srv`**: Discard boolean -> total recorded frames.
* **`FinalizeDataset.srv`**: `Trigger` -> success boolean, message.
* **`PushToHub.srv`**: Optional API key string -> success boolean, message.

### Actions (`action/`)
* **`RunPolicy.action`**: Policy name and task string -> success flag, status feedback.
* **`StoreEpisodes.action`**: Empty goal -> episodes stored count, total episodes count.
* **`ReplayEpisode.action`**: Episode ID and repetitions -> progress status feedback.

---

## Hardware Driver Nodes (`so101`)

* **`leader_node`**: [`so101_leader.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/so101/so101/so101_leader.py)
  * Reads joint states from the lead manipulator arm and publishes them.
* **`follower_node`**: [`so101_follower.py`](file:///workspaces/ros-fhnw-autonomy/src/lerobot_ros/src/so101/so101/so101_follower.py)
  * Listens to command topics and controls the follower manipulator arm.
