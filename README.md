# LeRobot <-> ROS

This repository contains some utilities to work with the [LeRobot](https://huggingface.co/lerobot) framework in [ROS](https://www.ros.org/).

LeRobot is a framework that makes deep learning-based policies for robotics easy, and ROS is a suite for robotics in general.

With the provided tools you can use your ROS setup to record datasets and train AI models to control your robots.

## Setup instructions

Installing LeRobot and ROS together has some caveats, as ROS does not really accommodate virtual Python environments, and LeRobot depends on them. It should also be possible to install the required packages globally, but this has not been tested.

```bash

# Go to your workspace and clone the repository
cd ros_ws/
git clone https://github.com/sacovo/lerobot_ros.git

# First install uv for Python dependency management
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"

uv venv --system-site-packages
source .venv/bin/activate

# Install lerobot_ros dependencies
uv pip install "./lerobot_ros[so101]" # or just ./lerobot_ros if you do not want to use the so101

# Build ros package
source /opt/ros/jazzy/setup.bash
colcon build

source install/setup.bash

# Use the file as starting point and customize as you see fit
ros2 launch so101 so101.py config:=lerobot_ros/config/so101/params.yml

ros2 run lerobot_ros dataset_recorder --ros-args -p config:=config/your_setup.toml

```

## Configuration

Configuration is done with a `toml` file for each setup. A setup uses the same topics and a set of policies, which all use these topics as in- and output.

An example is provided in `lerobot_ros/config/so101/so101.toml`.

LeRobot works with rerun for visualization. You can configure the nodes to visualize data with rerun by providing the address of your instance. This gives you an easy interface to see what you are currently recording or what your policies currently see.

### Topic Output Limits and Rounding

Policies output continuous values, but some robotic controllers/actuators require distinct inputs (e.g. `1.0` or `-1.0` for a gripper) or have physical range limits. You can specify rounding and output clamping on a per-topic basis under the `[topics]` section using:
- `limits`: Clamps output values between a minimum and maximum range (e.g., `[min, max]`). One-sided limits can be specified by setting either bound to `None`.
- `round_values`: Snaps output values to the closest number in a given list of allowed values.

These constraints can be configured in three ways:

1. **Topic-wide (applies to all elements):**
   ```toml
   [topics."/test/gripper"]
   msg_type = "Float32"
   tag = "action"
   limits = [-1.0, 1.0]
   round_values = [-1.0, 1.0]
   ```

2. **Element-wise (by list index):**
   ```toml
   [topics."/test/joints"]
   msg_type = "Float32MultiArray"
   names = ["joint_a", "joint_b"]
   tag = "action"
   limits = [ [-1.0, 1.0], [-2.0, 2.0] ]
   round_values = [ [-0.5, 0.5], [1.0, -1.0, 0.0] ]
   ```

3. **Named (by joint name or field part):**
   ```toml
   [topics."/follower/joint_states"]
   msg_type = "JointState"
   tag = "action"
   joints = ["shoulder", "gripper"]
   position = true
   limits = { shoulder = [-1.57, 1.57], gripper = [-1.0, 1.0] }
   round_values = { gripper = [-1.0, 1.0] }
   ```

### Input Transforms (observations)

`limits`/`round_values` above shape *action* outputs. For **observation** inputs you can
apply a nonlinear `transform` as the message is converted to a tensor. This is useful for
sensors whose interesting range is small relative to their full range — e.g. a gripper ToF
distance that reads up to 4000 mm but only matters below ~200 mm. Converting the distance
to a "closeness" hands most of the value range to the region that matters, giving the policy
larger gradients there.

Note this is *not* a plain sign-flip: normalization in the training pipeline already absorbs
any affine (linear) inversion for free, so only nonlinear reshaping adds information. The
transform is applied identically during recording and inference, so the dataset statistics
stay consistent. It is only valid on observation topics (there is no inverse for actions), and
a transformed topic is always stored as `float32`.

Available `type` values:
- `exp_decay`: `exp(-value / scale)` — bounded in `(0, 1]`, `0 → 1`, large values `→ ~0`. Recommended default; pick `scale` near your region of interest.
- `reciprocal`: `scale / (value + eps)` — sharper emphasis near zero but unbounded; `eps` guards division by zero.

```toml
# ToF distance (mm) inside the gripper; only 0–200mm is interesting
[topics."/gripper/tof_distance"]
msg_type = "Float32"
tag = "observation"
key = "tof_closeness"
transform = { type = "exp_decay", scale = 100.0 }
```

Like `limits`, a `transform` can also be applied per element by name for multi-value topics:

```toml
transform = { near_sensor = { type = "exp_decay", scale = 100.0 } }
```

## Quick guide

First you need to configure the topics you want to record. Usually these would be some images that a human needs to solve the task, as well as sensors and joint states of your motors. These are the input topics that should later be predicted by a policy.

If you have set everything up, you can start the recorder node:
```bash
ros2 run lerobot_ros dataset_recorder --ros-args -p config:=.../your_config.toml
```

In another terminal you can control the capture of your dataset. A dataset consists of episodes, where each episode is the completion of one task.

```bash
# 1) Create a new dataset
ros2 service call /new_dataset lerobot_interfaces/srv/NewDataset 'repo_id: "user/dataset-name"'

# 2) Start an episode
ros2 service call /start_episode lerobot_interfaces/srv/StartEpisode 'task: "pick up the ball"'

# Perform the task

# 3) End the episode
ros2 service call /end_episode lerobot_interfaces/srv/EndEpisode

# or discard the episode if something went wrong
ros2 service call /end_episode lerobot_interfaces/srv/EndEpisode 'discard: true'

# Repeat 2 and 3 until you have enough recordings

# When you are done, store the episodes to your disk (this might take some time)
ros2 service call /store_episodes std_srvs/srv/Trigger

```

You can upload your datasets to huggingface, so they are available for training on different machines.
```bash
# Upload from data/... to huggingface
hf upload --repo-type dataset username/ds-name ./path/to/data/username/ds-name

# Tag the repository
hf repo tag create --repo-type dataset username/ds-name v3.0 
```

Now you can train your policy on a device with a GPU or other accelerated hardware:

```bash
lerobot-train \
    --dataset.repo_id username/dataset-name \
    --policy.type=act \
    --job_name=lerobot_drive_act \
    --wandb.enable true \
    --policy.repo_id=username/model-name
```

This will train the policy and then push it to the huggingface hub where you can download it to use it on your robot.

For more information about training, policies, and best practices, refer to the [documentation](https://huggingface.co/docs/lerobot/index).

To use a policy add it to the `[policies]` section of your configuration and start the `policy_controller`.
```bash
ros2 run lerobot_ros policy_controller --ros-args -p config:=...

```

The node is controlled through the `run_policy` action. Sending a goal starts a
policy; cancelling the goal stops it. Available policies are listed on the
latched `policy_control/status` topic.

```bash

# see loaded policies and current state (latched)
ros2 topic echo --once /policy_control/status

# run a task (cancel with Ctrl-C to stop)
ros2 action send_goal /run_policy lerobot_interfaces/action/RunPolicy \
  '{policy_name: "key", task: "do the thing"}' --feedback
```

Actuation additionally requires a fresh `policy_control/heartbeat`
(`std_msgs/Empty`) published by the operator — see the autonomy/safety section
in the [repository README](../../README.md).

You can also replay a dataset to check whether the actions have been recorded correctly. Be aware that this will publish actions to a topic that controls your robot!

```bash

ros2 run lerobot_ros replay --ros-args\
     -p repo_id:=fhnwrover/so101-ros-red-ring-all \
     -p episode:=[2,3,5] \
     -p repetions:=3 \
     -p config:=lerobot_ros/config/so101/so101.toml
```

## SO101

The SO101 is a small and affordable robotic manipulator that has good integration with LeRobot. A manual and parts list can be found in this repository: https://github.com/TheRobotStudio/SO-ARM100

Two nodes are part of this package, one for controlling the follower and one for reading from the leader.

```bash

ros2 run so101 follower_node --ros-args \
    -p port:=/dev/ttyACM0 \
    -p calibration_dir:=config/so101/calibrations/ \
    -p calibrate:=True  \
    -p frequency:=20 # fps

ros2 run so101 leader_node --ros-args \
    -p port:=/dev/ttyACM1 \
    -p calibration_dir:=config/so101/calibrations/ \
    -p calibrate:=True  \
    -p frequency:=20 # fps
    
# Or start them together with a launch file and provide the parameters as yaml

ros2 launch so101 so101.py config:=config/so101/params.yml
```

## Annotation GUI

Instead of splitting a bag by a task topic, you can annotate episode ranges
interactively in a browser. Start the server (a dev-side tool) and open the
printed URL:

```bash
ros2 run lerobot_ros annotation_gui -- --bag-root data --config config/so101/so101.toml
```

Scrub the timeline, mark start/end times per episode, then **Convert & Build
Dataset**. The episode **timing** is stored independently of the feature/topic
config, so you can reuse it after changing the config (e.g. adding a new topic
as a feature) without re-annotating:

- **Auto-save**: converting writes `annotations.json` into the bag directory,
  and reloading that bag restores its episodes automatically.
- **Save / Export / Import** buttons let you write `annotations.json` on demand,
  download the timing as a portable JSON file, or load one into the GUI.

To rebuild with a changed config: edit the TOML, reload the same bag (the saved
timing is restored), and convert again.

## Development & Testing

Tests live under `test/`. Two kinds of test dependencies are needed:

- **Python** — `pytest` (a base dependency) plus, for the annotation-server
  tests, `fastapi`, `uvicorn` and `httpx` (FastAPI's `TestClient`). Those three
  live in the `dev` dependency group in `pyproject.toml`, since the annotation
  GUI is a dev-side tool; install the group with:

  ```bash
  uv pip install ./lerobot_ros --group dev
  # or, with a uv-managed project: `uv sync` (installs the dev group by default)
  ```

  The ROS-side test deps (`rclpy`, `rosbag2_py`, `std_msgs`) come from the
  sourced ROS install, not pip. `fastapi`/`uvicorn` are intentionally absent
  from the lean rover/Jetson image, so the annotation-server tests skip there
  and run in the dev container instead.

- **Node.js** — used to syntax-check / lint the annotation GUI's static JS, e.g.
  `node --check lerobot_ros/gui/app.js`. The GUI is vanilla JS (no bundler).

Both `nodejs`/`npm` (via `docker/apt-dev.txt`) and the Python test deps are
provided in the dev container (`docker/Dockerfile.dev`); rebuild the dev image
to pick them up. Run the tests with:

```bash
# ROS-aware suite (annotation server, bag → dataset), from the workspace root
colcon test --packages-select lerobot_ros && colcon test-result --verbose

# or directly with pytest, once ROS + the workspace are sourced
python -m pytest src/lerobot_ros/test
```

## About

This package is developed as part of the the [FHNW Rover](https://www.fhnw.ch/plattformen/erc-rover/blog/) project.