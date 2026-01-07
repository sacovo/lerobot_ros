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
colcon build

source install/setup.bash

# Use the file as starting point and customize as you see fit
ros2 launch so101 so101.py config:=lerobot_ros/config/so101/params.yml

ros2 run lerobot_ros recorder --ros-args -p config:=config/your_setup.toml

```

## Configuration

Configuration is done with a `toml` file for each setup. A setup uses the same topics and a set of policies, which all use these topics as in- and output.

An example is provided in `lerobot_ros/config/so101/so101.toml`.

LeRobot works with rerun for visualization. You can configure the nodes to visualize data with rerun by providing the address of your instance. This gives you an easy interface to see what you are currently recording or what your policies currently see.

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

The node is controlled by service calls:

```bash

# get the policies you can choose
ros2 service call /list_policies lerobot_interfaces/srv/ListPolicies

ros2 service call /set_active_policy lerobot_interfaces/srv/SetActivePolicy 'policy_name: "key"'

# enable autonomous control and disable it

ros2 service call /set_policy_running std_srvs/srv/SetBool 'data: true'

ros2 service call /set_policy_running std_srvs/srv/SetBool 'data: false'
```

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

## About

This package is developed as part of the the [FHNW Rover](https://www.fhnw.ch/plattformen/erc-rover/blog/) project.