# ROS2 Bag to LeRobot Dataset Conversion Tool

This document describes the design, options, and usage of the offline conversion tool `bag_to_dataset` in `lerobot_ros`.

## Overview

The `bag_to_dataset` tool is an offline utility that reads a ROS2 bag chronologically and converts it into a LeRobot dataset. It enables converting the recorded bag data into distinct training episodes mapped to specific tasks, completely filtering out reset phases or idle transitions between episodes.

Since it operates offline (by reading and deserializing the bag messages directly), it is fully deterministic, processes data much faster than real-time playback, and avoids the timing jitters associated with running live ROS2 nodes.

## How It Works

1. **Config Loading**: Reads the TOML configuration file to identify target topics, their types, Quality of Service (QoS) parameters, and target FPS.
2. **Bag Open & Mapping**: Autodetects the storage type (e.g. `mcap` or `sqlite3`) of the ROS2 bag and maps the bag's topics to those specified in the configuration.
3. **Resampling Logic**:
   - The tool maintains a zero-order hold (sample-and-hold) state for all configured topics.
   - It reads messages sequentially. When an episode/task is active, it writes out resampled dataset frames at a rate defined by `fps` (e.g., every `1 / fps` seconds).
   - If a topic has not received any message since the beginning of the bag, the tool populates it with zero tensors.
4. **Episode Splitting**:
   - If a static task name is provided via the command line (`--task`), the tool treats the entire bag as one single episode.
   - If a control topic is used (default `/task`), the tool monitors messages on this topic. It automatically starts an episode when a non-empty/non-idle task name is published, ends it when the topic becomes empty/idle, and seamlessly splits episodes if the task name changes.
   - Frames recorded during idle/reset task phases are completely excluded from the resulting dataset.

---

## Command Line Interface (CLI)

You can run the tool using `ros2 run`:

```bash
ros2 run lerobot_ros bag_to_dataset --bag-path <path_to_bag> --config <path_to_toml> --repo-id <dataset_name> [options]
```

### Options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--bag-path` | `str` | *Required* | Path to the ROS2 bag directory. |
| `--config` | `str` | *Required* | Path to the TOML configuration file. |
| `--repo-id` | `str` | *Required* | Name/repo_id of the LeRobot dataset (e.g. `user/dataset-name`). |
| `--task-topic` | `str` | `/task` | Topic to monitor for task names / episode transitions (supports `std_msgs/msg/String` and `std_msgs/msg/Bool`). |
| `--task` | `str` | `None` | If specified, overrides `--task-topic` and converts the entire bag as one episode with this task name. |
| `--storage-id` | `str` | `None` | Storage format of the bag (e.g. `mcap`, `sqlite3`). If omitted, it is automatically detected. |
| `--exclude-tasks` | `list[str]` | `["idle", "reset", "none", "false"]` | List of task names (case-insensitive) that signify idle or reset state, during which frames are discarded. |
| `--default-task-name`| `str` | `task` | Default task name to assign when using a boolean task topic that publishes `True`. |
| `--resume` | `flag` | `False` | Resume writing to an existing dataset instead of failing or creating a new one. |

---

## Examples

### Example 1: Splitting bag into episodes using `/task` topic
If your bag records string messages on `/task` (e.g. `"pick up the ball"`, `"idle"`, `"place the ball"`), you can split the bag automatically:

```bash
ros2 run lerobot_ros bag_to_dataset \
    --bag-path ./data/my_ros2_bag \
    --config ./src/lerobot_ros/config/so101/so101.toml \
    --repo-id fhnwrover/so101-manipulation-dataset \
    --task-topic /task
```

### Example 2: Converting the entire bag as one episode
If the bag does not contain any control/task topic, you can specify the task name manually for the entire duration:

```bash
ros2 run lerobot_ros bag_to_dataset \
    --bag-path ./data/single_trial_bag \
    --config ./src/lerobot_ros/config/so101/so101.toml \
    --repo-id fhnwrover/so101-single-trial \
    --task "open the drawer"
```

---

## Interactive Annotation GUI

When the task transitions are not recorded in the bag, or when a human annotator needs to manually identify and segment episodes, you can use the interactive Web Annotation GUI. 

The application is built using **FastAPI** on the backend and a premium, responsive glassmorphic UI on the frontend.

### Launching the GUI

Start the annotation server via the ROS2 console command:

```bash
ros2 run lerobot_ros annotation_gui --host 0.0.0.0 --port 8000
```

Once started, open [http://localhost:8000](http://localhost:8000) in your web browser.

### Key Features

1. **Bag Browser**: Scans the workspace automatically for directories containing `metadata.yaml`. Select a bag from the dropdown menu to load it.
2. **Timeline Scrubber**:
   - Scrub through the bag using the graphical timeline bar. Clicking or dragging seeks immediately to the target frame.
   - Play/Pause buttons with speed selectors (`0.25x`, `0.5x`, `1x`, `2x`).
3. **Synchronized Visualization**:
   - Displays all camera feeds configured in the TOML profile.
   - Displays all other non-image topic telemetry (e.g. joints, velocities) in a formatted, clean table (with underscores/internal ROS fields removed).
4. **Episode Management**:
   - Add range markers using `[ Start Marker` and `] End Marker` buttons or by pressing the `[` and `]` shortcut keys.
   - Enter a descriptive task name and click **"Save Episode"**.
   - Shaded timeline blocks indicate saved episodes. Click the red garbage icon to delete an episode range.
5. **Background Dataset Compilation**:
   - Provide the LeRobot dataset `Repo ID`, the configuration TOML file path, and target `FPS`.
   - Click **"Build Dataset"** to compile the annotated episodes. The task is executed as an asynchronous background worker, and a progress bar displays status in real-time.
