#!/usr/bin/env python3
import argparse
import os
import sys
import torch

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from lerobot_ros.config import load_toml_dict, parse_config
from lerobot_ros.core import FrameAssembler, DatasetWriter


def get_storage_id_from_bag(bag_path: str) -> str:
    metadata_path = os.path.join(bag_path, "metadata.yaml")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                for line in f:
                    if "storage_identifier:" in line:
                        return line.split("storage_identifier:")[-1].strip().strip("'\"")
        except Exception:
            pass
    if os.path.isdir(bag_path):
        for f in os.listdir(bag_path):
            if f.endswith(".mcap"):
                return "mcap"
            if f.endswith(".db3"):
                return "sqlite3"
    return "mcap"  # default to mcap for newer ROS2 versions


def main():
    parser = argparse.ArgumentParser(description="Convert a ROS2 bag into a LeRobot dataset.")
    parser.add_argument("--bag-path", type=str, required=True, help="Path to the ROS2 bag directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the TOML configuration file.")
    parser.add_argument("--repo-id", type=str, required=True, help="Name/repo_id of the dataset to write to.")
    parser.add_argument("--task-topic", type=str, default="/task", help="Topic indicating the current task.")
    parser.add_argument("--task", type=str, default=None, help="If specified, override task-topic and treat the entire bag as a single episode with this task name.")
    parser.add_argument("--storage-id", type=str, default=None, help="ROS2 bag storage ID (e.g. mcap, sqlite3). Auto-detected if omitted.")
    parser.add_argument("--exclude-tasks", type=str, nargs="*", default=["idle", "reset", "none", "false"], help="Task names (lowercase) that indicate idle/reset.")
    parser.add_argument("--default-task-name", type=str, default="task", help="Default task name when using a boolean task-topic.")
    parser.add_argument("--resume", action="store_true", help="Resume writing to an existing dataset.")
    parser.add_argument(
        "--human-intervention-topic",
        type=str,
        default=None,
        help="Topic (std_msgs/Bool) indicating human intervention frames. When set, "
             "an 'intervention' column (float32, 1=human, 0=policy) is added to the dataset. "
             "Overrides the value in the TOML config if provided.",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' does not exist.")
        sys.exit(1)

    config = parse_config(load_toml_dict(config_path))

    # CLI flag overrides TOML config value
    if args.human_intervention_topic:
        config.human_intervention_topic = args.human_intervention_topic

    print(f"Loaded config from {config_path}")

    # Determine storage ID
    bag_path = os.path.abspath(args.bag_path)
    if not os.path.exists(bag_path):
        print(f"Error: Bag path '{bag_path}' does not exist.")
        sys.exit(1)
        
    storage_id = args.storage_id or get_storage_id_from_bag(bag_path)
    print(f"Opening bag '{bag_path}' with storage format '{storage_id}'...")

    # Open bag
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Failed to open ROS2 bag: {e}")
        sys.exit(1)

    # Get topic types
    topic_types = {}
    for topic_info in reader.get_all_topics_and_types():
        try:
            topic_types[topic_info.name] = get_message(topic_info.type)
        except Exception as e:
            print(f"Warning: Could not load message class for type '{topic_info.type}': {e}")

    # Map bag topics to config topics (leading slash insensitive)
    bag_to_config_map = {}
    for bag_topic in topic_types.keys():
        bag_norm = bag_topic.strip("/")
        for config_topic in config.topics.keys():
            config_norm = config_topic.strip("/")
            if bag_norm == config_norm:
                bag_to_config_map[bag_topic] = config_topic
                break

    print(f"Mapped {len(bag_to_config_map)} topics from bag to configuration:")
    for b_top, c_top in bag_to_config_map.items():
        print(f"  {b_top} -> {c_top}")

    # Resolve human intervention topic
    bag_intervention_topic = None
    if config.human_intervention_topic:
        norm = config.human_intervention_topic.strip("/")
        for bag_topic in topic_types.keys():
            if bag_topic.strip("/") == norm:
                bag_intervention_topic = bag_topic
                break
        if bag_intervention_topic:
            print(f"Using intervention topic '{bag_intervention_topic}' to label human-controlled frames.")
        else:
            print(f"Warning: human_intervention_topic '{config.human_intervention_topic}' not found in bag.")

    # Map task/episode control topic
    bag_task_topic = None
    if not args.task:
        task_topic_norm = args.task_topic.strip("/")
        for bag_topic in topic_types.keys():
            if bag_topic.strip("/") == task_topic_norm:
                bag_task_topic = bag_topic
                break
        if bag_task_topic:
            print(f"Using control topic '{bag_task_topic}' for task/episode splitting.")
        else:
            print(f"Warning: Control topic '{args.task_topic}' not found in the bag.")
            print("To treat the entire bag as one episode, use the '--task' option.")

    # Initialize LeRobot dataset and assembler
    assembler = FrameAssembler(config.topics)
    try:
        writer = DatasetWriter(args.repo_id, config, resume=args.resume)
    except Exception as e:
        print(f"Failed to initialize LeRobotDataset: {e}")
        sys.exit(1)

    # Exclude tasks normalization
    exclude_tasks_lower = [t.lower() for t in args.exclude_tasks]

    # Processing loop
    latest_msgs = {}
    latest_intervention = False  # most-recent value from intervention topic
    episode_active = False
    current_task = None
    t_start = None
    t_next_sample = None
    episode_frames = []

    # If the user specified a static task name, we start active
    if args.task:
        episode_active = True
        current_task = args.task

    print("Processing bag messages chronologically...")
    count = 0
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        t_sec = timestamp / 1e9
        count += 1

        # Track human intervention signal
        if bag_intervention_topic and topic_name == bag_intervention_topic:
            try:
                msg_class = topic_types[topic_name]
                msg = deserialize_message(data, msg_class)
                val = msg.data if hasattr(msg, "data") else False
                latest_intervention = bool(val)
            except Exception as e:
                print(f"Error reading intervention topic: {e}")

        # Check if topic is a data topic
        if topic_name in bag_to_config_map:
            config_topic = bag_to_config_map[topic_name]
            topic_converter = config.topics[config_topic]
            try:
                msg_class = topic_types[topic_name]
                msg = deserialize_message(data, msg_class)
                tensor = topic_converter.to_tensor(msg)
                latest_msgs[config_topic] = tensor
            except Exception as e:
                print(f"Error converting message on '{topic_name}': {e}")
                continue

            # If args.task is set, initialize timeline on first data message
            if args.task and t_start is None:
                t_start = t_sec
                t_next_sample = t_start
                print(f"[{t_sec:.2f}s] Starting single episode for task: '{current_task}'")

        # Check if topic is the control topic
        elif not args.task and topic_name == bag_task_topic:
            try:
                msg_class = topic_types[topic_name]
                msg = deserialize_message(data, msg_class)
                
                if hasattr(msg, "data"):
                    val = msg.data
                else:
                    val = str(msg)
                
                if isinstance(val, bool):
                    task_active = val
                    task_name = args.default_task_name if task_active else ""
                else:
                    task_name = str(val).strip()
                    task_active = (task_name.lower() not in exclude_tasks_lower) and (len(task_name) > 0)
                
                if task_active:
                    if not episode_active:
                        episode_active = True
                        current_task = task_name
                        t_start = t_sec
                        t_next_sample = t_start
                        episode_frames = []
                        print(f"[{t_sec:.2f}s] Starting episode for task: '{current_task}'")
                    elif task_name != current_task:
                        if len(episode_frames) > 0:
                            print(f"[{t_sec:.2f}s] Ending episode for task '{current_task}' with {len(episode_frames)} frames (task changed)")
                            writer.save_episode(episode_frames, current_task, success=True)
                        current_task = task_name
                        t_start = t_sec
                        t_next_sample = t_start
                        episode_frames = []
                        print(f"[{t_sec:.2f}s] Starting new episode for task: '{current_task}'")
                else:
                    if episode_active:
                        if len(episode_frames) > 0:
                            print(f"[{t_sec:.2f}s] Ending episode for task '{current_task}' with {len(episode_frames)} frames")
                            writer.save_episode(episode_frames, current_task, success=True)
                        episode_active = False
                        current_task = None
                        t_start = None
                        t_next_sample = None
                        episode_frames = []
            except Exception as e:
                print(f"Error parsing task message: {e}")

        # Check if we need to write frames
        if episode_active and t_start is not None:
            while t_sec >= t_next_sample:
                frame = assembler.assemble(latest_msgs)
                if writer.track_intervention:
                    frame["intervention"] = torch.tensor([1.0 if latest_intervention else 0.0])
                episode_frames.append((frame, current_task, t_next_sample))
                t_next_sample += 1.0 / config.fps

    # Finalize any remaining active episode at the end of the bag
    if episode_active and len(episode_frames) > 0:
        print(f"Ending final episode for task '{current_task}' with {len(episode_frames)} frames")
        writer.save_episode(episode_frames, current_task, success=True)

    print("Finalizing dataset (computing statistics, etc.)...")
    try:
        writer.finalize()
        print("Dataset finalized and saved successfully!")
    except Exception as e:
        print(f"Failed to finalize dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
