#!/usr/bin/env python3
import os
import sys
import base64
import io
import typing
import traceback
import numpy as np
import torch
import cv2
import yaml
from PIL import Image as PILImage

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import DEFAULT_FEATURES
from lerobot_ros.config import load_toml_dict, parse_config
from lerobot_ros.convert.image import ros_image_to_numpy, ImageTopic, ImageCompressedTopic
from lerobot_ros.bag_to_dataset import get_storage_id_from_bag
from lerobot_ros.core import FrameAssembler, DatasetWriter

# Initialize FastAPI App
app = FastAPI(title="LeRobot Bag Annotator")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global status tracking for conversion task
conversion_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "Idle"
}

def pil_to_base64_jpeg(pil_img: PILImage.Image) -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def ros_message_to_dict(msg):
    if hasattr(msg, "__slots__"):
        res = {}
        for slot in msg.__slots__:
            display_field = slot[1:] if slot.startswith('_') else slot
            if display_field == "check_fields":
                continue
            val = getattr(msg, slot)
            res[display_field] = ros_message_to_dict(val)
        return res
    elif hasattr(msg, "__iter__") and not isinstance(msg, (str, bytes, dict)):
        return [ros_message_to_dict(x) for x in msg]
    elif isinstance(msg, (float, np.float32, np.float64)):
        return float(msg)
    elif isinstance(msg, (int, np.int32, np.int64)):
        return int(msg)
    else:
        return msg

@app.get("/api/bags")
def list_bags():
    bags = []
    root_dir = "/workspaces/ros-fhnw-autonomy"
    for root, dirs, files in os.walk(root_dir):
        # Limit search depth to 3 levels to avoid traversing node_modules/build dirs
        depth = root[len(root_dir):].count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        if "metadata.yaml" in files:
            bags.append(os.path.abspath(root))
    return {"bags": bags}

@app.get("/api/bag-info")
def get_bag_info(path: str = Query(..., description="Absolute path to the bag directory"), config: typing.Optional[str] = Query(None, description="Path to the TOML configuration file")):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Bag path not found")
    metadata_path = os.path.join(path, "metadata.yaml")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=400, detail="Not a valid ROS2 bag directory (metadata.yaml missing)")
    
    try:
        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)
        
        bag_info = meta.get("rosbag2_bagfile_information", {})
        start_time_ns = bag_info.get("starting_time", {}).get("nanoseconds_since_epoch", 0)
        duration_ns = bag_info.get("duration", {}).get("nanoseconds", 0)
        
        start_time_sec = start_time_ns / 1e9
        end_time_sec = start_time_sec + duration_ns / 1e9
        
        topics = []
        for topic_info in bag_info.get("topics_with_message_count", []):
            topic = topic_info.get("topic_metadata", {})
            topics.append({
                "name": topic.get("name"),
                "type": topic.get("type"),
                "message_count": topic_info.get("message_count")
            })
            
        config_topics = {}
        if config and os.path.exists(config):
            try:
                cfg = parse_config(load_toml_dict(config))
                for t_name, t_obj in cfg.topics.items():
                    config_topics[t_name] = {
                        "tag": "action" if t_obj.is_action else ("meta" if t_obj.is_meta else "observation")
                    }
            except Exception:
                pass
                
        return {
            "path": path,
            "start_time": start_time_sec,
            "end_time": end_time_sec,
            "duration": duration_ns / 1e9,
            "topics": topics,
            "config_topics": config_topics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse bag metadata: {str(e)}")

@app.get("/api/frame")
def get_frame(path: str = Query(...), time: float = Query(...), config: str = Query(...)):
    if not os.path.exists(config):
        raise HTTPException(status_code=404, detail="TOML configuration file not found")
    try:
        cfg = parse_config(load_toml_dict(config))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse configuration TOML: {str(e)}")
        
    storage_id = get_storage_id_from_bag(path)
    
    try:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )
        reader.open(storage_options, converter_options)
        
        topic_types = {}
        for topic_info in reader.get_all_topics_and_types():
            try:
                topic_types[topic_info.name] = get_message(topic_info.type)
            except Exception:
                pass
                
        bag_to_config_map = {}
        for bag_topic in topic_types.keys():
            bag_norm = bag_topic.strip("/")
            for config_topic in cfg.topics.keys():
                config_norm = config_topic.strip("/")
                if bag_norm == config_norm:
                    bag_to_config_map[bag_topic] = config_topic
                    break
                    
        if not bag_to_config_map:
            raise HTTPException(
                status_code=400,
                detail="No topics in the configuration file match the topics in the loaded bag file. "
                       "Please ensure you have entered the correct TOML Config path in the header bar."
            )
                    
        # Determine starting timestamp
        metadata_path = os.path.join(path, "metadata.yaml")
        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)
        bag_info = meta.get("rosbag2_bagfile_information", {})
        start_time_ns = bag_info.get("starting_time", {}).get("nanoseconds_since_epoch", 0)
        
        target_ts_ns = start_time_ns + int(time * 1e9)
        
        # Seek to 1s before to fill sample-and-hold cache
        seek_ts_ns = max(start_time_ns, target_ts_ns - int(1.0 * 1e9))
        reader.seek(seek_ts_ns)
        
        latest_msgs = {}
        while reader.has_next():
            topic_name, msg_data, timestamp = reader.read_next()
            if timestamp > target_ts_ns:
                # If we haven't found any messages for our configured topics yet,
                # allow reading up to 2.0s past target_ts_ns to populate the initial frame.
                if len(latest_msgs) > 0 or timestamp > target_ts_ns + int(2.0 * 1e9):
                    break
            if topic_name in bag_to_config_map:
                latest_msgs[bag_to_config_map[topic_name]] = (msg_data, topic_name)
                
        cameras = {}
        telemetry = {}
        
        for config_topic_name, (msg_data, bag_topic_name) in latest_msgs.items():
            topic_converter = cfg.topics[config_topic_name]
            msg_class = topic_types[bag_topic_name]
            try:
                msg = deserialize_message(msg_data, msg_class)
                if isinstance(topic_converter, (ImageTopic, ImageCompressedTopic)):
                    if isinstance(topic_converter, ImageCompressedTopic):
                        pil_img = PILImage.open(io.BytesIO(msg.data))
                    else:
                        pil_img = ros_image_to_numpy(msg)
                        
                    if topic_converter.rotate:
                        pil_img = pil_img.rotate(topic_converter.rotate, expand=True)
                        
                    cameras[config_topic_name] = pil_to_base64_jpeg(pil_img)
                else:
                    # Non-camera topic, format fields recursively to JSON-serializable types
                    telemetry[config_topic_name] = ros_message_to_dict(msg)
            except Exception as e:
                print(f"Error decoding topic {config_topic_name}: {e}")
                
        return {
            "timestamp": time,
            "cameras": cameras,
            "telemetry": telemetry
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error reading bag at {time}s: {str(e)}")

# Pydantic models for Conversion REST API
class AnnotationSegment(BaseModel):
    start_time: float
    end_time: float
    task: str

class ConvertRequest(BaseModel):
    bag_path: str
    config: str
    repo_id: str
    resume: bool
    annotations: list[AnnotationSegment]

def run_conversion_task(bag_path, config_path, repo_id, resume, annotations):
    global conversion_status
    conversion_status["running"] = True
    conversion_status["progress"] = 0
    conversion_status["total"] = len(annotations)
    conversion_status["message"] = "Initializing LeRobot Dataset..."
    
    try:
        config = parse_config(load_toml_dict(config_path))
        
        dataset_name = repo_id
        
        conversion_status["message"] = f"Initializing dataset {dataset_name}..."
        writer = DatasetWriter(dataset_name, config, resume=resume)
        dataset = writer.dataset
        assembler = FrameAssembler(config.topics)
            
        storage_id = get_storage_id_from_bag(bag_path)
        
        # Sort annotations chronologically
        sorted_ann = sorted(annotations, key=lambda a: a["start_time"])
        
        for idx, ann in enumerate(sorted_ann):
            start_t = float(ann["start_time"])
            end_t = float(ann["end_time"])
            task = ann["task"]
            
            conversion_status["progress"] = idx
            conversion_status["message"] = f"Converting episode {idx+1}/{len(sorted_ann)}: '{task}'"
            
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr"
            )
            reader.open(storage_options, converter_options)
            
            topic_types = {}
            for topic_info in reader.get_all_topics_and_types():
                try:
                    topic_types[topic_info.name] = get_message(topic_info.type)
                except Exception:
                    pass
                    
            bag_to_config_map = {}
            for bag_topic in topic_types.keys():
                bag_norm = bag_topic.strip("/")
                for config_topic in config.topics.keys():
                    config_norm = config_topic.strip("/")
                    if bag_norm == config_norm:
                        bag_to_config_map[bag_topic] = config_topic
                        break
                        
            # Determine starting timestamp
            metadata_path = os.path.join(bag_path, "metadata.yaml")
            with open(metadata_path, "r") as f:
                meta = yaml.safe_load(f)
            bag_info = meta.get("rosbag2_bagfile_information", {})
            start_time_ns = bag_info.get("starting_time", {}).get("nanoseconds_since_epoch", 0)
            
            # Seek to start_t - 1.0s to accumulate initial values
            start_ts_ns = start_time_ns + int(start_t * 1e9)
            seek_ts_ns = max(start_time_ns, start_ts_ns - int(1.0 * 1e9))
            reader.seek(seek_ts_ns)
            
            latest_msgs = {}
            t_next_sample = start_t
            episode_frames = []
            
            while reader.has_next():
                topic_name, data, timestamp = reader.read_next()
                t_sec = (timestamp - start_time_ns) / 1e9
                
                if t_sec > end_t:
                    break
                    
                if topic_name in bag_to_config_map:
                    config_topic = bag_to_config_map[topic_name]
                    topic_converter = config.topics[config_topic]
                    try:
                        msg_class = topic_types[topic_name]
                        msg = deserialize_message(data, msg_class)
                        tensor = topic_converter.to_tensor(msg)
                        latest_msgs[config_topic] = tensor
                    except Exception as e:
                        print(f"Error converting message: {e}")
                        
                if t_sec >= start_t:
                    while t_sec >= t_next_sample:
                        if t_next_sample <= end_t:
                            frame = assembler.assemble(latest_msgs)
                            episode_frames.append((frame, task, t_next_sample))
                        t_next_sample += 1.0 / config.fps
                        
            while t_next_sample <= end_t:
                frame = assembler.assemble(latest_msgs)
                episode_frames.append((frame, task, t_next_sample))
                t_next_sample += 1.0 / config.fps
                
            if len(episode_frames) > 0:
                writer.save_episode(episode_frames, task)
                
        conversion_status["message"] = "Finalizing LeRobot Dataset..."
        writer.finalize()
        
        conversion_status["running"] = False
        conversion_status["progress"] = len(annotations)
        conversion_status["message"] = "Dataset created successfully!"
        
    except Exception as e:
        traceback.print_exc()
        conversion_status["running"] = False
        conversion_status["message"] = f"Conversion failed: {str(e)}"

@app.post("/api/convert")
def start_convert(req: ConvertRequest, background_tasks: BackgroundTasks):
    if conversion_status["running"]:
        raise HTTPException(status_code=400, detail="A conversion task is already running")
    
    background_tasks.add_task(
        run_conversion_task,
        req.bag_path,
        req.config,
        req.repo_id,
        req.resume,
        [ann.model_dump() for ann in req.annotations]
    )
    return {"status": "started"}

@app.get("/api/status")
def get_status():
    return conversion_status

# Serve static files from the gui directory at the root URL
gui_dir = os.path.join(os.path.dirname(__file__), "gui")
app.mount("/", StaticFiles(directory=gui_dir, html=True), name="gui")

def main():
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Start LeRobot Bag Annotation Server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args(args=sys.argv[1:] if len(sys.argv) > 1 else [])
    
    print(f"Starting LeRobot Bag Annotation Server on http://{args.host}:{args.port}")
    uvicorn.run("lerobot_ros.annotation_server:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
