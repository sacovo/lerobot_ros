#!/usr/bin/env python3
import os
import json
import time
import base64
import typing
import logging
import threading
import traceback
import numpy as np
import torch
import cv2
import yaml

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from lerobot_ros.config import load_toml_dict, parse_config
from lerobot_ros.convert.image import ros_image_to_numpy, ImageTopic, ImageCompressedTopic, _rotate_expand
from lerobot_ros.bag_to_dataset import get_storage_id_from_bag
from lerobot_ros.core import FrameAssembler, DatasetWriter

# Module-level Argument Parsing (ROS-safe)
import argparse
parser = argparse.ArgumentParser(description="Start LeRobot Bag Annotation Server.")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Binding host")
parser.add_argument("--port", type=int, default=8000, help="Server port")
parser.add_argument("--bag-root", type=str, default=os.getcwd(), help="Root directory to search for bags")
parser.add_argument("--config", type=str, default="", help="Default TOML configuration path")
parser.add_argument("--reload", action="store_true", help="Enable reload for development")
args, _ = parser.parse_known_args()

# Normalize root path
args.bag_root = os.path.abspath(args.bag_root)

# Initialize FastAPI App and state
app = FastAPI(title="LeRobot Bag Annotator")
app.state.bag_root = args.bag_root
app.state.default_config = args.config

logger = logging.getLogger("lerobot_ros.annotation_server")


@app.exception_handler(StarletteHTTPException)
async def _log_and_handle_http_exception(request, exc: StarletteHTTPException):
    """Log 4xx/5xx detail before returning it.

    FastAPI's default handler only emits the access-log line for an
    HTTPException, so the actual reason (exc.detail) never reaches the server
    log. Log it here, then delegate to the default JSON response so clients
    still receive {"detail": ...}.
    """
    if exc.status_code >= 400:
        logger.warning(
            "%s %s -> %d: %s",
            request.method, request.url.path, exc.status_code, exc.detail,
        )
    return await http_exception_handler(request, exc)

# Path-traversal protection helper
def validate_path_under_root(path: str):
    abs_path = os.path.abspath(path)
    if os.path.commonpath([app.state.bag_root, abs_path]) != app.state.bag_root:
        raise HTTPException(status_code=403, detail="Access denied: path must be inside the bag root directory.")

# Global status tracking for conversion task
conversion_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "Idle",
    "state": "idle", # "idle" | "running" | "done" | "error"
    "cancel_requested": False
}

# Caching for ROS2 Bag Readers
BAG_CACHE = {}

def get_bag_reader_cached(path: str, config_path: str):
    validate_path_under_root(path)
    
    cfg_path = config_path or app.state.default_config
    if not cfg_path:
        raise HTTPException(status_code=400, detail="TOML Configuration path is required (none specified on server or request)")
    if not os.path.exists(cfg_path):
        raise HTTPException(status_code=404, detail=f"TOML configuration file not found: {cfg_path}")
        
    config_mtime = os.path.getmtime(cfg_path)
    cache_key = (path, config_mtime)
    
    if cache_key in BAG_CACHE:
        return BAG_CACHE[cache_key]
        
    entry = setup_bag_reader(path, cfg_path)
    
    if len(BAG_CACHE) > 5:
        oldest_key = next(iter(BAG_CACHE))
        del BAG_CACHE[oldest_key]
        
    BAG_CACHE[cache_key] = entry
    return entry

def setup_bag_reader(path: str, cfg_path: str):
    try:
        cfg = parse_config(load_toml_dict(cfg_path))
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
        topic_type_strs = {}
        for topic_info in reader.get_all_topics_and_types():
            try:
                topic_types[topic_info.name] = get_message(topic_info.type)
                topic_type_strs[topic_info.name] = topic_info.type
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
            cfg_topic_list = ", ".join(sorted(cfg.topics.keys())) or "(none)"
            bag_topic_list = ", ".join(sorted(topic_types.keys())) or "(none resolved)"
            raise HTTPException(
                status_code=400,
                detail=(
                    "No topics in the configuration file match the topics in the loaded bag. "
                    "Check that the TOML config path is correct for this bag.\n"
                    f"Config expects: {cfg_topic_list}\n"
                    f"Bag provides: {bag_topic_list}"
                )
            )
            
        metadata_path = os.path.join(path, "metadata.yaml")
        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)
        bag_info = meta.get("rosbag2_bagfile_information", {})
        start_time_ns = bag_info.get("starting_time", {}).get("nanoseconds_since_epoch", 0)
        duration_sec = bag_info.get("duration", {}).get("nanoseconds", 0) / 1e9
        
        return {
            "reader": reader,
            "topic_types": topic_types,
            "topic_type_strs": topic_type_strs,
            "bag_to_config_map": bag_to_config_map,
            "start_time_ns": start_time_ns,
            "duration_sec": duration_sec,
            "cfg": cfg,
            # Serialize access to the shared SequentialReader: sync endpoints run
            # in a threadpool and could otherwise interleave seeks/reads.
            "lock": threading.Lock(),
            # Incremental "latch" cursor for /api/frame. Mirrors how the dataset
            # export carries the most recent message of every topic forward into
            # every frame. Advanced in place on forward scrubbing/playback (O(1)
            # per frame); reset and replayed from the start only on a backward
            # seek. See _advance_latch.
            "latch_pos_ns": None,          # bag consumed up to (and incl.) here
            "latch_latest": {},            # config_topic -> (msg_data, bag_topic)
            "latch_last_update_ns": {},    # config_topic -> timestamp of that msg
            "latch_pending": None,         # one read-ahead (topic, data, ts)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open/parse bag: {str(e)}")

# Bags List Cache
BAGS_LIST_CACHE = None
BAGS_LIST_CACHE_TIME = 0
BAGS_LIST_CACHE_TTL = 5.0 # Cache list_bags for 5 seconds

def bgr_to_base64_jpeg(bgr_img) -> str:
    """Encode a BGR (OpenCV convention) numpy array as a base64 JPEG data URI."""
    ok, buffer = cv2.imencode(".jpg", bgr_img)
    if not ok:
        raise ValueError("Failed to encode image as JPEG")
    img_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
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

@app.get("/api/config")
def get_server_config():
    return {
        "default_config": app.state.default_config,
        "bag_root": app.state.bag_root
    }

@app.get("/api/bags")
def list_bags():
    global BAGS_LIST_CACHE, BAGS_LIST_CACHE_TIME
    now = time.time()
    if BAGS_LIST_CACHE is not None and (now - BAGS_LIST_CACHE_TIME) < BAGS_LIST_CACHE_TTL:
        return {"bags": BAGS_LIST_CACHE}
        
    bags = []
    root_dir = app.state.bag_root
    for root, dirs, files in os.walk(root_dir):
        depth = root[len(root_dir):].count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        if "metadata.yaml" in files:
            abs_path = os.path.abspath(root)
            rel_path = os.path.relpath(abs_path, root_dir).replace(os.sep, "/")
            bags.append({"path": abs_path, "rel_path": rel_path})

    # Newest-looking (lexicographically largest, e.g. by embedded date) first,
    # with bags in the same subfolder grouped together for the GUI.
    bags.sort(key=lambda b: b["rel_path"], reverse=True)

    BAGS_LIST_CACHE = bags
    BAGS_LIST_CACHE_TIME = now
    return {"bags": bags}

@app.get("/api/bag-info")
def get_bag_info(path: str = Query(..., description="Absolute path to the bag directory"), config: typing.Optional[str] = Query(None, description="Path to the TOML configuration file")):
    validate_path_under_root(path)
    
    cfg_path = config or app.state.default_config
    if not cfg_path:
        raise HTTPException(status_code=400, detail="TOML Configuration path is required (none specified on server or request)")
        
    bag_data = setup_bag_reader(path, cfg_path)
    
    metadata_path = os.path.join(path, "metadata.yaml")
    with open(metadata_path, "r") as f:
        meta = yaml.safe_load(f)
    
    bag_info = meta.get("rosbag2_bagfile_information", {})
    start_time_ns = bag_data["start_time_ns"]
    duration_ns = bag_info.get("duration", {}).get("nanoseconds", 0)
    
    start_time_sec = start_time_ns / 1e9
    end_time_sec = start_time_sec + duration_ns / 1e9
    
    topics = []
    for topic_name, msg_class in bag_data["topic_types"].items():
        msg_count = 0
        for topic_info in bag_info.get("topics_with_message_count", []):
            if topic_info.get("topic_metadata", {}).get("name") == topic_name:
                msg_count = topic_info.get("message_count", 0)
                break
        topics.append({
            "name": topic_name,
            "type": bag_data["topic_type_strs"].get(topic_name, msg_class.__name__ if hasattr(msg_class, "__name__") else str(msg_class)),
            "message_count": msg_count
        })
        
    config_topics = {}
    for t_name, t_obj in bag_data["cfg"].topics.items():
        config_topics[t_name] = {
            "tag": "action" if t_obj.is_action else ("meta" if t_obj.is_meta else "observation")
        }
            
    return {
        "path": path,
        "start_time": start_time_sec,
        "end_time": end_time_sec,
        "duration": duration_ns / 1e9,
        "topics": topics,
        "config_topics": config_topics
    }

def _advance_latch(bag_data: dict, target_ts_ns: int):
    """Return the latched (most-recent-at-or-before target) message and update
    time for every configured topic, mirroring the dataset export.

    The export replays a bag start-to-finish and keeps the last message of each
    topic in ``latest_msgs``, so a sparse, event-driven topic (e.g. a velocity
    "Set" published only when it changes) keeps contributing its held value to
    every frame. This reproduces that here for the annotation preview.

    The scan state lives on ``bag_data`` and is advanced in place: a forward
    request continues reading from where the previous one stopped (O(1) per
    frame during playback); only a backward seek resets and replays from the
    start of the bag. Must be called while holding ``bag_data["lock"]``.
    """
    reader = bag_data["reader"]
    bag_to_config_map = bag_data["bag_to_config_map"]
    start_time_ns = bag_data["start_time_ns"]

    pos = bag_data["latch_pos_ns"]
    if pos is None or target_ts_ns < pos:
        # First request, or scrubbing backwards -> replay from the start.
        reader.seek(start_time_ns)
        bag_data["latch_latest"] = {}
        bag_data["latch_last_update_ns"] = {}
        bag_data["latch_pending"] = None

    latest = bag_data["latch_latest"]
    last_update = bag_data["latch_last_update_ns"]

    def _apply(topic_name, data, ts):
        if topic_name in bag_to_config_map:
            config_topic = bag_to_config_map[topic_name]
            latest[config_topic] = (data, topic_name)
            last_update[config_topic] = ts

    # A message previously read one step past the old target may now be in range.
    pending = bag_data["latch_pending"]
    if pending is not None:
        p_topic, p_data, p_ts = pending
        if p_ts <= target_ts_ns:
            _apply(p_topic, p_data, p_ts)
            bag_data["latch_pending"] = None
        else:
            bag_data["latch_pos_ns"] = target_ts_ns
            return latest, last_update

    while reader.has_next():
        topic_name, data, ts = reader.read_next()
        if ts > target_ts_ns:
            # Remember it so the next forward request doesn't have to re-seek.
            bag_data["latch_pending"] = (topic_name, data, ts)
            break
        _apply(topic_name, data, ts)

    bag_data["latch_pos_ns"] = target_ts_ns
    return latest, last_update


@app.get("/api/frame")
def get_frame(path: str = Query(...), time: float = Query(...), config: typing.Optional[str] = Query(None)):
    validate_path_under_root(path)

    bag_data = get_bag_reader_cached(path, config)
    topic_types = bag_data["topic_types"]
    start_time_ns = bag_data["start_time_ns"]
    cfg = bag_data["cfg"]

    try:
        target_ts_ns = start_time_ns + int(time * 1e9)

        with bag_data["lock"]:
            latest_msgs, last_update_ns = _advance_latch(bag_data, target_ts_ns)
            # Snapshot so decoding happens outside the mutating scan state.
            latest_msgs = dict(latest_msgs)
            last_update_ns = dict(last_update_ns)

        # One dataset-frame period; a value not refreshed within it is being
        # held (latched) into this frame rather than freshly sampled here.
        fps = getattr(cfg, "fps", 0) or 0
        frame_period_ns = int(1e9 / fps) if fps else 0

        cameras = {}
        telemetry = {}
        # Parallel per-topic status the GUI uses to flag held/default values,
        # without changing the {topic: {feature: value}} telemetry shape.
        #   "live"    -> refreshed at (or within one frame of) this timestamp
        #   "held"    -> latched from an earlier message; written to every frame
        #   "default" -> no message yet; the export pads this with zeros
        telemetry_meta = {}
        warnings = []

        for config_topic_name, topic_converter in cfg.topics.items():
            is_image = isinstance(topic_converter, (ImageTopic, ImageCompressedTopic))
            entry = latest_msgs.get(config_topic_name)

            if entry is None:
                # Not yet seen at this time. Images just keep their last frame;
                # numeric topics show the zero pad the export would write, so
                # the pane still lists a value in every frame.
                if not is_image:
                    try:
                        desc = topic_converter.feature_description()
                        names = desc.get("names") or []
                        shape = desc.get("shape", (1,))
                        n = shape[0] if shape else 1
                        telemetry[config_topic_name] = {
                            (names[i] if i < len(names) else f"[{i}]"): 0.0
                            for i in range(n)
                        }
                        telemetry_meta[config_topic_name] = {"status": "default", "age": None}
                    except Exception as e:
                        warnings.append(f"Error building default for {config_topic_name}: {str(e)}")
                continue

            msg_data, bag_topic_name = entry
            msg_class = topic_types[bag_topic_name]
            try:
                msg = deserialize_message(msg_data, msg_class)
                if is_image:
                    if isinstance(topic_converter, ImageCompressedTopic):
                        np_arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
                        bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if bgr_img is None:
                            raise ValueError("Failed to decode compressed image")
                    else:
                        bgr_img = cv2.cvtColor(ros_image_to_numpy(msg), cv2.COLOR_RGB2BGR)

                    if topic_converter.rotate:
                        bgr_img = _rotate_expand(bgr_img, topic_converter.rotate)

                    cameras[config_topic_name] = bgr_to_base64_jpeg(bgr_img)
                else:
                    # Show exactly what the dataset will contain for this topic --
                    # the converter's feature values (after any input transform),
                    # labeled by the dataset feature names -- rather than the full
                    # raw bag message. This keeps the panel in sync with the columns
                    # that actually get written to the LeRobot dataset.
                    tensor = topic_converter.to_tensor(msg)
                    flat = tensor.flatten().tolist()
                    names = topic_converter.feature_description().get("names") or [
                        f"{config_topic_name}[{i}]" for i in range(len(flat))
                    ]
                    telemetry[config_topic_name] = {
                        (names[i] if i < len(names) else f"[{i}]"): flat[i]
                        for i in range(len(flat))
                    }
                    age_ns = target_ts_ns - last_update_ns[config_topic_name]
                    held = frame_period_ns > 0 and age_ns > frame_period_ns
                    telemetry_meta[config_topic_name] = {
                        "status": "held" if held else "live",
                        "age": age_ns / 1e9,
                    }
            except Exception as e:
                warn_msg = f"Error decoding topic {config_topic_name}: {str(e)}"
                print(warn_msg)
                warnings.append(warn_msg)

        return {
            "timestamp": time,
            "cameras": cameras,
            "telemetry": telemetry,
            "telemetry_meta": telemetry_meta,
            "warnings": warnings
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error reading bag at {time}s: {str(e)}")

# Pydantic models for Conversion REST API
class AnnotationSegment(BaseModel):
    start_time: float
    end_time: float
    task: str
    success: typing.Optional[bool] = None  # None means treat as success

class ConvertRequest(BaseModel):
    bag_path: str
    config: typing.Optional[str] = None
    repo_id: str
    resume: bool
    annotations: list[AnnotationSegment]

def _resolve_intervention_topic(bag_topic_types: dict, config_intervention_topic: typing.Optional[str]) -> typing.Optional[str]:
    """Return the bag topic name matching the configured intervention topic, or None."""
    if not config_intervention_topic:
        return None
    norm = config_intervention_topic.strip("/")
    for bag_topic in bag_topic_types.keys():
        if bag_topic.strip("/") == norm:
            return bag_topic
    print(f"Warning: human_intervention_topic '{config_intervention_topic}' not found in bag.")
    return None


def run_conversion_task(bag_path, config_path, repo_id, resume, annotations):
    global conversion_status
    conversion_status["running"] = True
    conversion_status["state"] = "running"
    conversion_status["progress"] = 0
    conversion_status["total"] = len(annotations)
    conversion_status["cancel_requested"] = False
    conversion_status["message"] = "Initializing LeRobot Dataset..."

    try:
        config = parse_config(load_toml_dict(config_path))
        dataset_name = repo_id

        conversion_status["message"] = f"Initializing dataset {dataset_name}..."
        writer = DatasetWriter(dataset_name, config, resume=resume)
        assembler = FrameAssembler(config.topics)

        sorted_ann = sorted(annotations, key=lambda a: a["start_time"])

        for idx, ann in enumerate(sorted_ann):
            if conversion_status.get("cancel_requested", False):
                raise Exception("Cancelled by user")

            start_t = float(ann["start_time"])
            end_t = float(ann["end_time"])
            task = ann["task"]
            # None from the client means the episode succeeded (default)
            success = ann.get("success", None)
            if success is None:
                success = True

            conversion_status["progress"] = idx
            conversion_status["message"] = f"Converting episode {idx+1}/{len(sorted_ann)}: '{task}'"

            bag_data = setup_bag_reader(bag_path, config_path)
            reader = bag_data["reader"]
            topic_types = bag_data["topic_types"]
            bag_to_config_map = bag_data["bag_to_config_map"]
            start_time_ns = bag_data["start_time_ns"]

            intervention_bag_topic = _resolve_intervention_topic(topic_types, config.human_intervention_topic)

            start_ts_ns = start_time_ns + int(start_t * 1e9)
            seek_ts_ns = max(start_time_ns, start_ts_ns - int(1.0 * 1e9))
            reader.seek(seek_ts_ns)

            latest_msgs = {}
            latest_intervention = False  # tracks most recent value of intervention topic
            t_next_sample = start_t
            episode_frames = []

            while reader.has_next():
                if conversion_status.get("cancel_requested", False):
                    raise Exception("Cancelled by user")

                topic_name, data, timestamp = reader.read_next()
                t_sec = (timestamp - start_time_ns) / 1e9

                if t_sec > end_t:
                    break

                # Track human intervention signal
                if intervention_bag_topic and topic_name == intervention_bag_topic:
                    try:
                        msg_class = topic_types[topic_name]
                        msg = deserialize_message(data, msg_class)
                        val = msg.data if hasattr(msg, "data") else False
                        latest_intervention = bool(val)
                    except Exception as e:
                        print(f"Error reading intervention topic: {e}")

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
                            if writer.track_intervention:
                                frame["intervention"] = torch.tensor([1.0 if latest_intervention else 0.0])
                            episode_frames.append((frame, task, t_next_sample))
                        t_next_sample += 1.0 / config.fps

            while t_next_sample <= end_t:
                frame = assembler.assemble(latest_msgs)
                if writer.track_intervention:
                    frame["intervention"] = torch.tensor([0.0])
                episode_frames.append((frame, task, t_next_sample))
                t_next_sample += 1.0 / config.fps

            if len(episode_frames) > 0:
                writer.save_episode(episode_frames, task, success=success)

        conversion_status["message"] = "Finalizing LeRobot Dataset..."
        writer.finalize()

        conversion_status["running"] = False
        conversion_status["progress"] = len(annotations)
        conversion_status["state"] = "done"
        conversion_status["message"] = "Dataset created successfully!"

    except Exception as e:
        traceback.print_exc()
        conversion_status["running"] = False
        if str(e) == "Cancelled by user":
            conversion_status["state"] = "error"
            conversion_status["message"] = "Conversion cancelled by user."
        else:
            conversion_status["state"] = "error"
            conversion_status["message"] = f"Conversion failed: {str(e)}"

@app.post("/api/convert")
def start_convert(req: ConvertRequest, background_tasks: BackgroundTasks):
    validate_path_under_root(req.bag_path)
    
    if conversion_status["running"]:
        raise HTTPException(status_code=400, detail="A conversion task is already running")
        
    cfg_path = req.config or app.state.default_config
    if not cfg_path:
        raise HTTPException(status_code=400, detail="TOML Configuration path is required (none specified on server or request)")
        
    bag_data = setup_bag_reader(req.bag_path, cfg_path)
    bag_duration_sec = bag_data["duration_sec"]
    
    # Validation of annotations
    sorted_ann = sorted(req.annotations, key=lambda a: a.start_time)
    last_end_time = -1.0
    for ann in sorted_ann:
        if ann.start_time < 0.0:
            raise HTTPException(status_code=400, detail=f"Invalid segment start time ({ann.start_time}s): must be non-negative.")
        if ann.end_time <= ann.start_time:
            raise HTTPException(status_code=400, detail=f"Invalid segment ({ann.start_time}s - {ann.end_time}s): end_time must be greater than start_time.")
        if ann.end_time > bag_duration_sec + 0.1:
            raise HTTPException(status_code=400, detail=f"Segment ({ann.start_time}s - {ann.end_time}s) exceeds bag duration ({bag_duration_sec:.2f}s).")
        if ann.start_time < last_end_time:
            raise HTTPException(status_code=400, detail=f"Overlapping segments detected: ({ann.start_time}s - {ann.end_time}s) overlaps with previous segment ending at {last_end_time}s.")
        last_end_time = ann.end_time
        
    background_tasks.add_task(
        run_conversion_task,
        req.bag_path,
        cfg_path,
        req.repo_id,
        req.resume,
        [ann.model_dump() for ann in req.annotations]
    )
    return {"status": "started"}

@app.post("/api/convert/cancel")
def cancel_convert():
    global conversion_status
    if conversion_status["running"]:
        conversion_status["cancel_requested"] = True
        conversion_status["message"] = "Cancellation requested..."
        return {"status": "cancelling"}
    return {"status": "not_running"}

@app.get("/api/status")
def get_status():
    return conversion_status

# Annotation timing persistence (decoupled from the TOML config, so segments
# can be reused after the feature/topic config changes). Stored as a small
# JSON file inside the bag directory next to metadata.yaml.
ANNOTATIONS_FILENAME = "annotations.json"
ANNOTATIONS_VERSION = 1

class SaveAnnotationsRequest(BaseModel):
    bag_path: str
    annotations: list[AnnotationSegment]
    # Optional context, stored for convenience so a reopened file knows which
    # dataset/config it was last built with. Not required to rebuild.
    repo_id: typing.Optional[str] = None
    config: typing.Optional[str] = None

@app.get("/api/annotations")
def get_annotations(path: str = Query(..., description="Absolute path to the bag directory")):
    validate_path_under_root(path)
    ann_path = os.path.join(path, ANNOTATIONS_FILENAME)
    if not os.path.exists(ann_path):
        return {"exists": False, "annotations": []}
    try:
        with open(ann_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read {ANNOTATIONS_FILENAME}: {str(e)}")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"Malformed {ANNOTATIONS_FILENAME}: expected a JSON object.")
    return {
        "exists": True,
        "annotations": data.get("annotations", []),
        "repo_id": data.get("repo_id"),
        "config": data.get("config"),
    }

@app.post("/api/annotations")
def save_annotations(req: SaveAnnotationsRequest):
    validate_path_under_root(req.bag_path)
    if not os.path.isdir(req.bag_path):
        raise HTTPException(status_code=404, detail=f"Bag directory not found: {req.bag_path}")
    ann_path = os.path.join(req.bag_path, ANNOTATIONS_FILENAME)
    payload = {
        "version": ANNOTATIONS_VERSION,
        "repo_id": req.repo_id,
        "config": req.config,
        "annotations": [a.model_dump() for a in req.annotations],
    }
    try:
        with open(ann_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write {ANNOTATIONS_FILENAME}: {str(e)}")
    return {"status": "saved", "path": ann_path, "count": len(req.annotations)}

# Serve static files from the gui directory at the root URL
gui_dir = os.path.join(os.path.dirname(__file__), "gui")
app.mount("/", StaticFiles(directory=gui_dir, html=True), name="gui")

def main():
    import uvicorn
    print(f"Starting LeRobot Bag Annotation Server on http://{args.host}:{args.port} (root: {args.bag_root})")
    uvicorn.run("lerobot_ros.annotation_server:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
