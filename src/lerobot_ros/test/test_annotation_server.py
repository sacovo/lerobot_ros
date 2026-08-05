import os
import shutil
import tempfile
import time
import pytest
import sys

import rosbag2_py
from rclpy.serialization import serialize_message
from std_msgs.msg import Float32, String

# fastapi is only needed by the annotation server (a dev-side tool) and is
# deliberately not part of the rover image venv. Do NOT use a module-level
# pytest.importorskip here: launch_testing's collection hook imports every
# test module during collection, and a Skipped (or ImportError) raised there
# aborts collection of the ENTIRE test directory, silently dropping every
# other test file. Guard the import and skip the tests individually instead.
try:
    from fastapi.testclient import TestClient
    from lerobot_ros.annotation_server import app
except ImportError:
    TestClient = None
    app = None

pytestmark = pytest.mark.skipif(
    TestClient is None, reason="fastapi not installed (lean rover image)")

from test_bag_to_dataset import create_test_bag


def test_annotation_endpoints():
    # Setup temporary directory inside workspace so that scanner finds it
    workspace_test_dir = "/workspaces/ros-fhnw-autonomy"
    temp_dir = tempfile.mkdtemp(dir=workspace_test_dir)
    
    try:
        bag_path = os.path.join(temp_dir, "test_bag")
        config_path = os.path.join(temp_dir, "test_config.toml")
        dataset_root = os.path.join(temp_dir, "datasets")
        
        # Write configuration TOML
        config_content = f"""
        fps = 10
        dataset_root = "{dataset_root}"

        [topics]
        "/test/float" = {{ msg_type = "Float32" }}
        """
        with open(config_path, "w") as f:
            f.write(config_content)
            
        # Create mock bag
        create_test_bag(bag_path, storage_id="mcap")
        
        # Initialize test client
        client = TestClient(app)
        
        # 1. Test GET /api/bags
        response = client.get("/api/bags")
        assert response.status_code == 200
        bags_data = response.json()
        assert "bags" in bags_data
        # Ensure our temporary bag is discovered
        assert any(bag_path in os.path.abspath(b["path"]) for b in bags_data["bags"])
        assert all("rel_path" in b for b in bags_data["bags"])
        
        # 2. Test GET /api/bag-info
        response = client.get(f"/api/bag-info?path={bag_path}")
        assert response.status_code == 200
        info = response.json()
        assert info["path"] == bag_path
        assert info["duration"] == 4.0
        # Check topic list
        topics = {t["name"]: t["type"] for t in info["topics"]}
        assert "/test/float" in topics
        assert "/task" in topics
        
        # 3. Test GET /api/frame
        response = client.get(f"/api/frame?path={bag_path}&time=1.5&config={config_path}")
        assert response.status_code == 200
        frame_data = response.json()
        assert "timestamp" in frame_data
        assert "telemetry" in frame_data
        assert "cameras" in frame_data
        assert "/test/float" in frame_data["telemetry"]
        assert frame_data["telemetry"]["/test/float"]["data"] == 20.0
        
        # 4. Test POST /api/convert
        convert_payload = {
            "bag_path": bag_path,
            "config": config_path,
            "repo_id": "test_gui_dataset",
            "resume": False,
            "annotations": [
                {
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "task": "gui_task_1"
                },
                {
                    "start_time": 3.0,
                    "end_time": 4.0,
                    "task": "gui_task_2"
                }
            ]
        }
        response = client.post("/api/convert", json=convert_payload)
        assert response.status_code == 200
        assert response.json()["status"] == "started"
        
        # 5. Poll GET /api/status until conversion completes
        max_attempts = 15
        for _ in range(max_attempts):
            response = client.get("/api/status")
            assert response.status_code == 200
            status_data = response.json()
            if not status_data["running"]:
                break
            time.sleep(0.5)
            
        assert not status_data["running"]
        assert "successfully" in status_data["message"].lower()
        
        # 6. Verify resulting dataset exists on disk
        dataset_path = os.path.join(dataset_root, "test_gui_dataset")
        assert os.path.exists(dataset_path)

    finally:
        shutil.rmtree(temp_dir)
